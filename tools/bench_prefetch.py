# The MIT License (MIT)
#
# Copyright (c) 2026 Xiaoyang Chen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""What SDXL costs when its weights live in host memory and are fetched a layer ahead.

The idea being measured: keep the U-Net's weights on the host, and while layer n computes, copy
layer n+1 over PCIe on a second stream. If the copy of the next layer finishes before the current
one does, the transfer is free and only the memory is saved; if it does not, every layer waits.
Which of the two happens is a property of the machine -- the ratio of its PCIe bandwidth to its
arithmetic -- so this measures rather than argues.

Four ways of running the same U-Net forward, all timed the same way:

  resident    every weight on the GPU, which is what the pipeline does today.
  prefetch    weights on pinned host memory, copied a group ahead on a second stream.
  serial      weights on pinned host memory, copied in front of each group on the compute
              stream. No overlap at all, which is what diffusers' sequential CPU offload does,
              and the thing prefetching is meant to beat.
  model       not run: the sum over groups of max(compute, bytes / bandwidth), from the measured
              per-group compute and the measured bandwidth. What perfect overlap would cost.

The granularity is a knob. Whole blocks copy fast but a pair of them barely fits in less memory
than the model itself; single convolutions save the most memory and waste the most bandwidth on
per-copy overhead. `-group-size` sets the largest a group may be, and the sweep prints what each
one costs.
"""

import argparse
import time

import torch
from diffusers import UNet2DConditionModel

BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"


def megabytes(n):
    return n / (1 << 20)


def bytes_of(module, recurse=True):
    return sum(p.numel() * p.element_size() for p in module.parameters(recurse=recurse))


# ---------------------------------------------------------------------------- the machine

def measure_bandwidth(device, pinned=True):
    """Host to device bandwidth, at the sizes a layer of weights actually is.

    Pinned memory is what an asynchronous copy needs -- a pageable copy goes through a staging
    buffer and cannot overlap with compute at all -- so that is the number the rest of this uses.
    """
    results = {}
    for mb in (4, 16, 64, 256):
        n = mb * (1 << 20) // 2
        host = torch.empty(n, dtype=torch.float16, pin_memory=pinned)
        gpu = torch.empty(n, dtype=torch.float16, device=device)

        for _ in range(3):
            gpu.copy_(host, non_blocking=pinned)
        torch.cuda.synchronize()

        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(10):
            gpu.copy_(host, non_blocking=pinned)
        end.record()
        torch.cuda.synchronize()

        seconds = start.elapsed_time(end) / 1e3 / 10
        results[mb] = n * 2 / seconds / 1e9
        del host, gpu
    return results


# ---------------------------------------------------------------------------- the layers

def call_order(unet, inputs):
    """The order the modules are actually called in, discovered by running one forward.

    Reading the order off the module tree would be a guess: `nn.Module`'s children are in
    definition order, which is not always execution order, and a group fetched after it was
    needed is a wrong answer rather than a slow one. So the order comes from the model.
    """
    order = {}
    handles = []

    # A pre-hook that returns something is taken to be returning replacement arguments, so this
    # one is careful to return nothing at all.
    def note(name):
        def hook(module, args):
            order.setdefault(name, len(order))
        return hook

    for name, module in unet.named_modules():
        handles.append(module.register_forward_pre_hook(note(name)))

    with torch.no_grad():
        unet(**inputs)
    for handle in handles:
        handle.remove()
    return order


def group_modules(unet, order, limit_bytes):
    """Cut the model into consecutive groups of at most `limit_bytes` of weights.

    Walks down from the root, taking a module whole when it is small enough and splitting it when
    it is not. What comes back is a partition: every parameter is in exactly one group, and the
    groups are in the order the model runs them.

    A group has to be a module that is actually called, because a hook is what fetches it. That
    rules out the `ModuleList`s -- `down_blocks.2.resnets` holds a third of a gigabyte and would
    make a fine group by size, but nothing calls it, so its hook would never fire and its weights
    would never arrive.
    """
    chosen = []
    stranded = []

    def visit(name, module):
        size = bytes_of(module)
        if size == 0:
            return

        runs = name in order
        leaf = not any(bytes_of(child) for child in module.children())
        if runs and (size <= limit_bytes or leaf):
            chosen.append((name, module))
            return
        if leaf:
            stranded.append((name, module))
            return

        # Split. Anything the module holds directly would be lost by descending, so this asserts
        # rather than silently dropping it; in this U-Net only leaves own parameters.
        direct = sum(p.numel() * p.element_size() for p in module.parameters(recurse=False))
        assert direct == 0, f"{name} owns {direct} bytes and has children too"
        for child_name, child in module.named_children():
            visit(f"{name}.{child_name}" if name else child_name, child)

    visit("", unet)
    chosen.sort(key=lambda pair: order[pair[0]])
    return chosen, stranded


# ---------------------------------------------------------------------------- the offload

class Offload:
    """The weights of one group, held on the host and copied into a slot on the GPU.

    Each group is one flat buffer, so a group is one copy rather than one per tensor: at these
    sizes the per-copy overhead is what decides whether the bandwidth is the bandwidth.

    Page-locked, unless `pinned` says otherwise. The driver's DMA engine reads physical addresses
    and cannot have the pages move underneath it, so a copy out of ordinary pageable memory goes
    through a staging buffer the driver owns -- and it has to fill that buffer before it can
    return, which is what makes such a copy synchronous however it was asked for. Pinning is
    therefore not about bandwidth, which barely moves; it is what makes a copy able to overlap
    with anything at all. `-host pageable` is here to show that rather than assert it.

    A group's slot is its index around a ring, and a group's index never changes, so which slot a
    group lands in is fixed. That is worth more than it looks: the parameter views into the slot
    can be built once here rather than reassigned on every forward, and a forward touches no
    Python at all beyond issuing the copy.
    """

    def __init__(self, name, module, slot, pinned=True):
        self.name = name
        self.module = module
        self.params = list(module.parameters(recurse=True))
        # Remembered here rather than read back off the parameters later. `unbind` replaces
        # `p.data` with an empty tensor, which takes the shape with it, so a `bind` that asked
        # `p.shape` after the first unbind would rebuild every parameter as a 1-D nothing.
        self.shapes = [p.shape for p in self.params]
        self.numels = [p.numel() for p in self.params]
        self.numel = sum(self.numels)
        self.nbytes = self.numel * 2
        self.host = torch.empty(self.numel, dtype=torch.float16, pin_memory=pinned)

        at = 0
        for p, numel in zip(self.params, self.numels):
            self.host[at:at + numel].copy_(p.detach().reshape(-1))
            at += numel
        self.slot = slot
        self.empty = torch.empty(0, dtype=torch.float16, device="cuda")

    def bind(self, arena):
        """Point the parameters at the slot, and let go of what they held on the GPU."""
        at = 0
        for p, shape, numel in zip(self.params, self.shapes, self.numels):
            p.data = arena[at:at + numel].view(shape)
            at += numel

    def unbind(self):
        """Point the parameters away, so that the buffer they were on can be given back.

        Needed only when there is no arena. Dropping the last handle to a buffer is not enough
        while the parameters still point into it -- which is why letting go of the memory costs a
        second pass over every parameter, on top of the one that bound them.
        """
        for p in self.params:
            p.data = self.empty

    def view(self, arena):
        return arena[:self.numel]

    def restore(self, device):
        """Put the weights back on the GPU as ordinary parameters, undoing everything above."""
        at = 0
        for p, shape, numel in zip(self.params, self.shapes, self.numels):
            p.data = self.host[at:at + numel].view(shape).to(device)
            at += numel


def install(unet, groups, arenas, copy_stream, overlap):
    """Hang the fetching off the modules, and give back what starts a forward.

    With `overlap`, copies are issued ahead on `copy_stream` and waited for when each group's turn
    comes. How far ahead is decided by how many slots there are: with S slots, the copy for group
    j may be issued once group j-S has finished computing, so standing at group i it is safe to
    issue everything up to group i+S-1. Two slots is one group of lookahead, which is only enough
    when every group's copy fits inside the group in front of it -- and in a U-Net they do not,
    because a resnet is mostly weights and an attention is mostly arithmetic. More slots let a
    heavy copy borrow time from several groups of compute, which is the whole point.

    Without `overlap`, the copy is issued on the compute stream right in front of the group that
    needs it: the same traffic with none of it hidden.
    """
    slots = len(arenas)
    events = [torch.cuda.Event() for _ in groups]
    handles = []

    def fetch(index):
        if index >= len(groups):
            return
        group = groups[index]
        # Read before the stream is switched. Inside the context below `current_stream()` is the
        # copy stream itself, and waiting on itself is a no-op -- which lets the copy run ahead
        # and overwrite a slot the compute stream is still reading from. It costs nothing and
        # gives a wrong picture, which is the worst way for a benchmark to be wrong.
        compute = torch.cuda.current_stream()
        with torch.cuda.stream(copy_stream):
            # The slot this is about to write was last used by group index-S. Waiting on the
            # compute stream here waits for everything enqueued so far, which is up to and
            # including the group before this one -- and not the group about to run, whose work
            # is enqueued after this returns.
            copy_stream.wait_stream(compute)
            group.view(arenas[group.slot]).copy_(group.host, non_blocking=True)
            events[index].record(copy_stream)

    def before(index):
        if overlap:
            torch.cuda.current_stream().wait_event(events[index])
            fetch(index + slots - 1)
        else:
            group = groups[index]
            group.view(arenas[group.slot]).copy_(group.host, non_blocking=True)

    def hook_for(index):
        def hook(module, args):
            before(index)
        return hook

    for index, group in enumerate(groups):
        handles.append(group.module.register_forward_pre_hook(hook_for(index)))

    def start():
        if overlap:
            for index in range(slots - 1):
                fetch(index)

    return start, handles


def install_without_arena(unet, groups, copy_stream, lookahead, overlap):
    """The same fetching, but with the memory asked for and given back a group at a time.

    What the arena buys, measured by taking it away. Two things go with it:

      - the allocation itself. The caching allocator usually answers from what it already holds,
        but it is still a lookup and a stream-ordered handover on the critical path of every
        group, several hundred times per pass.
      - the binding. With fixed slots a parameter's view into its slot is built once at load and
        never touched again; with a fresh buffer every time, every parameter of every group has
        to be pointed at the new address on every pass. That is a few thousand Python assignments
        per step, and unlike the fetching they cannot overlap with anything.

    What it does not buy is the bound on the memory, which is worth saying because it is the
    obvious thing to expect. S slots do cap the footprint by construction, and nothing here does;
    every call into the GPU is asynchronous, so in principle the host can run the whole forward,
    allocating for all 420 groups, while the GPU is still near the start. It does not happen: the
    buffer is handed back in the forward hook, one group after it was taken, and the allocator is
    free to answer the next group out of the same block. The measured peak without an arena comes
    out below the peak with one, because a slot has to be as large as the largest group while an
    allocation is only as large as the group asking.

    The buffers are kept alive in a dict until the group that reads them has run, because letting
    go earlier would hand the memory back while a copy or a kernel is still using it.
    """
    events = [torch.cuda.Event() for _ in groups]
    live = {}
    handles = []
    device = torch.device("cuda")

    def fetch(index):
        if index >= len(groups) or index in live:
            return
        group = groups[index]
        buffer = torch.empty(group.numel, dtype=torch.float16, device=device)
        live[index] = buffer

        compute = torch.cuda.current_stream()
        with torch.cuda.stream(copy_stream):
            copy_stream.wait_stream(compute)
            buffer.copy_(group.host, non_blocking=True)
            events[index].record(copy_stream)

        # The block was taken on the compute stream and is being written on the copy stream, so
        # the allocator has to be told before it may consider the block idle again.
        buffer.record_stream(copy_stream)

    def before(index):
        if overlap:
            fetch(index)
            torch.cuda.current_stream().wait_event(events[index])
            fetch(index + lookahead)
        else:
            fetch(index)
            torch.cuda.current_stream().wait_event(events[index])

        groups[index].bind(live[index])

    def after(index):
        # The group has run, so the memory may go back -- but only once nothing points at it, and
        # the parameters still do.
        groups[index].unbind()
        live.pop(index, None)

    for index, group in enumerate(groups):
        handles.append(group.module.register_forward_pre_hook(
            (lambda i: lambda m, a: before(i))(index)))
        handles.append(group.module.register_forward_hook(
            (lambda i: lambda m, a, o: after(i))(index)))

    def start():
        live.clear()
        if overlap:
            for index in range(lookahead):
                fetch(index)

    return start, handles


def hook_overhead(unet, modules, inputs, repeats):
    """What the hooks cost by themselves, with every weight already resident.

    A pre-hook per group is Python on the critical path, and at the finer granularities there are
    hundreds of them. That cost belongs to this benchmark rather than to the idea being measured:
    a runtime that fetches from its own scheduler pays none of it. Measured by installing hooks
    that wait on an event that is already behind them, which is the same handful of calls with
    nothing to wait for.
    """
    done = torch.cuda.Event()
    done.record()
    torch.cuda.synchronize()

    def hook(module, args):
        torch.cuda.current_stream().wait_event(done)

    handles = [module.register_forward_pre_hook(hook) for module in modules]
    with torch.no_grad():
        cost = timed(lambda: unet(**inputs), repeats)
    for handle in handles:
        handle.remove()
    return cost


# ---------------------------------------------------------------------------- the timing

def timed(run, repeats, warmup=2):
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(repeats):
        run()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats * 1e3


def per_group_compute(unet, modules, inputs, repeats=3):
    """How long each group's own work takes, with everything resident.

    Timed with events around each group rather than by subtraction, so a group that is mostly
    waiting for the one before it is not credited with the wait.
    """
    totals = [0.0] * len(modules)
    starts = [torch.cuda.Event(True) for _ in modules]
    ends = [torch.cuda.Event(True) for _ in modules]
    handles = []

    def mark(event):
        def hook(*_):
            event.record()
        return hook

    for index, module in enumerate(modules):
        handles.append(module.register_forward_pre_hook(mark(starts[index])))
        handles.append(module.register_forward_hook(mark(ends[index])))

    with torch.no_grad():
        for _ in range(repeats):
            unet(**inputs)
            torch.cuda.synchronize()
            for index in range(len(modules)):
                totals[index] += starts[index].elapsed_time(ends[index])

    for handle in handles:
        handle.remove()
    return [total / repeats for total in totals]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-model", default=BASE_MODEL, help="which SDXL to read the U-Net from.")
    parser.add_argument("-size", type=str, default="1024",
                        help="image sides to try, in pixels, comma separated.")
    parser.add_argument("-batch", type=int, default=2,
                        help="latents per step. Two is what guidance runs.")
    parser.add_argument("-repeats", type=int, default=5, help="timed forwards per measurement.")
    parser.add_argument("-group-size", dest="group_size", type=str, default="1024,256,64",
                        help="largest weight bytes per group, in MB, comma separated.")
    parser.add_argument("-arena", type=str, default="on",
                        help="whether the slots are preallocated and the parameter views built "
                             "once, as on, off, or both. Off asks for the memory a group at a "
                             "time and points every parameter at it again on every pass.")
    parser.add_argument("-host", type=str, default="pinned",
                        help="where the weights wait, as pinned, pageable, or both. Pageable is "
                             "there to show what page-locking buys, which is not bandwidth but "
                             "the ability to overlap at all.")
    parser.add_argument("-slots", type=str, default="2,4,8",
                        help="how many groups fit on the GPU at once, comma separated. S slots "
                             "is S-1 groups of lookahead, and S times the largest group of "
                             "memory.")
    args = parser.parse_args()

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("host to device bandwidth (GB/s):")
    for pinned in (True, False):
        rates = measure_bandwidth(device, pinned)
        row = "  ".join(f"{mb}MB {rate:5.1f}" for mb, rate in rates.items())
        print(f"  {'pinned  ' if pinned else 'pageable'}  {row}")
    bandwidth = max(measure_bandwidth(device, True).values()) * 1e9

    unet = UNet2DConditionModel.from_pretrained(
        args.model, subfolder="unet", variant="fp16", torch_dtype=torch.float16).to(device).eval()
    weights = bytes_of(unet)
    print(f"\nU-Net: {megabytes(weights):.0f} MB of weights")

    for size in [int(x) for x in args.size.split(",")]:
        sweep(unet, device, size, bandwidth, args)


def sweep(unet, device, size, bandwidth, args):
    latent = size // 8
    inputs = {
        "sample": torch.randn(args.batch, 4, latent, latent, dtype=torch.float16, device=device),
        "timestep": torch.tensor(981, device=device),
        "encoder_hidden_states":
            torch.randn(args.batch, 77, 2048, dtype=torch.float16, device=device),
        "added_cond_kwargs": {
            "text_embeds": torch.randn(args.batch, 1280, dtype=torch.float16, device=device),
            "time_ids": torch.zeros(args.batch, 6, dtype=torch.float16, device=device),
        },
    }

    with torch.no_grad():
        run = lambda: unet(**inputs)
        torch.cuda.reset_peak_memory_stats()
        resident = timed(run, args.repeats)
        resident_peak = torch.cuda.max_memory_allocated()

        # What the answer is meant to be. Every mode below computes it again from weights that
        # took a different road to the GPU, and a mode that does not reproduce this is not a
        # cheaper way of running the model, it is a different model.
        reference = unet(**inputs).sample.float()

    print(f"\n{args.batch}x{size}x{size}: resident {resident:.1f} ms/step, "
          f"peak {megabytes(resident_peak):.0f} MB")

    with torch.no_grad():
        order = call_order(unet, inputs)

    print(f"{'group MB':>9} {'groups':>7} {'hooks':>8} {'model':>8} {'arena':>6} {'host':>9} "
          f"{'slots':>6} {'prefetch':>9} {'serial':>8} {'peak MB':>8}")
    for limit_mb in [int(x) for x in args.group_size.split(",")]:
        chosen, stranded = group_modules(unet, order, limit_mb * (1 << 20))
        if stranded:
            print(f"  left resident, never called: {[name for name, _ in stranded]}")

        # Per-group compute is measured while the weights are still resident, so that it is the
        # arithmetic alone and not the arithmetic plus a wait for a copy.
        modules = [module for _, module in chosen]
        compute = per_group_compute(unet, modules, inputs)
        overhead = hook_overhead(unet, modules, inputs, args.repeats)

        copy_stream = torch.cuda.Stream()
        biggest = max(bytes_of(module) for module in modules)
        modelled = sum(max(c, bytes_of(module) / bandwidth * 1e3)
                       for c, module in zip(compute, modules))

        first = True
        rounds = [(int(s), h == "pinned", a)
                  for a in args.arena.split(",")
                  for h in args.host.split(",")
                  for s in args.slots.split(",")]
        for slots, pinned, arena in rounds:
            groups = [Offload(name, module, index % slots, pinned)
                      for index, (name, module) in enumerate(chosen)]
            arenas = [torch.empty(biggest // 2, dtype=torch.float16, device=device)
                      for _ in range(slots)] if arena == "on" else []
            # Either way the parameters have to stop pointing at the resident weights, or the
            # round measures the offloading with the whole model still on the GPU beside it.
            # Binding does that for an arena; without one there is nothing yet to point at, so
            # they are pointed at nothing and each group's own hook binds it when its turn comes.
            for group in groups:
                group.bind(arenas[group.slot]) if arena == "on" else group.unbind()
            torch.cuda.empty_cache()

            rows = {}
            for overlap in (True, False):
                if arena == "on":
                    start, handles = install(unet, groups, arenas, copy_stream, overlap)
                else:
                    start, handles = install_without_arena(
                        unet, groups, copy_stream, slots - 1, overlap)

                def run():
                    start()
                    with torch.no_grad():
                        unet(**inputs)

                torch.cuda.reset_peak_memory_stats()
                rows[overlap] = (timed(run, args.repeats), torch.cuda.max_memory_allocated())

                start()
                with torch.no_grad():
                    got = unet(**inputs).sample.float()
                gap = (got - reference).abs().max().item()
                assert gap == 0.0, (
                    f"{'prefetch' if overlap else 'serial'} at {limit_mb}MB, {slots} slots "
                    f"differs by {gap}")

                for handle in handles:
                    handle.remove()

            head = (f"{limit_mb:>9} {len(chosen):>7} {overhead:7.1f}m {modelled:7.1f}m"
                    if first else " " * 34)
            first = False
            print(f"{head} {arena:>6} "
                  f"{'pinned' if pinned else 'pageable':>9} {slots:>6} "
                  f"{rows[True][0]:8.1f}m {rows[False][0]:7.1f}m "
                  f"{megabytes(rows[True][1]):8.0f}")

            # Put the weights back where they were, so the next run starts from the model rather
            # than from the last arena.
            for group in groups:
                group.restore(device)
            del groups, arenas
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
