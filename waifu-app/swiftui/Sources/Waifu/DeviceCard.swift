// The MIT License (MIT)
//
// Copyright (c) 2026 Xiaoyang Chen
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software
// and associated documentation files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
// BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// The Device card, under the Model card: where the model runs, and -- for metal_disk_offload --
// how much of it stays on the GPU. Nothing is read from here: the next job reads the model where
// this says, again if it has changed.

import SwiftUI

struct DeviceCard: View {
    @Bindable var session: Session

    var body: some View {
        FormGroup {
            LabeledContent("Device") {
                Picker("Device", selection: $session.device) {
                    ForEach(Session.devices, id: \.self) { Text($0).tag($0) }
                }
                .labelsHidden()
                // The one choice that has to say what it costs: picked without being told, it is
                // a run that looks broken.
                .help(session.device == "metal_disk_offload"
                    ? "Leaves the weights in the model's files and reads each one off the disk as it is used: for a model larger than this Mac's memory, and slower."
                    : "Where the model runs")
            }
            .disabled(session.canStop)
            // Only for the one device that reads weights off the disk: every other keeps all of
            // them on the GPU already.
            if session.device == "metal_disk_offload" {
                SliderRow(
                    label: "vram (GB)", value: keptOnGPU, range: 0...gpuLimit, step: 0.5,
                    decimals: 1,
                    explanation: """
                        How much of the model's weights stay on the GPU, the smallest first. \
                        The rest are read off the disk as they are used, so more is faster.

                        This is the weights alone, not a limit on the run: drawing needs \
                        memory of its own on top of it.

                        Takes effect when the model is next read.
                        """
                )
                .disabled(session.canStop)
            }
        } header: {
            Header("Device")
        }
    }

    /// The most the slider goes to, in gigabytes: the model, where its size is known -- the end of
    /// the slider is all of it -- and this Mac's memory before it is downloaded. In half gigabytes,
    /// rounded up, so that the end is a step.
    private var gpuLimit: Double {
        let bytes = session.model?.bytes ?? ProcessInfo.processInfo.physicalMemory
        return max(0.5, (Double(bytes) / 1e9 * 2).rounded(.up) / 2)
    }

    /// What is kept, as far as the slider goes: a value left from a larger model is all of this one.
    private var keptOnGPU: Binding<Double> {
        Binding(
            get: { min(session.deviceWeightGB, gpuLimit) },
            set: { session.deviceWeightGB = $0 })
    }
}
