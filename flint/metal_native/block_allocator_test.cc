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

#include "flint/metal_native/block_allocator.h"

#include "catch2/catch_amalgamated.hpp"

// The bookkeeping alone, with no device: chunks are named by a number rather than a buffer.

namespace fl {
namespace op {
namespace metal_native {

namespace {

void *handle(intptr_t n) {
  return reinterpret_cast<void *>(n);
}

}  // namespace

CATCH_TEST_CASE("BlockAllocator cuts blocks from a chunk and merges them back", "[metal-alloc]") {
  BlockAllocator blocks;
  CATCH_REQUIRE(blocks.take(256) == nullptr);

  Chunk *chunk = blocks.addChunk(handle(1), 4096);
  Block *a = blocks.take(1000);  // rounded up to 1024
  Block *b = blocks.take(1024);
  Block *c = blocks.take(1024);
  CATCH_REQUIRE((a && b && c));
  CATCH_CHECK(a->size == 1024);
  CATCH_CHECK(a->offset == 0);
  CATCH_CHECK(b->offset == 1024);
  CATCH_CHECK(c->offset == 2048);
  CATCH_CHECK(blocks.getUsedBytes() == 3072);
  CATCH_CHECK(!chunk->empty());

  // Two neighbours given back are one free run, which a block twice the size fits in.
  blocks.give(a, 0);
  blocks.give(b, 0);
  Block *wide = blocks.take(2048);
  CATCH_REQUIRE(wide);
  CATCH_CHECK(wide->offset == 0);

  // And one block given back is two smaller ones -- once the run that fits exactly, the 1024 at
  // the end of the chunk, has gone first.
  blocks.give(wide, 0);
  Block *tail = blocks.take(1024);
  Block *d = blocks.take(1024);
  Block *e = blocks.take(1024);
  CATCH_REQUIRE((tail && d && e));
  CATCH_CHECK(tail->offset == 3072);
  CATCH_CHECK(d->offset == 0);
  CATCH_CHECK(e->offset == 1024);

  blocks.give(c, 0);
  blocks.give(d, 0);
  blocks.give(e, 0);
  blocks.give(tail, 0);
  CATCH_CHECK(chunk->empty());
  CATCH_CHECK(blocks.getUsedBytes() == 0);
  CATCH_CHECK(blocks.getEmptyBytes() == 4096);
}

CATCH_TEST_CASE("BlockAllocator takes the smallest free block that fits", "[metal-alloc]") {
  BlockAllocator blocks;
  blocks.addChunk(handle(1), 8192);
  Block *a = blocks.take(4096);
  Block *gap = blocks.take(256);
  Block *b = blocks.take(1024);
  Block *rest = blocks.take(2816);  // the remainder of the chunk
  CATCH_REQUIRE((a && gap && b && rest));
  CATCH_CHECK(blocks.take(256) == nullptr);

  // Free runs of 4096 and 1024, apart: a block of 1024 goes in the smaller.
  blocks.give(a, 0);
  blocks.give(b, 0);
  Block *small = blocks.take(1024);
  CATCH_REQUIRE(small);
  CATCH_CHECK(small->offset == 4096 + 256);
}

CATCH_TEST_CASE("BlockAllocator carries the last use with the bytes", "[metal-alloc]") {
  BlockAllocator blocks;
  Chunk *chunk = blocks.addChunk(handle(1), 4096);
  Block *a = blocks.take(1024);
  Block *b = blocks.take(1024);
  blocks.give(a, 7);

  // The bytes a had, handed out again: whoever has them waits for batch 7.
  Block *again = blocks.take(512);
  CATCH_REQUIRE(again);
  CATCH_CHECK(again->offset == 0);
  CATCH_CHECK(again->lastUse == 7);

  blocks.give(again, 3);
  blocks.give(b, 5);
  CATCH_CHECK(chunk->empty());
  CATCH_CHECK(BlockAllocator::lastUseOf(chunk) == 7);
}

CATCH_TEST_CASE("BlockAllocator lists empty chunks oldest first and forgets them", "[metal-alloc]") {
  BlockAllocator blocks;
  Chunk *first = blocks.addChunk(handle(1), 1024);
  Chunk *second = blocks.addChunk(handle(2), 1024);
  Block *a = blocks.take(1024);
  Block *b = blocks.take(1024);
  CATCH_REQUIRE((a && b));
  CATCH_CHECK(blocks.emptyChunks().empty());

  // Emptied in the other order from how they were made.
  Chunk *emptiedFirst = b->chunk;
  blocks.give(b, 0);
  blocks.give(a, 0);
  std::vector<Chunk *> empty = blocks.emptyChunks();
  CATCH_REQUIRE(empty.size() == 2);
  CATCH_CHECK(empty[0] == emptiedFirst);

  blocks.removeChunk(first);
  blocks.removeChunk(second);
  CATCH_CHECK(blocks.getHeldBytes() == 0);
  CATCH_CHECK(blocks.take(256) == nullptr);
}

CATCH_TEST_CASE("BlockAllocator keeps a dedicated chunk for one block", "[metal-alloc]") {
  BlockAllocator blocks;
  Chunk *big = blocks.addChunk(handle(1), 10240, true);

  // Its bytes are not the free ones: take() does not see them, before or after it is used.
  CATCH_CHECK(blocks.take(256) == nullptr);
  Block *a = blocks.takeDedicated(9000, 10240);  // rounded up to 9216
  CATCH_REQUIRE(a);
  CATCH_CHECK(a->chunk == big);
  CATCH_CHECK(a->offset == 0);
  CATCH_CHECK(a->size == 9216);
  CATCH_CHECK(blocks.getUsedBytes() == 9216);
  CATCH_CHECK(blocks.take(256) == nullptr);
  CATCH_CHECK(blocks.takeDedicated(256, 10240) == nullptr);

  // Given back, it is empty and whole again -- and still nobody's but takeDedicated's.
  blocks.give(a, 0);
  CATCH_CHECK(big->empty());
  CATCH_CHECK(blocks.take(256) == nullptr);

  // Only to a block that asks for a chunk no larger than this one.
  CATCH_CHECK(blocks.takeDedicated(4096, 4096 * 5 / 4) == nullptr);
  Block *b = blocks.takeDedicated(8192, 8192 * 5 / 4);
  CATCH_REQUIRE(b);
  CATCH_CHECK(b->chunk == big);
  blocks.give(b, 0);

  // The smallest of those that fit, and the ordinary chunks are untouched by it.
  Chunk *bigger = blocks.addChunk(handle(2), 12288, true);
  Chunk *ordinary = blocks.addChunk(handle(3), 4096);
  Block *c = blocks.takeDedicated(9216, 12288);
  CATCH_REQUIRE(c);
  CATCH_CHECK(c->chunk == big);
  Block *d = blocks.takeDedicated(9216, 12288);
  CATCH_REQUIRE(d);
  CATCH_CHECK(d->chunk == bigger);
  Block *e = blocks.take(4096);
  CATCH_REQUIRE(e);
  CATCH_CHECK(e->chunk == ordinary);

  blocks.give(c, 0);
  blocks.give(d, 0);
  blocks.give(e, 0);
  CATCH_CHECK(blocks.emptyChunks().size() == 3);
  for (Chunk *chunk : blocks.emptyChunks()) blocks.removeChunk(chunk);
  CATCH_CHECK(blocks.getHeldBytes() == 0);
}

}  // namespace metal_native
}  // namespace op
}  // namespace fl
