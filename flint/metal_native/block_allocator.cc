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

#include <algorithm>

#include "lutil/log.h"

namespace fl {
namespace op {
namespace metal_native {

bool BlockAllocator::BySize::operator()(const Block *a, const Block *b) const {
  // By size, and then by where, so that of two that fit the same the one earlier in an earlier
  // chunk is taken, and the set can hold blocks of one size.
  if (a->size != b->size) return a->size < b->size;
  if (a->chunk != b->chunk) return a->chunk < b->chunk;
  return a->offset < b->offset;
}

BlockAllocator::~BlockAllocator() {
  for (Chunk *chunk : _chunks) {
    Block *block = chunk->first;
    while (block) {
      Block *next = block->next;
      delete block;
      block = next;
    }
    delete chunk;
  }
}

int64_t BlockAllocator::roundUp(int64_t bytes) {
  return (bytes + kAlignment - 1) / kAlignment * kAlignment;
}

Block *BlockAllocator::take(int64_t size) {
  size = roundUp(size);
  CHECK(size > 0);

  // The smallest free block of at least `size`: a key of that size, at the lowest address there
  // can be, sorts before every block of that size.
  Block key;
  key.size = size;
  key.chunk = nullptr;
  key.offset = 0;
  auto it = _free.lower_bound(&key);
  if (it == _free.end()) return nullptr;

  Block *block = *it;
  _free.erase(it);
  split(block, size);
  block->free = false;
  _usedBytes += block->size;
  return block;
}

Block *BlockAllocator::takeDedicated(int64_t size, int64_t largestChunk) {
  size = roundUp(size);
  CHECK(size > 0);

  Chunk *best = nullptr;
  for (Chunk *chunk : _chunks) {
    if (!chunk->dedicated || !chunk->empty()) continue;
    if (chunk->size < size || chunk->size > largestChunk) continue;
    if (!best || chunk->size < best->size) best = chunk;
  }
  if (!best) return nullptr;

  Block *block = best->first;
  split(block, size);
  block->free = false;
  _usedBytes += block->size;
  return block;
}

void BlockAllocator::split(Block *block, int64_t size) {
  if (block->size == size) return;

  Block *rest = new Block();
  rest->chunk = block->chunk;
  rest->offset = block->offset + size;
  rest->size = block->size - size;
  rest->free = true;
  rest->lastUse = block->lastUse;  // the same bytes, used by the same batches
  rest->prev = block;
  rest->next = block->next;
  if (block->next) block->next->prev = rest;
  block->next = rest;
  block->size = size;
  if (!block->chunk->dedicated) _free.insert(rest);
}

Chunk *BlockAllocator::addChunk(void *handle, int64_t size, bool dedicated) {
  CHECK(size > 0 && size % kAlignment == 0);

  Chunk *chunk = new Chunk();
  chunk->handle = handle;
  chunk->size = size;
  chunk->emptiedAt = ++_numGiven;
  chunk->dedicated = dedicated;

  Block *block = new Block();
  block->chunk = chunk;
  block->size = size;
  chunk->first = block;

  _chunks.push_back(chunk);
  if (!dedicated) _free.insert(block);
  _heldBytes += size;
  return chunk;
}

void BlockAllocator::give(Block *block, uint64_t lastUse) {
  CHECK(!block->free);
  _usedBytes -= block->size;
  block->free = true;
  block->lastUse = std::max(block->lastUse, lastUse);

  // Into whichever neighbour is free, the bytes and the batches that used them both.
  auto absorb = [this](Block *into, Block *from) {
    into->size += from->size;
    into->lastUse = std::max(into->lastUse, from->lastUse);
    into->next = from->next;
    if (from->next) from->next->prev = into;
    delete from;
  };
  if (block->next && block->next->free) {
    _free.erase(block->next);
    absorb(block, block->next);
  }
  if (block->prev && block->prev->free) {
    Block *prev = block->prev;
    _free.erase(prev);
    absorb(prev, block);
    block = prev;
  }
  // A dedicated chunk's free bytes are in no list: erasing them from it above found nothing.
  if (!block->chunk->dedicated) _free.insert(block);

  if (block->chunk->empty()) block->chunk->emptiedAt = ++_numGiven;
}

std::vector<Chunk *> BlockAllocator::emptyChunks() const {
  std::vector<Chunk *> empty;
  for (Chunk *chunk : _chunks) {
    if (chunk->empty()) empty.push_back(chunk);
  }
  std::sort(empty.begin(), empty.end(), [](const Chunk *a, const Chunk *b) {
    return a->emptiedAt < b->emptiedAt;
  });
  return empty;
}

void BlockAllocator::removeChunk(Chunk *chunk) {
  CHECK(chunk->empty());
  _free.erase(chunk->first);
  _chunks.erase(std::find(_chunks.begin(), _chunks.end(), chunk));
  _heldBytes -= chunk->size;
  delete chunk->first;
  delete chunk;
}

int64_t BlockAllocator::getEmptyBytes() const {
  int64_t bytes = 0;
  for (Chunk *chunk : _chunks) {
    if (chunk->empty()) bytes += chunk->size;
  }
  return bytes;
}

uint64_t BlockAllocator::lastUseOf(const Chunk *chunk) {
  uint64_t lastUse = 0;
  for (const Block *block = chunk->first; block; block = block->next) {
    lastUse = std::max(lastUse, block->lastUse);
  }
  return lastUse;
}

}  // namespace metal_native
}  // namespace op
}  // namespace fl
