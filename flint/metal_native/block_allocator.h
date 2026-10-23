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

#pragma once

#include <stdint.h>

#include <set>
#include <vector>

namespace fl {
namespace op {
namespace metal_native {

struct Chunk;

/// A run of a chunk: in use by one tensor, or free.
struct Block {
  Chunk *chunk = nullptr;
  int64_t offset = 0;
  int64_t size = 0;
  bool free = true;
  // The last batch that used any of these bytes, which whoever is handed them next has to wait
  // for before the host touches them, and the chunk before it goes back.
  uint64_t lastUse = 0;
  Block *prev = nullptr;  // the blocks either side, in the chunk
  Block *next = nullptr;
};

/// One allocation from the device, which blocks are cut from.
struct Chunk {
  void *handle = nullptr;  // the device's buffer, which BlockAllocator never looks at
  int64_t size = 0;
  Block *first = nullptr;
  uint64_t emptiedAt = 0;  // when it last became wholly free, in BlockAllocator::give calls
  // Holds one block at a time: what that block leaves free is nobody else's, and the chunk is
  // handed out whole again only by BlockAllocator::takeDedicated.
  bool dedicated = false;

  bool empty() const {
    return first->free && first->size == size;
  }
};

/// Sub-allocation of large chunks, best fit, with free neighbours merged: the bookkeeping only.
/// What a chunk is, and when it may go back to the device, is the caller's.
///
/// A block is cut from the smallest free block it fits in, and what is left of that block stays
/// free; a block given back joins the free blocks on either side. So the bytes one tensor held can
/// go to any tensor that fits in them, or to two smaller ones, or -- with a neighbour -- to one
/// larger, which a cache of whole buffers cannot do.
///
/// Except in a dedicated chunk, which holds one block at a time and is none of that: its free
/// bytes are never cut from by take(), and it is had again whole, by takeDedicated(), or not at
/// all.
class BlockAllocator {
 public:
  /// Every block and offset is a multiple of this.
  static constexpr int64_t kAlignment = 256;

  BlockAllocator() = default;
  ~BlockAllocator();
  BlockAllocator(const BlockAllocator &) = delete;
  BlockAllocator &operator=(const BlockAllocator &) = delete;

  /// `bytes` rounded up to the alignment.
  static int64_t roundUp(int64_t bytes);

  /// A block of `size` bytes, rounded up, from the free ones, or nullptr if none is large enough.
  /// Never from a dedicated chunk.
  Block *take(int64_t size);

  /// A block of `size` bytes, rounded up, at the start of the smallest empty dedicated chunk of no
  /// more than `largestChunk` bytes it fits in, or nullptr if there is none. What it leaves of the
  /// chunk stays free and is nobody's until the block is given back.
  Block *takeDedicated(int64_t size, int64_t largestChunk);

  /// Adds a chunk of `size` bytes, wholly free -- and, if `dedicated`, kept for one block at a
  /// time. It is not counted as emptied, so the next take that fits uses it before anything else
  /// is given back.
  Chunk *addChunk(void *handle, int64_t size, bool dedicated = false);

  /// Gives a block back, as last used by batch `lastUse`, and merges it with its free neighbours.
  void give(Block *block, uint64_t lastUse);

  /// The chunks with nothing in them, the one emptied longest ago first.
  std::vector<Chunk *> emptyChunks() const;

  /// Forgets a chunk with nothing in it, which the caller then hands back to the device.
  void removeChunk(Chunk *chunk);

  /// The bytes of every chunk, of the blocks in use, and of the chunks with nothing in them.
  int64_t getHeldBytes() const {
    return _heldBytes;
  }
  int64_t getUsedBytes() const {
    return _usedBytes;
  }
  int64_t getEmptyBytes() const;

  /// The latest batch that used any of the chunk's bytes.
  static uint64_t lastUseOf(const Chunk *chunk);

 private:
  struct BySize {
    bool operator()(const Block *a, const Block *b) const;
  };

  std::set<Block *, BySize> _free;
  std::vector<Chunk *> _chunks;
  int64_t _heldBytes = 0;
  int64_t _usedBytes = 0;
  uint64_t _numGiven = 0;

  void split(Block *block, int64_t size);
};

}  // namespace metal_native
}  // namespace op
}  // namespace fl
