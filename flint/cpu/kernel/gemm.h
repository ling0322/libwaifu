// The MIT License (MIT)
//
// Copyright (c) 2023 Xiaoyang Chen
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

#include "lutil/log.h"
#include "lutil/time.h"
#include "flint/cpu/kernel/abstract.h"
#include "flint/cpu/kernel/block.h"
#include "flint/cpu/kernel/cvt.h"
#include "flint/cpu/kernel/gemv.h"

namespace fl {
namespace op {
namespace cpu {
namespace kernel {

// GEMM over the packed-block algorithm. TComp is the type the micro-kernel computes in, TA and TB
// are the types A and B are stored in; when they differ from TComp the conversion happens in
// Pack(), so a fp16 weight is turned into fp32 one KC x NC panel at a time rather than as a whole
// matrix.
template<
    int MC,
    int KC,
    int NC,
    int MR,
    int NR,
    typename TComp,
    CpuMathBackend TYPE,
    Mode MODE,
    typename TA = TComp,
    typename TB = TComp>
class Gemm {
 public:
  Gemm() {
    int packedSize = (MC * KC + KC * NC) * sizeof(TComp);
    _packedBuffer = (TComp *)malloc(packedSize);

    TComp *A = _packedBuffer;
    TComp *B = A + MC * KC;

    _bufferA = Block<TComp>{A, MR, (MC / MR) * KC, MR, false};
    _bufferB = Block<TComp>{B, NR, (NC / NR) * KC, NR, false};
  }

  ~Gemm() {
    free(_packedBuffer);
    _packedBuffer = nullptr;
  }

  void apply(const GemmArgs<TA, TB, TComp> &args) {
    _inputA = Block<TA>{(TA *)args.A, args.lda, args.M, args.K, args.transA};
    _inputB = Block<TB>{(TB *)args.B, args.ldb, args.K, args.N, args.transB};
    _inputC = Block<TComp>{(TComp *)args.C, args.ldc, args.M, args.N, false};

    split0ByNC();
  }

 private:
  TComp *_packedBuffer;

  Block<TComp> _bufferA;
  Block<TComp> _bufferB;

  Block<TA> _inputA;
  Block<TB> _inputB;
  Block<TComp> _inputC;

  void split0ByNC() {
    int nb = _inputB.numCols / NC;
    int nc = _inputB.numCols % NC;

    for (int i = 0; i < nb; ++i) {
      Block<TB> Bn = _inputB.sliceCol(i * NC, NC);
      Block<TComp> Cj = _inputC.sliceCol(i * NC, NC);
      split1ByKC(Bn, Cj);
    }

    if (nc) {
      Block<TB> Bn = _inputB.sliceCol(nb * NC, nc);
      Block<TComp> Cj = _inputC.sliceCol(nb * NC, nc);
      split1ByKC(Bn, Cj);
    }
  }

  void split1ByKC(Block<TB> Bn, Block<TComp> Cj) {
    int kb = Bn.numRows / KC;
    int kc = Bn.numRows % KC;

    for (int i = 0; i < kb; ++i) {
      Block<TB> Bkn = Bn.sliceRow(i * KC, KC);
      Block<TA> Ak = _inputA.sliceCol(i * KC, KC);
      PackedBlock<TComp> Bp = Pack<TB, TComp, MODE>(Bkn, _bufferB, NR);
      split2ByMC(Ak, Bp, Cj);
    }

    if (kc) {
      Block<TB> Bkn = Bn.sliceRow(kb * KC, kc);
      Block<TA> Ak = _inputA.sliceCol(kb * KC, kc);
      PackedBlock<TComp> Bp = Pack<TB, TComp, MODE>(Bkn, _bufferB, NR);
      split2ByMC(Ak, Bp, Cj);
    }
  }

  void split2ByMC(Block<TA> Ak, PackedBlock<TComp> Bp, Block<TComp> Cj) {
    int mb = Ak.numRows / MC;
    int mc = Ak.numRows % MC;

    for (int i = 0; i < mb; ++i) {
      Block<TA> Amk = Ak.sliceRow(i * MC, MC);
      Block<TComp> Cij = Cj.sliceRow(i * MC, MC);
      PackedBlock<TComp> Ap = Pack<TA, TComp, MODE>(Amk.t(), _bufferA, MR);
      macroKernel(Ap, Bp, Cij);
    }

    if (mc) {
      Block<TA> Amk = Ak.sliceRow(mb * MC, mc);
      Block<TComp> Cij = Cj.sliceRow(mb * MC, mc);
      PackedBlock<TComp> Ap = Pack<TA, TComp, MODE>(Amk.t(), _bufferA, MR);
      macroKernel(Ap, Bp, Cij);
    }
  }

  // GEMM macro-kernel: A(packed: MC, KC) DOT B(packed: KC, NC) -> C(MC, NC)
  void macroKernel(PackedBlock<TComp> A, PackedBlock<TComp> B, Block<TComp> C) {
    int np = (C.numCols + NR - 1) / NR;
    int mp = (C.numRows + MR - 1) / MR;
    int lastNr = C.numCols % NR;
    int lastMr = C.numRows % MR;

#pragma omp parallel for if (MODE == Mode::OMP) schedule(dynamic, 1)
    for (int i = 0; i < np; ++i) {
      for (int j = 0; j < mp; ++j) {
        int nr = (i != np - 1 || lastNr == 0) ? NR : lastNr;
        int mr = (j != mp - 1 || lastMr == 0) ? MR : lastMr;

        Block<TComp> Aj = A.block(j);
        Block<TComp> Bi = B.block(i);
        Block<TComp> Cji = C.slice(j * MR, i * NR, mr, nr);

        microKernel(Aj, Bi, Cji);
      }
    }
  }

  void microKernel(Block<TComp> A, Block<TComp> B, Block<TComp> C) {
    TComp dataCb[MR * NR];

    if (C.numRows < MR || C.numCols < NR) {
      Block<TComp> Cb{dataCb, NR, MR, NR, false};
      Cb.fillZero();

      Block<TComp> Cbs = Cb.slice(0, 0, C.numRows, C.numCols);
      C.copyTo(Cbs);

      gemmKernel<TComp, TComp, TComp, MR, NR, TYPE>(A.numRows, A.data, B.data, Cb.data, Cb.stride);
      Cbs.copyTo(C);
    } else {
      gemmKernel<TComp, TComp, TComp, MR, NR, TYPE>(A.numRows, A.data, B.data, C.data, C.stride);
    }
  }
};

/// @brief Provides GEMM interface with dispatcher for GEMM/GEMV.
template<int MC, int KC, int NC, int MR, int NR, typename T, CpuMathBackend TYPE, Mode MODE>
void gemm(const GemmArgs<T, T, T> &args) {
  if (args.M == 1) {
    std::fill(args.C, args.C + args.N, 0.0f);

    gemv<T, T, T, TYPE, MODE>(GemvArgs<T, T, T>{
        !args.transB,
        args.transB ? args.N : args.K,
        args.transB ? args.K : args.N,
        args.B,
        args.ldb,
        args.A,
        args.transA ? args.lda : 1,
        args.C,
        1});
  } else if (args.N == 1) {
    bool needPackC = args.ldc != 1;
    if (args.ldc != 1) {
      NOT_IMPL();
    } else {
      std::fill(args.C, args.C + args.M, 0.0f);
    }

    gemv<T, T, T, TYPE, MODE>(GemvArgs<T, T, T>{
        args.transA,
        args.transA ? args.K : args.M,
        args.transA ? args.M : args.K,
        args.A,
        args.lda,
        args.B,
        args.transB ? 1 : args.ldb,
        args.C,
        args.ldc});
  } else {
    Gemm<MC, KC, NC, MR, NR, T, TYPE, MODE>().apply(args);
  }
}

/// @brief GEMM with A (activation) in T and B (weight) in TW, computed in T. TW is converted to T
///        while packing, so B is never materialized as a whole T matrix.
template<
    int MC,
    int KC,
    int NC,
    int MR,
    int NR,
    typename T,
    typename TW,
    CpuMathBackend TYPE,
    Mode MODE>
void wgemm(const GemmArgs<T, TW, T> &args) {
  if (args.M == 1) {
    // fill C with zero.
    std::fill(args.C, args.C + args.N, 0.0f);

    gemv<TW, T, T, TYPE, MODE>(GemvArgs<TW, T, T>{
        !args.transB,
        args.transB ? args.N : args.K,
        args.transB ? args.K : args.N,
        args.B,
        args.ldb,
        args.A,
        args.transA ? args.lda : 1,
        args.C,
        1});
  } else {
    Gemm<MC, KC, NC, MR, NR, T, TYPE, MODE, T, TW>().apply(args);
  }
}

}  // namespace kernel
}  // namespace cpu
}  // namespace op
}  // namespace fl
