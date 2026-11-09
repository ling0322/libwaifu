// The MIT License (MIT)
//
// Copyright (c) 2026 Xiaoyang Chen
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
// of the Software, and to permit persons to whom the Software is furnished to do
// so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include <cuda_runtime.h>

#include "catch2/catch_amalgamated.hpp"
#include "lutil/time.h"
#include "flint/test_functional.h"
#include "flint/cuda/matmul.h"
#include "flint/device.h"
#include "flint/operators.h"

namespace fl {

namespace {

/// The CUDA operators, which the calls here that run on CUDA are asked of.
Operators *cudaOps() {
  return getOperators(Device::kCuda);
}

/// The CPU operators, which the calls here that run on CPU are asked of.
Operators *cpuOps() {
  return getOperators(Device::kCpu);
}

Tensor toCuda(const Tensor &a) {
  return fl::test::cast(cudaOps(), fl::test::toDevice(cudaOps(), a, Device::getCuda()), DType::kFloat16);
}

Tensor toCpu(const Tensor &a) {
  return fl::test::toDevice(cudaOps(), fl::test::cast(cudaOps(), a, DType::kFloat), Device::getCpu());
}

/// A float tensor moved to the device as it stands, which is what the autoencoder's attention
/// multiplies. It goes to cuBLAS through sgemm rather than hgemm, and never to the vector kernel.
Tensor toCudaFloat(const Tensor &a) {
  return fl::test::toDevice(cudaOps(), a, Device::getCuda());
}

}  // namespace

CATCH_TEST_CASE("test CUDA matmul", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  auto runCase = [](std::initializer_list<int> shapeA, std::initializer_list<int> shapeB) {
    Tensor a = fl::test::rand(Device::getCpu(), shapeA, DType::kFloat);
    Tensor b = fl::test::rand(Device::getCpu(), shapeB, DType::kFloat);
    Tensor xr = fl::test::matmul(cpuOps(), a, b.slice(-1, {8, 32}).transpose(-1, -2));

    Tensor y = toCuda(b).slice(-1, {8, 32}).transpose(-1, -2);
    Tensor x = fl::test::matmul(cudaOps(), toCuda(a), y);

    return cpuOps()->allClose(toCpu(x), xr, 5e-2);
  };

  CATCH_REQUIRE(runCase({10, 24}, {40, 64}));
  CATCH_REQUIRE(runCase({5, 10, 24}, {40, 64}));
  CATCH_REQUIRE(runCase({5, 10, 5, 24}, {10, 40, 64}));
}

CATCH_TEST_CASE("test CUDA matmul (2D shapes)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  auto runCase = [](int m, int k, int n) {
    Tensor a = fl::test::rand(Device::getCpu(), {m, k}, DType::kFloat);
    Tensor b = fl::test::rand(Device::getCpu(), {k, n}, DType::kFloat);

    Tensor x = fl::test::matmul(cudaOps(), toCuda(a), toCuda(b));
    CATCH_INFO("m = " << m << ", k = " << k << ", n = " << n);
    CATCH_REQUIRE(x.getShape() == std::vector<int>{m, n});
    return cpuOps()->allClose(toCpu(x), fl::test::matmul(cpuOps(), a, b), 5e-2);
  };

  // A single row or column collapses the output to a vector shape, and k == 1 makes the
  // accumulation a single product. A one-column B is the case that looks like a transposed
  // weight to the gemv dispatch (its leading stride is also 1) without being one.
  CATCH_REQUIRE(runCase(1, 8, 1));
  CATCH_REQUIRE(runCase(1, 8, 16));
  CATCH_REQUIRE(runCase(16, 8, 1));
  CATCH_REQUIRE(runCase(4, 1, 4));
  CATCH_REQUIRE(runCase(1, 1, 1));
  // a single row against a k that the vector kernel's eight-wide load cannot divide.
  CATCH_REQUIRE(runCase(1, 12, 16));
  // shapes that are not multiples of a tile.
  CATCH_REQUIRE(runCase(17, 33, 65));
}

CATCH_TEST_CASE("test CUDA matmul (dispatches a row to gemv)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // A single-row left operand against a transposed weight is the layout Linear::forward feeds
  // during decode, and it is routed to the vector kernel rather than the general GEMM.
  constexpr int K = 64;
  constexpr int N = 40;
  Tensor x = fl::test::rand(Device::getCpu(), {1, K}, DType::kFloat);
  Tensor w = fl::test::rand(Device::getCpu(), {N, K}, DType::kFloat);

  Tensor wT = toCuda(w).transpose(0, 1);
  CATCH_REQUIRE(wT.getStride(0) == 1);

  Tensor actual = fl::test::matmul(cudaOps(), toCuda(x), wT);
  CATCH_REQUIRE(actual.getShape() == std::vector<int>{1, N});
  CATCH_REQUIRE(cpuOps()->allClose(toCpu(actual), fl::test::matmul(cpuOps(), x, w.transpose(0, 1)), 5e-2));
}

CATCH_TEST_CASE("test CUDA matmul (batched)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // Both operands batched, with one and two batch axes: these are the two instantiations of the
  // pointer-array gather that feeds the batched GEMM.
  Tensor a3 = fl::test::rand(Device::getCpu(), {4, 6, 8}, DType::kFloat);
  Tensor b3 = fl::test::rand(Device::getCpu(), {4, 8, 5}, DType::kFloat);
  CATCH_REQUIRE(cpuOps()->allClose(
      toCpu(fl::test::matmul(cudaOps(), toCuda(a3), toCuda(b3))),
      fl::test::matmul(cpuOps(), a3, b3),
      5e-2));

  Tensor a4 = fl::test::rand(Device::getCpu(), {2, 3, 6, 8}, DType::kFloat);
  Tensor b4 = fl::test::rand(Device::getCpu(), {2, 3, 8, 5}, DType::kFloat);
  CATCH_REQUIRE(cpuOps()->allClose(
      toCpu(fl::test::matmul(cudaOps(), toCuda(a4), toCuda(b4))),
      fl::test::matmul(cpuOps(), a4, b4),
      5e-2));

  // a right operand with fewer batch axes is broadcast across the batch.
  Tensor b2 = fl::test::rand(Device::getCpu(), {8, 5}, DType::kFloat);
  CATCH_REQUIRE(cpuOps()->allClose(
      toCpu(fl::test::matmul(cudaOps(), toCuda(a3), toCuda(b2))),
      fl::test::matmul(cpuOps(), a3, b2),
      5e-2));

  // a single-element batch still has to take the batched path rather than degenerating.
  Tensor a1 = fl::test::rand(Device::getCpu(), {1, 6, 8}, DType::kFloat);
  Tensor b1 = fl::test::rand(Device::getCpu(), {1, 8, 5}, DType::kFloat);
  CATCH_REQUIRE(cpuOps()->allClose(
      toCpu(fl::test::matmul(cudaOps(), toCuda(a1), toCuda(b1))),
      fl::test::matmul(cpuOps(), a1, b1),
      5e-2));
}

CATCH_TEST_CASE("test CUDA matmul (strided operands)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // A non-contiguous left operand cannot be flattened into a plain GEMM, so it goes through the
  // batched path instead; the answer must not depend on which one was chosen.
  Tensor a = fl::test::rand(Device::getCpu(), {4, 6, 8}, DType::kFloat);
  Tensor b = fl::test::rand(Device::getCpu(), {8, 5}, DType::kFloat);

  Tensor strided = toCuda(a).slice(1, {1, 5});
  CATCH_REQUIRE(!strided.isContiguous());

  CATCH_REQUIRE(cpuOps()->allClose(
      toCpu(fl::test::matmul(cudaOps(), strided, toCuda(b))),
      fl::test::matmul(cpuOps(), a.slice(1, {1, 5}), b),
      5e-2));
}

CATCH_TEST_CASE("test CUDA matmul (float)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // The same three dispatches the half arm has -- a plain GEMM, a batched left operand folded
  // into one, and both operands batched -- with the transposed slice that makes the strides
  // interesting. Float against the host in float is nearly exact, so the tolerance is tight
  // enough to catch an operand swapped rather than merely a slower kernel.
  auto runCase = [](std::initializer_list<int> shapeA, std::initializer_list<int> shapeB) {
    Tensor a = fl::test::rand(Device::getCpu(), shapeA, DType::kFloat);
    Tensor b = fl::test::rand(Device::getCpu(), shapeB, DType::kFloat);
    Tensor expected = fl::test::matmul(cpuOps(), a, b.slice(-1, {8, 32}).transpose(-1, -2));

    Tensor y = toCudaFloat(b).slice(-1, {8, 32}).transpose(-1, -2);
    Tensor x = fl::test::matmul(cudaOps(), toCudaFloat(a), y);
    CATCH_REQUIRE(x.getDType() == DType::kFloat);

    return cpuOps()->allClose(fl::test::toDevice(cudaOps(), x, Device::getCpu()), expected, 1e-4f);
  };

  CATCH_REQUIRE(runCase({10, 24}, {40, 64}));
  CATCH_REQUIRE(runCase({5, 10, 24}, {40, 64}));
  CATCH_REQUIRE(runCase({5, 10, 5, 24}, {10, 40, 64}));
}

CATCH_TEST_CASE("test CUDA matmul (float takes the shapes gemv would)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // A single row against a transposed weight is the layout the vector kernel exists for, and
  // that kernel is half only. The float arm has to answer the same through the GEMM instead of
  // dispatching to something that cannot read its operands.
  constexpr int K = 64;
  constexpr int N = 40;
  Tensor x = fl::test::rand(Device::getCpu(), {1, K}, DType::kFloat);
  Tensor w = fl::test::rand(Device::getCpu(), {N, K}, DType::kFloat);

  Tensor wT = toCudaFloat(w).transpose(0, 1);
  CATCH_REQUIRE(wT.getStride(0) == 1);

  Tensor actual = fl::test::matmul(cudaOps(), toCudaFloat(x), wT);
  CATCH_REQUIRE(actual.getShape() == std::vector<int>{1, N});
  CATCH_REQUIRE(cpuOps()->allClose(
      fl::test::toDevice(cudaOps(), actual, Device::getCpu()),
      fl::test::matmul(cpuOps(), x, w.transpose(0, 1)),
      1e-4f));
}

CATCH_TEST_CASE("test CUDA matmul (float carries what half cannot)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // The point of the float arm: a product half would return as infinity. 300 * 300 * 4 is
  // 360000, which is past 65504.
  Tensor a = fl::test::rand(Device::getCpu(), {2, 4}, DType::kFloat);
  Tensor b = fl::test::rand(Device::getCpu(), {4, 3}, DType::kFloat);
  a = fl::test::mul(cpuOps(), a, 300.0f);
  b = fl::test::mul(cpuOps(), b, 300.0f);

  Tensor x = fl::test::toDevice(cudaOps(), fl::test::matmul(cudaOps(), toCudaFloat(a), toCudaFloat(b)), Device::getCpu());
  CATCH_REQUIRE(cpuOps()->allClose(x, fl::test::matmul(cpuOps(), a, b), 1e-4f));

  Tensor asHalf = toCpu(fl::test::matmul(cudaOps(), toCuda(a), toCuda(b)));
  CATCH_REQUIRE(!cpuOps()->allClose(asHalf, fl::test::matmul(cpuOps(), a, b), 1e-4f));
}

namespace {

/// `shape` at random, with every element in row 0 (`column` false) or column 0 (`column` true) of
/// each matrix set to `big`.
Tensor loud(std::vector<int> shape, bool column, float big) {
  Tensor x = fl::test::rand(Device::getCpu(), shape, DType::kFloat);
  int rows = shape[shape.size() - 2], cols = shape.back();
  int64_t matrices = x.getNumEl() / (static_cast<int64_t>(rows) * cols);
  float *data = x.getInternalData()->getData<float>(x.getInternalOffset());
  for (int64_t m = 0; m < matrices; ++m) {
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        if ((column ? c : r) == 0) data[(m * rows + r) * cols + c] = big;
      }
    }
  }
  return x;
}

}  // namespace

CATCH_TEST_CASE("test CUDA matmul (half kept in float)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");
  CATCH_REQUIRE(cudaOps()->matmulWidensHalf());

  // Half times half into float, as the composed attention takes its scores: the sum as it was,
  // past what half holds. A's first column is 300 and B's first row 1500, so every product is
  // 450,000 and more; the rest is random. B as it is, and along K -- the transpose of an (N, K),
  // as the keys are -- each as one matrix and as a batch of (2, 3) the way attention hands them
  // over. The K and N are the rungs of CUTLASS's ladder for one matrix: C read 16 bytes at a time
  // (N of 40), 8 (34) and 4 (33), and a K of 75 that no rung reads, which is copied into padding.
  std::vector<std::shared_ptr<op::cuda::MatMul>> backends = {op::cuda::MatMul::createCutlass()};
  try {
    backends.push_back(op::cuda::MatMul::createCublas());
  } catch (const lut::Error &e) {
    CATCH_WARN("cuBLAS is not usable, so only CUTLASS is asked: " << e.what());
  }

  struct Case {
    int k;
    int n;
  };
  for (size_t backend = 0; backend < backends.size(); ++backend) {
    for (Case c : {Case{64, 40}, Case{64, 34}, Case{64, 33}, Case{75, 33}}) {
      for (bool batched : {false, true}) {
        for (bool alongK : {false, true}) {
          CATCH_INFO(
              "backend " << (backend == 0 ? "CUTLASS" : "cuBLAS") << ", K = " << c.k << ", N = "
                         << c.n << ", batched = " << batched << ", B along K = " << alongK);
          std::vector<int> batch = batched ? std::vector<int>{2, 3} : std::vector<int>{};
          std::vector<int> shapeA = batch, shapeB = batch;
          shapeA.insert(shapeA.end(), {70, c.k});
          if (alongK) {
            shapeB.insert(shapeB.end(), {c.n, c.k});
          } else {
            shapeB.insert(shapeB.end(), {c.k, c.n});
          }
          Tensor a = toCuda(loud(shapeA, true, 300.0f));
          Tensor b = toCuda(loud(shapeB, alongK, 1500.0f));
          if (alongK) b = b.transpose(-2, -1);

          // The same operands, rounded to half, multiplied on the host in float.
          Tensor reference =
              fl::test::matmul(cpuOps(), toCpu(a), toCpu(fl::test::contiguous(cudaOps(), b)));

          std::vector<int> shape = a.getShape();
          shape.back() = b.getShape(-1);
          Tensor x = F::empty(Device::getCuda(), shape, DType::kFloat);
          backends[backend]->apply(a, b, x);
          Tensor got = fl::test::toDevice(cudaOps(), x, Device::getCpu());
          const float *data = got.getInternalData()->getData<float>(got.getInternalOffset());
          CATCH_REQUIRE(std::all_of(
              data, data + got.getNumEl(), [](float y) { return std::isfinite(y); }));
          CATCH_REQUIRE(cpuOps()->allClose(got, reference, 1e-4f, 1e-4f));
        }
      }
    }
  }
}

namespace {

/// `shape` of zeros on the CPU, and a pointer to write it through.
Tensor zerosOnCpu(std::vector<int> shape, float **data) {
  Tensor x = fl::test::zeros(Device::getCpu(), shape, DType::kFloat);
  *data = x.getInternalData()->getData<float>(x.getInternalOffset());
  return x;
}

/// Rows of 1 at columns 0 and 1 and 0 elsewhere -- the left operand of a product of 2049. See
/// "test CUDA matmul with a bias rounds once".
Tensor ones01(std::vector<int> shape) {
  float *data;
  Tensor x = zerosOnCpu(shape, &data);
  int cols = shape.back();
  for (int64_t r = 0; r < x.getNumEl() / cols; ++r) data[r * cols] = data[r * cols + 1] = 1.0f;
  return x;
}

/// Matrices whose row 0 is 2048 and row 1 is 1 -- the right operand of a product of 2049.
Tensor rows2048(std::vector<int> shape) {
  float *data;
  Tensor x = zerosOnCpu(shape, &data);
  int rows = shape[shape.size() - 2], cols = shape.back();
  for (int64_t m = 0; m < x.getNumEl() / (static_cast<int64_t>(rows) * cols); ++m) {
    for (int c = 0; c < cols; ++c) {
      data[(m * rows + 0) * cols + c] = 2048.0f;
      data[(m * rows + 1) * cols + c] = 1.0f;
    }
  }
  return x;
}

/// Whether every element of `x`, a half tensor on the device, is `value`.
bool allAre(const Tensor &x, float value) {
  Tensor host = toCpu(x);
  const float *data = host.getInternalData()->getData<float>(host.getInternalOffset());
  return std::all_of(data, data + host.getNumEl(), [value](float y) { return y == value; });
}

}  // namespace

// A product of 2049 in every element, which half cannot hold: it lies halfway between 2048 and
// 2050 and rounds to 2048. With a bias of 1 the sum is 2050, which half holds exactly. Added to
// the product's float sum and rounded once, as torch's addmm does, the answer is 2050; rounded
// after the product and again after the bias, it is 2048.
CATCH_TEST_CASE("test CUDA matmul with a bias rounds once", "[op][cuda]") {
  constexpr int K = 16, N = 136;
  float *ones;
  Tensor host = zerosOnCpu({N}, &ones);
  std::fill(ones, ones + N, 1.0f);
  Tensor bias = toCuda(host);

  // A weight held (n, k) and read transposed, as a linear layer reads it: one row goes to the
  // vector kernel, more to the GEMM, and a stack of them is flattened into one GEMM.
  Tensor weight = toCuda(fl::test::contiguous(cpuOps(), rows2048({K, N}).transpose(0, 1)));
  Tensor weightT = weight.transpose(0, 1);
  for (std::vector<int> shape : {std::vector<int>{1, K}, {3, K}, {130, K}, {2, 5, K}}) {
    CATCH_INFO("A = " << ones01(shape).getShapeString());
    Tensor x = toCuda(ones01(shape));
    CATCH_REQUIRE(allAre(fl::test::matmulAdd(cudaOps(), x, weightT, bias), 2050.0f));
    // What it is without the fusing: the product rounded, then the bias added and rounded again.
    Tensor twice = fl::test::add(cudaOps(), fl::test::matmul(cudaOps(), x, weightT), bias);
    CATCH_REQUIRE(allAre(twice, 2048.0f));

    // A residual of the output's own shape, and one of its last two dimensions -- which a stack
    // of rows flattened into one GEMM has to see in the shape it was handed, not the flat one.
    std::vector<int> out = shape;
    out.back() = N;
    std::vector<std::vector<int>> addends = {out};
    if (out.size() > 2) addends.push_back(std::vector<int>(out.end() - 2, out.end()));
    for (const std::vector<int> &addendShape : addends) {
      float *data;
      Tensor ones = zerosOnCpu(addendShape, &data);
      std::fill(data, data + ones.getNumEl(), 1.0f);
      CATCH_REQUIRE(allAre(fl::test::matmulAdd(cudaOps(), x, weightT, toCuda(ones)), 2050.0f));
    }
  }

  // A batch of right operands, which is the batched GEMM.
  Tensor x = toCuda(ones01({2, 3, K}));
  Tensor batched = toCuda(rows2048({2, K, N}));
  CATCH_REQUIRE(allAre(fl::test::matmulAdd(cudaOps(), x, batched, bias), 2050.0f));
}

CATCH_TEST_CASE("test CUDA matmul with an addend agrees with float", "[op][cuda]") {
  // 40 columns are read as they stand, and a bias goes to the GEMM as a source; 41 are not, and it
  // goes into C first.
  for (int N : {40, 41}) {
    for (std::vector<int> shape : {std::vector<int>{1, 64}, {7, 64}, {200, 64}, {3, 9, 64}}) {
      std::vector<int> out = shape;
      out.back() = N;
      // A bias, a residual of the output's shape, and one broadcast over its leading dimension.
      std::vector<std::vector<int>> addends = {{N}, out};
      if (out.size() > 2) addends.push_back({1, out[1], N});
      for (const std::vector<int> &addendShape : addends) {
        Tensor a = fl::test::rand(Device::getCpu(), shape, DType::kFloat);
        Tensor w = fl::test::rand(Device::getCpu(), {N, 64}, DType::kFloat);
        Tensor c = fl::test::rand(Device::getCpu(), addendShape, DType::kFloat);
        Tensor want =
            fl::test::add(cpuOps(), fl::test::matmul(cpuOps(), a, w.transpose(0, 1)), c);

        CATCH_INFO("A = " << a.getShapeString() << ", addend = " << c.getShapeString());
        Tensor got =
            fl::test::matmulAdd(cudaOps(), toCuda(a), toCuda(w).transpose(0, 1), toCuda(c));
        CATCH_REQUIRE(cpuOps()->allClose(toCpu(got), want, 5e-2));
      }
    }
  }
}

// What the addend costs a linear layer: the product alone, the product and then add (what a biased
// linear was before matmulAdd), and matmulAdd. Half, the weight (n, k) read transposed.
CATCH_TEST_CASE("benchmark CUDA matmul with an addend", "[.][cuda-benchmark]") {
  auto time = [](auto &&run) {
    // CudaOperators::synchronize() is the default, which waits for nothing.
    for (int i = 0; i < 3; ++i) run();
    cudaDeviceSynchronize();
    int repeat = 50;
    double t0 = lut::now();
    for (int i = 0; i < repeat; ++i) run();
    cudaDeviceSynchronize();
    return (lut::now() - t0) / repeat;
  };

  struct Case {
    const char *name;
    int M, N, K;
  };
  std::vector<Case> cases = {
      {"decode 1x1024x1024", 1, 1024, 1024},
      {"decode 1x3072x1024", 1, 3072, 1024},
      {"encoder 179x896x896", 179, 896, 896},
      {"encoder 179x3584x896", 179, 3584, 896},
      {"prefill 1024x2048x2048", 1024, 2048, 2048},
      {"linear 8192x640x640", 8192, 640, 640},
      {"linear 8192x5120x640", 8192, 5120, 640},
  };
  printf("%-26s %10s %12s %12s\n", "", "matmul", "matmul+add", "matmulAdd");
  for (const Case &c : cases) {
    Tensor x = toCuda(fl::test::rand(Device::getCpu(), {c.M, c.K}, DType::kFloat));
    Tensor w = toCuda(fl::test::rand(Device::getCpu(), {c.N, c.K}, DType::kFloat));
    Tensor bias = toCuda(fl::test::rand(Device::getCpu(), {c.N}, DType::kFloat));
    Tensor wT = w.transpose(0, 1);
    Tensor out = F::empty(Device::getCuda(), {c.M, c.N}, DType::kFloat16);
    TensorView rows = bias.view({1, c.N}).expand({c.M, c.N});  // as the graph broadcast it

    double alone = time([&]() { cudaOps()->matmul(x, wT, out); });
    double twice = time([&]() {
      cudaOps()->matmul(x, wT, out);
      cudaOps()->add(out, rows, out);
    });
    double fused = time([&]() { cudaOps()->matmulAdd(x, wT, bias, out); });
    printf(
        "%-26s %8.1f us %10.1f us %10.1f us\n", c.name, alone * 1e6, twice * 1e6, fused * 1e6);
  }
}

}  // namespace fl
