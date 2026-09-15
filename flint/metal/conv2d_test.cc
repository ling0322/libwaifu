#include <cmath>
#include <vector>

#include "catch2/catch_amalgamated.hpp"
#include "flint/device.h"
#include "flint/functional.h"
#include "flint/operators.h"

namespace fl {
namespace {

Tensor toMetal(const Tensor &a) {
  return F::cast(F::toDevice(Device::getMetal(), a), DType::kFloat16);
}

std::vector<float> readFloats(const Tensor &a) {
  Tensor c = F::contiguous(F::toDevice(Device::getCpu(), F::cast(a, DType::kFloat)));
  const float *data = c.getInternalData()->getData<float>(c.getInternalOffset());
  return std::vector<float>(data, data + c.getNumEl());
}

struct Shape4 { int n, c, h, w; };

std::vector<float> referenceConv2d(
    const std::vector<float> &x, Shape4 in,
    const std::vector<float> &w, Shape4 filter,
    const std::vector<float> *bias,
    int stride, int padding,
    Shape4 &out) {
  int r = filter.h, s = filter.w;
  out = {in.n, filter.n,
         (in.h + 2 * padding - r) / stride + 1,
         (in.w + 2 * padding - s) / stride + 1};

  std::vector<float> y(size_t(out.n) * out.c * out.h * out.w, 0.0f);
  for (int n = 0; n < out.n; ++n) {
    for (int k = 0; k < out.c; ++k) {
      for (int oh = 0; oh < out.h; ++oh) {
        for (int ow = 0; ow < out.w; ++ow) {
          float sum = bias ? (*bias)[k] : 0.0f;
          for (int ci = 0; ci < in.c; ++ci) {
            for (int kh = 0; kh < r; ++kh) {
              for (int kw = 0; kw < s; ++kw) {
                int ih = oh * stride - padding + kh;
                int iw = ow * stride - padding + kw;
                if (ih < 0 || ih >= in.h || iw < 0 || iw >= in.w) continue;
                sum += x[((size_t(n) * in.c + ci) * in.h + ih) * in.w + iw] *
                       w[((size_t(k) * in.c + ci) * r + kh) * s + kw];
              }
            }
          }
          y[((size_t(n) * out.c + k) * out.h + oh) * out.w + ow] = sum;
        }
      }
    }
  }
  return y;
}

bool matchesReference(Shape4 in, Shape4 filter, bool withBias, int stride, int padding) {
  Tensor input = F::rand({in.n, in.c, in.h, in.w}, DType::kFloat);
  Tensor weight = F::rand({filter.n, filter.c, filter.h, filter.w}, DType::kFloat);
  Tensor bias = withBias ? F::rand({filter.n}, DType::kFloat) : Tensor();

  std::vector<float> x = readFloats(input);
  std::vector<float> w = readFloats(weight);
  std::vector<float> b = withBias ? readFloats(bias) : std::vector<float>();

  Shape4 out{};
  std::vector<float> expected =
      referenceConv2d(x, in, w, filter, withBias ? &b : nullptr, stride, padding, out);

  Tensor got = F::conv2d(
      toMetal(input), toMetal(weight), withBias ? toMetal(bias) : Tensor(),
      stride, padding, 1, 1);
  if (got.getShape() != std::vector<int>{out.n, out.c, out.h, out.w}) return false;

  int accums = filter.c * filter.h * filter.w;
  float tol = std::max(5e-2f, accums * 5e-4f);

  std::vector<float> actual = readFloats(got);
  for (size_t i = 0; i < expected.size(); ++i) {
    if (std::fabs(actual[i] - expected[i]) > tol) return false;
  }
  return true;
}

}  // namespace

CATCH_TEST_CASE("test Metal conv2d (kernel shapes)", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  // 1x1, which is a per-pixel matrix multiply.
  CATCH_REQUIRE(matchesReference({2, 8, 7, 5}, {4, 8, 1, 1}, false, 1, 0));
  // 3x3 that keeps its resolution.
  CATCH_REQUIRE(matchesReference({2, 8, 7, 5}, {4, 8, 3, 3}, false, 1, 1));
  // A kernel that shrinks its input.
  CATCH_REQUIRE(matchesReference({1, 3, 8, 8}, {6, 3, 3, 3}, false, 1, 0));
  // Kernel that reaches the edge exactly.
  CATCH_REQUIRE(matchesReference({1, 4, 5, 5}, {2, 4, 5, 5}, false, 1, 0));
  // Not square.
  CATCH_REQUIRE(matchesReference({1, 4, 9, 4}, {8, 4, 3, 1}, false, 1, 0));
}

CATCH_TEST_CASE("test Metal conv2d (stride and padding)", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  // Halving the resolution, how a U-Net goes down a level.
  CATCH_REQUIRE(matchesReference({1, 8, 8, 8}, {16, 8, 3, 3}, true, 2, 1));
  // Padding wider than the kernel reaches.
  CATCH_REQUIRE(matchesReference({1, 2, 3, 3}, {2, 2, 3, 3}, false, 1, 2));
}

CATCH_TEST_CASE("test Metal conv2d (bias)", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  CATCH_REQUIRE(matchesReference({2, 4, 6, 6}, {8, 4, 3, 3}, true, 1, 1));
  CATCH_REQUIRE(matchesReference({1, 4, 5, 5}, {4, 4, 1, 1}, true, 1, 0));
}

CATCH_TEST_CASE("test Metal conv2d (SDXL shapes)", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  // The shapes the benchmark runs. A mismatch here and the benchmark measures the wrong thing.
  CATCH_REQUIRE(matchesReference({1, 320, 16, 16}, {320, 320, 3, 3}, true, 1, 1));
  CATCH_REQUIRE(matchesReference({1, 640, 8, 8}, {640, 640, 3, 3}, true, 1, 1));
  CATCH_REQUIRE(matchesReference({1, 320, 16, 16}, {320, 320, 3, 3}, true, 2, 1));
  CATCH_REQUIRE(matchesReference({1, 640, 8, 8}, {640, 640, 1, 1}, true, 1, 0));
}

}  // namespace fl
