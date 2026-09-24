#include <cmath>
#include <vector>

#include "catch2/catch_amalgamated.hpp"
#include "flint/device.h"
#include "flint/operators.h"

namespace fl {

namespace {

/// The Metal operators, which the calls here that run on Metal are asked of.
Operators *metalOps() {
  return getOperators(Device::kMetal);
}

/// The CPU operators, which the calls here that run on CPU are asked of.
Operators *cpuOps() {
  return getOperators(Device::kCpu);
}

Tensor toMetal(const Tensor &a) {
  return metalOps()->cast(metalOps()->toDevice(Device::getMetal(), a), DType::kFloat16);
}

std::vector<float> readFloats(const Tensor &a) {
  // Called on the CPU inputs as well as on what the card gave back, so a tensor already on the
  // CPU stays where it is: the Metal operators take only their own.
  Tensor host = a.getDevice().getType() == Device::kCpu
      ? a
      : metalOps()->toDevice(Device::getCpu(), metalOps()->cast(a, DType::kFloat));
  Tensor c = cpuOps()->contiguous(host);
  const float *data = c.getInternalData()->getData<float>(c.getInternalOffset());
  return std::vector<float>(data, data + c.getNumEl());
}

struct Shape3 { int n, c, l; };

/// The convolution as its definition, so that what this compares is MLX's answer and the layout
/// it is handed rather than two spellings of the same misreading. The transposes either side of
/// the MLX call are exactly where a 1-D convolution can go wrong silently -- channels last and a
/// weight of (K, R, C) both look like a reshape until the numbers come back.
std::vector<float> referenceConv1d(
    const std::vector<float> &x, Shape3 in,
    const std::vector<float> &w, Shape3 filter,
    const std::vector<float> *bias,
    int stride, int padding, int dilation, int groups,
    Shape3 &out) {
  int r = filter.l;
  out = {in.n, filter.n, (in.l + 2 * padding - dilation * (r - 1) - 1) / stride + 1};

  int channelsPerGroup = in.c / groups;
  int filtersPerGroup = filter.n / groups;

  std::vector<float> y(size_t(out.n) * out.c * out.l, 0.0f);
  for (int n = 0; n < out.n; ++n) {
    for (int k = 0; k < out.c; ++k) {
      int group = k / filtersPerGroup;
      for (int ot = 0; ot < out.l; ++ot) {
        float sum = bias ? (*bias)[k] : 0.0f;
        for (int c = 0; c < channelsPerGroup; ++c) {
          int inChannel = group * channelsPerGroup + c;
          for (int kt = 0; kt < r; ++kt) {
            int t = ot * stride - padding + kt * dilation;
            if (t < 0 || t >= in.l) continue;
            sum += x[(size_t(n) * in.c + inChannel) * in.l + t] *
                   w[(size_t(k) * channelsPerGroup + c) * r + kt];
          }
        }
        y[(size_t(n) * out.c + k) * out.l + ot] = sum;
      }
    }
  }
  return y;
}

bool matchesReference(
    Shape3 in, Shape3 filter, bool withBias,
    int stride, int padding, int dilation, int groups) {
  Tensor input = cpuOps()->rand({in.n, in.c, in.l}, DType::kFloat);
  Tensor weight = cpuOps()->rand({filter.n, filter.c, filter.l}, DType::kFloat);
  Tensor bias = withBias ? cpuOps()->rand({filter.n}, DType::kFloat) : Tensor();

  std::vector<float> x = readFloats(input);
  std::vector<float> w = readFloats(weight);
  std::vector<float> b = withBias ? readFloats(bias) : std::vector<float>();

  Shape3 out{};
  std::vector<float> expected = referenceConv1d(
      x, in, w, filter, withBias ? &b : nullptr, stride, padding, dilation, groups, out);

  Tensor got = metalOps()->conv1d(
      toMetal(input), toMetal(weight), withBias ? toMetal(bias) : Tensor(),
      stride, padding, dilation, groups);
  if (got.getShape() != std::vector<int>{out.n, out.c, out.l}) return false;

  // Half throughout, so the tolerance grows with what each output sums over -- the same rule the
  // conv2d test beside this one uses.
  int accums = filter.c * filter.l;
  float tol = std::max(5e-2f, accums * 5e-4f);

  std::vector<float> actual = readFloats(got);
  for (size_t i = 0; i < expected.size(); ++i) {
    if (std::fabs(actual[i] - expected[i]) > tol) return false;
  }
  return true;
}

}  // namespace

CATCH_TEST_CASE("test Metal conv1d (kernel shapes)", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  // A kernel of one, which is a per-sample matrix multiply.
  CATCH_REQUIRE(matchesReference({2, 8, 12}, {4, 8, 1}, false, 1, 0, 1, 1));
  // A kernel of three that keeps its length.
  CATCH_REQUIRE(matchesReference({2, 8, 12}, {4, 8, 3}, false, 1, 1, 1, 1));
  // A kernel that shrinks its input, and one that reaches the end exactly.
  CATCH_REQUIRE(matchesReference({1, 3, 8}, {6, 3, 3}, false, 1, 0, 1, 1));
  CATCH_REQUIRE(matchesReference({1, 4, 5}, {2, 4, 5}, false, 1, 0, 1, 1));
}

CATCH_TEST_CASE("test Metal conv1d (stride, padding and dilation)", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  // Halving the rate, which is how a codec goes down one.
  CATCH_REQUIRE(matchesReference({1, 8, 16}, {16, 8, 5}, true, 2, 2, 1, 1));
  // Padding wider than the kernel reaches, so most of the output sees only zeros.
  CATCH_REQUIRE(matchesReference({1, 2, 3}, {2, 2, 3}, false, 1, 2, 1, 1));
  // Dilation, which is what a conformer's wide receptive field is made of, and both at once.
  CATCH_REQUIRE(matchesReference({1, 4, 20}, {4, 4, 3}, false, 1, 2, 2, 1));
  CATCH_REQUIRE(matchesReference({1, 4, 20}, {4, 4, 3}, true, 2, 2, 4, 1));
}

CATCH_TEST_CASE("test Metal conv1d (groups)", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  CATCH_REQUIRE(matchesReference({2, 8, 10}, {8, 4, 3}, false, 1, 1, 1, 2));
  // The depthwise case -- one group per channel -- which is every grouped convolution
  // IndexTTS-2.5 asks for.
  CATCH_REQUIRE(matchesReference({1, 6, 9}, {6, 1, 3}, true, 1, 1, 1, 6));
  CATCH_REQUIRE(matchesReference({1, 16, 24}, {16, 1, 5}, true, 1, 2, 1, 16));
}

CATCH_TEST_CASE("test Metal conv1d (speech shapes)", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  // What IndexTTS-2.5 actually convolves: w2v-bert's pointwise pair around a depthwise 31, and
  // BigVGAN's widest residual block.
  CATCH_REQUIRE(matchesReference({1, 1024, 200}, {2048, 1024, 1}, false, 1, 0, 1, 1));
  CATCH_REQUIRE(matchesReference({1, 1024, 200}, {1024, 1, 31}, false, 1, 15, 1, 1024));
  CATCH_REQUIRE(matchesReference({1, 512, 128}, {512, 512, 3}, true, 1, 1, 1, 1));
}

}  // namespace fl
