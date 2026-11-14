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

// A rig's parts, each a grid of triangles over its premultiplied picture, as RigMetal.swift drew
// them. A mask pass writes each texel's alpha into one slice of an array of coverage images; the
// main pass writes texel * alpha, times the coverage of its mask's slice at its own pixel where
// it has one. Compiled at run time by render/metal.rs.

#include <metal_stdlib>
using namespace metal;

// What each draw is told; render/metal.rs's Uniforms, field for field.
struct Uniforms {
    float2 canvas;
    float alpha;
    float useMask;
    float maskOut;
    uint maskSlice;
};

struct Varyings {
    float4 position [[position]];
    float2 uv;
};

vertex Varyings rig_vertex(
    uint id [[vertex_id]], const device float2 *positions [[buffer(0)]], const device float2 *uv [[buffer(1)]],
    constant Uniforms &u [[buffer(2)]])
{
    Varyings out;
    float2 c = positions[id] / u.canvas * 2.0 - 1.0;
    out.position = float4(c.x, -c.y, 0.0, 1.0);
    out.uv = uv[id];
    return out;
}

fragment float4 rig_fragment(
    Varyings in [[stage_in]], texture2d<float> picture [[texture(0)]], texture2d_array<float> masks [[texture(1)]],
    sampler s [[sampler(0)]], constant Uniforms &u [[buffer(0)]])
{
    float4 c = picture.sample(s, in.uv);
    if (u.maskOut > 0.5) return float4(c.a);
    float m = 1.0;
    if (u.useMask > 0.5) {
        // This fragment's own pixel of its mask, unfiltered.
        m = masks.read(uint2(in.position.xy), u.maskSlice).r;
    }
    return c * (u.alpha * m);
}
