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

// A rig drawn with Metal, as Anime2.5DRig draws one with WebGL (lib/renderer.js): every layer a
// grid of triangles over its picture, moved vertex by vertex on the CPU, in premultiplied alpha.
// The eye whites are drawn first into a mask of their own -- the left one's into red, the right
// one's into green -- which the irises are cut by as they are drawn, so that an iris stays inside
// its white with a smooth edge whatever order the layers are in and however faint the white is.

import CoreGraphics
import Foundation
import Metal
import simd

final class RigRenderer {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let pixelFormat: MTLPixelFormat

    private let draw: MTLRenderPipelineState
    /// Writing a white into the mask's red, or its green.
    private let maskRed: MTLRenderPipelineState
    private let maskGreen: MTLRenderPipelineState
    private let sampler: MTLSamplerState
    private var mask: MTLTexture?

    /// A part as the GPU has it.
    private struct Mesh {
        var texture: MTLTexture
        var uv: MTLBuffer
        var indices: MTLBuffer
        var count: Int
    }
    private var meshes: [Mesh] = []
    private(set) var rig: Rig?

    /// What each draw is told; as the shaders have it.
    private struct Uniforms {
        var canvas: SIMD2<Float>
        var target: SIMD2<Float>
        var maskChannel: SIMD2<Float>
        var alpha: Float
        var useMask: Float
        var maskOut: Float
    }

    private static let shaders = """
        #include <metal_stdlib>
        using namespace metal;

        struct Uniforms {
            float2 canvas;
            float2 target;
            float2 maskChannel;
            float alpha;
            float useMask;
            float maskOut;
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
            Varyings in [[stage_in]], texture2d<float> picture [[texture(0)]], texture2d<float> mask [[texture(1)]],
            sampler s [[sampler(0)]], constant Uniforms &u [[buffer(0)]])
        {
            float4 c = picture.sample(s, in.uv);
            if (u.maskOut > 0.5) return float4(c.a);
            float m = 1.0;
            if (u.useMask > 0.5) {
                constexpr sampler exact(filter::nearest, address::clamp_to_edge);
                m = dot(mask.sample(exact, in.position.xy / u.target).rg, u.maskChannel);
            }
            return c * (u.alpha * m);
        }
        """

    /// A renderer drawing into `pixelFormat`, or nil where there is no GPU to draw with.
    init?(pixelFormat: MTLPixelFormat = .bgra8Unorm, device: MTLDevice? = MTLCreateSystemDefaultDevice()) {
        guard let device, let queue = device.makeCommandQueue(),
            let library = try? device.makeLibrary(source: Self.shaders, options: nil)
        else { return nil }
        self.device = device
        self.queue = queue
        self.pixelFormat = pixelFormat

        let pipeline = { (format: MTLPixelFormat, writes: MTLColorWriteMask) -> MTLRenderPipelineState? in
            let d = MTLRenderPipelineDescriptor()
            d.vertexFunction = library.makeFunction(name: "rig_vertex")
            d.fragmentFunction = library.makeFunction(name: "rig_fragment")
            let colour = d.colorAttachments[0]!
            colour.pixelFormat = format
            colour.writeMask = writes
            colour.isBlendingEnabled = true
            colour.rgbBlendOperation = .add
            colour.alphaBlendOperation = .add
            colour.sourceRGBBlendFactor = .one
            colour.sourceAlphaBlendFactor = .one
            colour.destinationRGBBlendFactor = .oneMinusSourceAlpha
            colour.destinationAlphaBlendFactor = .oneMinusSourceAlpha
            return try? device.makeRenderPipelineState(descriptor: d)
        }
        guard let draw = pipeline(pixelFormat, .all), let maskRed = pipeline(.rgba8Unorm, .red),
            let maskGreen = pipeline(.rgba8Unorm, .green)
        else { return nil }
        self.draw = draw
        self.maskRed = maskRed
        self.maskGreen = maskGreen

        let s = MTLSamplerDescriptor()
        s.minFilter = .linear
        s.magFilter = .linear
        s.mipFilter = .linear
        s.sAddressMode = .clampToEdge
        s.tAddressMode = .clampToEdge
        guard let sampler = device.makeSamplerState(descriptor: s) else { return nil }
        self.sampler = sampler
    }

    /// Hands `rig`'s pictures and grids to the GPU, once: its pictures premultiplied, with mipmaps
    /// for when it is drawn smaller than it is.
    func load(_ rig: Rig) {
        meshes = []
        self.rig = nil
        guard let commands = queue.makeCommandBuffer(), let blit = commands.makeBlitCommandEncoder() else { return }
        for part in rig.parts {
            let (w, h) = (max(1, part.pixels.width), max(1, part.pixels.height))
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: w, height: h, mipmapped: true)
            d.usage = .shaderRead
            guard let texture = device.makeTexture(descriptor: d) else { return }
            var premultiplied = part.pixels.rgba
            if premultiplied.count == w * h * 4 {
                for i in stride(from: 0, to: premultiplied.count, by: 4) {
                    let a = UInt16(premultiplied[i + 3])
                    premultiplied[i] = UInt8((UInt16(premultiplied[i]) * a + 127) / 255)
                    premultiplied[i + 1] = UInt8((UInt16(premultiplied[i + 1]) * a + 127) / 255)
                    premultiplied[i + 2] = UInt8((UInt16(premultiplied[i + 2]) * a + 127) / 255)
                }
            } else {
                premultiplied = [UInt8](repeating: 0, count: w * h * 4)
            }
            texture.replace(region: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0, withBytes: premultiplied, bytesPerRow: w * 4)
            blit.generateMipmaps(for: texture)
            guard
                let uv = device.makeBuffer(
                    bytes: part.mesh.uv, length: max(8, part.mesh.uv.count * MemoryLayout<SIMD2<Float>>.stride)),
                let indices = device.makeBuffer(
                    bytes: part.mesh.indices, length: max(4, part.mesh.indices.count * MemoryLayout<UInt32>.stride))
            else { return }
            meshes.append(Mesh(texture: texture, uv: uv, indices: indices, count: part.mesh.indices.count))
        }
        blit.endEncoding()
        commands.commit()
        self.rig = rig
    }

    /// The canvas fitted into `size` pixels, in the middle of it.
    static func fit(canvas: CGSize, into size: CGSize) -> CGRect {
        let scale = min(size.width / max(1, canvas.width), size.height / max(1, canvas.height))
        let (w, h) = (canvas.width * scale, canvas.height * scale)
        return CGRect(x: (size.width - w) / 2, y: (size.height - h) / 2, width: w, height: h)
    }

    /// Encodes `frame` of the loaded rig into `target`, fitted into it, over `clear`.
    func encode(_ frame: RigFrame, to target: MTLTexture, in commands: MTLCommandBuffer, clear: MTLClearColor) {
        guard let rig, meshes.count == rig.parts.count, frame.positions.count == meshes.count else { return }
        let size = SIMD2<Float>(Float(target.width), Float(target.height))
        let fitted = Self.fit(canvas: CGSize(width: rig.width, height: rig.height), into: CGSize(width: target.width, height: target.height))
        let viewport = MTLViewport(
            originX: Double(fitted.minX), originY: Double(fitted.minY), width: Double(fitted.width),
            height: Double(fitted.height), znear: 0, zfar: 1)

        // Every part's vertices where they are, in one buffer.
        var all: [SIMD2<Float>] = []
        var offsets: [Int] = []
        for positions in frame.positions {
            offsets.append(all.count * MemoryLayout<SIMD2<Float>>.stride)
            all += positions
        }
        guard !all.isEmpty,
            let vertices = device.makeBuffer(bytes: all, length: all.count * MemoryLayout<SIMD2<Float>>.stride)
        else { return }
        var uniforms = Uniforms(
            canvas: SIMD2(Float(rig.width), Float(rig.height)), target: size, maskChannel: .zero, alpha: 1, useMask: 0,
            maskOut: 0)

        // The whites, into the mask, as they are now -- whether they show or not -- where an iris
        // will be drawn that needs one.
        var masked: Set<Side> = []
        let whites = rig.parts.indices.filter { rig.parts[$0].role == .eyewhite && rig.parts[$0].side != nil }
        let clipped = rig.parts.indices.contains { rig.parts[$0].role == .irides && rig.parts[$0].side != nil && frame.alphas[$0] >= 0.004 }
        if !whites.isEmpty, clipped, let mask = maskTexture(width: target.width, height: target.height) {
            let pass = MTLRenderPassDescriptor()
            pass.colorAttachments[0].texture = mask
            pass.colorAttachments[0].loadAction = .clear
            pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
            pass.colorAttachments[0].storeAction = .store
            if let encoder = commands.makeRenderCommandEncoder(descriptor: pass) {
                encoder.setViewport(viewport)
                encoder.setFragmentSamplerState(sampler, index: 0)
                // Not read while the mask is drawn, but bound: never the mask itself, drawn into.
                encoder.setFragmentTexture(meshes[whites[0]].texture, index: 1)
                uniforms.maskOut = 1
                for i in whites {
                    let side = rig.parts[i].side!
                    masked.insert(side)
                    encoder.setRenderPipelineState(side == .left ? maskRed : maskGreen)
                    encodeMesh(i, encoder: encoder, vertices: vertices, offset: offsets[i], uniforms: uniforms)
                }
                encoder.endEncoding()
            }
        }

        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = target
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].clearColor = clear
        pass.colorAttachments[0].storeAction = .store
        guard let encoder = commands.makeRenderCommandEncoder(descriptor: pass) else { return }
        encoder.setRenderPipelineState(draw)
        encoder.setViewport(viewport)
        encoder.setFragmentSamplerState(sampler, index: 0)
        // Bound whether an iris reads it or not.
        encoder.setFragmentTexture(maskTexture(width: target.width, height: target.height), index: 1)
        uniforms.maskOut = 0
        for i in rig.parts.indices where frame.alphas[i] >= 0.004 {
            uniforms.alpha = frame.alphas[i]
            let part = rig.parts[i]
            if part.role == .irides, let side = part.side, masked.contains(side) {
                uniforms.useMask = 1
                uniforms.maskChannel = side == .left ? SIMD2(1, 0) : SIMD2(0, 1)
            } else {
                uniforms.useMask = 0
            }
            encodeMesh(i, encoder: encoder, vertices: vertices, offset: offsets[i], uniforms: uniforms)
        }
        encoder.endEncoding()
    }

    private func encodeMesh(_ i: Int, encoder: MTLRenderCommandEncoder, vertices: MTLBuffer, offset: Int, uniforms: Uniforms) {
        let mesh = meshes[i]
        guard mesh.count > 0 else { return }
        var uniforms = uniforms
        encoder.setVertexBuffer(vertices, offset: offset, index: 0)
        encoder.setVertexBuffer(mesh.uv, offset: 0, index: 1)
        encoder.setVertexBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 2)
        encoder.setFragmentBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 0)
        encoder.setFragmentTexture(mesh.texture, index: 0)
        encoder.drawIndexedPrimitives(
            type: .triangle, indexCount: mesh.count, indexType: .uint32, indexBuffer: mesh.indices, indexBufferOffset: 0)
    }

    /// The mask, the size of what is drawn into.
    private func maskTexture(width: Int, height: Int) -> MTLTexture? {
        if let mask, mask.width == width, mask.height == height { return mask }
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: width, height: height, mipmapped: false)
        d.usage = [.renderTarget, .shaderRead]
        d.storageMode = .private
        mask = device.makeTexture(descriptor: d)
        return mask
    }

    /// `frame` of the loaded rig drawn `width` by `height`, over `background` (premultiplied RGBA),
    /// for a picture rather than the screen. Needs a renderer made for `.rgba8Unorm`.
    func image(_ frame: RigFrame, width: Int, height: Int, background: MTLClearColor = MTLClearColor()) -> CGImage? {
        guard pixelFormat == .rgba8Unorm else { return nil }
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: width, height: height, mipmapped: false)
        d.usage = [.renderTarget, .shaderRead]
        d.storageMode = .managed
        guard let target = device.makeTexture(descriptor: d), let commands = queue.makeCommandBuffer() else { return nil }
        encode(frame, to: target, in: commands, clear: background)
        if let blit = commands.makeBlitCommandEncoder() {
            blit.synchronize(resource: target)
            blit.endEncoding()
        }
        commands.commit()
        commands.waitUntilCompleted()
        var bytes = [UInt8](repeating: 0, count: width * height * 4)
        target.getBytes(&bytes, bytesPerRow: width * 4, from: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0)
        guard let provider = CGDataProvider(data: Data(bytes) as CFData) else { return nil }
        return CGImage(
            width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
            space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue), provider: provider, decode: nil,
            shouldInterpolate: false, intent: .defaultIntent)
    }
}
