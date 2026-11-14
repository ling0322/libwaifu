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

// The Swift face of libwaifu2d's C API (waifu2d.h, docs/waifu2d.md): a drawing in layers rigged
// to move, the rig moving on a stage, and a renderer drawing the stage into a CAMetalLayer or a
// picture. Everything the rig does is the library's; this only holds its handles.

import CWaifu
import Foundation
import QuartzCore

public enum Waifu2D {
    /// What libwaifu2d said went wrong.
    public struct Failure: Error, LocalizedError, Equatable {
        public var message: String
        public var errorDescription: String? { message }
    }

    /// Why the last call on this thread failed.
    static func failure() -> Failure {
        Failure(message: waifu2d_last_error().map { String(cString: $0) } ?? "libwaifu2d refused")
    }

    /// The version the library was built with, and whether it is one this was written for.
    public static var abiVersion: UInt32 { waifu2d_abi_version() }
    public static var isCompatible: Bool { abiVersion >> 16 == UInt32(WAIFU2D_ABI_VERSION_MAJOR) }

    /// A drawing in layers, read and rigged. Never changed, so it may be handed between threads.
    public final class Rig: @unchecked Sendable {
        let handle: OpaquePointer

        /// Reads and rigs the Photoshop document at `url`. Slow: call it off the main thread.
        public init(contentsOf url: URL) throws {
            guard let handle = url.withUnsafeFileSystemRepresentation({ path in path.flatMap { waifu2d_rig_open($0) } })
            else { throw Waifu2D.failure() }
            self.handle = handle
        }

        /// A document already in memory.
        public init(psd: Data) throws {
            let made = psd.withUnsafeBytes { bytes in
                waifu2d_rig_read(bytes.bindMemory(to: UInt8.self).baseAddress, bytes.count)
            }
            guard let made else { throw Waifu2D.failure() }
            handle = made
        }

        deinit { waifu2d_rig_free(handle) }

        private func attribute(_ which: Waifu2dRigAttribute) -> Int {
            var value: Int64 = 0
            return waifu2d_rig_get_int_attribute(handle, which, &value) ? Int(value) : 0
        }

        public var width: Int { attribute(WAIFU2D_RIG_WIDTH) }
        public var height: Int { attribute(WAIFU2D_RIG_HEIGHT) }
        /// How many layers it draws.
        public var partCount: Int { attribute(WAIFU2D_RIG_PART_COUNT) }
        /// How many of its eyes blink, 0 to 2.
        public var eyeCount: Int { attribute(WAIFU2D_RIG_EYE_COUNT) }
        /// What it could not do, to say under the picture.
        public var notes: [String] {
            (0..<attribute(WAIFU2D_RIG_NOTE_COUNT)).compactMap { i in waifu2d_rig_note(handle, i).map { String(cString: $0) } }
        }
    }

    /// What moves on its own.
    public enum Switch: Sendable {
        case followPointer, idle, lookAround, physics, blink, wind, autoTalk, gusts

        var c: Waifu2dSwitch {
            switch self {
            case .followPointer: WAIFU2D_SWITCH_FOLLOW_POINTER
            case .idle: WAIFU2D_SWITCH_IDLE
            case .lookAround: WAIFU2D_SWITCH_LOOK_AROUND
            case .physics: WAIFU2D_SWITCH_PHYSICS
            case .blink: WAIFU2D_SWITCH_BLINK
            case .wind: WAIFU2D_SWITCH_WIND
            case .autoTalk: WAIFU2D_SWITCH_AUTO_TALK
            case .gusts: WAIFU2D_SWITCH_GUSTS
            }
        }
    }

    /// What the settings say, each a number in its range (waifu2d.h has them).
    public enum Control: Sendable {
        case eyesOpen, closesAt, nod, bodyFollows, talkLevel, mouthOpen, smile, brows, twist, winkLeft, winkRight,
            windStrength

        var c: Waifu2dControl {
            switch self {
            case .eyesOpen: WAIFU2D_CONTROL_EYES_OPEN
            case .closesAt: WAIFU2D_CONTROL_CLOSES_AT
            case .nod: WAIFU2D_CONTROL_NOD
            case .bodyFollows: WAIFU2D_CONTROL_BODY_FOLLOWS
            case .talkLevel: WAIFU2D_CONTROL_TALK_LEVEL
            case .mouthOpen: WAIFU2D_CONTROL_MOUTH_OPEN
            case .smile: WAIFU2D_CONTROL_SMILE
            case .brows: WAIFU2D_CONTROL_BROWS
            case .twist: WAIFU2D_CONTROL_TWIST
            case .winkLeft: WAIFU2D_CONTROL_WINK_LEFT
            case .winkRight: WAIFU2D_CONTROL_WINK_RIGHT
            case .windStrength: WAIFU2D_CONTROL_WIND_STRENGTH
            }
        }
    }

    /// What it can be told to do once.
    public enum Action: Sendable {
        case blink, jump, shake, nod

        var c: Waifu2dAction {
            switch self {
            case .blink: WAIFU2D_ACTION_BLINK
            case .jump: WAIFU2D_ACTION_JUMP
            case .shake: WAIFU2D_ACTION_SHAKE
            case .nod: WAIFU2D_ACTION_NOD
            }
        }
    }

    /// A rig moving. Used from one thread at a time.
    public final class Stage {
        let handle: OpaquePointer
        public let rig: Rig

        /// `seed` is where its looking around and its blinks come from.
        public init(rig: Rig, seed: UInt64 = UInt64.random(in: 0...UInt64.max)) throws {
            guard let handle = waifu2d_stage_new(rig.handle, seed) else { throw Waifu2D.failure() }
            self.handle = handle
            self.rig = rig
        }

        deinit { waifu2d_stage_free(handle) }

        public func set(_ which: Switch, _ on: Bool) {
            _ = waifu2d_stage_set_switch(handle, which.c, on)
        }

        /// Throws for a value outside the control's range, and leaves it as it was.
        public func set(_ which: Control, _ value: Double) throws {
            guard waifu2d_stage_set_control(handle, which.c, Float(value)) else { throw Waifu2D.failure() }
        }

        public func act(_ action: Action) {
            _ = waifu2d_stage_act(handle, action.c)
        }

        /// The view, in the unit the pointer comes in.
        public func setViewSize(_ size: CGSize) {
            waifu2d_stage_set_view_size(handle, Float(size.width), Float(size.height))
        }

        /// The pointer in the view, origin top left, y down: an AppKit view's y is flipped first.
        public func pointerMoved(to point: CGPoint) {
            waifu2d_stage_pointer_moved(handle, Float(point.x), Float(point.y))
        }

        public func pointerLeft() {
            waifu2d_stage_pointer_left(handle)
        }

        /// True where the press took hold of a part.
        @discardableResult
        public func pointerDown(at point: CGPoint) -> Bool {
            waifu2d_stage_pointer_down(handle, Float(point.x), Float(point.y))
        }

        public func pointerUp() {
            waifu2d_stage_pointer_up(handle)
        }

        /// Moves on by `dt` seconds.
        public func step(by dt: Double) {
            waifu2d_stage_step(handle, dt)
        }
    }

    /// Which API draws.
    public enum API: Sendable, Equatable {
        case platformDefault, cpu, metal, d3d11, vulkan

        var c: Waifu2dApi {
            switch self {
            case .platformDefault: WAIFU2D_API_DEFAULT
            case .cpu: WAIFU2D_API_CPU
            case .metal: WAIFU2D_API_METAL
            case .d3d11: WAIFU2D_API_D3D11
            case .vulkan: WAIFU2D_API_VULKAN
            }
        }

        public var isAvailable: Bool { waifu2d_api_is_available(c) }
    }

    /// One backend drawing into one surface, on the thread the surface belongs to.
    public final class Renderer {
        let handle: OpaquePointer
        /// Kept as long as the renderer draws into it.
        private let layer: CAMetalLayer?
        public private(set) var pixelSize: (width: Int, height: Int)

        /// Draws into `layer`, which takes the renderer's device and pixel format.
        public init(layer: CAMetalLayer, api: API = .metal) throws {
            var surface = Waifu2dSurface(
                kind: WAIFU2D_SURFACE_METAL_LAYER, handle: Unmanaged.passUnretained(layer).toOpaque(), handle2: nil,
                width: 0, height: 0)
            guard let handle = waifu2d_renderer_new(api.c, &surface) else { throw Waifu2D.failure() }
            self.handle = handle
            self.layer = layer
            pixelSize = (0, 0)
        }

        /// Draws into a picture `width` by `height`, read back with `rgba()`.
        public init(width: Int, height: Int, api: API = .platformDefault) throws {
            var surface = Waifu2dSurface(
                kind: WAIFU2D_SURFACE_OFFSCREEN, handle: nil, handle2: nil, width: UInt32(width), height: UInt32(height))
            guard let handle = waifu2d_renderer_new(api.c, &surface) else { throw Waifu2D.failure() }
            self.handle = handle
            layer = nil
            pixelSize = (width, height)
        }

        deinit { waifu2d_renderer_free(handle) }

        /// Pixels; cheap when unchanged.
        public func resize(width: Int, height: Int) throws {
            guard width > 0, height > 0 else { return }
            guard waifu2d_renderer_resize(handle, UInt32(width), UInt32(height)) else { throw Waifu2D.failure() }
            pixelSize = (width, height)
        }

        /// Draws the stage as it is now over `clear`, premultiplied RGBA, and presents it.
        public func draw(_ stage: Stage, clear: (Float, Float, Float, Float) = (0, 0, 0, 0)) throws {
            let colour = [clear.0, clear.1, clear.2, clear.3]
            guard waifu2d_renderer_draw(handle, stage.handle, colour) else { throw Waifu2D.failure() }
        }

        /// An offscreen renderer's last draw: premultiplied RGBA, top row first.
        public func rgba() throws -> RGBAImage {
            var bytes = [UInt8](repeating: 0, count: pixelSize.width * pixelSize.height * 4)
            guard waifu2d_renderer_read_rgba(handle, &bytes, bytes.count) else { throw Waifu2D.failure() }
            return RGBAImage(width: pixelSize.width, height: pixelSize.height, rgba: bytes)
        }
    }
}
