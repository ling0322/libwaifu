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

//! The Metal backend: what waifu-app/swiftui's RigMetal.swift did, behind [`Backend`].
//!
//! Its shader and pipelines are RigMetal's, with two changes the backend contract asks for: the
//! pictures' mip chains are uploaded level by level from [`Assets`] rather than generated, and
//! the masks are an R8 array texture with a slice a mask, found by id, rather than the red and
//! green of one RGBA texture.

use std::collections::HashMap;
use std::ffi::c_void;
use std::mem::size_of;
use std::ptr::NonNull;

use objc2::rc::{autoreleasepool, Retained};
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSSize, NSString};
use objc2_metal::{
    MTLBlendFactor, MTLBlendOperation, MTLBlitCommandEncoder, MTLBuffer, MTLClearColor,
    MTLColorWriteMask, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder,
    MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice, MTLIndexType, MTLLibrary,
    MTLLoadAction, MTLOrigin, MTLPixelFormat, MTLPrimitiveType, MTLRegion, MTLRenderCommandEncoder,
    MTLRenderPassDescriptor, MTLRenderPipelineDescriptor, MTLRenderPipelineState,
    MTLResourceOptions, MTLSamplerAddressMode, MTLSamplerDescriptor, MTLSamplerMinMagFilter,
    MTLSamplerMipFilter, MTLSamplerState, MTLSize, MTLStorageMode, MTLStoreAction, MTLTexture,
    MTLTextureDescriptor, MTLTextureType, MTLTextureUsage, MTLViewport,
};
use objc2_quartz_core::{CAMetalDrawable, CAMetalLayer};

use super::{Api, Assets, Backend, DrawList, Loaded, Surface};
use crate::Error;

/// The shaders, compiled when a backend is made, as RigMetal.swift compiled its own.
const SHADERS: &str = include_str!("shaders/rig.metal");

type Device = ProtocolObject<dyn MTLDevice>;
type Texture = ProtocolObject<dyn MTLTexture>;
type Buffer = ProtocolObject<dyn MTLBuffer>;
type Pipeline = ProtocolObject<dyn MTLRenderPipelineState>;
type Encoder = ProtocolObject<dyn MTLRenderCommandEncoder>;

/// Whether there is a GPU to draw with: a system default Metal device.
pub fn is_available() -> bool {
    autoreleasepool(|_| MTLCreateSystemDefaultDevice().is_some())
}

/// What each draw is told; rig.metal's `Uniforms`, field for field.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct Uniforms {
    canvas: [f32; 2],
    alpha: f32,
    use_mask: f32,
    mask_out: f32,
    mask_slice: u32,
}

/// A part as the GPU has it.
struct Mesh {
    texture: Retained<Texture>,
    uv: Retained<Buffer>,
    indices: Retained<Buffer>,
    count: usize,
}

/// Where the main pass draws.
enum Target {
    /// An RGBA8 texture of the backend's own, read back with `read_rgba`; `None` until it has a
    /// size.
    Offscreen(Option<Retained<Texture>>),
    /// The host's layer, a drawable of which each draw takes and presents.
    Layer(Retained<CAMetalLayer>),
}

pub struct MetalBackend {
    device: Retained<Device>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    draw: Retained<Pipeline>,
    /// Writing a part's texel alpha into a slice of the masks.
    mask: Retained<Pipeline>,
    sampler: Retained<ProtocolObject<dyn MTLSamplerState>>,
    target: Target,
    size: [u32; 2],
    meshes: Vec<Mesh>,
    loaded: Option<Loaded>,
    /// The masks, a slice each, the size of the target; grown when a frame has more.
    masks: Option<Retained<Texture>>,
    /// Bound where a frame has no masks: never read.
    no_masks: Retained<Texture>,
    /// The last draw's commands, waited on before the offscreen texture is read.
    last: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
}

fn gpu(what: &str) -> Error {
    Error::Gpu(format!("Metal could not {what}"))
}

impl MetalBackend {
    pub fn new(surface: Surface) -> Result<MetalBackend, Error> {
        autoreleasepool(|_| Self::make(surface))
    }

    fn make(surface: Surface) -> Result<MetalBackend, Error> {
        let device = MTLCreateSystemDefaultDevice().ok_or_else(|| gpu("find a device"))?;
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| gpu("make a command queue"))?;
        let library = device
            .newLibraryWithSource_options_error(&NSString::from_str(SHADERS), None)
            .map_err(|e| Error::Gpu(format!("rig.metal: {}", e.localizedDescription())))?;
        let function = |name: &str| {
            library
                .newFunctionWithName(&NSString::from_str(name))
                .ok_or_else(|| gpu(name))
        };
        let (vertex, fragment) = (function("rig_vertex")?, function("rig_fragment")?);
        let pipeline = |format: MTLPixelFormat| -> Result<Retained<Pipeline>, Error> {
            let d = MTLRenderPipelineDescriptor::new();
            d.setVertexFunction(Some(&vertex));
            d.setFragmentFunction(Some(&fragment));
            let colour = unsafe { d.colorAttachments().objectAtIndexedSubscript(0) };
            colour.setPixelFormat(format);
            colour.setWriteMask(MTLColorWriteMask::All);
            colour.setBlendingEnabled(true);
            colour.setRgbBlendOperation(MTLBlendOperation::Add);
            colour.setAlphaBlendOperation(MTLBlendOperation::Add);
            colour.setSourceRGBBlendFactor(MTLBlendFactor::One);
            colour.setSourceAlphaBlendFactor(MTLBlendFactor::One);
            colour.setDestinationRGBBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
            colour.setDestinationAlphaBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
            device
                .newRenderPipelineStateWithDescriptor_error(&d)
                .map_err(|e| Error::Gpu(format!("a pipeline: {}", e.localizedDescription())))
        };

        let (target, format, size) = match surface {
            Surface::Offscreen { width, height } => (
                Target::Offscreen(None),
                MTLPixelFormat::RGBA8Unorm,
                [width, height],
            ),
            Surface::MetalLayer { layer } => {
                // SAFETY: the host hands over a CAMetalLayer, which it keeps alive at least until
                // this retains it.
                let layer = unsafe { Retained::retain(layer.as_ptr().cast::<CAMetalLayer>()) }
                    .ok_or_else(|| Error::InvalidArgument("a NULL CAMetalLayer".into()))?;
                layer.setDevice(Some(&device));
                layer.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
                let drawable = layer.drawableSize();
                let size = [
                    drawable.width.max(0.0) as u32,
                    drawable.height.max(0.0) as u32,
                ];
                (Target::Layer(layer), MTLPixelFormat::BGRA8Unorm, size)
            }
            other => {
                return Err(Error::Unsupported(format!(
                    "Metal does not draw into {other:?}"
                )))
            }
        };
        let draw = pipeline(format)?;
        let mask = pipeline(MTLPixelFormat::R8Unorm)?;

        let s = MTLSamplerDescriptor::new();
        s.setMinFilter(MTLSamplerMinMagFilter::Linear);
        s.setMagFilter(MTLSamplerMinMagFilter::Linear);
        s.setMipFilter(MTLSamplerMipFilter::Linear);
        s.setSAddressMode(MTLSamplerAddressMode::ClampToEdge);
        s.setTAddressMode(MTLSamplerAddressMode::ClampToEdge);
        let sampler = device
            .newSamplerStateWithDescriptor(&s)
            .ok_or_else(|| gpu("make a sampler"))?;
        let no_masks = mask_array(&device, 1, 1, 1)?;

        let mut backend = MetalBackend {
            device,
            queue,
            draw,
            mask,
            sampler,
            target,
            size: [0, 0],
            meshes: Vec::new(),
            loaded: None,
            masks: None,
            no_masks,
            last: None,
        };
        if size[0] > 0 && size[1] > 0 {
            backend.resize(size[0], size[1])?;
        }
        Ok(backend)
    }

    /// A buffer holding `bytes`, or at least `min` bytes where there are none.
    fn buffer(&self, bytes: &[u8], min: usize) -> Result<Retained<Buffer>, Error> {
        let made = match NonNull::new(bytes.as_ptr() as *mut c_void) {
            Some(pointer) if !bytes.is_empty() => unsafe {
                self.device.newBufferWithBytes_length_options(
                    pointer,
                    bytes.len(),
                    MTLResourceOptions::StorageModeShared,
                )
            },
            _ => self
                .device
                .newBufferWithLength_options(min, MTLResourceOptions::StorageModeShared),
        };
        made.ok_or_else(|| gpu("make a buffer"))
    }

    /// The masks with at least `slices` slices, the size of the target.
    fn masks(&mut self, slices: usize) -> Result<Retained<Texture>, Error> {
        let [w, h] = [self.size[0] as usize, self.size[1] as usize];
        if let Some(masks) = &self.masks {
            if masks.width() == w && masks.height() == h && masks.arrayLength() >= slices {
                return Ok(masks.clone());
            }
        }
        let have = self.masks.as_ref().map_or(0, |m| m.arrayLength());
        let masks = mask_array(&self.device, w, h, slices.max(have))?;
        self.masks = Some(masks.clone());
        Ok(masks)
    }

    fn encode(&mut self, list: &DrawList, clear: [f32; 4]) -> Result<(), Error> {
        let loaded = self
            .loaded
            .as_ref()
            .ok_or_else(|| Error::InvalidArgument("drawn before a rig was loaded".into()))?;
        loaded.check(list)?;
        if list.target != self.size {
            return Err(Error::InvalidArgument(format!(
                "a frame for {:?} pixels drawn into {:?}",
                list.target, self.size
            )));
        }
        let masks = if list.masks.is_empty() {
            None
        } else {
            Some(self.masks(list.masks.len())?)
        };

        let (texture, drawable) = match &self.target {
            Target::Offscreen(texture) => (
                texture
                    .clone()
                    .ok_or_else(|| gpu("draw before it had a size"))?,
                None,
            ),
            Target::Layer(layer) => {
                let drawable = layer.nextDrawable().ok_or_else(|| gpu("get a drawable"))?;
                (drawable.texture(), Some(drawable))
            }
        };
        let commands = self
            .queue
            .commandBuffer()
            .ok_or_else(|| gpu("make a command buffer"))?;

        // Every part's vertices where they are, in one buffer.
        let positions = &list.frame.positions;
        let bytes = unsafe {
            std::slice::from_raw_parts(positions.as_ptr().cast::<u8>(), positions.len() * 8)
        };
        let vertices = self.buffer(bytes, 8)?;
        let v = &list.viewport;
        let viewport = MTLViewport {
            originX: v.x,
            originY: v.y,
            width: v.width,
            height: v.height,
            znear: 0.0,
            zfar: 1.0,
        };
        let canvas = [list.canvas_size[0] as f32, list.canvas_size[1] as f32];
        let offset = |part: crate::PartId| list.frame.offsets[part.index()] as usize * 8;

        // The masks, each into its own slice, its parts as they are now whether they show or not.
        let mut slice_of = HashMap::new();
        if let Some(masks) = &masks {
            for (slice, mask) in list.masks.iter().enumerate() {
                slice_of.insert(mask.id, slice as u32);
                let pass = MTLRenderPassDescriptor::new();
                let colour = unsafe { pass.colorAttachments().objectAtIndexedSubscript(0) };
                colour.setTexture(Some(masks));
                colour.setSlice(slice);
                colour.setLoadAction(MTLLoadAction::Clear);
                colour.setClearColor(MTLClearColor {
                    red: 0.0,
                    green: 0.0,
                    blue: 0.0,
                    alpha: 0.0,
                });
                colour.setStoreAction(MTLStoreAction::Store);
                let encoder = commands
                    .renderCommandEncoderWithDescriptor(&pass)
                    .ok_or_else(|| gpu("begin a mask pass"))?;
                encoder.setRenderPipelineState(&self.mask);
                encoder.setViewport(viewport);
                unsafe {
                    encoder.setFragmentSamplerState_atIndex(Some(&self.sampler), 0);
                    // Not read while a mask is drawn, but bound: never the masks, drawn into.
                    encoder.setFragmentTexture_atIndex(Some(&self.no_masks), 1);
                }
                let uniforms = Uniforms {
                    canvas,
                    alpha: 1.0,
                    mask_out: 1.0,
                    ..Default::default()
                };
                for &part in &mask.parts {
                    self.encode_mesh(&encoder, part.index(), &vertices, offset(part), &uniforms);
                }
                encoder.endEncoding();
            }
        }

        // Then the parts, over the clear colour.
        let pass = MTLRenderPassDescriptor::new();
        let colour = unsafe { pass.colorAttachments().objectAtIndexedSubscript(0) };
        colour.setTexture(Some(&texture));
        colour.setLoadAction(MTLLoadAction::Clear);
        colour.setClearColor(MTLClearColor {
            red: clear[0] as f64,
            green: clear[1] as f64,
            blue: clear[2] as f64,
            alpha: clear[3] as f64,
        });
        colour.setStoreAction(MTLStoreAction::Store);
        let encoder = commands
            .renderCommandEncoderWithDescriptor(&pass)
            .ok_or_else(|| gpu("begin a pass"))?;
        encoder.setRenderPipelineState(&self.draw);
        encoder.setViewport(viewport);
        unsafe {
            encoder.setFragmentSamplerState_atIndex(Some(&self.sampler), 0);
            encoder.setFragmentTexture_atIndex(Some(masks.as_ref().unwrap_or(&self.no_masks)), 1);
        }
        for draw in &list.draws {
            let mut uniforms = Uniforms {
                canvas,
                alpha: draw.alpha,
                ..Default::default()
            };
            if let Some(id) = draw.mask {
                uniforms.use_mask = 1.0;
                uniforms.mask_slice = slice_of[&id];
            }
            self.encode_mesh(
                &encoder,
                draw.part.index(),
                &vertices,
                offset(draw.part),
                &uniforms,
            );
        }
        encoder.endEncoding();

        match drawable {
            Some(drawable) => {
                commands.presentDrawable(ProtocolObject::from_ref(&*drawable));
            }
            None => {
                let blit = commands
                    .blitCommandEncoder()
                    .ok_or_else(|| gpu("begin a blit"))?;
                blit.synchronizeResource(ProtocolObject::from_ref(&*texture));
                blit.endEncoding();
            }
        }
        commands.commit();
        self.last = Some(commands);
        Ok(())
    }

    fn encode_mesh(
        &self,
        encoder: &Encoder,
        part: usize,
        vertices: &Buffer,
        offset: usize,
        uniforms: &Uniforms,
    ) {
        let mesh = &self.meshes[part];
        if mesh.count == 0 {
            return;
        }
        let bytes = NonNull::from(uniforms).cast::<c_void>();
        // SAFETY: the buffers outlive the command buffer, which retains them; `Loaded::check`
        // made sure every index is inside this part's vertices this frame.
        unsafe {
            encoder.setVertexBuffer_offset_atIndex(Some(vertices), offset, 0);
            encoder.setVertexBuffer_offset_atIndex(Some(&mesh.uv), 0, 1);
            encoder.setVertexBytes_length_atIndex(bytes, size_of::<Uniforms>(), 2);
            encoder.setFragmentBytes_length_atIndex(bytes, size_of::<Uniforms>(), 0);
            encoder.setFragmentTexture_atIndex(Some(&mesh.texture), 0);
            encoder.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset(
                MTLPrimitiveType::Triangle,
                mesh.count,
                MTLIndexType::UInt32,
                &mesh.indices,
                0,
            );
        }
    }
}

/// An R8 array texture of `slices` coverage images, `width` by `height`, to draw into and read.
fn mask_array(
    device: &Device,
    width: usize,
    height: usize,
    slices: usize,
) -> Result<Retained<Texture>, Error> {
    let d = unsafe {
        MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
            MTLPixelFormat::R8Unorm,
            width,
            height,
            false,
        )
    };
    d.setTextureType(MTLTextureType::Type2DArray);
    unsafe { d.setArrayLength(slices.max(1)) };
    d.setUsage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
    d.setStorageMode(MTLStorageMode::Private);
    device
        .newTextureWithDescriptor(&d)
        .ok_or_else(|| gpu("make the masks"))
}

fn region(width: usize, height: usize) -> MTLRegion {
    MTLRegion {
        origin: MTLOrigin { x: 0, y: 0, z: 0 },
        size: MTLSize {
            width,
            height,
            depth: 1,
        },
    }
}

impl Backend for MetalBackend {
    fn api(&self) -> Api {
        Api::Metal
    }

    fn load(&mut self, assets: &Assets) -> Result<(), Error> {
        self.meshes.clear();
        self.loaded = None;
        autoreleasepool(|_| {
            for (i, part) in assets.parts.iter().enumerate() {
                let base = part
                    .mips
                    .first()
                    .ok_or_else(|| Error::InvalidArgument(format!("part {i} has no picture")))?;
                let (w, h) = (base.width, base.height);
                for (level, mip) in part.mips.iter().enumerate() {
                    let (lw, lh) = ((w >> level).max(1), (h >> level).max(1));
                    if mip.width != lw || mip.height != lh || mip.rgba.len() != lw * lh * 4 {
                        return Err(Error::InvalidArgument(format!(
                            "part {i}'s level {level} is {} by {}, not {lw} by {lh}",
                            mip.width, mip.height
                        )));
                    }
                }
                let d = unsafe {
                    MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                        MTLPixelFormat::RGBA8Unorm,
                        w,
                        h,
                        part.mips.len() > 1,
                    )
                };
                unsafe { d.setMipmapLevelCount(part.mips.len()) };
                d.setUsage(MTLTextureUsage::ShaderRead);
                let texture = self
                    .device
                    .newTextureWithDescriptor(&d)
                    .ok_or_else(|| gpu("make a texture"))?;
                for (level, mip) in part.mips.iter().enumerate() {
                    let bytes = NonNull::new(mip.rgba.as_ptr() as *mut c_void).unwrap();
                    unsafe {
                        texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow(
                            region(mip.width, mip.height),
                            level,
                            bytes,
                            mip.width * 4,
                        )
                    };
                }
                let uv = unsafe {
                    std::slice::from_raw_parts(part.uv.as_ptr().cast::<u8>(), part.uv.len() * 8)
                };
                let indices = unsafe {
                    std::slice::from_raw_parts(
                        part.indices.as_ptr().cast::<u8>(),
                        part.indices.len() * 4,
                    )
                };
                self.meshes.push(Mesh {
                    texture,
                    uv: self.buffer(uv, 8)?,
                    indices: self.buffer(indices, 4)?,
                    count: part.indices.len() / 3 * 3,
                });
            }
            Ok(())
        })?;
        self.loaded = Some(Loaded::of(assets));
        Ok(())
    }

    fn resize(&mut self, width: u32, height: u32) -> Result<(), Error> {
        if width == 0 || height == 0 {
            return Err(Error::InvalidArgument(format!(
                "a {width} by {height} surface"
            )));
        }
        autoreleasepool(|_| {
            match &mut self.target {
                Target::Offscreen(texture) => {
                    let d = unsafe {
                        MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                            MTLPixelFormat::RGBA8Unorm,
                            width as usize,
                            height as usize,
                            false,
                        )
                    };
                    d.setUsage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
                    d.setStorageMode(MTLStorageMode::Managed);
                    *texture = Some(
                        self.device
                            .newTextureWithDescriptor(&d)
                            .ok_or_else(|| gpu("make the target"))?,
                    );
                }
                Target::Layer(layer) => {
                    layer.setDrawableSize(NSSize::new(width as f64, height as f64));
                }
            }
            self.size = [width, height];
            self.last = None;
            Ok(())
        })
    }

    fn draw(&mut self, list: &DrawList, clear: [f32; 4]) -> Result<(), Error> {
        autoreleasepool(|_| self.encode(list, clear))
    }

    fn read_rgba(&mut self, out: &mut [u8]) -> Result<(), Error> {
        let Target::Offscreen(texture) = &self.target else {
            return Err(Error::Unsupported("read_rgba on a CAMetalLayer".into()));
        };
        let [w, h] = [self.size[0] as usize, self.size[1] as usize];
        if out.len() != w * h * 4 {
            return Err(Error::InvalidArgument(format!(
                "{} bytes for {}",
                out.len(),
                w * h * 4
            )));
        }
        let (Some(texture), Some(commands)) = (texture, &self.last) else {
            out.fill(0);
            return Ok(());
        };
        commands.waitUntilCompleted();
        if commands.status() == MTLCommandBufferStatus::Error {
            let why = commands
                .error()
                .map(|e| e.localizedDescription().to_string());
            return Err(Error::Gpu(format!(
                "a draw failed: {}",
                why.unwrap_or_default()
            )));
        }
        let bytes = NonNull::new(out.as_mut_ptr().cast::<c_void>()).unwrap();
        unsafe {
            texture.getBytes_bytesPerRow_fromRegion_mipmapLevel(bytes, w * 4, region(w, h), 0)
        };
        Ok(())
    }
}
