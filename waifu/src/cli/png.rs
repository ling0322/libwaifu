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

//! Writing a picture out as a PNG, without compressing it.
//!
//! A PNG holds a zlib stream, and a zlib stream is allowed to say that a run of bytes was left
//! alone -- a "stored" deflate block. That is the whole trick here: the file is a few headers, the
//! pixels copied in verbatim, and two checksums. It is the size of the pixels rather than a third
//! of it, which for a picture nobody is going to keep more than a handful of is a fair trade for
//! not carrying a compressor.

/// The eight bytes every PNG starts with.
const SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];

/// The most a stored deflate block may hold, as its length is two bytes.
const BLOCK_MAX: usize = 0xffff;

/// The PNG file for `pixels`, which is `width * height` pixels of three bytes each, row by row.
///
/// # Panics
///
/// If `pixels` is not exactly as long as the size says it is.
pub fn encode(width: u32, height: u32, pixels: &[u8]) -> Vec<u8> {
    assert_eq!(
        pixels.len(),
        width as usize * height as usize * 3,
        "the pixels do not match the size they are said to be"
    );

    let mut header = Vec::with_capacity(13);
    header.extend_from_slice(&width.to_be_bytes());
    header.extend_from_slice(&height.to_be_bytes());
    header.push(8); // Eight bits a channel,
    header.push(2); // three channels, no palette and no alpha,
    header.push(0); // deflated, which is the only thing a PNG may be,
    header.push(0); // with the filters every PNG has,
    header.push(0); // and not interlaced.

    let mut png = Vec::from(SIGNATURE);
    chunk(&mut png, b"IHDR", &header);
    chunk(&mut png, b"IDAT", &zlib(&scanlines(width, height, pixels)));
    chunk(&mut png, b"IEND", &[]);
    png
}

/// The pixels as PNG wants them: each row behind the byte that says how it was filtered, which
/// here is always zero, meaning not at all.
fn scanlines(width: u32, height: u32, pixels: &[u8]) -> Vec<u8> {
    let stride = width as usize * 3;
    let mut out = Vec::with_capacity((stride + 1) * height as usize);
    for row in 0..height as usize {
        out.push(0);
        out.extend_from_slice(&pixels[row * stride..(row + 1) * stride]);
    }
    out
}

/// `data` wrapped in a zlib stream that stores it rather than compressing it.
fn zlib(data: &[u8]) -> Vec<u8> {
    // Deflate, a 32k window, no preset dictionary, and a check byte that makes the pair divide
    // by 31 -- which is all the header of a zlib stream says.
    let mut out = vec![0x78, 0x01];

    // An empty stream still needs the one block that says the stream has ended.
    let mut blocks = data.chunks(BLOCK_MAX).peekable();
    loop {
        let block: &[u8] = blocks.next().unwrap_or(&[]);
        let last = blocks.peek().is_none();

        // The low bit says this is the last block, and the two above it that it is a stored one.
        out.push(u8::from(last));
        let length = block.len() as u16;
        out.extend_from_slice(&length.to_le_bytes());
        out.extend_from_slice(&(!length).to_le_bytes());
        out.extend_from_slice(block);

        if last {
            break;
        }
    }

    out.extend_from_slice(&adler32(data).to_be_bytes());
    out
}

/// Appends a chunk: its length, its name, what it holds, and the check over the last two.
fn chunk(out: &mut Vec<u8>, name: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(name);
    out.extend_from_slice(data);

    let mut crc = Crc32::new();
    crc.update(name);
    crc.update(data);
    out.extend_from_slice(&crc.finish().to_be_bytes());
}

/// The checksum zlib ends a stream with: two running sums, one of the bytes and one of the sum.
fn adler32(data: &[u8]) -> u32 {
    const MODULUS: u32 = 65521;

    let (mut low, mut high) = (1u32, 0u32);
    // Far enough apart to keep the sums inside a u32 between the divisions.
    for block in data.chunks(5552) {
        for byte in block {
            low += u32::from(*byte);
            high += low;
        }
        low %= MODULUS;
        high %= MODULUS;
    }

    (high << 16) | low
}

/// The check every PNG chunk ends with, which is the ordinary CRC-32.
struct Crc32(u32);

impl Crc32 {
    fn new() -> Crc32 {
        Crc32(0xffff_ffff)
    }

    fn update(&mut self, data: &[u8]) {
        for byte in data {
            let mut value = (self.0 ^ u32::from(*byte)) & 0xff;
            for _ in 0..8 {
                // The polynomial, written backwards, because the bits go through it backwards.
                value = if value & 1 != 0 {
                    (value >> 1) ^ 0xedb8_8320
                } else {
                    value >> 1
                };
            }
            self.0 = value ^ (self.0 >> 8);
        }
    }

    fn finish(self) -> u32 {
        self.0 ^ 0xffff_ffff
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The chunks of a PNG, as name and contents.
    fn chunks(png: &[u8]) -> Vec<(String, Vec<u8>)> {
        assert_eq!(&png[..8], SIGNATURE, "the signature is not a PNG's");

        let mut out = Vec::new();
        let mut at = 8;
        while at < png.len() {
            let length = u32::from_be_bytes(png[at..at + 4].try_into().unwrap()) as usize;
            let name = String::from_utf8(png[at + 4..at + 8].to_vec()).unwrap();
            let data = png[at + 8..at + 8 + length].to_vec();

            let mut crc = Crc32::new();
            crc.update(&png[at + 4..at + 8]);
            crc.update(&data);
            let stated =
                u32::from_be_bytes(png[at + 8 + length..at + 12 + length].try_into().unwrap());
            assert_eq!(crc.finish(), stated, "{name} does not check out");

            out.push((name, data));
            at += 12 + length;
        }
        out
    }

    /// What a zlib stream of stored blocks holds. Only stored blocks, which is all this writes.
    fn inflate_stored(stream: &[u8]) -> Vec<u8> {
        assert_eq!(&stream[..2], &[0x78, 0x01], "not a zlib header");

        let mut out = Vec::new();
        let mut at = 2;
        loop {
            let header = stream[at];
            assert_eq!(header & 0x06, 0, "not a stored block");
            let length = u16::from_le_bytes(stream[at + 1..at + 3].try_into().unwrap());
            let complement = u16::from_le_bytes(stream[at + 3..at + 5].try_into().unwrap());
            assert_eq!(length, !complement, "the length does not check out");

            out.extend_from_slice(&stream[at + 5..at + 5 + length as usize]);
            at += 5 + length as usize;

            if header & 1 != 0 {
                break;
            }
        }

        let stated = u32::from_be_bytes(stream[at..at + 4].try_into().unwrap());
        assert_eq!(adler32(&out), stated, "the stream does not check out");
        assert_eq!(at + 4, stream.len(), "there is something after the stream");
        out
    }

    #[test]
    fn writes_the_chunks_a_png_is_made_of() {
        let pixels: Vec<u8> = (0..2 * 3 * 3).map(|i| i as u8).collect();
        let chunks = chunks(&encode(3, 2, &pixels));

        let names: Vec<&str> = chunks.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(names, ["IHDR", "IDAT", "IEND"]);

        let header = &chunks[0].1;
        assert_eq!(u32::from_be_bytes(header[0..4].try_into().unwrap()), 3);
        assert_eq!(u32::from_be_bytes(header[4..8].try_into().unwrap()), 2);
        assert_eq!(&header[8..], &[8, 2, 0, 0, 0], "eight bit RGB, unfiltered");
    }

    #[test]
    fn the_pixels_come_back_out_row_by_row() {
        let pixels: Vec<u8> = (0..2 * 3 * 3).map(|i| i as u8).collect();
        let chunks = chunks(&encode(3, 2, &pixels));
        let raw = inflate_stored(&chunks[1].1);

        // Each row is the filter byte and then the row itself.
        assert_eq!(raw[0], 0);
        assert_eq!(&raw[1..10], &pixels[0..9]);
        assert_eq!(raw[10], 0);
        assert_eq!(&raw[11..20], &pixels[9..18]);
        assert_eq!(raw.len(), 20);
    }

    #[test]
    fn a_picture_too_big_for_one_block_is_split_into_several() {
        // Three rows of a size that no two of them fit in one stored block.
        let width = 20_000;
        let pixels = vec![7u8; width * 3 * 3];
        let png = encode(width as u32, 3, &pixels);

        let raw = inflate_stored(&chunks(&png)[1].1);
        assert_eq!(raw.len(), (width * 3 + 1) * 3);
        assert!(raw.iter().all(|byte| *byte == 7 || *byte == 0));
    }

    #[test]
    fn the_checksums_are_the_ones_everybody_elses_files_carry() {
        // The end of every PNG ever written is this chunk, check included.
        let png = encode(1, 1, &[0, 0, 0]);
        assert_eq!(
            &png[png.len() - 12..],
            &[0, 0, 0, 0, b'I', b'E', b'N', b'D', 0xae, 0x42, 0x60, 0x82]
        );

        // Which leaves the other checksum, on the example the zlib specification uses.
        assert_eq!(adler32(b"abc"), 0x024d_0127);
        assert_eq!(adler32(b""), 1);
    }

    #[test]
    #[should_panic(expected = "the pixels do not match")]
    fn a_size_that_is_not_the_picture_is_refused() {
        encode(2, 2, &[0; 3]);
    }
}
