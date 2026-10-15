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

//! The two libraries' C APIs, which are all of them this tool calls.
//!
//! Both are linked into the binary statically, so these are the very functions waifu.h and
//! waifu_modelmanager.h declare, with the types they declare, called as a C program would call
//! them. Nothing else of either crate is reached from here or from anywhere in the tool.

#![allow(unused_imports)]

use std::ffi::{c_char, CStr, CString};

pub use waifu::c_api::*;
pub use waifu_modelmanager::c_api::*;

/// A string the library handed back for the caller to read, or None for NULL.
pub unsafe fn text(pointer: *const c_char) -> Option<String> {
    (!pointer.is_null()).then(|| CStr::from_ptr(pointer).to_string_lossy().into_owned())
}

/// `text` as a C string, with any NUL in it -- which C would read as the end -- made a space.
pub fn c_string(text: &str) -> CString {
    CString::new(text.replace('\0', " ")).unwrap_or_default()
}

/// This thread's last message from libwaifu.
pub fn waifu_error() -> String {
    unsafe { text(waifu_last_error()) }.unwrap_or_default()
}

/// This thread's last message from libwaifu_modelmanager.
pub fn manager_error() -> String {
    unsafe { text(waifu_modelmanager_last_error()) }.unwrap_or_default()
}

/// A string libwaifu returned, read and freed.
pub fn waifu_string(said: *mut c_char) -> Option<String> {
    let read = unsafe { text(said) };
    unsafe { waifu_string_free(said) };
    read
}

/// A string libwaifu_modelmanager returned, read and freed.
pub fn manager_string(said: *mut c_char) -> Option<String> {
    let read = unsafe { text(said) };
    unsafe { waifu_modelmanager_string_free(said) };
    read
}

/// JSON libwaifu returned, read and freed.
pub fn waifu_json(said: *mut c_char) -> Option<serde_json::Value> {
    waifu_string(said).and_then(|said| serde_json::from_str(&said).ok())
}

/// JSON libwaifu_modelmanager returned, read and freed.
pub fn manager_json(said: *mut c_char) -> Option<serde_json::Value> {
    manager_string(said).and_then(|said| serde_json::from_str(&said).ok())
}
