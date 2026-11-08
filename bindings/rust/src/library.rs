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

//! Finding libwaifu and loading it, once a process. It is never unloaded: the library keeps
//! threads, thread-locals and device contexts that are not built to be.

use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use crate::sys::{Functions, WAIFU_ABI_VERSION};
use crate::{Error, ErrorCode, Result};

/// The file the platform's loader knows libwaifu by.
#[cfg(target_os = "macos")]
const FILE_NAME: &str = "libwaifu.dylib";
#[cfg(windows)]
const FILE_NAME: &str = "waifu.dll";
#[cfg(not(any(target_os = "macos", windows)))]
const FILE_NAME: &str = "libwaifu.so";

/// The library loaded, and where it was.
struct Loaded {
    path: PathBuf,
    functions: Functions,
}

static LOADED: OnceLock<Loaded> = OnceLock::new();
/// Held while a library is looked for and loaded, so that two threads calling `init` load one.
static LOADING: Mutex<()> = Mutex::new(());

fn library_error(message: impl Into<String>) -> Error {
    Error {
        code: ErrorCode::Library,
        message: message.into(),
    }
}

pub(crate) fn functions() -> Result<&'static Functions> {
    LOADED.get().map(|loaded| &loaded.functions).ok_or_else(|| {
        library_error("libwaifu is not loaded: call waifu::init() or waifu::init_from_path() first")
    })
}

pub fn is_initialized() -> bool {
    LOADED.get().is_some()
}

pub fn library_path() -> Option<&'static Path> {
    LOADED.get().map(|loaded| loaded.path.as_path())
}

/// The places [`init`] looks, in order, each a path to load as it is: a full path, or for the last
/// a file name alone, for the loader to search for.
fn places() -> Vec<PathBuf> {
    let mut places = Vec::new();
    if let Some(named) = std::env::var_os("LIBWAIFU_PATH").filter(|named| !named.is_empty()) {
        places.push(PathBuf::from(named));
    }
    if let Some(directory) = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(Path::to_path_buf))
    {
        places.push(directory.join(FILE_NAME));
        if cfg!(target_os = "macos") {
            places.push(directory.join("../Frameworks").join(FILE_NAME));
        }
    }
    places
}

/// Where `file_name` would be in each directory `path` names, in order -- a PATH-style list -- for
/// the directories that are absolute only: a relative one ("plugins", "..\\shared", "C:foo", ".",
/// "") is found from the current directory, which is never searched.
#[cfg_attr(not(windows), allow(dead_code))]
fn on_path(path: &std::ffi::OsStr, file_name: &str) -> Vec<PathBuf> {
    std::env::split_paths(path)
        .filter(|directory| directory.is_absolute())
        .map(|directory| directory.join(file_name))
        .collect()
}

pub fn init() -> Result<()> {
    let _loading = LOADING.lock().unwrap_or_else(|held| held.into_inner());
    if LOADED.get().is_some() {
        return Ok(());
    }

    let mut tried = Vec::new();
    for place in places() {
        if !place.is_file() {
            tried.push(format!("{}: not there", place.display()));
            continue;
        }
        match load(&place) {
            Ok(loaded) => return keep(loaded),
            Err(why) => tried.push(format!("{}: {why}", place.display())),
        }
    }
    match search() {
        Ok(loaded) => keep(loaded),
        Err(why) => {
            tried.push(why);
            Err(library_error(format!(
                "libwaifu was not found; tried {}",
                tried.join("; ")
            )))
        }
    }
}

pub fn init_from_path(library_path: &Path) -> Result<()> {
    let _loading = LOADING.lock().unwrap_or_else(|held| held.into_inner());
    if let Some(loaded) = LOADED.get() {
        let asked = std::fs::canonicalize(library_path).ok();
        return match asked.as_deref() == Some(loaded.path.as_path()) {
            true => Ok(()),
            false => Err(library_error(format!(
                "libwaifu is already loaded from {}, and one process has one: {} is another",
                loaded.path.display(),
                library_path.display()
            ))),
        };
    }
    let loaded = load(library_path)
        .map_err(|why| library_error(format!("{}: {why}", library_path.display())))?;
    keep(loaded)
}

fn keep(loaded: Loaded) -> Result<()> {
    let _ = LOADED.set(loaded);
    Ok(())
}

/// The library at `path`, its functions found and its ABI the one this crate speaks.
fn load(path: &Path) -> std::result::Result<Loaded, String> {
    let handle = platform::open(path)?;
    found(
        handle,
        std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf()),
    )
}

/// The library the platform's loader finds by its file name alone.
fn search() -> std::result::Result<Loaded, String> {
    let (handle, path) = platform::search(FILE_NAME)?;
    let functions = Functions::find(|name| platform::symbol(handle, name))
        .map_err(|missing| format!("{FILE_NAME}, as the loader found it, has no {missing}"))?;
    let path = path
        .or_else(|| platform::path_of(functions.waifu_abi_version as *const c_void))
        .map(|path| std::fs::canonicalize(&path).unwrap_or(path))
        .unwrap_or_else(|| PathBuf::from(FILE_NAME));
    checked(functions, path)
}

fn found(handle: platform::Handle, path: PathBuf) -> std::result::Result<Loaded, String> {
    let functions = Functions::find(|name| platform::symbol(handle, name))
        .map_err(|missing| format!("it has no {missing}: it is not libwaifu, or not ABI 4"))?;
    checked(functions, path)
}

fn checked(functions: Functions, path: PathBuf) -> std::result::Result<Loaded, String> {
    let abi = unsafe { (functions.waifu_abi_version)() };
    if abi != WAIFU_ABI_VERSION {
        return Err(format!(
            "it speaks ABI {abi}, and this crate speaks {WAIFU_ABI_VERSION}"
        ));
    }
    Ok(Loaded { path, functions })
}

#[cfg(unix)]
mod platform {
    use std::ffi::{c_char, c_int, c_void, CStr, CString};
    use std::os::unix::ffi::OsStrExt;
    use std::path::{Path, PathBuf};

    pub type Handle = *mut c_void;

    const RTLD_NOW: c_int = 2;
    #[cfg(target_os = "macos")]
    const RTLD_GLOBAL: c_int = 8;
    #[cfg(not(target_os = "macos"))]
    const RTLD_GLOBAL: c_int = 0x100;

    #[repr(C)]
    struct DlInfo {
        dli_fname: *const c_char,
        dli_fbase: *mut c_void,
        dli_sname: *const c_char,
        dli_saddr: *mut c_void,
    }

    #[cfg_attr(any(target_os = "linux", target_os = "android"), link(name = "dl"))]
    extern "C" {
        fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
        fn dlerror() -> *const c_char;
        fn dladdr(address: *const c_void, info: *mut DlInfo) -> c_int;
    }

    fn opened(name: &CStr) -> Result<Handle, String> {
        let handle = unsafe { dlopen(name.as_ptr(), RTLD_NOW | RTLD_GLOBAL) };
        if handle.is_null() {
            let why = unsafe { dlerror() };
            return Err(match why.is_null() {
                true => "it could not be loaded".to_string(),
                false => unsafe { CStr::from_ptr(why) }
                    .to_string_lossy()
                    .into_owned(),
            });
        }
        Ok(handle)
    }

    pub fn open(path: &Path) -> Result<Handle, String> {
        // A path with a slash in it is loaded from there, and searched for nowhere.
        let path = match path.is_absolute() {
            true => path.to_path_buf(),
            false => Path::new(".").join(path),
        };
        let name = CString::new(path.as_os_str().as_bytes())
            .map_err(|_| "the path has a NUL in it".to_string())?;
        opened(&name)
    }

    /// The loader's own search, and no path: [`path_of`] asks the loader where it found it.
    pub fn search(file_name: &str) -> Result<(Handle, Option<PathBuf>), String> {
        let name = CString::new(file_name).expect("no NUL");
        let handle =
            opened(&name).map_err(|why| format!("the loader's search for {file_name}: {why}"))?;
        Ok((handle, None))
    }

    pub fn symbol(handle: Handle, name: &str) -> Option<*mut c_void> {
        let name = CString::new(name).ok()?;
        let found = unsafe { dlsym(handle, name.as_ptr()) };
        (!found.is_null()).then_some(found)
    }

    pub fn path_of(address: *const c_void) -> Option<PathBuf> {
        let mut info = DlInfo {
            dli_fname: std::ptr::null(),
            dli_fbase: std::ptr::null_mut(),
            dli_sname: std::ptr::null(),
            dli_saddr: std::ptr::null_mut(),
        };
        if unsafe { dladdr(address, &mut info) } == 0 || info.dli_fname.is_null() {
            return None;
        }
        let name = unsafe { CStr::from_ptr(info.dli_fname) };
        Some(PathBuf::from(std::ffi::OsStr::from_bytes(name.to_bytes())))
    }
}

#[cfg(windows)]
mod platform {
    use std::ffi::c_void;
    use std::os::windows::ffi::OsStrExt;
    use std::path::{Path, PathBuf};

    pub type Handle = *mut c_void;

    /// What the DLL needs is looked for in its own directory, the program's and System32 -- and
    /// not in the current directory, nor on PATH. libwaifu links cudart statically, and asks the
    /// loader for nothing a machine keeps anywhere else: the CUDA driver, Vulkan's loader and the
    /// C runtime are in System32.
    const LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR: u32 = 0x100;
    const LOAD_LIBRARY_SEARCH_DEFAULT_DIRS: u32 = 0x1000;

    #[link(name = "kernel32")]
    extern "system" {
        fn LoadLibraryExW(name: *const u16, file: *mut c_void, flags: u32) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const u8) -> *mut c_void;
        fn GetLastError() -> u32;
    }

    fn wide(path: &Path) -> Vec<u16> {
        path.as_os_str().encode_wide().chain([0]).collect()
    }

    pub fn open(path: &Path) -> Result<Handle, String> {
        // A full path, so that nothing is looked for by its name -- in the current directory
        // least of all.
        let path = std::path::absolute(path).map_err(|error| error.to_string())?;
        let handle = unsafe {
            LoadLibraryExW(
                wide(&path).as_ptr(),
                std::ptr::null_mut(),
                LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS,
            )
        };
        match handle.is_null() {
            true => Err(format!("it could not be loaded (error {})", unsafe {
                GetLastError()
            })),
            false => Ok(handle),
        }
    }

    /// Each absolute directory on PATH, in order, for `file_name`: the loader's search, without
    /// the current directory it would otherwise look in, nor any directory found from it.
    pub fn search(file_name: &str) -> Result<(Handle, Option<PathBuf>), String> {
        let path = std::env::var_os("PATH").unwrap_or_default();
        for candidate in super::on_path(&path, file_name) {
            if candidate.is_file() {
                if let Ok(handle) = open(&candidate) {
                    return Ok((handle, Some(candidate)));
                }
            }
        }
        Err(format!("{file_name} is in no directory on PATH"))
    }

    pub fn symbol(handle: Handle, name: &str) -> Option<*mut c_void> {
        let name: Vec<u8> = name.bytes().chain([0]).collect();
        let found = unsafe { GetProcAddress(handle, name.as_ptr()) };
        (!found.is_null()).then_some(found)
    }

    /// The search says where it found the file, so nothing has to be asked.
    pub fn path_of(_address: *const c_void) -> Option<PathBuf> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_relative_directory_on_path_is_never_searched() {
        let root = std::env::temp_dir();
        let path = std::env::join_paths([
            Path::new("plugins"),
            Path::new("../shared"),
            Path::new("."),
            root.as_path(),
        ])
        .unwrap();
        assert_eq!(on_path(&path, FILE_NAME), [root.join(FILE_NAME)]);
        assert!(on_path(std::ffi::OsStr::new(""), FILE_NAME).is_empty());
    }
}
