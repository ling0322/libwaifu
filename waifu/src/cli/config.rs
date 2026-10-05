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

//! `config.toml`: what is kept from one run of the tool to the next.
//!
//! Today that is one thing, `model_dir`, the directory fetched models go into. The model screen
//! writes it and the hub reads it; anybody can also write it by hand.
//!
//! ```toml
//! model_dir = "D:\\models"
//! ```
//!
//! Where the file lives:
//!
//! - Windows: beside `waifu.exe`. A Windows install is a folder somebody unzipped, and the
//!   settings of a folder like that belong in the folder, where they go along when it is moved
//!   and go away when it is deleted.
//! - Everywhere else: `$XDG_CONFIG_HOME/libwaifu/config.toml`, which is
//!   `~/.config/libwaifu/config.toml` when that is not set -- beside the cache the models go into
//!   by default, `~/.cache/libwaifu/models`, in the place for things that cannot be fetched again.
//!
//! A relative `model_dir` is relative to the directory the file is in rather than to wherever the
//! tool happens to be started from, so `model_dir = "models"` next to `waifu.exe` means the folder
//! beside it, whichever directory the shell is in.
//!
//! A file that is there and cannot be read is an error rather than a file that says nothing: one
//! typo would otherwise send the next download of several gigabytes somewhere nobody asked for.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

type Error = Box<dyn std::error::Error>;

/// What the file is called, wherever it is.
pub const FILE_NAME: &str = "config.toml";

/// The key that says where fetched models go.
const MODEL_DIR: &str = "model_dir";

/// Where the settings file is, whether or not it exists. None only where the environment gives no
/// way to tell: no executable path on Windows, no home directory elsewhere.
pub fn path() -> Option<PathBuf> {
    if cfg!(windows) {
        let exe = env::current_exe().ok()?;
        return Some(exe.parent()?.join(FILE_NAME));
    }

    let base = env::var_os("XDG_CONFIG_HOME")
        .filter(|base| !base.is_empty())
        .map(PathBuf::from)
        .or_else(|| env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")))?;
    Some(base.join("libwaifu").join(FILE_NAME))
}

/// The directory the settings file says models go into, or None where it says nothing.
pub fn model_dir() -> Result<Option<PathBuf>, Error> {
    match path() {
        Some(file) => model_dir_in(&file),
        None => Ok(None),
    }
}

/// Says that models go into `directory` from now on, or, given None, takes that back so that they
/// go wherever they would by default. Hands back the file it wrote.
///
/// Every other key in the file is kept. The comments are not: the file is read into a table and
/// written back out of one, and a table has nowhere to keep them.
pub fn set_model_dir(directory: Option<&Path>) -> Result<PathBuf, Error> {
    let file = path().ok_or("cannot tell where the settings file goes")?;
    set_model_dir_in(&file, directory)?;
    Ok(file)
}

/// [`model_dir`], of a file given rather than found.
fn model_dir_in(file: &Path) -> Result<Option<PathBuf>, Error> {
    let Some(table) = read(file)? else {
        return Ok(None);
    };

    match table.get(MODEL_DIR) {
        None => Ok(None),
        Some(toml::Value::String(directory)) if directory.is_empty() => Ok(None),
        Some(toml::Value::String(directory)) => {
            // Relative to the file, not to where the tool was started. See the module's header.
            let base = file.parent().unwrap_or(Path::new("."));
            Ok(Some(base.join(directory)))
        }
        Some(_) => Err(format!(
            "{}: {MODEL_DIR} has to be a string, the path of a directory",
            file.display()
        )
        .into()),
    }
}

/// [`set_model_dir`], of a file given rather than found.
fn set_model_dir_in(file: &Path, directory: Option<&Path>) -> Result<(), Error> {
    let mut table = read(file)?.unwrap_or_default();
    match directory {
        Some(directory) => {
            let directory = directory.to_str().ok_or_else(|| {
                format!(
                    "{} is not a path that can be written down",
                    directory.display()
                )
            })?;
            table.insert(
                MODEL_DIR.to_string(),
                toml::Value::String(directory.to_string()),
            );
        }
        None => {
            table.remove(MODEL_DIR);
        }
    }

    if let Some(parent) = file.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("cannot make {}: {error}", parent.display()))?;
    }
    fs::write(file, table.to_string())
        .map_err(|error| format!("cannot write {}: {error}", file.display()))?;
    Ok(())
}

/// The file as a table, or None where there is no file.
fn read(file: &Path) -> Result<Option<toml::Table>, Error> {
    let text = match fs::read_to_string(file) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(format!("cannot read {}: {error}", file.display()).into()),
    };

    text.parse::<toml::Table>()
        .map(Some)
        .map_err(|error| format!("{} is not valid TOML: {error}", file.display()).into())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A directory of its own for one test, so that two of them running at once do not share a
    /// file.
    fn scratch(name: &str) -> PathBuf {
        let directory = env::temp_dir().join(format!("waifu-config-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&directory);
        directory
    }

    #[test]
    fn no_file_says_nothing() {
        let directory = scratch("none");
        assert_eq!(model_dir_in(&directory.join(FILE_NAME)).unwrap(), None);
    }

    #[test]
    fn what_is_written_is_read_back_and_the_rest_is_kept() {
        let directory = scratch("round-trip");
        let file = directory.join("deeper").join(FILE_NAME);
        fs::create_dir_all(file.parent().unwrap()).unwrap();
        fs::write(&file, "# a comment\nsomething_else = 3\n").unwrap();

        set_model_dir_in(&file, Some(Path::new("/data/models"))).unwrap();
        assert_eq!(
            model_dir_in(&file).unwrap(),
            Some(PathBuf::from("/data/models"))
        );
        assert!(fs::read_to_string(&file)
            .unwrap()
            .contains("something_else = 3"));

        set_model_dir_in(&file, None).unwrap();
        assert_eq!(model_dir_in(&file).unwrap(), None);
        assert!(fs::read_to_string(&file)
            .unwrap()
            .contains("something_else = 3"));
        let _ = fs::remove_dir_all(&directory);
    }

    #[test]
    fn the_file_and_its_directory_are_made_when_there_are_none() {
        let directory = scratch("made");
        let file = directory.join("libwaifu").join(FILE_NAME);
        set_model_dir_in(&file, Some(Path::new("/data/models"))).unwrap();
        assert!(file.exists());
        let _ = fs::remove_dir_all(&directory);
    }

    #[test]
    fn a_relative_directory_is_relative_to_the_file() {
        let directory = scratch("relative");
        fs::create_dir_all(&directory).unwrap();
        let file = directory.join(FILE_NAME);
        fs::write(&file, "model_dir = \"models\"\n").unwrap();
        assert_eq!(model_dir_in(&file).unwrap(), Some(directory.join("models")));
        let _ = fs::remove_dir_all(&directory);
    }

    #[test]
    fn a_windows_path_reads_as_written() {
        let directory = scratch("windows");
        fs::create_dir_all(&directory).unwrap();
        let file = directory.join(FILE_NAME);
        fs::write(&file, "model_dir = 'D:\\models'\n").unwrap();
        let read = model_dir_in(&file).unwrap().unwrap();
        assert!(read.to_string_lossy().ends_with("D:\\models"), "{read:?}");
        let _ = fs::remove_dir_all(&directory);
    }

    #[test]
    fn a_broken_file_is_an_error_and_not_silence() {
        let directory = scratch("broken");
        fs::create_dir_all(&directory).unwrap();
        let file = directory.join(FILE_NAME);

        fs::write(&file, "model_dir = \n").unwrap();
        let error = model_dir_in(&file).unwrap_err().to_string();
        assert!(error.contains("not valid TOML"), "{error}");

        fs::write(&file, "model_dir = 3\n").unwrap();
        let error = model_dir_in(&file).unwrap_err().to_string();
        assert!(error.contains("has to be a string"), "{error}");
        let _ = fs::remove_dir_all(&directory);
    }

    #[test]
    fn an_empty_directory_is_the_default() {
        let directory = scratch("empty");
        fs::create_dir_all(&directory).unwrap();
        let file = directory.join(FILE_NAME);
        fs::write(&file, "model_dir = \"\"\n").unwrap();
        assert_eq!(model_dir_in(&file).unwrap(), None);
        let _ = fs::remove_dir_all(&directory);
    }
}
