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

//! What the program keeps: every job it was asked for, what each one made, and the files posted
//! for a job to start from.
//!
//! On the disk, under the output directory, so that it is all still there after a restart:
//!
//! ```text
//! waifu-output/
//!   jobs/<id>/job.json        what was asked, and how it went
//!   jobs/<id>/output.png      what it made: a PNG with its parameters in it, or output.wav
//!   uploads/<id>.png          a picture to draw from, or a .jpg, or a .wav to sound like
//! ```
//!
//! Kept until somebody deletes it, or until the directory is over its limit -- at which point
//! the oldest goes first. Nothing here knows who asked for what. An id is 128 random bits, and
//! whoever holds one can read and delete what it names; which ids belong to whom is the business
//! of whatever sits in front of this, which for the page is the browser that posted them.

use std::collections::VecDeque;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};

/// What a job is for. One program serves one model, so it can do one of these; which one is
/// the task chosen in the terminal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Image,
    Speech,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::Image => "image",
            Kind::Speech => "speech",
        }
    }

    pub fn named(name: &str) -> Option<Kind> {
        match name {
            "image" => Some(Kind::Image),
            "speech" => Some(Kind::Speech),
            _ => None,
        }
    }

    /// The file a job of this kind makes, and what it is.
    fn output(self) -> (&'static str, &'static str) {
        match self {
            Kind::Image => ("output.png", "image/png"),
            Kind::Speech => ("output.wav", "audio/wav"),
        }
    }
}

/// Where a job is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Queued,
    Running,
    Done,
    Failed,
    Cancelled,
}

impl Status {
    pub fn name(self) -> &'static str {
        match self {
            Status::Queued => "queued",
            Status::Running => "running",
            Status::Done => "done",
            Status::Failed => "failed",
            Status::Cancelled => "cancelled",
        }
    }

    fn named(name: &str) -> Option<Status> {
        [
            Status::Queued,
            Status::Running,
            Status::Done,
            Status::Failed,
            Status::Cancelled,
        ]
        .into_iter()
        .find(|status| status.name() == name)
    }

    /// Whether it has stopped moving: nothing will happen to it now but being deleted.
    pub fn finished(self) -> bool {
        matches!(self, Status::Done | Status::Failed | Status::Cancelled)
    }
}

/// One job, as it is kept.
#[derive(Clone, Debug)]
pub struct Job {
    pub id: String,
    pub kind: Kind,
    /// What was asked for, every setting spelled out: what the request said, and the model's
    /// own defaults for what it did not. The worker runs from this and nothing else.
    pub asked: Value,
    pub status: Status,
    /// Milliseconds since 1970.
    pub created: u64,
    pub started: Option<u64>,
    pub finished: Option<u64>,
    /// Why it failed, for one that did.
    pub error: Option<String>,
    /// What it made, described: the picture's size, the clip's length, the line of parameters.
    /// Present for a job that is done.
    pub made: Option<Value>,
    /// How big the output file is.
    pub bytes: u64,
    /// Deleted while it was running: gone as soon as it stops.
    doomed: bool,
}

impl Job {
    /// The file it made, and what it is.
    pub fn output(&self) -> (&'static str, &'static str) {
        self.kind.output()
    }

    /// As it is written to `job.json`, and -- with what only the running program knows added to
    /// it -- as it is answered with.
    pub fn json(&self) -> Value {
        let mut described = json!({
            "id": self.id,
            "kind": self.kind.name(),
            "status": self.status.name(),
            "created": self.created,
            "started": self.started,
            "finished": self.finished,
            "asked": self.asked,
        });
        if let Some(error) = &self.error {
            described["error"] = json!(error);
        }
        if let Some(made) = &self.made {
            let (_, mime) = self.output();
            described["output"] = json!({
                "url": format!("/api/jobs/{}/output", self.id),
                "type": mime,
                "bytes": self.bytes,
                "made": made,
            });
        }

        described
    }

    fn from_json(said: &Value) -> Option<Job> {
        let number = |field: &str| said.get(field).and_then(Value::as_u64);
        Some(Job {
            id: said.get("id")?.as_str()?.to_string(),
            kind: Kind::named(said.get("kind")?.as_str()?)?,
            asked: said.get("asked").cloned().unwrap_or(Value::Null),
            status: Status::named(said.get("status")?.as_str()?)?,
            created: number("created")?,
            started: number("started"),
            finished: number("finished"),
            error: said
                .get("error")
                .and_then(Value::as_str)
                .map(str::to_string),
            made: said
                .get("output")
                .and_then(|output| output.get("made"))
                .cloned(),
            bytes: said
                .get("output")
                .and_then(|output| output.get("bytes"))
                .and_then(Value::as_u64)
                .unwrap_or(0),
            doomed: false,
        })
    }
}

/// A file posted for a job to start from.
#[derive(Clone, Debug)]
pub struct Upload {
    pub id: String,
    /// `image/png`, `image/jpeg` or `audio/wav`, read off the bytes.
    pub mime: &'static str,
    pub bytes: u64,
    pub created: u64,
}

impl Upload {
    pub fn json(&self) -> Value {
        json!({
            "id": self.id,
            "type": self.mime,
            "bytes": self.bytes,
            "created": self.created,
            "url": format!("/api/uploads/{}", self.id),
        })
    }

    fn file(&self) -> String {
        format!("{}.{}", self.id, extension(self.mime))
    }
}

/// What can go wrong keeping something.
#[derive(Debug)]
pub enum Refused {
    /// Not a PNG, a JPEG or a WAV.
    NotAFile,
    Io(io::Error),
}

impl std::fmt::Display for Refused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Refused::NotAFile => write!(f, "that is not a PNG, a JPEG or a WAV file"),
            Refused::Io(error) => write!(f, "{error}"),
        }
    }
}

impl From<io::Error> for Refused {
    fn from(error: io::Error) -> Refused {
        Refused::Io(error)
    }
}

/// The jobs and uploads, in the order they arrived, and the directory they are kept in.
pub struct Store {
    root: PathBuf,
    /// How many bytes of outputs and uploads the directory may hold. `None` is no limit.
    limit: Option<u64>,
    kept: Mutex<Kept>,
}

struct Kept {
    /// Oldest first.
    jobs: VecDeque<Job>,
    uploads: VecDeque<Upload>,
}

impl Store {
    /// The store in `root`, made if it is not there, with whatever an earlier run left in it.
    ///
    /// A job that was queued or running when that run stopped did not finish, and will not: it
    /// is said to have failed, which is what it did.
    pub fn open(root: &Path, limit: Option<u64>) -> io::Result<Store> {
        std::fs::create_dir_all(root.join("jobs"))?;
        std::fs::create_dir_all(root.join("uploads"))?;

        let mut jobs = Vec::new();
        for entry in std::fs::read_dir(root.join("jobs"))? {
            let path = entry?.path().join("job.json");
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue;
            };
            let Some(mut job) = serde_json::from_str(&text)
                .ok()
                .as_ref()
                .and_then(Job::from_json)
            else {
                continue;
            };
            if !job.status.finished() {
                job.status = Status::Failed;
                job.error = Some("the program stopped before it finished".to_string());
                job.finished = Some(now());
                write_job(root, &job)?;
            }
            jobs.push(job);
        }
        jobs.sort_by_key(|job| job.created);

        let mut uploads = Vec::new();
        for entry in std::fs::read_dir(root.join("uploads"))? {
            let entry = entry?;
            let path = entry.path();
            let (Some(id), Some(mime)) = (
                path.file_stem().and_then(|stem| stem.to_str()),
                path.extension()
                    .and_then(|extension| extension.to_str())
                    .and_then(mime_of_extension),
            ) else {
                continue;
            };
            let metadata = entry.metadata()?;
            uploads.push(Upload {
                id: id.to_string(),
                mime,
                bytes: metadata.len(),
                created: metadata
                    .modified()
                    .ok()
                    .and_then(|at| at.duration_since(UNIX_EPOCH).ok())
                    .map(|since| since.as_millis() as u64)
                    .unwrap_or(0),
            });
        }
        uploads.sort_by_key(|upload| upload.created);

        let store = Store {
            root: root.to_path_buf(),
            limit,
            kept: Mutex::new(Kept {
                jobs: jobs.into(),
                uploads: uploads.into(),
            }),
        };
        store.keep_to_the_limit(None);

        Ok(store)
    }

    #[cfg(test)]
    pub fn root(&self) -> &Path {
        &self.root
    }

    fn kept(&self) -> MutexGuard<'_, Kept> {
        self.kept.lock().unwrap_or_else(|held| held.into_inner())
    }

    // -- uploads ----------------------------------------------------------------------------

    /// Keeps `bytes` as an upload, if they are a file a job can start from.
    pub fn upload(&self, bytes: &[u8]) -> Result<Upload, Refused> {
        let mime = sniff(bytes).ok_or(Refused::NotAFile)?;
        let upload = Upload {
            id: a_key(),
            mime,
            bytes: bytes.len() as u64,
            created: now(),
        };
        std::fs::write(self.root.join("uploads").join(upload.file()), bytes)?;

        self.kept().uploads.push_back(upload.clone());
        self.keep_to_the_limit(Some(&upload.id));
        Ok(upload)
    }

    /// Where an upload is on the disk, and what it is.
    pub fn upload_file(&self, id: &str) -> Option<(&'static str, PathBuf)> {
        let kept = self.kept();
        let upload = kept.uploads.iter().find(|upload| upload.id == id)?;
        Some((upload.mime, self.root.join("uploads").join(upload.file())))
    }

    pub fn find_upload(&self, id: &str) -> Option<Upload> {
        self.kept()
            .uploads
            .iter()
            .find(|upload| upload.id == id)
            .cloned()
    }

    /// Deletes an upload. False where there was none by that id.
    pub fn delete_upload(&self, id: &str) -> bool {
        let mut kept = self.kept();
        let Some(at) = kept.uploads.iter().position(|upload| upload.id == id) else {
            return false;
        };
        let upload = kept.uploads.remove(at).expect("found above");
        let _ = std::fs::remove_file(self.root.join("uploads").join(upload.file()));
        true
    }

    // -- jobs -------------------------------------------------------------------------------

    /// A new job, queued. Written to the disk before it is answered with, so that one that was
    /// accepted is one that a restart still knows about.
    ///
    /// Each one a millisecond after the last at least, so that two posted in the same millisecond
    /// still have an order -- which is the order they are listed in, and read back in after a
    /// restart, where nothing else says which came first.
    pub fn submit(&self, kind: Kind, asked: Value) -> io::Result<Job> {
        let mut kept = self.kept();
        let after = kept.jobs.back().map_or(0, |last| last.created + 1);
        let job = Job {
            id: a_key(),
            kind,
            asked,
            status: Status::Queued,
            created: now().max(after),
            started: None,
            finished: None,
            error: None,
            made: None,
            bytes: 0,
            doomed: false,
        };
        std::fs::create_dir_all(self.root.join("jobs").join(&job.id))?;
        write_job(&self.root, &job)?;

        kept.jobs.push_back(job.clone());
        Ok(job)
    }

    pub fn find(&self, id: &str) -> Option<Job> {
        self.kept().jobs.iter().find(|job| job.id == id).cloned()
    }

    /// Every job, oldest first.
    pub fn jobs(&self) -> Vec<Job> {
        self.kept().jobs.iter().cloned().collect()
    }

    /// Where a finished job's output is, and what it is.
    pub fn output_file(&self, id: &str) -> Option<(&'static str, PathBuf)> {
        let kept = self.kept();
        let job = kept.jobs.iter().find(|job| job.id == id)?;
        job.made.as_ref()?;
        let (file, mime) = job.output();
        Some((mime, self.root.join("jobs").join(&job.id).join(file)))
    }

    /// Marks a job as running, and says what it asked for. None where it is not queued any more.
    pub fn start(&self, id: &str) -> Option<Job> {
        self.change(id, |job| {
            (job.status == Status::Queued).then(|| {
                job.status = Status::Running;
                job.started = Some(now());
            })
        })?;
        self.find(id)
    }

    /// Keeps what a job made, and marks it done.
    pub fn done(&self, id: &str, bytes: &[u8], made: Value) -> io::Result<()> {
        let Some(job) = self.find(id) else {
            return Ok(());
        };
        let (file, _) = job.output();
        std::fs::write(self.root.join("jobs").join(id).join(file), bytes)?;

        self.change(id, |job| {
            job.status = Status::Done;
            job.finished = Some(now());
            job.made = Some(made);
            job.bytes = bytes.len() as u64;
            Some(())
        });
        self.keep_to_the_limit(Some(id));
        Ok(())
    }

    pub fn failed(&self, id: &str, error: impl Into<String>) {
        let error = error.into();
        self.change(id, |job| {
            job.status = Status::Failed;
            job.finished = Some(now());
            job.error = Some(error);
            Some(())
        });
    }

    /// Marks a job cancelled. Only one that has not finished: a job that is done stays done.
    pub fn cancelled(&self, id: &str) -> bool {
        self.change(id, |job| {
            (!job.status.finished()).then(|| {
                job.status = Status::Cancelled;
                job.finished = Some(now());
            })
        })
        .is_some()
    }

    /// Deletes a job and what it made. One that is running is deleted when it stops: its output
    /// is being written into the directory this would remove. False where there is none.
    pub fn delete(&self, id: &str) -> bool {
        let mut kept = self.kept();
        let Some(at) = kept.jobs.iter().position(|job| job.id == id) else {
            return false;
        };
        if kept.jobs[at].status == Status::Running {
            kept.jobs[at].doomed = true;
            return true;
        }
        kept.jobs.remove(at);
        let _ = std::fs::remove_dir_all(self.root.join("jobs").join(id));
        true
    }

    /// Changes a job, writes it out, and deletes it if it was deleted while it ran and has now
    /// stopped. `change` answering None changes nothing.
    fn change(&self, id: &str, change: impl FnOnce(&mut Job) -> Option<()>) -> Option<()> {
        let mut kept = self.kept();
        let at = kept.jobs.iter().position(|job| job.id == id)?;
        let job = &mut kept.jobs[at];
        change(job)?;

        if job.doomed && job.status.finished() {
            kept.jobs.remove(at);
            let _ = std::fs::remove_dir_all(self.root.join("jobs").join(id));
            return Some(());
        }
        if let Err(error) = write_job(&self.root, job) {
            crate::cli::webui::log::line(format_args!(
                "error: could not write down job {id}: {error}"
            ));
        }
        Some(())
    }

    // -- the limit ----------------------------------------------------------------------------

    /// How many bytes of outputs and uploads are kept.
    pub fn used(&self) -> u64 {
        let kept = self.kept();
        kept.jobs.iter().map(|job| job.bytes).sum::<u64>()
            + kept.uploads.iter().map(|upload| upload.bytes).sum::<u64>()
    }

    pub fn limit(&self) -> Option<u64> {
        self.limit
    }

    /// Deletes the oldest of what is kept until it is under the limit.
    ///
    /// Only what is finished with: a job that is done, failed or cancelled, and an upload no
    /// queued or running job starts from. Never `keeping`, which is what was just added -- a
    /// single file bigger than the whole limit is still somebody's picture, and it stays until
    /// the next one pushes it out.
    fn keep_to_the_limit(&self, keeping: Option<&str>) {
        let Some(limit) = self.limit else {
            return;
        };
        let mut kept = self.kept();

        loop {
            let used = kept.jobs.iter().map(|job| job.bytes).sum::<u64>()
                + kept.uploads.iter().map(|upload| upload.bytes).sum::<u64>();
            if used <= limit {
                return;
            }

            let wanted: Vec<String> = kept
                .jobs
                .iter()
                .filter(|job| !job.status.finished())
                .flat_map(|job| uploads_named(&job.asked))
                .collect();

            let oldest_job = kept
                .jobs
                .iter()
                .filter(|job| job.status.finished() && job.bytes > 0)
                .filter(|job| Some(job.id.as_str()) != keeping)
                .map(|job| (job.created, job.id.clone()))
                .next();
            let oldest_upload = kept
                .uploads
                .iter()
                .filter(|upload| Some(upload.id.as_str()) != keeping)
                .filter(|upload| !wanted.contains(&upload.id))
                .map(|upload| (upload.created, upload.id.clone()))
                .next();

            match (oldest_job, oldest_upload) {
                (Some(job), Some(upload)) if upload.0 < job.0 => {
                    self.forget_upload(&mut kept, &upload.1)
                }
                (Some(job), _) => self.forget_job(&mut kept, &job.1),
                (None, Some(upload)) => self.forget_upload(&mut kept, &upload.1),
                (None, None) => return,
            }
        }
    }

    fn forget_job(&self, kept: &mut Kept, id: &str) {
        crate::cli::webui::log::line(format_args!(
            "note: over the output limit, deleting job {id}"
        ));
        kept.jobs.retain(|job| job.id != id);
        let _ = std::fs::remove_dir_all(self.root.join("jobs").join(id));
    }

    fn forget_upload(&self, kept: &mut Kept, id: &str) {
        crate::cli::webui::log::line(format_args!(
            "note: over the output limit, deleting upload {id}"
        ));
        if let Some(at) = kept.uploads.iter().position(|upload| upload.id == id) {
            let upload = kept.uploads.remove(at).expect("found above");
            let _ = std::fs::remove_file(self.root.join("uploads").join(upload.file()));
        }
    }
}

/// The uploads a job's settings name, which are what it will read when it runs.
fn uploads_named(asked: &Value) -> Vec<String> {
    ["init_image", "reference"]
        .into_iter()
        .filter_map(|field| asked.get(field).and_then(Value::as_str))
        .map(str::to_string)
        .collect()
}

fn write_job(root: &Path, job: &Job) -> io::Result<()> {
    let directory = root.join("jobs").join(&job.id);
    // Written beside and then moved over, so that a program stopped mid-write leaves the last
    // good one rather than half of a new one.
    let partial = directory.join("job.json.partial");
    std::fs::write(&partial, job.json().to_string())?;
    std::fs::rename(partial, directory.join("job.json"))
}

/// What a file is, from the bytes every file of its kind starts with.
fn sniff(bytes: &[u8]) -> Option<&'static str> {
    if bytes.starts_with(&[0x89, b'P', b'N', b'G']) {
        Some("image/png")
    } else if bytes.starts_with(&[0xff, 0xd8, 0xff]) {
        Some("image/jpeg")
    } else if bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WAVE" {
        Some("audio/wav")
    } else {
        None
    }
}

fn extension(mime: &str) -> &'static str {
    match mime {
        "image/png" => "png",
        "image/jpeg" => "jpg",
        _ => "wav",
    }
}

fn mime_of_extension(extension: &str) -> Option<&'static str> {
    match extension {
        "png" => Some("image/png"),
        "jpg" => Some("image/jpeg"),
        "wav" => Some("audio/wav"),
        _ => None,
    }
}

/// Milliseconds since 1970.
pub fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|since| since.as_millis() as u64)
        .unwrap_or(0)
}

/// A fresh id: 128 bits, as hex.
///
/// It is the whole of what makes a job or an upload readable by whoever holds it and nobody else,
/// so it comes from the operating system where there is a `/dev/urandom` to read. Elsewhere it is
/// two finished `RandomState` hashers -- keyed from the operating system's randomness as well,
/// once per thread, and a keyed hash that nothing outside this process has the key to.
pub fn a_key() -> String {
    use std::io::Read;

    let mut bytes = [0u8; 16];
    let from_the_system = std::fs::File::open("/dev/urandom")
        .and_then(|mut random| random.read_exact(&mut bytes))
        .is_ok();

    if !from_the_system {
        use std::hash::{BuildHasher, Hasher};
        let mut halves = (0..2).map(|_| {
            std::collections::hash_map::RandomState::new()
                .build_hasher()
                .finish()
        });
        bytes[..8].copy_from_slice(&halves.next().unwrap_or_default().to_le_bytes());
        bytes[8..].copy_from_slice(&halves.next().unwrap_or_default().to_le_bytes());
    }

    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Whether `id` looks like one this store gives out: 32 hex digits, and nothing a path could
    /// be made of.
    fn looks_like_an_id(id: &str) -> bool {
        id.len() == 32 && id.bytes().all(|byte| byte.is_ascii_hexdigit())
    }

    /// An empty store in a directory of the calling test's own.
    fn a_store(called: &str, limit: Option<u64>) -> Store {
        let root =
            std::env::temp_dir().join(format!("libwaifu-store-{called}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        Store::open(&root, limit).expect("a store")
    }

    const PNG: &[u8] = &[0x89, b'P', b'N', b'G', 0, 0, 0, 0];

    fn a_done_job(store: &Store, bytes: usize) -> String {
        let job = store
            .submit(Kind::Image, json!({"prompt": "a cat"}))
            .unwrap();
        store.start(&job.id).unwrap();
        store
            .done(&job.id, &vec![7; bytes], json!({"seed": 1}))
            .unwrap();
        job.id
    }

    #[test]
    fn a_job_goes_from_queued_to_done_and_keeps_what_it_made() {
        let store = a_store("lifecycle", None);
        let job = store
            .submit(Kind::Image, json!({"prompt": "a cat"}))
            .unwrap();
        assert_eq!(job.status, Status::Queued);
        assert!(looks_like_an_id(&job.id));
        assert!(store.output_file(&job.id).is_none());

        let started = store.start(&job.id).expect("it was queued");
        assert_eq!(started.status, Status::Running);
        assert_eq!(started.asked["prompt"], "a cat");
        // Not started twice.
        assert!(store.start(&job.id).is_none());

        store.done(&job.id, PNG, json!({"seed": 7})).unwrap();
        let done = store.find(&job.id).unwrap();
        assert_eq!(done.status, Status::Done);
        let described = done.json();
        assert_eq!(
            described["output"]["url"],
            format!("/api/jobs/{}/output", job.id)
        );
        assert_eq!(described["output"]["type"], "image/png");
        assert_eq!(described["output"]["made"]["seed"], 7);

        let (mime, path) = store.output_file(&job.id).unwrap();
        assert_eq!(mime, "image/png");
        assert_eq!(std::fs::read(path).unwrap(), PNG);
    }

    #[test]
    fn what_is_kept_is_still_there_after_a_restart_and_what_was_running_failed() {
        let store = a_store("restart", None);
        let done = a_done_job(&store, 3);
        let running = store.submit(Kind::Image, json!({})).unwrap();
        store.start(&running.id);
        let queued = store.submit(Kind::Speech, json!({"text": "hi"})).unwrap();
        let upload = store.upload(PNG).unwrap();
        let root = store.root().to_path_buf();
        drop(store);

        let again = Store::open(&root, None).unwrap();
        assert_eq!(again.find(&done).unwrap().status, Status::Done);
        assert!(again.output_file(&done).is_some());
        for id in [&running.id, &queued.id] {
            let job = again.find(id).unwrap();
            assert_eq!(job.status, Status::Failed);
            assert!(job.error.unwrap().contains("stopped before it finished"));
        }
        assert_eq!(again.find(&queued.id).unwrap().kind, Kind::Speech);
        assert_eq!(again.find_upload(&upload.id).unwrap().mime, "image/png");
        // In the order they arrived.
        let order: Vec<_> = again.jobs().into_iter().map(|job| job.id).collect();
        assert_eq!(order, vec![done, running.id, queued.id]);
    }

    #[test]
    fn deleting_takes_the_files_with_it_and_a_running_job_goes_when_it_stops() {
        let store = a_store("delete", None);
        let done = a_done_job(&store, 3);
        let directory = store.root().join("jobs").join(&done);
        assert!(directory.is_dir());

        assert!(store.delete(&done));
        assert!(!directory.exists());
        assert!(store.find(&done).is_none());
        assert!(!store.delete(&done), "nothing by that id any more");

        let running = store.submit(Kind::Image, json!({})).unwrap();
        store.start(&running.id);
        assert!(store.delete(&running.id));
        assert!(store.find(&running.id).is_some(), "still running");
        store.cancelled(&running.id);
        assert!(store.find(&running.id).is_none());
        assert!(!store.root().join("jobs").join(&running.id).exists());
    }

    #[test]
    fn a_finished_job_is_not_cancelled_after_the_fact() {
        let store = a_store("cancel-done", None);
        let done = a_done_job(&store, 1);
        assert!(!store.cancelled(&done));
        assert_eq!(store.find(&done).unwrap().status, Status::Done);

        let queued = store.submit(Kind::Image, json!({})).unwrap();
        assert!(store.cancelled(&queued.id));
        assert_eq!(store.find(&queued.id).unwrap().status, Status::Cancelled);
        assert!(
            store.start(&queued.id).is_none(),
            "a cancelled job does not run"
        );
    }

    #[test]
    fn only_a_picture_or_a_wav_is_kept_as_an_upload() {
        let store = a_store("uploads", None);
        let png = store.upload(PNG).unwrap();
        assert_eq!(png.mime, "image/png");
        let jpeg = store.upload(&[0xff, 0xd8, 0xff, 0xe0]).unwrap();
        assert_eq!(jpeg.mime, "image/jpeg");
        let wav = store.upload(b"RIFF\0\0\0\0WAVEfmt ").unwrap();
        assert_eq!(wav.mime, "audio/wav");
        assert!(matches!(store.upload(b"#!/bin/sh"), Err(Refused::NotAFile)));

        let (mime, path) = store.upload_file(&jpeg.id).unwrap();
        assert_eq!(mime, "image/jpeg");
        assert!(path.ends_with(format!("{}.jpg", jpeg.id)));

        assert!(store.delete_upload(&png.id));
        assert!(store.upload_file(&png.id).is_none());
        assert!(!store.delete_upload(&png.id));
    }

    #[test]
    fn over_the_limit_the_oldest_goes_first() {
        let store = a_store("limit", Some(10));
        let first = a_done_job(&store, 4);
        let second = a_done_job(&store, 4);
        assert!(store.find(&first).is_some());

        // Twelve bytes: the oldest goes, and the one just made stays.
        let third = a_done_job(&store, 4);
        assert!(store.find(&first).is_none());
        assert!(!store.root().join("jobs").join(&first).exists());
        assert!(store.find(&second).is_some());
        assert!(store.find(&third).is_some());
        assert!(store.used() <= 10);
    }

    #[test]
    fn an_upload_a_waiting_job_needs_is_not_what_the_limit_takes() {
        let store = a_store("limit-wanted", Some(12));
        let wanted = store.upload(PNG).unwrap(); // 8 bytes
        store
            .submit(Kind::Image, json!({"init_image": wanted.id}))
            .unwrap();

        // Over the limit, and the only thing old enough to go is wanted: the finished job goes
        // instead, and the new one stays.
        let old = a_done_job(&store, 3);
        let new = a_done_job(&store, 3);
        assert!(store.find_upload(&wanted.id).is_some());
        assert!(store.find(&old).is_none());
        assert!(store.find(&new).is_some());
    }

    #[test]
    fn something_bigger_than_the_whole_limit_is_still_kept_until_the_next_one() {
        let store = a_store("limit-huge", Some(4));
        let huge = a_done_job(&store, 100);
        assert!(store.find(&huge).is_some());

        let next = a_done_job(&store, 1);
        assert!(store.find(&huge).is_none());
        assert!(store.find(&next).is_some());
    }

    #[test]
    fn an_id_is_only_ever_hex() {
        assert!(looks_like_an_id(&a_key()));
        assert!(!looks_like_an_id("../../etc/passwd"));
        assert!(!looks_like_an_id("0123"));
        assert!(!looks_like_an_id(&"g".repeat(32)));
        assert_ne!(a_key(), a_key());
    }
}
