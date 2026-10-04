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

use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Condvar, Mutex, MutexGuard};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};

/// What a job is for. One program serves one model, so it can do one of these; which one is
/// the task chosen in the terminal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Image,
    Speech,
    /// One recording said again in the voice of another.
    Conversion,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::Image => "image",
            Kind::Speech => "speech",
            Kind::Conversion => "conversion",
        }
    }

    pub fn named(name: &str) -> Option<Kind> {
        match name {
            "image" => Some(Kind::Image),
            "speech" => Some(Kind::Speech),
            "conversion" => Some(Kind::Conversion),
            _ => None,
        }
    }

    /// The file a job of this kind makes, and what it is.
    fn output(self) -> (&'static str, &'static str) {
        match self {
            Kind::Image => ("output.png", "image/png"),
            Kind::Speech | Kind::Conversion => ("output.wav", "audio/wav"),
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
    /// Where it comes in the order of jobs: when it was posted, and its id for two posted in the
    /// same millisecond.
    fn order(&self) -> (u64, &str) {
        (self.created, &self.id)
    }

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

/// The jobs and uploads, by id, and the directory they are kept in.
///
/// One map for each and nothing beside them: whether a job is waiting its turn is its status, and
/// which came first is its `(created, id)`. The line of waiting jobs, the order jobs are listed
/// in, and which are the oldest when the directory is over its limit are all found by looking --
/// a pass over a few thousand jobs at most, which is nothing beside the job it is finding.
///
/// `(created, id)` rather than `created` alone, since two jobs can be posted in the same
/// millisecond: which of those two goes first is not something anybody could tell, but it is the
/// same answer every time it is asked, and after a restart too.
pub struct Store {
    root: PathBuf,
    /// How many bytes of outputs and uploads the directory may hold. `None` is no limit.
    limit: Option<u64>,
    kept: Mutex<Kept>,
    /// Rung when a job is queued, which is what the worker waits on between jobs.
    queued: Condvar,
}

struct Kept {
    jobs: HashMap<String, Job>,
    uploads: HashMap<String, Upload>,
}

/// What asking a job to stop came to.
#[derive(Debug, PartialEq, Eq)]
pub enum Cancelled {
    /// It was waiting, and is not any more.
    Dequeued,
    /// It is running, and will stop after the step it is on.
    Stopping,
    /// It is neither: finished already, or never was.
    NotGoing,
}

impl Store {
    /// The store in `root`, made if it is not there, with whatever an earlier run left in it.
    ///
    /// A job that was queued or running when that run stopped did not finish, and will not: it
    /// is said to have failed, which is what it did.
    pub fn open(root: &Path, limit: Option<u64>) -> io::Result<Store> {
        std::fs::create_dir_all(root.join("jobs"))?;
        std::fs::create_dir_all(root.join("uploads"))?;

        let mut jobs = HashMap::new();
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
            jobs.insert(job.id.clone(), job);
        }

        let mut uploads = HashMap::new();
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
            let upload = Upload {
                id: id.to_string(),
                mime,
                bytes: metadata.len(),
                created: metadata
                    .modified()
                    .ok()
                    .and_then(|at| at.duration_since(UNIX_EPOCH).ok())
                    .map(|since| since.as_millis() as u64)
                    .unwrap_or(0),
            };
            uploads.insert(upload.id.clone(), upload);
        }

        let store = Store {
            root: root.to_path_buf(),
            limit,
            kept: Mutex::new(Kept { jobs, uploads }),
            queued: Condvar::new(),
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

        self.kept()
            .uploads
            .insert(upload.id.clone(), upload.clone());
        self.keep_to_the_limit(Some(&upload.id));
        Ok(upload)
    }

    /// Where an upload is on the disk, and what it is.
    pub fn upload_file(&self, id: &str) -> Option<(&'static str, PathBuf)> {
        let kept = self.kept();
        let upload = kept.uploads.get(id)?;
        Some((upload.mime, self.root.join("uploads").join(upload.file())))
    }

    pub fn find_upload(&self, id: &str) -> Option<Upload> {
        self.kept().uploads.get(id).cloned()
    }

    /// Deletes an upload. False where there was none by that id.
    pub fn delete_upload(&self, id: &str) -> bool {
        let Some(upload) = self.kept().uploads.remove(id) else {
            return false;
        };
        let _ = std::fs::remove_file(self.root.join("uploads").join(upload.file()));
        true
    }

    // -- jobs -------------------------------------------------------------------------------

    /// A new job, queued, and the worker woken for it. Written to the disk before it is answered
    /// with, so that one that was accepted is one that a restart still knows about.
    pub fn submit(&self, kind: Kind, asked: Value) -> io::Result<Job> {
        let job = Job {
            id: a_key(),
            kind,
            asked,
            status: Status::Queued,
            created: now(),
            started: None,
            finished: None,
            error: None,
            made: None,
            bytes: 0,
            doomed: false,
        };
        std::fs::create_dir_all(self.root.join("jobs").join(&job.id))?;
        write_job(&self.root, &job)?;

        self.kept().jobs.insert(job.id.clone(), job.clone());
        self.queued.notify_one();
        Ok(job)
    }

    pub fn find(&self, id: &str) -> Option<Job> {
        self.kept().jobs.get(id).cloned()
    }

    /// Every job, oldest first.
    pub fn jobs(&self) -> Vec<Job> {
        let mut jobs: Vec<Job> = self.kept().jobs.values().cloned().collect();
        jobs.sort_by(|a, b| a.order().cmp(&b.order()));
        jobs
    }

    /// Where a finished job's output is, and what it is.
    pub fn output_file(&self, id: &str) -> Option<(&'static str, PathBuf)> {
        let kept = self.kept();
        let job = kept.jobs.get(id)?;
        job.made.as_ref()?;
        let (file, mime) = job.output();
        Some((mime, self.root.join("jobs").join(&job.id).join(file)))
    }

    /// How many jobs are waiting their turn.
    pub fn queued(&self) -> usize {
        self.kept()
            .jobs
            .values()
            .filter(|job| job.status == Status::Queued)
            .count()
    }

    /// Where a queued job is in line: nought for next. None for one that is not waiting.
    pub fn position(&self, id: &str) -> Option<usize> {
        let kept = self.kept();
        let job = kept.jobs.get(id)?;
        (job.status == Status::Queued).then(|| {
            kept.jobs
                .values()
                .filter(|other| other.status == Status::Queued && other.order() < job.order())
                .count()
        })
    }

    /// The job that has waited longest, marked as running -- waiting for one where none is.
    ///
    /// `taken` is called with it before the lock is let go of, which is where whatever else has to
    /// know it is running is told: a request asking it to stop, which takes the same lock, then
    /// finds it either still waiting or running and known to be, and never in between.
    pub fn take_next(&self, taken: impl FnOnce(&Job)) -> Job {
        let mut kept = self.kept();
        loop {
            let next = kept
                .jobs
                .values()
                .filter(|job| job.status == Status::Queued)
                .min_by(|a, b| a.order().cmp(&b.order()))
                .map(|job| job.id.clone());

            if let Some(id) = next {
                let job = kept.jobs.get_mut(&id).expect("found above");
                job.status = Status::Running;
                job.started = Some(now());
                let job = job.clone();
                self.write_down(&job);
                taken(&job);
                return job;
            }

            kept = self
                .queued
                .wait(kept)
                .unwrap_or_else(|held| held.into_inner());
        }
    }

    /// Asks a job to stop: marked cancelled where it is waiting, and `stop` called where it is
    /// running, to ask it to stop between steps. Under the one lock [`Store::take_next`] takes a
    /// job under, so that it is always one or the other here.
    pub fn cancel(&self, id: &str, stop: impl FnOnce()) -> Cancelled {
        let mut kept = self.kept();
        let Some(job) = kept.jobs.get_mut(id) else {
            return Cancelled::NotGoing;
        };
        match job.status {
            Status::Queued => {
                job.status = Status::Cancelled;
                job.finished = Some(now());
                let job = job.clone();
                self.write_down(&job);
                Cancelled::Dequeued
            }
            Status::Running => {
                stop();
                Cancelled::Stopping
            }
            _ => Cancelled::NotGoing,
        }
    }

    /// Marks a job as running, by id. The worker takes the next one with [`Store::take_next`];
    /// this is for a test that wants a particular one done.
    #[cfg(test)]
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

    /// Marks a job cancelled: one that was running and stopped. A job that is done stays done.
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
        let Some(job) = kept.jobs.get_mut(id) else {
            return false;
        };
        if job.status == Status::Running {
            job.doomed = true;
            return true;
        }
        kept.jobs.remove(id);
        let _ = std::fs::remove_dir_all(self.root.join("jobs").join(id));
        true
    }

    /// Changes a job, writes it out, and deletes it if it was deleted while it ran and has now
    /// stopped. `change` answering None changes nothing.
    fn change(&self, id: &str, change: impl FnOnce(&mut Job) -> Option<()>) -> Option<()> {
        let mut kept = self.kept();
        let job = kept.jobs.get_mut(id)?;
        change(job)?;

        if job.doomed && job.status.finished() {
            kept.jobs.remove(id);
            let _ = std::fs::remove_dir_all(self.root.join("jobs").join(id));
            return Some(());
        }
        let job = job.clone();
        self.write_down(&job);
        Some(())
    }

    /// Writes a job out, saying so in the terminal where it could not be: what is kept in memory
    /// is still right, and the next change writes it again.
    fn write_down(&self, job: &Job) {
        if let Err(error) = write_job(&self.root, job) {
            crate::cli::webui::log::line(format_args!(
                "error: could not write down job {}: {error}",
                job.id
            ));
        }
    }

    // -- the limit ----------------------------------------------------------------------------

    /// How many bytes of outputs and uploads are kept.
    pub fn used(&self) -> u64 {
        let kept = self.kept();
        kept.jobs.values().map(|job| job.bytes).sum::<u64>()
            + kept
                .uploads
                .values()
                .map(|upload| upload.bytes)
                .sum::<u64>()
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

        let mut used = kept.jobs.values().map(|job| job.bytes).sum::<u64>()
            + kept
                .uploads
                .values()
                .map(|upload| upload.bytes)
                .sum::<u64>();
        if used <= limit {
            return;
        }

        let wanted: Vec<String> = kept
            .jobs
            .values()
            .filter(|job| !job.status.finished())
            .flat_map(|job| uploads_named(&job.asked))
            .collect();

        // Everything that may go, oldest first: finished jobs with something to free, and uploads
        // nothing waiting needs.
        let mut going: Vec<(u64, String, bool)> = kept
            .jobs
            .values()
            .filter(|job| job.status.finished() && job.bytes > 0)
            .map(|job| (job.created, job.id.clone(), true))
            .chain(
                kept.uploads
                    .values()
                    .filter(|upload| !wanted.contains(&upload.id))
                    .map(|upload| (upload.created, upload.id.clone(), false)),
            )
            .filter(|(_, id, _)| Some(id.as_str()) != keeping)
            .collect();
        going.sort();

        for (_, id, is_a_job) in going {
            if used <= limit {
                return;
            }
            used -= match is_a_job {
                true => self.forget_job(&mut kept, &id),
                false => self.forget_upload(&mut kept, &id),
            };
        }
    }

    /// Deletes a job for the limit, and says how many bytes that freed.
    fn forget_job(&self, kept: &mut Kept, id: &str) -> u64 {
        crate::cli::webui::log::line(format_args!(
            "note: over the output limit, deleting job {id}"
        ));
        let _ = std::fs::remove_dir_all(self.root.join("jobs").join(id));
        kept.jobs.remove(id).map_or(0, |job| job.bytes)
    }

    fn forget_upload(&self, kept: &mut Kept, id: &str) -> u64 {
        crate::cli::webui::log::line(format_args!(
            "note: over the output limit, deleting upload {id}"
        ));
        let Some(upload) = kept.uploads.remove(id) else {
            return 0;
        };
        let _ = std::fs::remove_file(self.root.join("uploads").join(upload.file()));
        upload.bytes
    }
}

/// The uploads a job's settings name, which are what it will read when it runs.
fn uploads_named(asked: &Value) -> Vec<String> {
    ["init_image", "reference", "source"]
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

    /// A finished job, posted a moment after the last so that which is older is the order they
    /// were made in: two in the same millisecond go by id, which would make "the oldest" a coin
    /// toss.
    fn a_done_job(store: &Store, bytes: usize) -> String {
        std::thread::sleep(std::time::Duration::from_millis(2));
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
        // All three, in the same order as before the restart: by when they were posted, and by id
        // for any posted in the same millisecond.
        let before: Vec<_> = [&done, &running.id, &queued.id]
            .into_iter()
            .map(|id| {
                let job = again.find(id).unwrap();
                (job.created, job.id)
            })
            .collect();
        let mut sorted = before.clone();
        sorted.sort();
        let listed: Vec<_> = again
            .jobs()
            .into_iter()
            .map(|job| (job.created, job.id))
            .collect();
        assert_eq!(listed, sorted);
    }

    #[test]
    fn the_worker_is_handed_the_job_that_has_waited_longest() {
        let store = a_store("take-next", None);
        let first = store.submit(Kind::Image, json!({})).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(2));
        let second = store.submit(Kind::Image, json!({})).unwrap();
        assert_eq!(store.queued(), 2);
        assert_eq!(store.position(&first.id), Some(0));
        assert_eq!(store.position(&second.id), Some(1));

        let mut told = None;
        let taken = store.take_next(|job| told = Some(job.id.clone()));
        assert_eq!(taken.id, first.id);
        assert_eq!(taken.status, Status::Running);
        assert_eq!(told.as_deref(), Some(first.id.as_str()));
        assert_eq!(store.find(&first.id).unwrap().status, Status::Running);
        assert_eq!(store.position(&first.id), None);
        assert_eq!(store.position(&second.id), Some(0));
        assert_eq!(store.queued(), 1);
    }

    #[test]
    fn two_jobs_posted_in_the_same_millisecond_go_by_id() {
        // Which of the two came first is not something anybody could tell. What matters is that
        // the answer is the same every time: by id.
        let store = a_store("same-millisecond", None);
        let a = store.submit(Kind::Image, json!({})).unwrap();
        let b = store.submit(Kind::Image, json!({})).unwrap();
        {
            let mut kept = store.kept();
            let created = kept.jobs[&a.id].created;
            kept.jobs.get_mut(&b.id).unwrap().created = created;
        }
        let (early, late) = match a.id < b.id {
            true => (&a.id, &b.id),
            false => (&b.id, &a.id),
        };

        assert_eq!(store.position(early), Some(0));
        assert_eq!(store.position(late), Some(1));
        assert_eq!(&store.take_next(|_| {}).id, early);
    }

    #[test]
    fn asking_a_job_to_stop_depends_on_where_it_is() {
        let store = a_store("cancel-where", None);
        let waiting = store.submit(Kind::Image, json!({})).unwrap();
        let mut stopped = false;
        assert_eq!(
            store.cancel(&waiting.id, || stopped = true),
            Cancelled::Dequeued
        );
        assert!(!stopped, "nothing was running to stop");
        assert_eq!(store.find(&waiting.id).unwrap().status, Status::Cancelled);
        assert_eq!(store.queued(), 0);

        let running = store.submit(Kind::Image, json!({})).unwrap();
        store.take_next(|_| {});
        assert_eq!(
            store.cancel(&running.id, || stopped = true),
            Cancelled::Stopping
        );
        assert!(stopped);
        // Still running until it stops between steps and says so.
        assert_eq!(store.find(&running.id).unwrap().status, Status::Running);

        assert_eq!(store.cancel(&waiting.id, || {}), Cancelled::NotGoing);
        assert_eq!(store.cancel(&"0".repeat(32), || {}), Cancelled::NotGoing);
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
    fn every_kind_is_kept_under_a_name_it_is_read_back_by() {
        for kind in [Kind::Image, Kind::Speech, Kind::Conversion] {
            assert_eq!(Kind::named(kind.name()), Some(kind));
        }
        // A conversion makes a recording, as a reading does.
        assert_eq!(Kind::Conversion.output(), Kind::Speech.output());
    }

    #[test]
    fn the_recordings_a_waiting_conversion_needs_are_not_what_the_limit_takes() {
        let store = a_store("limit-conversion", Some(20));
        let source = store.upload(PNG).unwrap();
        let reference = store.upload(PNG).unwrap();
        store
            .submit(
                Kind::Conversion,
                json!({"source": source.id, "reference": reference.id}),
            )
            .unwrap();

        a_done_job(&store, 3);
        a_done_job(&store, 3);
        assert!(store.find_upload(&source.id).is_some());
        assert!(store.find_upload(&reference.id).is_some());
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
