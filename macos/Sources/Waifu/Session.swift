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

// One window's worth of working with the loaded model: what is in the boxes, the jobs asked for
// and what they made, and the machine they run on.
//
// A job is a call on the engine, awaited in a task of its own: its progress arrives as it goes, and
// cancelling the task cancels the job. What a job makes is this app's to keep -- the engine keeps
// nothing -- so each one is written to the output folder as it finishes, a PNG or a WAV beside a
// JSON of what made it, and read back from there when a window opens on a model of the same kind.

import AppKit
import Foundation
import Observation
import WaifuKit

/// The tabs, which what the model does decides.
enum Tab: String, CaseIterable, Identifiable {
    case txt2img, img2img, text2speech, speech2speech

    var id: String { rawValue }

    var symbol: String {
        switch self {
        case .txt2img: "text.below.photo"
        case .img2img: "photo.on.rectangle"
        case .text2speech: "waveform"
        case .speech2speech: "person.wave.2"
        }
    }

    /// What a job of this tab is, as jobs are kept.
    var kind: String {
        switch self {
        case .txt2img, .img2img: "image"
        case .text2speech: "speech"
        case .speech2speech: "conversion"
        }
    }
}

/// What is in the boxes. Everything here is somebody's typing until it is sent.
struct JobForm {
    var prompt = ""
    var negative = ""
    var steps = 20
    var guidance = 7.0
    var strength = 0.8
    var seed = "-1"
    var width = 1024
    var height = 1024
    var text = ""
    var speed = 1.0
    var temperature = 0.8
    var conversionSteps = 30
    var convertStyle = false
    var style = ""
}

/// A picture or a recording this window is holding to start from: as the engine takes it, and as
/// the screen shows or plays it.
struct Held {
    var bytes: Data
    var image: NSImage?
    var rgb: RGBImage?
    var clip: AudioClip?
}

/// How far along things are, as the bar draws it.
struct BarState {
    var busy = false
    var fraction: Double?
    var doing = ""
    var seconds: Double?
    var stopping = false
}

@MainActor @Observable
final class Session {
    /// The engine, and what is loaded into it.
    let launcher = Launcher()
    /// The model chosen for each kind of job -- "image", "speech", "conversion" -- as it is asked
    /// for: a published name or a manifest's path. Kept from one run of the app to the next.
    private(set) var chosen: [String: String] = [:]
    /// What each chosen model is, as the library says without reading it -- which is all the
    /// settings need. It is fetched and read when a job first needs it.
    private(set) var described: [String: ModelInfo] = [:]
    /// Why a chosen model could not be described: a name nobody published, a manifest moved.
    private(set) var notDescribed: [String: String] = [:]
    /// What the models run on.
    var device: String {
        didSet { UserDefaults.standard.set(device, forKey: "device") }
    }
    /// The model the tab on show is for.
    var model: ModelInfo? { described[tab.kind] }
    /// Where what this window makes is kept, and what it made before is read back from.
    let folder: URL

    /// This window's jobs and those read back for the model's kinds, newest first.
    private(set) var jobs: [Job] = []
    private(set) var machine: Machine?
    /// What went wrong with the last thing that was clicked.
    var complaint: String?

    var tab: Tab = .txt2img
    /// Whether the Model Manager page is on show rather than a task.
    var showingModels = false
    var form = JobForm()

    /// The picture in the big frame and the clip in the player, until something new comes in.
    var showing: String?
    var playing: String?

    private(set) var picture: Held?
    private(set) var recording: Held?
    private(set) var source: Held?

    /// What finished jobs made, read once each.
    private(set) var outputs: [String: Data] = [:]
    private(set) var images: [String: NSImage] = [:]

    /// The running job's last report.
    private var progress: (job: String, said: JobProgress)?
    /// Each job's task, for cancelling it.
    private var tasks: [String: Task<Void, Never>] = [:]
    private var stopping: Set<String> = []
    private var polling: Task<Void, Never>?
    /// What the prompts were last filled in with, so that a box still holding it is the model's
    /// and a box holding anything else is somebody's.
    private var suggested: (prompt: String, negative: String) = ("", "")

    /// The last job asked for, which the next waits on: one at a time, so that a model is loaded
    /// for the job that needs it and not out from under the one before.
    private var last: Task<Void, Never>?

    static let kinds = ["image", "speech", "conversion"]

    init(folder: URL) {
        self.folder = folder
        // Before any model is described: every published model's manifest, fetched when the app
        // was built, so that what each suggests is in the boxes the moment it is chosen.
        if let manifests = Bundle.main.resourceURL?.appendingPathComponent("manifests"),
            FileManager.default.fileExists(atPath: manifests.path)
        {
            try? ModelManager.setBundledManifests(manifests.path)
        }
        device = UserDefaults.standard.string(forKey: "device") ?? "auto"
        for kind in Session.kinds {
            if let name = UserDefaults.standard.string(forKey: "model.\(kind)"), !name.isBlank {
                chosen[kind] = name
                describe(kind)
                if let info = described[kind] { adopt(info) }
            }
        }
        // The one model the launch screen kept, before there was one for each kind of job.
        if chosen.isEmpty, let old = UserDefaults.standard.string(forKey: "model"), !old.isBlank {
            choose(old)
            UserDefaults.standard.removeObject(forKey: "model")
        }
        readHistory()
        polling = Task { [weak self] in
            while !Task.isCancelled {
                let machine = await Task.detached { Machine.now() }.value
                if let machine { self?.machine = machine }
                try? await Task.sleep(for: .seconds(2))
            }
        }
    }

    // MARK: - the model

    /// Chooses `name` for jobs of `kind` -- or, where no kind is given, for whichever it is first,
    /// showing that tab -- and puts its own numbers in the boxes. Nothing is fetched or read: the
    /// first job that needs it does that.
    func choose(_ name: String, for kind: String? = nil) {
        let name = name.trimmingCharacters(in: .whitespaces)
        guard !name.isBlank else { return }
        var kind = kind
        if kind == nil {
            guard let info = try? ModelInfo.describe(name, device: device) else {
                complaint = "\(name) is neither a published model nor a manifest on this Mac"
                return
            }
            kind = info.kind
            tab = tabs.first { $0.kind == info.kind } ?? tab
        }
        guard let kind else { return }
        chosen[kind] = name
        UserDefaults.standard.set(name, forKey: "model.\(kind)")
        describe(kind)
        if let info = described[kind] { adopt(info) }
    }

    private func describe(_ kind: String) {
        guard let name = chosen[kind] else { return }
        do {
            described[kind] = try ModelInfo.describe(name, device: device)
            notDescribed[kind] = nil
        } catch {
            described[kind] = nil
            notDescribed[kind] = error.localizedDescription
        }
    }

    /// Describes `name` again wherever it is chosen, now that it has been fetched and read: what
    /// was guessed from its name is what its package says now. The boxes are left as they are.
    private func refresh(_ name: String) {
        for (kind, chosen) in chosen where chosen == name { describe(kind) }
    }

    /// Puts the model's own numbers in the boxes, and its card's suggestions in the prompts: in
    /// place of the last model's suggestions, or into an empty box, but never over something
    /// somebody typed.
    private func adopt(_ model: ModelInfo) {
        if let chosen = model.model {
            form.width = chosen.width
            form.height = chosen.height
            form.steps = chosen.steps
            form.guidance = chosen.guidance
            if form.prompt.isBlank || form.prompt == suggested.prompt { form.prompt = chosen.prompt ?? "" }
            if form.negative.isBlank || form.negative == suggested.negative { form.negative = chosen.avoid ?? "" }
            suggested = (chosen.prompt ?? "", chosen.avoid ?? "")
        }
        if let voice = model.voice {
            form.speed = voice.speed
            form.temperature = voice.temperature
        }
        if let converter = model.converter {
            form.conversionSteps = converter.steps
        }
    }

    // MARK: - what is kept on the disk

    private func file(_ id: String, kind: String) -> URL {
        folder.appendingPathComponent("\(id).\(kind == "image" ? "png" : "wav")")
    }

    private func recordFile(_ id: String) -> URL {
        folder.appendingPathComponent("\(id).json")
    }

    /// The jobs kept in the folder, newest first. What each made is read when it is first shown.
    private func readHistory() {
        let names = (try? FileManager.default.contentsOfDirectory(atPath: folder.path)) ?? []
        let decoder = JSONDecoder()
        jobs = names.filter { $0.hasSuffix(".json") }.compactMap { name -> Job? in
            guard let data = try? Data(contentsOf: folder.appendingPathComponent(name)),
                let record = try? decoder.decode(Record.self, from: data)
            else { return nil }
            return Job(
                id: record.id, kind: record.kind, status: .done, created: record.created,
                finished: record.finished, made: record.made)
        }
        .sorted { $0.created > $1.created }
    }

    /// Reads what a finished job made, once.
    func load(_ job: Job) async {
        guard job.isDone, outputs[job.id] == nil else { return }
        let url = file(job.id, kind: job.kind)
        guard let bytes = await Task.detached(operation: { try? Data(contentsOf: url) }).value else { return }
        outputs[job.id] = bytes
        if job.kind == "image" { images[job.id] = NSImage(data: bytes) }
    }

    /// Writes what a job made, and what made it, and shows it.
    private func keep(_ id: String, bytes: Data, made: Made) {
        guard let index = jobs.firstIndex(where: { $0.id == id }) else { return }
        let finished = Date().timeIntervalSince1970 * 1000
        jobs[index].status = .done
        jobs[index].finished = finished
        jobs[index].made = made
        outputs[id] = bytes
        if jobs[index].kind == "image" { images[id] = NSImage(data: bytes) }
        // What was asked for last is what should be on show.
        showing = nil
        playing = nil

        let record = Record(id: id, kind: jobs[index].kind, created: jobs[index].created, finished: finished, made: made)
        do {
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            try bytes.write(to: file(id, kind: record.kind))
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(record).write(to: recordFile(id))
        } catch {
            complaint = "could not keep it in \(folder.path): \(error.localizedDescription)"
        }
    }

    // MARK: - what the screen reads

    var going: [Job] { jobs.filter(\.isGoing) }
    var running: Job? { jobs.first { $0.status == .running } }
    var waiting: [Job] { jobs.filter { $0.status == .queued } }
    /// What is on show in the strip: the current tab's kind only.
    var done: [Job] { jobs.filter { $0.isDone && $0.kind == tab.kind } }

    /// The tasks the sidebar offers. Only drawing from words for now; the others are written and
    /// come back by being listed here.
    var tabs: [Tab] { [.txt2img] }

    /// The picture or clip on show: the one picked, or the newest.
    var shown: Job? {
        let picked = tab.kind == "image" ? showing : playing
        return done.first { $0.id == picked } ?? done.first
    }

    var bar: BarState {
        var bar = BarState(busy: !going.isEmpty)
        // Fetching or reading the model, which the job waits on before it has a step to report.
        if case .loading(_, let line, let fraction) = launcher.state {
            bar.busy = true
            bar.doing = line
            bar.fraction = fraction
            if let running { bar.stopping = stopping.contains(running.id) }
            return bar
        }
        if let running, let progress, progress.job == running.id {
            bar.fraction = progress.said.fraction
            bar.seconds = progress.said.seconds
            bar.doing = progress.said.words
            bar.stopping = stopping.contains(running.id)
        } else if running != nil {
            bar.doing = "starting"
        } else if !waiting.isEmpty {
            bar.doing = "waiting for the job before it"
        }
        return bar
    }

    /// What to say over the output: a complaint about the last click, or what the last job
    /// came to.
    var note: (said: String, bad: Bool)? {
        if let complaint { return (complaint, true) }
        switch jobs.first?.status {
        case .failed: return (jobs.first?.error ?? "it failed", true)
        case .cancelled: return ("stopped where it was", false)
        default: return nil
        }
    }

    /// Whether there is anything of this window's to stop.
    var canStop: Bool { !going.isEmpty }

    /// A job can be asked for while another runs -- it waits its turn -- but not while one is
    /// already waiting: a second click is not a second job.
    var canGo: Bool {
        guard waiting.isEmpty, let model else { return false }
        switch tab {
        case .txt2img:
            return model.model != nil && !form.prompt.isBlank
        case .img2img:
            return model.model?.drawsFromAPicture == true && picture != nil && !form.prompt.isBlank
        case .text2speech:
            return model.voice != nil && !form.text.isBlank
        case .speech2speech:
            return model.converter != nil && source != nil && recording != nil
        }
    }

    // MARK: - doing

    func go() {
        guard canGo else { return }
        switch tab {
        case .txt2img, .img2img: generate()
        case .text2speech: speak()
        case .speech2speech: convert()
        }
    }

    /// The seed in the box, or a fresh one where it asks for one: empty, or the -1 every tool of
    /// this kind takes to mean "surprise me".
    private var seed: UInt64 {
        let typed = form.seed.trimmingCharacters(in: .whitespaces)
        return UInt64(typed) ?? UInt64.random(in: 0...UInt64.max)
    }

    /// Starts a job of `kind` on `model`: listed at once, run when the one before it has ended --
    /// fetching and reading the model first where it is not the one loaded -- and kept when it
    /// ends.
    private func start(
        kind: String, model: String,
        _ run: @escaping (Engine, @escaping ProgressHandler) async throws -> (Data, Made)?
    ) {
        complaint = nil
        let id = UUID().uuidString.lowercased()
        jobs.insert(Job(id: id, kind: kind, status: .queued, created: Date().timeIntervalSince1970 * 1000), at: 0)

        let device = device
        let previous = last
        let task = Task { [weak self] in
            await previous?.value
            guard let self else { return }
            defer {
                tasks[id] = nil
                stopping.remove(id)
            }
            guard !Task.isCancelled else { return ended(id, as: .cancelled) }
            begin(id)

            let report: ProgressHandler = { said in
                Task { @MainActor in self.heard(said, of: id) }
            }
            do {
                let wasLoaded = launcher.has(model, on: device)
                let engine = try await launcher.ensure(model, on: device)
                if !wasLoaded { refresh(model) }
                if let (bytes, made) = try await run(engine, report) {
                    keep(id, bytes: bytes, made: made)
                } else {
                    ended(id, as: .cancelled)
                }
            } catch is CancellationError {
                ended(id, as: .cancelled)
            } catch {
                ended(id, as: .failed, error.localizedDescription)
            }
        }
        tasks[id] = task
        last = task
    }

    /// A job's turn has come.
    private func begin(_ id: String) {
        guard let index = jobs.firstIndex(where: { $0.id == id }), jobs[index].isGoing else { return }
        jobs[index].status = .running
        progress = nil
    }

    private func heard(_ said: JobProgress, of id: String) {
        guard let index = jobs.firstIndex(where: { $0.id == id }), jobs[index].isGoing else { return }
        jobs[index].status = .running
        progress = (id, said)
    }

    private func ended(_ id: String, as status: Job.Status, _ error: String? = nil) {
        guard let index = jobs.firstIndex(where: { $0.id == id }), jobs[index].isGoing else { return }
        jobs[index].status = status
        jobs[index].error = error
        jobs[index].finished = Date().timeIntervalSince1970 * 1000
    }

    private func generate() {
        guard let chosen = model?.model, let modelName = model?.name else { return }
        let seed = seed
        var request = DrawRequest(
            prompt: form.prompt, width: form.width, height: form.height, steps: form.steps,
            guidance: form.guidance, seed: seed)
        // Both or neither, and neither for a model with no second pass to steer.
        if chosen.guided { request.negative = form.negative }
        let fromAPicture = tab == .img2img
        if fromAPicture, let rgb = picture?.rgb {
            request.images = [(key: "start_from", image: rgb)]
            request.strength = form.strength
        }

        var parameters = [form.prompt]
        if chosen.guided, !form.negative.isBlank { parameters.append("Negative prompt: \(form.negative)") }
        var settings = "Steps: \(form.steps), CFG scale: \(chosen.guided ? form.guidance : chosen.guidance), Seed: \(seed), Size: \(form.width)x\(form.height), Model: \(modelName)"
        if fromAPicture { settings += ", Denoising strength: \(form.strength)" }
        parameters.append(settings)
        let made = Made(
            seed: seed, seconds: 0, model: modelName, parameters: parameters.joined(separator: "\n"),
            prompt: form.prompt, negative: chosen.guided ? form.negative : "", width: form.width,
            height: form.height, steps: form.steps, guidance: chosen.guided ? form.guidance : chosen.guidance,
            strength: fromAPicture ? form.strength : nil)

        start(kind: "image", model: modelName) { engine, report in
            let started = Date()
            guard let image = try await engine.draw(request, progress: report),
                let png = Media.png(image)
            else { return nil }
            var made = made
            made.seconds = Date().timeIntervalSince(started)
            return (png, made)
        }
    }

    private func speak() {
        guard let voice = model?.voice, let modelName = model?.name else { return }
        let seed = seed
        let style = (voice.styles ?? []).isEmpty || form.style.isBlank ? nil : form.style.trimmingCharacters(in: .whitespaces)
        let request = SpeakRequest(
            text: form.text, like: recording?.clip, speed: form.speed, temperature: form.temperature,
            style: style, seed: seed)

        var settings = "Speed: \(form.speed), Temperature: \(form.temperature), Seed: \(seed), Rate: \(voice.rate) Hz, Voice: \(modelName)"
        if recording != nil { settings += ", From a recording: yes" }
        if let style { settings += ", Style: \(style)" }
        let made = Made(
            seed: seed, seconds: 0, model: modelName, parameters: "\(form.text)\n\(settings)",
            text: form.text, speed: form.speed, temperature: form.temperature, style: style,
            fromARecording: recording != nil)

        start(kind: "speech", model: modelName) { engine, report in
            let started = Date()
            guard let clip = try await engine.speak(request, progress: report) else { return nil }
            var made = made
            made.seconds = Date().timeIntervalSince(started)
            made.length = clip.seconds
            made.rate = clip.rate
            return (Media.wav(clip.samples, rate: clip.rate), made)
        }
    }

    private func convert() {
        guard let converter = model?.converter, let modelName = model?.name,
            let source = source?.clip, let reference = recording?.clip
        else { return }
        let seed = seed
        let style = (converter.convertsStyle ?? false) && form.convertStyle
        let request = VoiceConversionRequest(
            source: source, reference: reference, steps: form.conversionSteps, convertStyle: style, seed: seed)
        let made = Made(
            seed: seed, seconds: 0, model: modelName,
            parameters: "Steps: \(form.conversionSteps), Style: \(style ? "yes" : "no"), Seed: \(seed), Rate: \(converter.rate) Hz, Converter: \(modelName)",
            steps: form.conversionSteps, convertStyle: style)

        start(kind: "conversion", model: modelName) { engine, report in
            let started = Date()
            guard let clip = try await engine.voiceConversion(request, progress: report) else { return nil }
            var made = made
            made.seconds = Date().timeIntervalSince(started)
            made.length = clip.seconds
            made.rate = clip.rate
            return (Media.wav(clip.samples, rate: clip.rate), made)
        }
    }

    /// Stops the running job after the step it is on, or takes the waiting one out of line.
    func stop() {
        guard let job = running ?? waiting.first else { return }
        stopping.insert(job.id)
        tasks[job.id]?.cancel()
        // One that has not had its turn is out of line at once, not when the one before it ends.
        if job.status == .queued { ended(job.id, as: .cancelled) }
    }

    /// Deletes a picture or a clip, from the disk and from the list.
    func forget(_ job: Job) {
        try? FileManager.default.removeItem(at: file(job.id, kind: job.kind))
        try? FileManager.default.removeItem(at: recordFile(job.id))
        jobs.removeAll { $0.id == job.id }
        outputs[job.id] = nil
        images[job.id] = nil
    }

    /// Puts what a job was made with back in the boxes, which is how one is made again.
    func reuse(_ job: Job) {
        guard let made = job.made else { return }
        form.seed = String(made.seed)
        switch job.kind {
        case "image":
            form.prompt = made.prompt ?? form.prompt
            form.negative = made.negative ?? form.negative
            form.steps = made.steps ?? form.steps
            form.guidance = made.guidance ?? form.guidance
            form.width = made.width ?? form.width
            form.height = made.height ?? form.height
            form.strength = made.strength ?? form.strength
        case "speech":
            form.text = made.text ?? form.text
            form.speed = made.speed ?? form.speed
            form.temperature = made.temperature ?? form.temperature
            form.style = made.style ?? ""
        default:
            form.conversionSteps = made.steps ?? form.conversionSteps
            form.convertStyle = made.convertStyle ?? false
        }
    }

    func lastSeed() {
        if let seed = shown?.made?.seed { form.seed = String(seed) }
    }

    func save(_ job: Job) {
        guard let bytes = outputs[job.id] else { return }
        Media.save(bytes, suggesting: Media.fileName(created: job.finished ?? job.created, kind: job.kind))
    }

    /// Takes a picture that was drawn back round to the box it can be drawn from.
    func sendToImg2Img(_ job: Job) {
        guard let bytes = outputs[job.id] else { return }
        holdPicture(bytes)
        tab = .img2img
    }

    // MARK: - what is held

    enum Which { case picture, recording, source }

    func holdPicture(at url: URL) {
        do {
            holdPicture(try Media.picture(at: url))
        } catch {
            complaint = error.localizedDescription
        }
    }

    func holdPicture(_ bytes: Data) {
        guard let image = NSImage(data: bytes), let rgb = Media.rgb(image) else {
            complaint = "that is not a picture this Mac can read"
            return
        }
        complaint = nil
        picture = Held(bytes: bytes, image: image, rgb: rgb)
        tab = .img2img
    }

    /// Holds a recording -- to sound like, or to convert -- decoded into the samples the engine
    /// takes, and a WAV of them for the player.
    func holdAudio(_ which: Which, at url: URL) async {
        let seconds = which == .source ? Media.sourceSeconds : Media.recordingSeconds
        let clip: AudioClip
        do {
            clip = try await Task.detached { try Media.clip(from: url, seconds: seconds) }.value
        } catch {
            complaint = error.localizedDescription
            return
        }
        complaint = nil
        let kept = Held(bytes: Media.wav(clip.samples, rate: clip.rate), clip: clip)
        if which == .source { source = kept } else { recording = kept }
    }

    func letGo(_ which: Which) {
        switch which {
        case .picture: picture = nil
        case .recording: recording = nil
        case .source: source = nil
        }
    }
}

extension String {
    var isBlank: Bool { trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
}
