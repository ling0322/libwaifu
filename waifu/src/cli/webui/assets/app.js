// What the page does.
//
// It is one client of the program's API, and it keeps one thing of its own: the ids of the jobs it
// posted and the files it uploaded, in localStorage. The server knows nothing of who is asking --
// whoever holds an id can read what it names -- so which pictures are this browser's is this
// list. A second tab of the same browser shares it; another browser has its own.
//
// It asks for the model once, when it opens; what the worker is doing, twice a second; and how its
// own jobs are getting on, for as long as any of them is still waiting or running.
//
// React draws it, from the three files under vendor/: React itself, its renderer, and htm, which
// is what stands in for JSX -- a tagged template the browser parses on its own. All three are
// built into the binary beside this file, so there is still no build step between the source and
// the thing that runs, and the page still asks for nothing outside this machine.

const { Fragment, useCallback, useEffect, useRef, useState } = React;
const html = htm.bind(React.createElement);

/** What the stop button says while the card is busy with another page's run. */
const THEIRS =
  "Another page is using the model. Only the page that started a run can stop it; this one can start its own once that one is done";

/** How often to ask what is happening. Short enough that the bar moves, long enough to be free. */
const TICK = 500;

/** And how often to ask what is left of the machine. Slower, because memory fills over seconds
 *  rather than over frames and because reading it is a file and, on a card, a driver call. */
const MACHINE_TICK = 2000;

// -- talking to the program -------------------------------------------------------------------

/** A request, and what it answered. An error is a value here rather than a throw: every one of
 *  them ends the same way, which is a line on the screen. */
async function ask(method, path, body) {
  try {
    const sent = {
      method,
      headers: {
        "Content-Type":
          body instanceof Blob ? body.type || "application/octet-stream" : "application/json",
      },
      body: body instanceof Blob ? body : body === undefined ? undefined : JSON.stringify(body),
    };
    const answer = await fetch(path, sent);
    const said = answer.headers.get("Content-Type")?.includes("json") ? await answer.json() : {};

    return answer.ok ? { ok: true, said } : { ok: false, error: said.error || answer.statusText };
  } catch (failed) {
    // The program is gone, most likely -- somebody pressed ctrl-c in the window it is running in.
    return { ok: false, error: `could not reach waifu: ${failed.message}` };
  }
}

/** How much room something takes, in the unit that says it in two or three digits. A model is
 *  gigabytes and the manifest beside it is kilobytes, and "0.0 GB" is neither of them. */
function room(bytes) {
  if (bytes >= 1_000_000_000) return `${(bytes / 1_000_000_000).toFixed(1)} GB`;
  if (bytes >= 1_000_000) return `${(bytes / 1_000_000).toFixed(0)} MB`;
  return `${(bytes / 1_000).toFixed(0)} kB`;
}


/**
 * How much of a recording is worth keeping, in seconds.
 *
 * A speech model listening to somebody wants seconds of them, not minutes -- what it is taking is
 * how a voice sounds, which a sentence already says. Trimmed here rather than on the far side so
 * that what crosses the wire is seconds of uncompressed samples rather than a podcast.
 */
const RECORDING_SECONDS = 30;

/**
 * And how much of a recording to convert. Longer, because this one is not a sample of a voice but
 * the thing being said again -- a converter works through it a window at a time -- and still a
 * limit, because five minutes of 16-bit samples at 48 kHz is already 29 MB on the wire.
 */
const SOURCE_SECONDS = 300;

/**
 * Turns whatever file was dropped into a WAV the program can read.
 *
 * The browser has a decoder for every format it will play -- mp3, m4a, ogg, flac, webm -- and the
 * program has one for none of them: a codec is several thousand lines and a licence to read, and
 * this crate carries neither. So the decoding happens on the side that already knows how, and
 * what is posted is samples.
 *
 * Down to one channel and to half a minute on the way, which are the two things the far side
 * would otherwise have to do to a file it did not ask for the size of.
 */
async function asWav(file, seconds = RECORDING_SECONDS) {
  const context = new (window.AudioContext || window.webkitAudioContext)();
  let decoded;
  try {
    decoded = await context.decodeAudioData(await file.arrayBuffer());
  } finally {
    // Closed whether or not it decoded. A browser allows a handful of these at once, and a page
    // where somebody has tried six files is a page that has opened six.
    context.close();
  }

  const channels = [];
  for (let channel = 0; channel < decoded.numberOfChannels; channel++) {
    channels.push(decoded.getChannelData(channel));
  }

  const length = Math.min(decoded.length, Math.floor(decoded.sampleRate * seconds));
  const mono = new Float32Array(length);
  for (let at = 0; at < length; at++) {
    let sum = 0;
    // The mean rather than the sum: the same thing in two channels added together is that thing
    // at twice the amplitude, which clips.
    for (const channel of channels) sum += channel[at];
    mono[at] = sum / (channels.length || 1);
  }

  return asWavBytes(mono, decoded.sampleRate);
}

/** Samples and a rate, as the bytes of a 16-bit mono WAV file. */
function asWavBytes(samples, rate) {
  const bytes = new ArrayBuffer(44 + samples.length * 2);
  const at = new DataView(bytes);
  const label = (where, said) => {
    for (let letter = 0; letter < said.length; letter++) {
      at.setUint8(where + letter, said.charCodeAt(letter));
    }
  };

  label(0, "RIFF");
  at.setUint32(4, 36 + samples.length * 2, true);
  label(8, "WAVE");
  label(12, "fmt ");
  at.setUint32(16, 16, true);
  at.setUint16(20, 1, true);
  at.setUint16(22, 1, true);
  at.setUint32(24, rate, true);
  at.setUint32(28, rate * 2, true);
  at.setUint16(32, 2, true);
  at.setUint16(34, 16, true);
  label(36, "data");
  at.setUint32(40, samples.length * 2, true);

  for (let sample = 0; sample < samples.length; sample++) {
    // Held inside the range before it is scaled, because the top of it wraps to the bottom --
    // which is a click, and a loud one.
    const held = Math.max(-1, Math.min(1, samples[sample]));
    at.setInt16(44 + sample * 2, Math.round(held * 32767), true);
  }

  return new Blob([bytes], { type: "audio/wav" });
}

/** A limit in seconds, in the words it is said in: "30 seconds", "5 minutes". */
function howLong(seconds) {
  return seconds < 120 ? `${seconds} seconds` : `${Math.round(seconds / 60)} minutes`;
}

/** How long something is, in the shape a clip's length is read in. */
function clock(seconds) {
  const whole = Math.max(0, Math.round(seconds));

  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}

// -- the small pieces ---------------------------------------------------------------------------

/** A labelled box. Every control on the settings side is one of these; `kind` is how wide it sits
 *  in the row it is in -- `grow` takes what is left, `number` is the narrow one beside it. */
function Field({ label, kind, children }) {
  return html`
    <label className=${kind ? `field ${kind}` : "field"}>
      <span className="label">${label}</span>
      ${children}
    </label>
  `;
}

/** A number and the slider under it, which are two ways of saying one value.
 *
 *  Both are handed the same value and the same way of changing it, so they cannot drift apart --
 *  which is the whole of what the pair of listeners this replaced was for. */
function NumberBox({ label, value, onChange, box, kind, disabled }) {
  return html`
    <${Field} label=${label} kind=${kind}>
      <input
        type="number"
        ...${box}
        value=${value}
        disabled=${disabled}
        onChange=${(e) => onChange(e.target.value)}
      />
    <//>
  `;
}

function Slider({ value, onChange, limits, disabled }) {
  return html`
    <input
      className="slider"
      type="range"
      ...${limits}
      value=${value}
      disabled=${disabled}
      onChange=${(e) => onChange(e.target.value)}
    />
  `;
}

// -- the name, and the column on the left ------------------------------------------------------

/** Where this came from, for somebody who arrived at the page before the repository. */
const HOME = "https://github.com/ling0322/libwaifu";

function TopBar({ state }) {
  return html`
    <header className="topbar">
      <div className="middle">
        <span className="name">libwaifu</span>
        <span className="dim">${state?.built_from ?? ""}</span>
        <a className="dim" href=${HOME} target="_blank" rel="noreferrer">github.com/ling0322/libwaifu</a>
      </div>
    </header>
  `;
}

/** The tabs that draw a picture, which there always are. */
const DRAWS = ["txt2img", "img2img"];
const SPEAKS = "text2speech";
const CONVERTS = "speech2speech";

/**
 * The tabs this session has, which the terminal decided: the task was chosen there, and the model
 * with it. A voice has its one tab. A picture model has txt2img, and img2img beside it where it
 * can start from a picture -- the same model either way, so moving between the two throws nothing
 * away, and a picture drawn on one can be sent to the other.
 */
function tabsFor(state) {
  if (!state) return [];
  if (state.task === SPEAKS) return [SPEAKS];
  if (state.task === CONVERTS) return [CONVERTS];
  return state.model?.draws_from_a_picture ? DRAWS : [DRAWS[0]];
}

/** The kinds of run this session can do, one under the other. */
function Nav({ tabs, tab, onTab }) {
  return html`
    <nav className="nav">
      ${tabs.map(
        (page) => html`
          <button
            key=${page}
            className=${tab === page ? "on" : ""}
            onClick=${() => onTab(page)}
          >
            ${page}
          </button>
        `,
      )}
    </nav>
  `;
}

/**
 * What this is running on, under the tabs.
 *
 * Three rows and two bars. The processor and how much memory is fitted are what somebody compares
 * against the machine they read a number on somewhere else; what is *left* of the memory and of
 * the card is what answers the question that actually gets asked, which is why a run that worked
 * yesterday now aborts. A card sitting at fifteen of sixteen gigabytes answers it on sight, and
 * before this the page gave no way to see that at all short of another window with nvidia-smi in
 * it.
 *
 * Every row is drawn only where there is something to put in it. A machine this has not been
 * taught to read says nothing rather than saying "unknown" four times.
 */
function Machine({ machine }) {
  if (!machine) return null;

  const { cpu, cores, threads, memory, gpu, vram } = machine;
  const counted = [cores && `${cores} cores`, threads && `${threads} threads`].filter(Boolean);

  return html`
    <section className="hardware">
      <h2>this machine</h2>

      ${(cpu || counted.length > 0) &&
      html`
        <div className="part">
          <div className="what">cpu</div>
          ${/* The full name in the tooltip: the tidying took the trademarks off, and a Xeon's
               name can still be longer than the column is wide. */ ""}
          ${cpu && html`<div className="named" title=${cpu}>${cpu}</div>`}
          ${counted.length > 0 && html`<div className="dim">${counted.join(" · ")}</div>`}
        </div>
      `}

      ${memory?.total &&
      html`
        <div className="part">
          <div className="what">memory</div>
          <${Meter} used=${memory.used} total=${memory.total} />
        </div>
      `}

      ${gpu &&
      html`
        <div className="part">
          <div className="what">gpu</div>
          <div className="named" title=${gpu.name}>${gpu.name}</div>
          ${/* Apple's card has the machine's memory rather than any of its own, so the bar for
               it would be the same bar again with the same numbers in it. */ ""}
          ${gpu.unified
            ? html`<div className="dim">memory is shared with the cpu</div>`
            : vram
              ? html`
                  <div className="vram">vram</div>
                  <${Meter} used=${vram.used} total=${vram.total} />
                  ${/* What of it is this program's, which is the half of the number a size that
                       aborted is about: the rest is the desktop and whatever else is on the
                       card. */ ""}
                  ${vram.ours > 0 && html`<div className="dim">${room(vram.ours)} is ours</div>`}
                `
              : html`<div className="dim">${machine.why_no_vram}</div>`}
        </div>
      `}

      ${/* What this build can send a run to, which is not the same as what is plugged in: a card
           in a build without CUDA is a card this page can name and cannot use. Named here as
           well as offered in the Device box, because the box says which one is chosen and this
           says whether the card in the line above is a card anything can be asked of. */ ""}
      <div className="part">
        <div className="what">accelerators</div>
        <div className="dim">
          ${(machine.accelerators ?? []).length > 0
            ? machine.accelerators.join(" · ")
            : "none -- runs go to the processor"}
        </div>
      </div>
    </section>
  `;
}

/** How much of something is gone, as a bar with the two numbers over it.
 *
 *  The same trough and fill the run's own bar is drawn from, at a third of the height: it is the
 *  same thing being said -- how much of a whole -- and a second visual language for it in the
 *  column beside would be two things to learn. */
function Meter({ used, total }) {
  // Only the total is ever certain. What is spare is a figure the system may not offer, and a bar
  // drawn as empty would be saying the machine has all of its memory free.
  const known = typeof used === "number" && total > 0;
  const part = known ? Math.max(0, Math.min(1, used / total)) : 0;

  return html`
    <${Fragment}>
      <div className="dim">${known ? `${room(used)} of ${room(total)}` : room(total)}</div>
      ${known &&
      html`
        <div className="meter" title=${`${Math.round(part * 100)}% in use`}>
          <div className="meter-fill" style=${{ width: `${part * 100}%` }} />
        </div>
      `}
    <//>
  `;
}

/**
 * What a run is going to be drawn or read with, and where, as they were chosen in the terminal.
 *
 * Said and not offered. The model and the device are settled before the page opens -- the model
 * fetched there, under its own bar -- and a box here that could change them would be a second
 * place to decide what the first had already decided. To run something else, the terminal is
 * where to go back to.
 */
function Chosen({ label, chosen, device, standing, children }) {
  return html`
    <div className="card">
      <div className="row">
        <${Field} label=${label} kind="grow">
          <div className="fixed" title=${chosen?.name ?? ""}>${chosen?.full_name ?? "none"}</div>
        <//>
        <${Field} label="Device" kind="short">
          <div className="fixed">${device ?? ""}</div>
        <//>
      </div>
      <p className="about">${standing}</p>
      ${children}
    </div>
  `;
}

/** How far a model or a voice is from being ready, in the words the card under it carries. */
function standingOf(chosen, first) {
  if (!chosen) return "nothing was chosen";
  if (chosen.in_memory) return "read, and on the device";
  if (chosen.on_disk) return `on the disk -- read ${first}`;
  return "not on the disk -- quit, and run waifu again to fetch it";
}

function ModelAndDevice({ state }) {
  const chosen = state?.model;
  return html`<${Chosen}
    label="Model"
    chosen=${chosen}
    device=${state?.device}
    standing=${standingOf(chosen, "before the first run")}
  />`;
}

// -- what to draw -------------------------------------------------------------------------------

/**
 * What to draw, and the button that draws it.
 *
 * Only on the screen once a model has been chosen -- the page above decides that, the same way
 * the settings below decide it for themselves. Two empty boxes that cannot be typed in are an
 * invitation to type in them, and somebody who accepts it has written their prompt into a screen
 * that was never going to take it.
 *
 * `guided` is the same thought one box further in: a model that answers in a single pass has
 * nowhere to put a negative prompt, so it does not get one to write in either.
 */
function Prompts({
  form,
  change,
  canDraw,
  drawing,
  fetching,
  theirs,
  guided,
  onDraw,
  onInterrupt,
}) {
  return html`
    <section className="prompts">
      <div className="prompt-boxes">
        ${/* Labelled, because "the big box" and "the other big box" is a thing somebody has to be
             told once by somebody else. The same height as each other: what goes in the second is
             as long as what goes in the first, and a box half the size of its neighbour says it
             matters half as much. */ ""}
        <${Field} label="Prompt">
          <textarea
            rows="3"
            placeholder="What to draw. A list of tags reads better to these models than a sentence, and the earlier a tag comes the more of the picture it tends to decide."
            value=${form.prompt}
            onChange=${(e) => change("prompt", e.target.value)}
          ></textarea>
        <//>
        ${/* And gone entirely for a model that runs one pass. What goes in this box is what the
             second pass is given, so a model without one has nowhere to put it: the text would be
             typed, sent, and dropped, and the picture would come back no different. Which is the
             reason above applied one box further in -- a box that cannot do anything is an
             invitation to type into it. The one above fills the height either way. */ ""}
        ${guided &&
        html`
          <${Field} label="Negative prompt">
            <textarea
              rows="3"
              placeholder="What to keep out. Left empty the model is still steered away from the empty prompt, which is not the same as steering away from nothing at all."
              value=${form.negative}
              onChange=${(e) => change("negative", e.target.value)}
            ></textarea>
          <//>
        `}
      </div>
      <div className="go">
        <button className="generate" disabled=${!canDraw} onClick=${onDraw}>Generate</button>
        ${/* Always under it, rather than appearing only once there is something to stop: a button
             that is not there until the moment it is needed is a button nobody knows about, and
             its place on the screen moves the thing above it when it arrives.

             One button for the two things there are to stop, because from here they are one
             thing -- the program is busy and this is how to stop it being busy. What each of
             them leaves behind is not the same, so it says which one it is about. */ ""}
        <button
          className="interrupt"
          disabled=${!drawing && !fetching}
          title=${drawing
            ? "Stop after the step it is on. The model comes off the card with it, so that whatever else wants the card can have it -- and so the next run reads the model again"
            : fetching
              ? "Stop the download. The packages that have come down are kept, and fetching it again carries on from there"
              : theirs
                ? THEIRS
                : "Nothing to stop. A run can be stopped while it is drawing and a model while it is coming down; reading one onto the card cannot be stopped part way"}
          onClick=${onInterrupt}
        >
          Cancel
        </button>
      </div>
    </section>
  `;
}

/**
 * text2speech: the one box, and the button that reads it out.
 *
 * Its own component rather than a second mode of the two-box one above. There is no second box --
 * nothing here is steered away from anything -- and a screen with one box beside an empty one is
 * a screen asking to be typed in twice.
 */
function SayBox({ form, change, canSpeak, speaking, theirs, onSpeak, onInterrupt }) {
  return html`
    <section className="prompts">
      <div className="prompt-boxes">
        <${Field} label="What to say">
          <textarea
            rows="6"
            placeholder="The text to read out. Punctuation is where it pauses, so a sentence that has commas in it is read with them."
            value=${form.text}
            onChange=${(e) => change("text", e.target.value)}
          ></textarea>
        <//>
      </div>
      <div className="go">
        <button className="generate" disabled=${!canSpeak} onClick=${onSpeak}>Speak</button>
        <button
          className="interrupt"
          disabled=${!speaking}
          title=${speaking
            ? "Stop where it is. Nothing is kept: half a sentence is not a clip"
            : theirs
              ? THEIRS
              : "Nothing to stop"}
          onClick=${onInterrupt}
        >
          Cancel
        </button>
      </div>
    </section>
  `;
}

/** img2img only: what the run starts from instead of noise. */
function FromPicture({ holding, revision, off, onHold, onClear }) {
  const [over, setOver] = useState(false);

  const dragged = (event) => {
    event.preventDefault();
    setOver(true);
  };

  return html`
    <div className="card drop">
      <div className="card-title">Picture to draw from</div>
      <label
        className=${`dropzone${over ? " over" : ""}${off ? " off" : ""}`}
        onDragEnter=${dragged}
        onDragOver=${dragged}
        onDragLeave=${() => setOver(false)}
        onDrop=${(event) => {
          event.preventDefault();
          setOver(false);
          onHold(event.dataTransfer.files[0]);
        }}
      >
        <input
          type="file"
          accept="image/png,image/jpeg"
          hidden
          disabled=${off}
          onChange=${(event) => onHold(event.target.files[0])}
        />
        ${holding
          ? // By the upload's id: a different picture is a different address, so the browser
            // fetches it again when it changes and not otherwise.
            html`<img alt="" src=${`/api/uploads/${revision}`} />`
          : html`<span>Drop a picture here, or click to choose one</span>`}
      </label>
      ${holding &&
      html`
        <button
          className="plain wide"
          onClick=${(event) => {
            // The button sits inside the label that opens the file chooser, so without this,
            // clearing the picture would immediately ask for another one.
            event.preventDefault();
            onClear();
          }}
        >
          Remove the picture
        </button>
      `}
    </div>
  `;
}

/** What a voice takes and writes, in the one line there is room for under it. */
function voiceSays(voice) {
  return `${voice.rate} Hz -- ${
    voice.takes_a_recording ? "takes a recording to sound like" : "takes no recording"
  }`;
}

/** What is going to speak, and where -- the head of the settings column on the speech tab. */
function VoiceAndDevice({ state }) {
  const voice = state?.voice;
  return html`
    <${Chosen}
      label="Voice"
      chosen=${voice}
      device=${state?.device}
      standing=${standingOf(voice, "at the first reading")}
    >
      ${voice && html`<p className="about">${voiceSays(voice)}</p>`}
    <//>
  `;
}

/** A length of sound as the player says it: minutes and seconds, `1:05`. */
function clock(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return "0:00";
  const whole = Math.floor(seconds);
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}

const PLAY_ICON = html`
  <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true">
    <path d="M4.5 2.5v11l9-5.5z" fill="currentColor" />
  </svg>
`;
const PAUSE_ICON = html`
  <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true">
    <path d="M3.5 2.5h3.2v11H3.5zM9.3 2.5h3.2v11H9.3z" fill="currentColor" />
  </svg>
`;

/**
 * A player drawn by the page rather than by the browser.
 *
 * The browser's own controls are a different widget in every browser, and none of them takes the
 * page's colours: a grey pill from another design sitting in the middle of this one. So the
 * `<audio>` element is kept for what it is good at -- decoding, buffering, asking the server for
 * byte ranges -- and hidden, and what is on screen is a play button, a bar and the time, drawn
 * here.
 *
 * Never plays by itself. Somebody presses play; see `ClipOutput` for why.
 *
 * The bar follows the sound every frame while it plays. `timeupdate` alone comes four times a
 * second, which is a bar that moves in visible steps. It can be clicked or dragged to move, and
 * when it has the focus the arrow keys move five seconds and Home and End go to either end.
 */
function Player({ src, className = "" }) {
  const audio = useRef(null);
  const track = useRef(null);
  const dragging = useRef(false);
  const [playing, setPlaying] = useState(false);
  const [at, setAt] = useState(0);
  const [length, setLength] = useState(0);
  const [broken, setBroken] = useState(false);

  // A new address is a new sound, started from nothing rather than from where the last one was.
  useEffect(() => {
    setPlaying(false);
    setAt(0);
    setLength(0);
    setBroken(false);
  }, [src]);

  useEffect(() => {
    if (!playing) return undefined;

    let frame = 0;
    const follow = () => {
      if (audio.current && !dragging.current) setAt(audio.current.currentTime);
      frame = requestAnimationFrame(follow);
    };
    frame = requestAnimationFrame(follow);
    return () => cancelAnimationFrame(frame);
  }, [playing]);

  const toggle = () => {
    const element = audio.current;
    if (!element || broken) return;
    if (element.paused) element.play().catch(() => setBroken(true));
    else element.pause();
  };

  const moveTo = (seconds) => {
    const element = audio.current;
    if (!element || !length) return;
    element.currentTime = Math.min(length, Math.max(0, seconds));
    setAt(element.currentTime);
  };

  const seekTo = (clientX) => {
    const box = track.current?.getBoundingClientRect();
    if (!box || !box.width) return;
    moveTo(((clientX - box.left) / box.width) * length);
  };

  const keys = (event) => {
    const step = { ArrowLeft: -5, ArrowRight: 5, Home: -Infinity, End: Infinity }[event.key];
    if (step === undefined) return;
    event.preventDefault();
    moveTo((audio.current?.currentTime ?? 0) + step);
  };

  const measured = (event) => {
    const seconds = event.currentTarget.duration;
    if (Number.isFinite(seconds)) setLength(seconds);
  };

  const fraction = length ? Math.min(1, at / length) : 0;

  return html`
    <div className=${`player-bar${className ? ` ${className}` : ""}`}>
      <audio
        ref=${audio}
        src=${src}
        preload="metadata"
        onPlay=${() => setPlaying(true)}
        onPause=${() => setPlaying(false)}
        onEnded=${(event) => {
          setPlaying(false);
          setAt(event.currentTarget.duration || 0);
        }}
        onLoadedMetadata=${measured}
        onDurationChange=${measured}
        onTimeUpdate=${(event) => {
          if (!dragging.current) setAt(event.currentTarget.currentTime);
        }}
        onError=${() => setBroken(true)}
      ></audio>

      <button
        type="button"
        className="play"
        aria-label=${playing ? "Pause" : "Play"}
        disabled=${broken}
        onClick=${toggle}
      >
        ${playing ? PAUSE_ICON : PLAY_ICON}
      </button>

      <div
        ref=${track}
        className="track"
        role="slider"
        tabIndex="0"
        aria-label="Position"
        aria-valuemin="0"
        aria-valuemax=${Math.round(length)}
        aria-valuenow=${Math.round(at)}
        aria-valuetext=${`${clock(at)} of ${clock(length)}`}
        onPointerDown=${(event) => {
          dragging.current = true;
          event.currentTarget.setPointerCapture(event.pointerId);
          seekTo(event.clientX);
        }}
        onPointerMove=${(event) => {
          if (dragging.current) seekTo(event.clientX);
        }}
        onPointerUp=${(event) => {
          dragging.current = false;
          event.currentTarget.releasePointerCapture(event.pointerId);
        }}
        onPointerCancel=${() => {
          dragging.current = false;
        }}
        onKeyDown=${keys}
      >
        <div className="rail">
          <div className="rail-fill" style=${{ width: `${fraction * 100}%` }}></div>
        </div>
      </div>

      <span className="time">
        ${broken ? "cannot play this" : `${clock(at)} / ${clock(length)}`}
      </span>
    </div>
  `;
}

/**
 * A recording to hold: on text2speech the one a reading is to sound like, and on speech2speech
 * both the one to convert and the voice to convert it to. `seconds` is how much of it is kept.
 */
function FromRecording({
  holding,
  revision,
  why,
  onHold,
  onClear,
  title = "Recording to sound like",
  seconds = RECORDING_SECONDS,
}) {
  const [over, setOver] = useState(false);
  const [reading, setReading] = useState(false);

  if (why) {
    return html`
      <div className="card">
        <div className="card-title">This voice cannot be handed a recording</div>
        <p className="about">${why}.</p>
      </div>
    `;
  }

  const dragged = (event) => {
    event.preventDefault();
    setOver(true);
  };
  const take = async (file) => {
    if (!file) return;
    // Decoding happens here, in the browser, and a long file takes a moment. Without this the
    // box sits unchanged and the click reads as one that did nothing.
    setReading(true);
    try {
      await onHold(file);
    } finally {
      setReading(false);
    }
  };

  return html`
    <div className="card drop">
      <div className="card-title">${title}</div>
      ${/* Above the box rather than inside it, which is where the picture's thumbnail goes. That
           box is a label around a file input, so everything inside it opens the file chooser when
           it is clicked -- which is what should happen to a thumbnail and is the opposite of what
           should happen to a play button. */ ""}
      ${holding &&
      html`<${Player}
        className="held"
        src=${
          // By the upload's id, which changes when the recording does and not otherwise: a
          // player whose address changes reloads.
          `/api/uploads/${revision}`
        }
      />`}
      <label
        className=${`dropzone short${over ? " over" : ""}`}
        onDragEnter=${dragged}
        onDragOver=${dragged}
        onDragLeave=${() => setOver(false)}
        onDrop=${(event) => {
          event.preventDefault();
          setOver(false);
          take(event.dataTransfer.files[0]);
        }}
      >
        <input
          type="file"
          accept="audio/*"
          hidden
          onChange=${(event) => take(event.target.files[0])}
        />
        ${reading
          ? html`<span>reading it...</span>`
          : holding
            ? html`<span>Drop another one here, or click to choose one</span>`
            : html`<span>Drop a recording here, or click to choose one</span>`}
      </label>
      <p className="about">
        Any format this browser can play: it is decoded here and the samples are what cross, so
        the program needs no codec of its own. The first ${howLong(seconds)} of it are kept.
      </p>
      ${holding &&
      html`
        <button
          className="plain wide"
          onClick=${(event) => {
            event.preventDefault();
            onClear();
          }}
        >
          Remove the recording
        </button>
      `}
    </div>
  `;
}

/** How to say it, for a voice that was taught some ways: a list of those, and the instruction
 *  itself in a box under it. The list writes into the box rather than standing beside it, so
 *  what is sent is always what can be read, and one of the list's can be changed by hand. */
function StyleCard({ styles, style, onChange }) {
  const chosen = styles.find((one) => one.instruction === style.trim());
  return html`
    <div className="card">
      <div className="card-title">Style</div>
      <div className="row">
        <${Field} label="Say it" kind="grow">
          <select
            value=${!style.trim() ? "" : chosen ? chosen.instruction : "custom"}
            onChange=${(e) => e.target.value !== "custom" && onChange(e.target.value)}
          >
            <option value="">the voice's own way</option>
            ${!!style.trim() && !chosen && html`<option value="custom">as written below</option>`}
            ${styles.map(
              (one) =>
                html`<option key=${one.instruction} value=${one.instruction}>${one.label}</option>`,
            )}
          </select>
        <//>
      </div>
      <div className="row">
        <${Field} label="Instruction" kind="grow">
          <input
            type="text"
            value=${style}
            placeholder="none"
            onChange=${(e) => onChange(e.target.value)}
          />
        <//>
      </div>
      <p className="about">
        Words the model follows rather than reads. The list is what it was taught, in its own
        words; anything else written here is a guess at what it might follow. With a style, the
        recording gives the voice and the instruction gives the manner.
      </p>
    </div>
  `;
}

/** text2speech: everything a reading is asked for that is not the text itself. */
function SpeechSettings({
  state,
  progress,
  form,
  change,
  onHold,
  onClear,
  onAnySeed,
}) {
  const voice = state?.voice;

  return html`
    <div className="settings">
      <${VoiceAndDevice} state=${state} />

      ${/* Nothing below this until there is a voice on the disk, for the reason there is nothing
           below the model on the other tabs until there is one. */ ""}
      ${!!voice?.on_disk &&
      html`
        <${Fragment}>

      ${/* The first thing on the tab, above every setting, for as long as it is true. It comes
           out of the voice itself rather than being written here, so the day a real model is
           loaded it goes away on its own rather than by somebody remembering to delete it. */ ""}
      ${voice?.not_a_voice_because &&
      html`
        <div className="card warn">
          <div className="card-title">This is not a voice yet</div>
          <p className="about">${voice.not_a_voice_because}.</p>
        </div>
      `}

      <${FromRecording}
        holding=${!!state?.holding_a_recording}
        revision=${state?.recording_upload}
        why=${voice?.no_likeness_because}
        onHold=${onHold}
        onClear=${onClear}
      />

      ${!!voice?.styles?.length &&
      html`<${StyleCard}
        styles=${voice.styles}
        style=${form.style}
        onChange=${(value) => change("style", value)}
      />`}

      <div className="card">
        <div className="row">
          <${NumberBox}
            label="Speed"
            kind="number"
            box=${{ min: 0.25, max: 4, step: 0.05 }}
            value=${form.speed}
            onChange=${(value) => change("speed", value)}
          />
          <${Slider}
            limits=${{ min: 0.25, max: 4, step: 0.05 }}
            value=${form.speed}
            onChange=${(value) => change("speed", value)}
          />
        </div>
      </div>

      <div className="card">
        <div className="row">
          <${NumberBox}
            label="Temperature"
            kind="number"
            box=${{ min: 0, max: 2, step: 0.05 }}
            value=${form.temperature}
            onChange=${(value) => change("temperature", value)}
          />
          <${Slider}
            limits=${{ min: 0, max: 2, step: 0.05 }}
            value=${form.temperature}
            onChange=${(value) => change("temperature", value)}
          />
        </div>
      </div>

      <div className="card">
        <div className="row">
          <${Field} label="Seed" kind="grow">
            <input
              type="text"
              inputMode="numeric"
              value=${form.seed}
              onChange=${(e) => change("seed", e.target.value)}
            />
          <//>
          <button className="plain" title="A different reading every time" onClick=${onAnySeed}>
            🎲
          </button>
        </div>
      </div>

        <//>
      `}
    </div>
  `;
}

/** speech2speech: what converts, and where -- the head of the settings column. */
function ConverterAndDevice({ state }) {
  const converter = state?.converter;
  return html`
    <${Chosen}
      label="Converter"
      chosen=${converter}
      device=${state?.device}
      standing=${standingOf(converter, "at the first conversion")}
    >
      ${converter && html`<p className="about">${converter.rate} Hz</p>`}
    <//>
  `;
}

/**
 * speech2speech: the recording to convert, and the button that converts it.
 *
 * Where the prompt is on the other tabs, because it is what the run is of: the voice it is turned
 * into is a setting beside it, the way a picture to start from is.
 */
function ConvertBox({
  state,
  canConvert,
  converting,
  theirs,
  onHold,
  onClear,
  onConvert,
  onInterrupt,
}) {
  return html`
    <section className="prompts">
      <div className="prompt-boxes">
        <${FromRecording}
          title="Recording to convert"
          seconds=${SOURCE_SECONDS}
          holding=${!!state?.holding_a_source}
          revision=${state?.source_upload}
          onHold=${onHold}
          onClear=${onClear}
        />
      </div>
      <div className="go">
        <button className="generate" disabled=${!canConvert} onClick=${onConvert}>Convert</button>
        <button
          className="interrupt"
          disabled=${!converting}
          title=${converting
            ? "Stop after the step it is on. Nothing is kept: half a recording is not a clip"
            : theirs
              ? THEIRS
              : "Nothing to stop"}
          onClick=${onInterrupt}
        >
          Cancel
        </button>
      </div>
    </section>
  `;
}

/** speech2speech: everything a conversion is asked for that is not the recording it converts. */
function ConversionSettings({ state, form, change, onHold, onClear, onAnySeed }) {
  const converter = state?.converter;

  return html`
    <div className="settings">
      <${ConverterAndDevice} state=${state} />

      ${!!converter?.on_disk &&
      html`
        <${Fragment}>
          <${FromRecording}
            title="Voice to convert it to"
            holding=${!!state?.holding_a_recording}
            revision=${state?.recording_upload}
            onHold=${onHold}
            onClear=${onClear}
          />

          <div className="card">
            <div className="row">
              <${NumberBox}
                label="Steps"
                kind="number"
                box=${{ min: 1, max: 100, step: 1 }}
                value=${form.conversionSteps}
                onChange=${(value) => change("conversionSteps", value)}
              />
              <${Slider}
                limits=${{ min: 1, max: 100, step: 1 }}
                value=${form.conversionSteps}
                onChange=${(value) => change("conversionSteps", value)}
              />
            </div>
          </div>

          <div className="card">
            <label className="check">
              <input
                type="checkbox"
                checked=${!!form.convertStyle}
                onChange=${(e) => change("convertStyle", e.target.checked)}
              />
              <span>Convert the style too</span>
            </label>
            <p className="about">
              Off, only whose voice it is changes: the timing and the accent are the recording's
              own. On, it is said again with the accent and pacing of the voice as well, which
              takes longer and keeps less of the original timing.
            </p>
          </div>

          <div className="card">
            <div className="row">
              <${Field} label="Seed" kind="grow">
                <input
                  type="text"
                  inputMode="numeric"
                  value=${form.seed}
                  onChange=${(e) => change("seed", e.target.value)}
                />
              <//>
              <button
                className="plain"
                title="A different conversion every time"
                onClick=${onAnySeed}
              >
                🎲
              </button>
            </div>
          </div>
        <//>
      `}
    </div>
  `;
}

function Settings({
  state,
  progress,
  tab,
  form,
  change,
  onHold,
  onClear,
  onAnySeed,
  onLastSeed,
}) {
  const model = state?.model;
  const why = model?.no_picture_because;

  // Whether there is any guidance to ask this one for. A distilled release answers in one pass,
  // already as though it had been guided; there is no second answer to push away from, so the
  // dial has no position that means anything and the model would discard the number. Absent --
  // a page talking to a build older than the key -- reads as yes, which is what every model was
  // before there was a way to say otherwise.
  const guided = model ? model.takes_guidance !== false : true;

  // Every knob here is a knob for a model. Until one is chosen they are all off, and the button
  // that chooses one is the only thing on this side of the page that does anything.
  const off = !model;

  // Whether the size in the boxes is one of the model's own. The list says "custom" when it is
  // not, rather than going on showing the last shape that was picked from it.
  const preset = (model?.sizes ?? []).some(
    ([width, height]) => String(width) === String(form.width) && String(height) === String(form.height),
  );

  return html`
    <div className="settings">
      <${ModelAndDevice} state=${state} />

      ${/* Nothing below this is worth showing until there is a model on the disk: every one of
           them is a setting for one, and until it is downloaded the thing to do is the Download
           button where Generate will be. */ ""}
      ${!!model?.on_disk &&
      html`
        <${Fragment}>

        ${/* The dropzone, or the sentence that says why there is no point in one. Said here, where
             somebody who came to this tab to draw from a picture is looking, rather than by a tab
             that has been quietly greyed out. */ ""}
        ${tab === "img2img" &&
        (why
          ? html`
              <div className="card">
                <div className="card-title">This model cannot start from a picture</div>
                <p className="about">${why}.</p>
              </div>
            `
          : html`<${FromPicture}
              holding=${!!state?.holding_a_picture}
              revision=${state?.picture_upload}
              off=${off}
              onHold=${onHold}
              onClear=${onClear}
            />`)}

        ${/*
          One knob to a card, and nothing under it but the knob. They used to be paired
          two to a row -- the sampler beside the steps, the size beside the guidance -- which was a
          shape borrowed from another tool: a wide box and a narrow one filling one line. Nothing
          related the size to the guidance except the width left over beside the size.
        */ ""}
        <div className="card">
          <div className="row">
            <${Field} label="Size" kind="grow">
              ${/*
                The sizes the model says it was trained at, where its card names enough of them, and
                SDXL's own list otherwise. Far from these shapes these models start drawing a body
                twice rather than one body larger -- which is why the list is what the box offers,
                and the two beside it are for somebody who knows they are leaving it.
              */ ""}
              <select
                value=${preset ? `${form.width}x${form.height}` : "custom"}
                disabled=${off}
                onChange=${(e) => {
                  const [width, height] = e.target.value.split("x");
                  change("width", width);
                  change("height", height);
                }}
              >
                ${!model && html`<option value="">once a model is chosen</option>`}
                ${!preset && model && html`<option value="custom">custom</option>`}
                ${(model?.sizes ?? []).map(
                  ([width, height]) => html`
                    <option key=${`${width}x${height}`} value=${`${width}x${height}`}>
                      ${`${width} x ${height}`}
                    </option>
                  `,
                )}
              </select>
            <//>
            <${NumberBox}
              label="Width"
              kind="number"
              box=${{ min: 64, max: 2048, step: 64 }}
              disabled=${off}
              value=${form.width}
              onChange=${(value) => change("width", value)}
            />
            <${NumberBox}
              label="Height"
              kind="number"
              box=${{ min: 64, max: 2048, step: 64 }}
              disabled=${off}
              value=${form.height}
              onChange=${(value) => change("height", value)}
            />
          </div>
        </div>

        <div className="card">
          <div className="row">
            <${NumberBox}
              label="Sampling steps"
              kind="number"
              box=${{ min: 1, max: 150, step: 1 }}
              disabled=${off}
              value=${form.steps}
              onChange=${(value) => change("steps", value)}
            />
            <${Slider}
              limits=${{ min: 1, max: 80, step: 1 }}
              disabled=${off}
              value=${form.steps}
              onChange=${(value) => change("steps", value)}
            />
          </div>
        </div>

        ${guided &&
        html`
          <div className="card">
            <div className="row">
              <${NumberBox}
                label="CFG Scale"
                kind="number"
                box=${{ min: 1, max: 30, step: 0.1 }}
                disabled=${off}
                value=${form.guidance}
                onChange=${(value) => change("guidance", value)}
              />
              ${/* A tenth at a time: most of the difference is between five and eight, and half a
                   point across that range is four choices. */ ""}
              <${Slider}
                limits=${{ min: 1, max: 30, step: 0.1 }}
                disabled=${off}
                value=${form.guidance}
                onChange=${(value) => change("guidance", value)}
              />
            </div>
          </div>
        `}

        ${tab === "img2img" &&
        !why &&
        html`
          <div className="card">
            <div className="row">
              <${NumberBox}
                label="Denoising strength"
                kind="number"
                box=${{ min: 0, max: 1, step: 0.05 }}
                disabled=${off}
                value=${form.strength}
                onChange=${(value) => change("strength", value)}
              />
              <${Slider}
                limits=${{ min: 0, max: 1, step: 0.05 }}
                disabled=${off}
                value=${form.strength}
                onChange=${(value) => change("strength", value)}
              />
            </div>
          </div>
        `}

        <div className="card">
          <div className="row">
            <${Field} label="Seed" kind="grow">
              <input
                type="text"
                inputMode="numeric"
                value=${form.seed}
                disabled=${off}
                onChange=${(e) => change("seed", e.target.value)}
              />
            <//>
            <button
              className="plain"
              title="A different picture every time"
              disabled=${off}
              onClick=${onAnySeed}
            >
              🎲
            </button>
            <button
              className="plain"
              title="The seed of the last picture"
              disabled=${off}
              onClick=${onLastSeed}
            >
              ♻
            </button>
          </div>
        </div>
        <//>
      `}
    </div>
  `;
}

// -- what came of it ----------------------------------------------------------------------------

/** The bar, which is the only thing on the page that moves on its own. */
function Bar({ progress }) {
  if (!progress.busy) return null;

  // A fetch and a run both know how far along they are; reading a model does not -- it is one
  // call into the tensor library that returns when it returns. That bar fills the whole width and
  // says so by moving, rather than by making a fraction up.
  const fraction = progress.fraction;

  // What is being stopped decides what the wait is for: a run is stopped between steps, and a
  // fetch between whatever it is in the middle of and the next thing it would have asked for.
  const stopping = progress.fetching ? "stopping the download..." : "stopping after this step...";
  const words = progress.interrupting
    ? stopping
    : [
        progress.doing,
        fraction === null ? null : `${Math.round(fraction * 100)}%`,
        progress.seconds === null ? null : `${progress.seconds.toFixed(1)}s`,
      ]
        .filter(Boolean)
        .join("   ");

  return html`
    <div className=${`bar${fraction === null ? " waiting" : ""}`}>
      <div
        id="bar-fill"
        style=${{ width: fraction === null ? "100%" : `${Math.round(fraction * 100)}%` }}
      ></div>
      <span id="bar-words">${words}</span>
    </div>
  `;
}

function Output({ state, pictures, progress, note, showing, onShow, onSend, onReuse, onDelete }) {
  // The newest is what somebody is looking at, unless they have clicked another since it came in
  // and it is still there: a run that finishes takes the frame back (useNewestWins).
  const picture = pictures.find((one) => one.id === showing) ?? pictures[0] ?? null;

  return html`
    <div className="output">
      <${Bar} progress=${progress} />
      ${note && html`<div className=${`note${note.bad ? " bad" : ""}`}>${note.said}</div>`}

      <div className="canvas">
        ${picture
          ? html`<img alt="" src=${picture.url} />`
          : html`<div className="nothing">Nothing drawn yet.</div>`}
      </div>

      ${picture &&
      html`
        <div className="actions">
          <a className="plain" href=${picture.url} download=${fileName(picture)}>Save</a>
          ${/* Only where there is an img2img tab to send it to, which is a model that can start
               from a picture. */ ""}
          ${state?.model?.draws_from_a_picture &&
          html`<button className="plain" onClick=${onSend}>Send to img2img</button>`}
          <button className="plain" onClick=${() => onReuse(picture)}>Reuse these settings</button>
          ${/* Off to the side of the three that keep it, because it is the one that does not. */ ""}
          <button className="plain away last" onClick=${() => onDelete(picture)}>Delete</button>
        </div>
        <textarea
          className="parameters"
          rows="4"
          readOnly
          value=${`${picture.parameters}\nTime taken: ${picture.seconds.toFixed(2)}s`}
        ></textarea>
      `}

      <div className="gallery">
        ${pictures.map(
          (one) => html`
            <img
              key=${one.id}
              src=${one.url}
              alt=${one.prompt}
              title=${`${fileName(one)} -- ${one.seed}`}
              className=${one.id === (picture?.id ?? null) ? "on" : ""}
              onClick=${() => onShow(one.id)}
            />
          `,
        )}
      </div>
    </div>
  `;
}

/**
 * text2speech: what came of it -- the clip in the player, and the ones before it under it.
 *
 * The same three parts the picture side has: the bar, the newest thing where it can be looked at
 * -- listened to, here -- and everything else this session made in a strip below. A clip cannot
 * be shown the way a picture can, so what the strip holds is the first words of each rather than
 * a thumbnail of it.
 */
function ClipOutput({
  clips,
  progress,
  note,
  showing,
  onShow,
  onReuse,
  onDelete,
  nothing = "Nothing said yet.",
}) {
  const clip = clips.find((one) => one.id === showing) ?? clips[0] ?? null;

  return html`
    <div className="output">
      <${Bar} progress=${progress} />
      ${note && html`<div className=${`note${note.bad ? " bad" : ""}`}>${note.said}</div>`}

      <div className="canvas player">
        ${clip
          ? html`
              ${/* Keyed by the clip, so that a new clip replaces the player rather than leaving
                   the old one loaded under a new address -- which is a player that goes on
                   playing what it had.

                   Not autoPlay. With it, every change of the clip on show started a sound: a new
                   reading finishing, but also a click on an old one in the strip, and a page
                   reload with clips already in it. A sound is played when somebody presses
                   play. */ ""}
              <${Player} key=${clip.id} src=${clip.url} />
              <p className="said">${clip.text}</p>
            `
          : html`<div className="nothing">${nothing}</div>`}
      </div>

      ${clip &&
      html`
        <div className="actions">
          <a className="plain" href=${clip.url} download=${fileName(clip)}>Save</a>
          <button className="plain" onClick=${() => onReuse(clip)}>Reuse these settings</button>
          <button className="plain away last" onClick=${() => onDelete(clip)}>Delete</button>
        </div>
        <textarea
          className="parameters"
          rows="4"
          readOnly
          value=${`${clip.parameters}\nTime taken: ${clip.seconds.toFixed(2)}s -- ${clock(
            clip.length,
          )} long`}
        ></textarea>
      `}

      <div className="clips">
        ${clips.map(
          (one) => html`
            <button
              key=${one.id}
              className=${`clip${one.id === (clip?.id ?? null) ? " on" : ""}`}
              title=${`${fileName(one)} -- ${one.seed}`}
              onClick=${() => onShow(one.id)}
            >
              <span className="clip-said">${one.text}</span>
              <span className="dim">${clock(one.length)}</span>
            </button>
          `,
        )}
      </div>
    </div>
  `;
}

// -- what this browser keeps --------------------------------------------------------------------

/**
 * The ids of this browser's jobs and uploads, kept in localStorage.
 *
 * The server knows nothing of who is asking: a job is found by its id, and whoever holds the id can
 * look at it and delete it. So "my pictures" is the list of ids this browser posted, kept here.
 * Per address, host and port -- which is also where the server keeps them, since one program
 * serves one port.
 *
 * Read and written through try, because a browser that will not keep anything (some private
 * windows) still has a page that works: it forgets its jobs when the tab closes, which is all.
 */
const KEPT_JOBS = "waifu-jobs";
const KEPT_INPUTS = "waifu-inputs";

function readKept(key, otherwise) {
  try {
    return JSON.parse(localStorage.getItem(key)) ?? otherwise;
  } catch {
    return otherwise;
  }
}

function writeKept(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Kept for this tab, in React's state, and nowhere else.
  }
}

/** What a thing is called when it is saved out of the page: when it was made, to the second. */
function fileName(one) {
  const at = new Date(one.created);
  const two = (n) => String(n).padStart(2, "0");
  const stamp =
    `${at.getFullYear()}${two(at.getMonth() + 1)}${two(at.getDate())}` +
    `-${two(at.getHours())}${two(at.getMinutes())}${two(at.getSeconds())}`;
  return `waifu-${stamp}.${one.kind === "clip" ? "wav" : "png"}`;
}

/**
 * What a conversion is called in the strip, which has no words of its own to show. Not the seed:
 * it arrives as a JSON number, which is a double, and a sixty-four bit one would be shown wrong.
 */
function conversionSaid(made) {
  return `${made.style ? "Voice and style" : "Voice only"}, ${made.steps} steps`;
}

/** A finished job, as the gallery shows it: what it made, and where to fetch it. */
function asMade(job) {
  const made = job.output.made;
  return {
    ...made,
    id: job.id,
    url: job.output.url,
    created: job.finished ?? job.created,
    kind: job.kind === "image" ? "picture" : "clip",
    ...(job.kind === "conversion" ? { text: conversionSaid(made) } : {}),
  };
}

/** Lets go of the one picked out of `made` whenever something turns up in it that was not there
 *  before, so the frame falls back to the newest. A deletion adds nothing and keeps the pick. */
function useNewestWins(made, choose) {
  const seen = useRef(null);
  const ids = made.map((one) => one.id).join(",");
  useEffect(() => {
    const now = ids ? ids.split(",") : [];
    // Before the first run there is nothing to compare with, and nothing picked to let go of.
    if (seen.current && now.some((id) => !seen.current.has(id))) choose(null);
    seen.current = new Set(now);
  }, [ids, choose]);
}

// -- the whole of it ----------------------------------------------------------------------------

function App() {
  /** The model this program serves, as it describes itself. Asked for once: it was read before
   *  the page was served, and it does not change while the program runs. */
  const [model, setModel] = useState(null);

  /** What the worker is doing, for anybody's job. Read twice a second. */
  const [worker, setWorker] = useState({ busy: false, queued: 0, progress: {} });

  /** This browser's jobs, newest first, as the server last described them. */
  const [jobs, setJobs] = useState([]);

  /** The uploads this browser is holding to start from: a picture, and a recording. */
  const [inputs, setInputs] = useState(() =>
    readKept(KEPT_INPUTS, { image: null, voice: null, source: null }),
  );

  /** What this page has to say about the last thing that was clicked, if it went wrong. */
  const [complaint, setComplaint] = useState(null);

  /** What the machine is and how much of it is left. Null until the first answer, which is what
   *  keeps the column from flashing a row of empty labels while the page opens. */
  const [machine, setMachine] = useState(null);

  /** Which tab is on top. It starts on the task chosen in the terminal, once the model has said
   *  which that was; after that it is the page's own. */
  const [tab, setTab] = useState(null);

  /** The picture in the big frame, which is the newest one until somebody clicks another. */
  const [showing, setShowing] = useState(null);

  /** And the clip in the player, kept apart from it: they are two lists and two frames. */
  const [playing, setPlaying] = useState(null);

  /** What is in the boxes. Everything here is somebody's typing until it is sent. */
  const [form, setForm] = useState({
    prompt: "",
    negative: "",
    steps: 20,
    guidance: 7,
    strength: 0.8,
    seed: "-1",
    width: 1024,
    height: 1024,
    // What the speech tab is asked for. In the one form beside the rest, because the seed is
    // shared between them and a second form would be a second seed to keep in step with it.
    text: "",
    speed: 1,
    temperature: 0.8,
    // And the conversion tab's. Steps of its own, because a picture's twenty and a conversion's
    // thirty are two different numbers that happen to share a name; and whether to convert the
    // style too, which is a switch where the speech tab's style above is an instruction.
    conversionSteps: 30,
    convertStyle: false,
    // The instruction a reading follows, for a voice that offers styles. Empty is none.
    style: "",
  });

  /** The model whose numbers have been put in the boxes, so that they go in once rather than on
   *  every draw -- which would type over somebody mid-sentence. */
  const adopted = useRef(null);

  const change = useCallback((what, value) => {
    setForm((form) => ({ ...form, [what]: value }));
  }, []);

  const holdInputs = useCallback((next) => {
    setInputs((inputs) => {
      const held = { ...inputs, ...next };
      writeKept(KEPT_INPUTS, held);
      return held;
    });
  }, []);

  /** Reads this browser's jobs from the server, and forgets the ones it no longer has: deleted
   *  from another tab, or pushed out by the output limit. */
  const readJobs = useCallback(async () => {
    const ids = readKept(KEPT_JOBS, []);
    if (!ids.length) return setJobs([]);

    const answer = await ask("GET", `/api/jobs?ids=${ids.join(",")}`);
    if (!answer.ok) return setComplaint(answer.error);

    const found = answer.said.jobs.sort((a, b) => b.created - a.created);
    const still = new Set(found.map((job) => job.id));
    writeKept(
      KEPT_JOBS,
      ids.filter((id) => still.has(id)),
    );
    setJobs(found);
  }, []);

  /** Asks after the jobs that are still going, and takes what has changed. */
  const readGoing = useCallback(async (going) => {
    const answer = await ask("GET", `/api/jobs?ids=${going.map((job) => job.id).join(",")}`);
    if (!answer.ok) return;
    const fresh = new Map(answer.said.jobs.map((job) => [job.id, job]));
    setJobs((jobs) => jobs.map((job) => fresh.get(job.id) ?? job));
  }, []);

  // What is here when the page opens: the model, this browser's jobs, and whether the uploads it
  // was holding are still there to hold.
  useEffect(() => {
    (async () => {
      const answer = await ask("GET", "/api/model");
      if (!answer.ok) return setComplaint(answer.error);
      setModel(answer.said);
      setTab(answer.said.task);

      const held = readKept(KEPT_INPUTS, { image: null, voice: null, source: null });
      for (const which of ["image", "voice", "source"]) {
        if (held[which] && (await fetch(`/api/uploads/${held[which]}`)).status === 404) {
          held[which] = null;
        }
      }
      // The picture -i named, for a page that has not been given one since.
      const start = answer.said.starting_picture;
      if (start && !held.image && readKept("waifu-started-from", null) !== start) {
        held.image = start;
        writeKept("waifu-started-from", start);
      }
      holdInputs(held);
    })();
    readJobs();

    // Another tab of this browser posted or deleted something.
    const elsewhere = (event) => {
      if (event.key === KEPT_JOBS) readJobs();
      if (event.key === KEPT_INPUTS) {
        setInputs(readKept(KEPT_INPUTS, { image: null, voice: null, source: null }));
      }
    };
    window.addEventListener("storage", elsewhere);
    return () => window.removeEventListener("storage", elsewhere);
  }, [readJobs, holdInputs]);

  // How far along things are, twice a second: the worker, and this browser's jobs that are still
  // going. Only those -- a finished job does not change.
  const going = jobs.filter((job) => job.status === "queued" || job.status === "running");
  const goingNow = useRef(going);
  goingNow.current = going;
  useEffect(() => {
    const timer = setInterval(async () => {
      const answer = await ask("GET", "/api/worker");
      if (answer.ok) setWorker(answer.said);
      if (goingNow.current.length) readGoing(goingNow.current);
    }, TICK);
    return () => clearInterval(timer);
  }, [readGoing]);

  // And what is left of the machine, which nothing this page does decides -- so it is asked for
  // on a clock of its own. A failed read is left alone: the column keeps the last answer.
  useEffect(() => {
    const read = async () => {
      const answer = await ask("GET", "/api/machine");
      if (answer.ok) setMachine(answer.said);
    };

    read();
    const timer = setInterval(read, MACHINE_TICK);

    return () => clearInterval(timer);
  }, []);

  // What the components read as the state: the model, and what this browser is holding.
  const state = model && {
    ...model,
    holding_a_picture: !!inputs.image,
    picture_upload: inputs.image,
    holding_a_recording: !!inputs.voice,
    recording_upload: inputs.voice,
    holding_a_source: !!inputs.source,
    source_upload: inputs.source,
  };

  // Puts the chosen model's own numbers in the boxes, and its card's suggestions in the prompts
  // -- which is what it asks to be drawn with, not a setting anybody chose. Once: the model does
  // not change while the program runs.
  const chosen = state?.model ?? null;
  useEffect(() => {
    if (!chosen || adopted.current === chosen.name) return;
    adopted.current = chosen.name;
    setForm((form) => ({
      ...form,
      width: chosen.width,
      height: chosen.height,
      steps: chosen.steps,
      guidance: chosen.guidance,
      strength: 0.8,
      // Only into a box nobody has typed in.
      prompt: form.prompt.trim() ? form.prompt : chosen.prompt ?? "",
      negative: form.negative.trim() ? form.negative : chosen.avoid ?? "",
    }));
  }, [chosen]);

  // And the voice's own numbers, the same way.
  const voice = state?.voice ?? null;
  useEffect(() => {
    if (!voice || adopted.current === voice.name) return;
    adopted.current = voice.name;
    // A style is in one voice's words, and another voice would refuse it or not know it.
    setForm((form) => ({ ...form, speed: voice.speed, temperature: voice.temperature, style: "" }));
  }, [voice]);

  // And the converter's, the same way again.
  const converter = state?.converter ?? null;
  useEffect(() => {
    if (!converter || adopted.current === converter.name) return;
    adopted.current = converter.name;
    setForm((form) => ({ ...form, conversionSteps: converter.steps }));
  }, [converter]);

  const tabs = tabsFor(state);

  // Which of this browser's jobs are going, and what the bar says about them.
  const running = going.find((job) => job.status === "running") ?? null;
  const waiting = going.filter((job) => job.status === "queued");
  const mine = !!running && worker.running === running.id;
  const speaks = model?.kind === "speech";
  const converts = model?.kind === "conversion";
  const theirs = worker.busy && !mine && !waiting.length;
  const doing = mine
    ? worker.progress.doing
    : waiting.length
      ? `waiting: ${waiting[0].position ?? 0} ahead of this one`
      : worker.busy
        ? `busy with another job${worker.queued ? `, ${worker.queued} waiting` : ""}`
        : "";
  const progress = {
    busy: worker.busy || going.length > 0,
    mine,
    // Whether this page has something to stop: its running job, or one waiting its turn.
    drawing: !speaks && !converts && going.length > 0,
    speaking: speaks && going.length > 0,
    converting: converts && going.length > 0,
    fetching: false,
    fraction: mine ? worker.progress.fraction ?? null : null,
    doing,
    seconds: mine ? worker.progress.seconds ?? null : null,
    interrupting: mine && !!running.stopping,
  };

  // What came of this browser's jobs, newest first.
  const done = jobs.filter((job) => job.status === "done" && job.output);
  const pictures = done.filter((job) => job.kind === "image").map(asMade);
  const clips = done.filter((job) => job.kind === "speech").map(asMade);
  const conversions = done.filter((job) => job.kind === "conversion").map(asMade);

  // A picture or clip that has just come in takes the frame back from one picked out of the
  // strip: what was asked for last is what the page should be showing.
  useNewestWins(pictures, setShowing);
  useNewestWins(clips, setPlaying);

  // What this page has to say beats what the last job came to: a complaint is about the click
  // that was just made.
  const latest = jobs[0];
  const note = complaint
    ? { said: complaint, bad: true }
    : latest?.status === "failed"
      ? { said: latest.error, bad: true }
      : latest?.status === "cancelled"
        ? { said: "stopped where it was", bad: false }
        : null;

  // A job can be posted while another is running -- it waits its turn -- but not while one of
  // this page's own is already waiting: a second click is not a second job.
  const canDraw =
    !!chosen?.on_disk && !waiting.length && !(tab === "img2img" && chosen.no_picture_because);
  const canSpeak = !!voice?.on_disk && !waiting.length && !!form.text.trim();
  const canConvert =
    !!converter?.on_disk && !waiting.length && !!inputs.source && !!inputs.voice;

  /** Posts a job, and keeps its id. */
  const post = useCallback(async (asked) => {
    const answer = await ask("POST", "/api/jobs", asked);
    if (!answer.ok) return setComplaint(answer.error);

    setComplaint(null);
    writeKept(KEPT_JOBS, [answer.said.id, ...readKept(KEPT_JOBS, [])]);
    setJobs((jobs) => [answer.said, ...jobs]);
  }, []);

  const generate = useCallback(() => {
    // Both or neither, and neither for a model with no second pass to steer.
    const guided = chosen ? chosen.takes_guidance !== false : true;
    post({
      kind: "image",
      prompt: form.prompt,
      ...(guided ? { negative: form.negative, guidance: Number(form.guidance) } : {}),
      steps: Number(form.steps),
      width: Number(form.width),
      height: Number(form.height),
      // As a string, because a seed is sixty-four bits and a JSON number is a double: the largest
      // ones there are would not survive the trip.
      seed: String(form.seed).trim(),
      strength: Number(form.strength),
      ...(tab === "img2img" && inputs.image ? { init_image: inputs.image } : {}),
    });
  }, [form, tab, inputs, chosen, post]);

  const speak = useCallback(() => {
    post({
      kind: "speech",
      text: form.text,
      speed: Number(form.speed),
      temperature: Number(form.temperature),
      seed: String(form.seed).trim(),
      ...(inputs.voice ? { reference: inputs.voice } : {}),
      ...(voice?.styles?.length && form.style.trim() ? { style: form.style.trim() } : {}),
    });
  }, [form, inputs, voice, post]);

  const convert = useCallback(() => {
    post({
      kind: "conversion",
      source: inputs.source,
      reference: inputs.voice,
      steps: Number(form.conversionSteps),
      style: !!form.convertStyle,
      seed: String(form.seed).trim(),
    });
  }, [form, inputs, post]);

  /** Stops this page's running job, or takes the one waiting out of line. */
  const interrupt = useCallback(async () => {
    const job = running ?? waiting[0];
    if (!job) return;
    const answer = await ask("POST", `/api/jobs/${job.id}/cancel`, {});
    if (!answer.ok) setComplaint(answer.error);
  }, [running, waiting]);

  // Ctrl-enter draws, from wherever the cursor is. The one keystroke every tool of this kind has.
  const draw = useRef(null);
  draw.current =
    tab === SPEAKS
      ? canSpeak
        ? speak
        : null
      : tab === CONVERTS
        ? canConvert
          ? convert
          : null
        : canDraw
          ? generate
          : null;
  useEffect(() => {
    const pressed = (key) => {
      if (key.key === "Enter" && (key.ctrlKey || key.metaKey)) {
        key.preventDefault();
        draw.current?.();
      }
    };
    document.addEventListener("keydown", pressed);
    return () => document.removeEventListener("keydown", pressed);
  }, []);

  /** Uploads a file to start from, and lets go of the one it replaces. */
  const upload = useCallback(
    async (which, file) => {
      const answer = await ask("POST", "/api/uploads", file);
      if (!answer.ok) return setComplaint(answer.error);

      setComplaint(null);
      if (inputs[which]) ask("DELETE", `/api/uploads/${inputs[which]}`);
      holdInputs({ [which]: answer.said.id });
    },
    [inputs, holdInputs],
  );

  const letGo = useCallback(
    async (which) => {
      if (inputs[which]) await ask("DELETE", `/api/uploads/${inputs[which]}`);
      holdInputs({ [which]: null });
    },
    [inputs, holdInputs],
  );

  /** Holds a picture to draw from. */
  const hold = useCallback(
    async (file) => {
      if (!file) return;
      await upload("image", file);
      setTab("img2img");
    },
    [upload],
  );

  const clearPicture = useCallback(() => letGo("image"), [letGo]);

  /**
   * Holds a recording for the voice to sound like.
   *
   * Decoded here rather than posted as it is: the browser has a decoder for every format it will
   * play and the program has one for none of them, so what crosses is samples. A file it cannot
   * decode is said so here, by name, rather than refused on the far side as "not a WAV file".
   */
  const holdAudio = useCallback(
    async (which, file, seconds) => {
      if (!file) return;

      let wav;
      try {
        wav = await asWav(file, seconds);
      } catch (failed) {
        return setComplaint(
          `${file.name} could not be read: this browser has no decoder for it. ` +
            `Anything it can play will work -- wav, mp3, m4a, ogg, flac`,
        );
      }
      await upload(which, wav);
    },
    [upload],
  );

  const holdRecording = useCallback((file) => holdAudio("voice", file), [holdAudio]);
  const clearRecording = useCallback(() => letGo("voice"), [letGo]);

  /** And the recording a conversion is of, which is kept for longer. */
  const holdSource = useCallback(
    (file) => holdAudio("source", file, SOURCE_SECONDS),
    [holdAudio],
  );
  const clearSource = useCallback(() => letGo("source"), [letGo]);

  /**
   * Deletes a picture or a clip, from the server and from this browser's list. Asked about first:
   * what it costs to make one again is a run, and nothing else keeps a copy.
   */
  const forget = useCallback(async (one) => {
    if (!confirm(`Delete this ${one.kind}? Nothing else keeps a copy.`)) return;

    const answer = await ask("DELETE", `/api/jobs/${one.id}`);
    if (!answer.ok && !answer.error.includes("no job")) return setComplaint(answer.error);

    writeKept(
      KEPT_JOBS,
      readKept(KEPT_JOBS, []).filter((id) => id !== one.id),
    );
    setJobs((jobs) => jobs.filter((job) => job.id !== one.id));
  }, []);

  /** Puts a clip's own settings back in the boxes, which is how one is said again. */
  const reuseClip = useCallback((clip) => {
    setForm((form) => ({
      ...form,
      text: clip.text,
      speed: clip.speed,
      temperature: clip.temperature,
      seed: String(clip.seed),
      style: clip.style ?? "",
    }));
  }, []);

  /** Puts a conversion's own settings back in the boxes. The recordings are whatever is held. */
  const reuseConversion = useCallback((clip) => {
    setForm((form) => ({
      ...form,
      conversionSteps: clip.steps,
      convertStyle: !!clip.style,
      seed: String(clip.seed),
    }));
  }, []);

  /** Takes a picture that was drawn back round to the box it can be drawn from. */
  const sendToImg2Img = useCallback(async () => {
    const picture = pictures.find((one) => one.id === showing) ?? pictures[0];
    if (!picture) return;
    const answer = await fetch(picture.url);
    if (!answer.ok) return setComplaint("that picture can no longer be read");
    await hold(await answer.blob());
  }, [hold, showing, pictures]);

  /** Puts a picture's own settings back in the boxes, which is how one is drawn again. */
  const reuse = useCallback((picture) => {
    setForm((form) => ({
      ...form,
      prompt: picture.prompt,
      negative: picture.negative,
      steps: picture.steps,
      guidance: picture.guidance,
      seed: String(picture.seed),
      width: picture.width,
      height: picture.height,
      strength: picture.strength ?? form.strength,
    }));
  }, []);

  const lastSeed = useCallback(() => {
    const picture = pictures.find((one) => one.id === showing) ?? pictures[0];
    if (picture) change("seed", String(picture.seed));
  }, [change, showing, pictures]);

  return html`
    <${TopBar} state=${state} />

    <div className="below">
      <aside className="side">
        <${Nav} tabs=${tabs} tab=${tab} onTab=${setTab} />
        ${/* Under the tabs rather than beside the settings. It is not a setting -- there is
             nothing on it to change -- and what it is is the ground everything else on the page
             stands on, which is where the column's other permanent thing already is. */ ""}
        <${Machine} machine=${machine} />
      </aside>

      ${/* The two halves of the page swap together. Which tab is on top decides what is typed
           at the top, what the column on the left asks for and what the frame on the right
           holds -- and a screen showing a prompt box over an audio player would be a screen
           that had swapped one of the three. */ ""}
      <main>
        ${tab === CONVERTS
          ? html`
              ${!!state?.converter &&
              html`<${ConvertBox}
                state=${state}
                canConvert=${canConvert}
                converting=${!!progress.converting}
                theirs=${theirs}
                onHold=${holdSource}
                onClear=${clearSource}
                onConvert=${convert}
                onInterrupt=${interrupt}
              />`}

              <section className="panes">
                <${ConversionSettings}
                  state=${state}
                  form=${form}
                  change=${change}
                  onHold=${holdRecording}
                  onClear=${clearRecording}
                  onAnySeed=${() => change("seed", "-1")}
                />
                <${ClipOutput}
                  clips=${conversions}
                  progress=${progress}
                  note=${note}
                  showing=${playing}
                  onShow=${setPlaying}
                  onReuse=${reuseConversion}
                  onDelete=${forget}
                  nothing="Nothing converted yet."
                />
              </section>
            `
          : tab === SPEAKS
          ? html`
              ${!!state?.voice &&
              html`<${SayBox}
                form=${form}
                change=${change}
                canSpeak=${canSpeak}
                speaking=${!!progress.speaking}
                theirs=${theirs}
                onSpeak=${speak}
                onInterrupt=${interrupt}
              />`}

              <section className="panes">
                <${SpeechSettings}
                  state=${state}
                  progress=${progress}
                  form=${form}
                  change=${change}
                  onHold=${holdRecording}
                  onClear=${clearRecording}
                  onAnySeed=${() => change("seed", "-1")}
                />
                <${ClipOutput}
                  clips=${clips}
                  progress=${progress}
                  note=${note}
                  showing=${playing}
                  onShow=${setPlaying}
                  onReuse=${reuseClip}
                  onDelete=${forget}
                />
              </section>
            `
          : html`
              ${!!chosen &&
              html`<${Prompts}
                form=${form}
                change=${change}
                canDraw=${canDraw}
                guided=${chosen.takes_guidance !== false}
                drawing=${!!progress.drawing}
                fetching=${!!progress.fetching}
                theirs=${theirs}
                onDraw=${generate}
                onInterrupt=${interrupt}
              />`}

              <section className="panes">
                <${Settings}
                  state=${state}
                  progress=${progress}
                  tab=${tab}
                  form=${form}
                  change=${change}
                  onHold=${hold}
                  onClear=${clearPicture}
                  onAnySeed=${() => change("seed", "-1")}
                  onLastSeed=${lastSeed}
                />
                <${Output}
                  state=${state}
                  pictures=${pictures}
                  progress=${progress}
                  note=${note}
                  showing=${showing}
                  onShow=${setShowing}
                  onSend=${sendToImg2Img}
                  onReuse=${reuse}
                  onDelete=${forget}
                />
              </section>
            `}
      </main>
    </div>
  `;
}

ReactDOM.createRoot(document.getElementById("app")).render(html`<${App} />`);
