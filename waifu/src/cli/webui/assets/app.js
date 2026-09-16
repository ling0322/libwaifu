// What the page does.
//
// It holds no state of its own beyond what is being looked at: everything that matters -- which
// model is loaded, what is happening, what has been drawn -- lives in the program, and this asks
// for it. Which means a second tab opened on the same address shows the same thing, and a tab
// left open across a run that started somewhere else catches up on its own.
//
// There are two things it asks for. The whole of the state, which is cheap but not free, is read
// when something has happened; and how far along a run is, which is a handful of numbers, is read
// twice a second. The `revision` in each says whether the other is worth asking for.
//
// React draws it, from the three files under vendor/: React itself, its renderer, and htm, which
// is what stands in for JSX -- a tagged template the browser parses on its own. All three are
// built into the binary beside this file, so there is still no build step between the source and
// the thing that runs, and the page still asks for nothing outside this machine.

const { Fragment, useCallback, useEffect, useRef, useState } = React;
const html = htm.bind(React.createElement);

/** How often to ask what is happening. Short enough that the bar moves, long enough to be free. */
const TICK = 500;

// -- talking to the program -------------------------------------------------------------------

/** A request, and what it answered. An error is a value here rather than a throw: every one of
 *  them ends the same way, which is a line on the screen. */
async function ask(method, path, body) {
  try {
    const sent = {
      method,
      headers: body instanceof Blob ? {} : { "Content-Type": "application/json" },
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

/**
 * The two kinds of run, one under the other.
 *
 * Neither is ever disabled. Which kind of run comes first and the model second -- and changing
 * the kind un-chooses the model, so a model that cannot start from a picture is not a reason to
 * bar the way to the page that starts from one. What that model cannot do is said on the page
 * it is chosen on, beside the button that would have asked for it.
 */
function Nav({ tab, onTab }) {
  return html`
    <nav className="nav">
      ${["txt2img", "img2img"].map(
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
 * What is going to be drawn with, and where: the two things a run needs before any of its own
 * settings mean anything, at the head of the column they belong to.
 *
 * The model is a button rather than a list. What the list has to say about each one -- what is on
 * the disk, what each costs to fetch, what can be deleted -- does not go in a dropdown, so the
 * dropdown is not offered anywhere; this is the one way to the list and the one place the answer
 * is shown.
 */
function ModelAndDevice({ state, progress, onDevice, onModels }) {
  const chosen = state?.model;

  // Chosen is not read. What it costs to start a run is worth saying before the run, because it
  // is the difference between four seconds and a download of several gigabytes.
  const standing = !chosen
    ? "nothing chosen yet -- a run needs one"
    : chosen.in_memory
      ? "read, and on the device"
      : chosen.on_disk
        ? "on the disk -- read at the first run"
        : "not fetched -- fetched and read at the first run";

  return html`
    <div className="card">
      <div className="row">
        <${Field} label="Model" kind="grow">
          ${/* Lit up while there is nothing chosen, because until there is, this is the only
               thing on the page that does anything; an ordinary button once it has been. */ ""}
          <button
            className=${`plain wide picker${chosen ? "" : " next"}`}
            onClick=${onModels}
          >
            ${chosen ? chosen.full_name : "Choose a model"}
          </button>
        <//>
        <${Field} label="Device" kind="short">
          ${/* Said in the tooltip rather than in a paragraph of its own: this is two words at the
               top of a column of settings, not one of the settings. */ ""}
          <select
            value=${state?.device ?? ""}
            disabled=${!!progress.busy}
            title="Where runs go. Changing it lets go of whatever weights are in memory; the next run reads them again on the device chosen."
            onChange=${(e) => onDevice(e.target.value)}
          >
            ${(state?.devices ?? []).map(
              (device) => html`<option key=${device} value=${device}>${device}</option>`,
            )}
          </select>
        <//>
      </div>
      <p className="about">${standing}</p>
      ${chosen?.no_picture_because &&
      html`<p className="about">Cannot start from a picture: ${chosen.no_picture_because}.</p>`}
    </div>
  `;
}

// -- what to draw -------------------------------------------------------------------------------

function Prompts({ form, change, canDraw, drawing, off, onDraw, onInterrupt }) {
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
            disabled=${off}
            onChange=${(e) => change("prompt", e.target.value)}
          ></textarea>
        <//>
        <${Field} label="Negative prompt">
          <textarea
            rows="3"
            placeholder="What to keep out. Left empty the model is still steered away from the empty prompt, which is not the same as steering away from nothing at all."
            value=${form.negative}
            disabled=${off}
            onChange=${(e) => change("negative", e.target.value)}
          ></textarea>
        <//>
      </div>
      <div className="go">
        <button className="generate" disabled=${!canDraw} onClick=${onDraw}>Generate</button>
        ${/* Always under it, rather than appearing only once there is something to stop: a button
             that is not there until the moment it is needed is a button nobody knows about, and
             its place on the screen moves the thing above it when it arrives. */ ""}
        <button
          className="interrupt"
          disabled=${!drawing}
          title=${drawing
            ? "Stop after the step it is on, and keep what has been drawn so far"
            : "Nothing to stop. A run can be stopped once it is drawing; fetching and reading a model cannot be stopped part way"}
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
          ? // With the revision on the end, because the address is the same every time and the
            // picture behind it is not.
            html`<img alt="" src=${`/api/upload?${revision}`} />`
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
  onDevice,
  onModels,
}) {
  const model = state?.model;
  const why = model?.no_picture_because;

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
      <${ModelAndDevice}
        state=${state}
        progress=${progress}
        onDevice=${onDevice}
        onModels=${onModels}
      />

      ${/* Nothing below this is worth showing until there is a model: every one of them is a
           setting for one, and a column of greyed-out boxes is a longer way of saying the same
           thing the button above says. */ ""}
      ${!!model &&
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
              revision=${state?.revision}
              off=${off}
              onHold=${onHold}
              onClear=${onClear}
            />`)}

        ${/*
          One knob to a card, each with the sentence that says what it does. They used to be paired
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
          <p className="about">
            What shape to draw. The list is what the model was trained at, which is where it draws
            one of a thing rather than two; a side typed into the boxes is taken down to the
            multiple of 64 below it, which is the shape the model can actually be asked for.
          </p>
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
          <p className="about">
            How many times the model is asked what to take out. More is more detail and costs its
            share of the time; past about forty there is little left to add.
          </p>
        </div>

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
          <p className="about">
            How hard to push towards the prompt. Five to eight is the usual range; higher burns the
            colours out, and one ignores the prompt and runs twice as fast.
          </p>
        </div>

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
            <p className="about">
              How far to walk from the picture above. Around 0.8 redraws it and keeps its
              composition; below about 0.3 there is little left for the prompt to do.
            </p>
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
          <p className="about">
            Which noise to start from. The same seed with everything else the same draws the same
            picture again; minus one is a new one every time.
          </p>
        </div>
        <//>
      `}
    </div>
  `;
}

// -- choosing a model ---------------------------------------------------------------------------

/** What is on the disk of one model, in the one line the row has for it. */
function onDisk(model) {
  if (model.cached) return `on the disk -- ${room(model.bytes)}`;
  // Not fetched and not nothing: a fetch that was stopped part way, which is worth saying,
  // because what is there is what the next fetch does not have to bring down again.
  if (model.bytes > 0) return `part fetched -- ${room(model.bytes)} of it is here`;
  return "not fetched -- it comes down at the first run";
}

/**
 * Every model this build knows, what is on the disk of each, and the two things that can be done
 * about it.
 *
 * Over the page rather than a page of its own: what is being drawn is decided before what it is
 * drawn with, and a screen that made somebody leave their prompt to go and change the model would
 * be putting the two the other way round. Choosing reads nothing -- it says which model the next
 * run is of, and the run is what reads it.
 */
function ModelPicker({ state, progress, note, onChoose, onForget, onRefresh, onClose }) {
  const chosen = state?.model;
  const busy = !!progress.busy;

  return html`
    <div className="veil" onClick=${onClose}>
      ${/* The box itself swallows the click that would close it: a list is something people click
           about in, and every one of those clicks lands on the veil underneath. */ ""}
      <div className="dialog models-dialog" onClick=${(event) => event.stopPropagation()}>
        <div className="dialog-top">
          <div className="card-title">Choose a model</div>
          <button className="plain" title="Look again at what is on the disk" onClick=${onRefresh}>
            ↻
          </button>
          <button className="plain" onClick=${onClose}>Close</button>
        </div>

        <p className="about">
          Choosing one reads nothing: the weights are read by the first run that needs them, and
          fetched first -- several gigabytes, kept afterwards -- where they are not here yet.
        </p>
        ${note?.bad && html`<div className="note bad">${note.said}</div>`}

        <div className="models">
          ${(state?.models ?? []).map((model) => {
            const here = chosen?.name === model.name;
            return html`
              <div key=${model.name} className="card model">
                <div className="model-what">
                  <div className="model-name">
                    ${model.full_name}
                    <span className="model-id">${model.name}</span>
                    ${here &&
                    html`<span className="badge">${chosen.in_memory ? "in memory" : "chosen"}</span>`}
                  </div>
                  <p className="about">${onDisk(model)}</p>
                  ${/* What this one will and will not do, as far as it is known before it is
                       read: out of its manifest where the package is here, and guessed from the
                       name where it is not. */ ""}
                  ${here &&
                  html`<p className="about">
                    ${`${chosen.sampler} -- ${chosen.steps} steps -- ${chosen.width} x ${chosen.height}`}
                    ${chosen.no_picture_because
                      ? ` -- cannot start from a picture: ${chosen.no_picture_because}`
                      : " -- draws from a prompt or from a picture"}
                  </p>`}
                </div>

                <div className="model-do">
                  <button className="plain" disabled=${here} onClick=${() => onChoose(model.name)}>
                    ${here ? "Chosen" : "Choose"}
                  </button>
                  <button
                    className="plain away"
                    disabled=${busy || model.bytes === 0}
                    onClick=${() => onForget(model.name)}
                  >
                    Delete
                  </button>
                </div>
              </div>
            `;
          })}
        </div>
      </div>
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
  const words = progress.interrupting
    ? "stopping after this step..."
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

function Output({ state, progress, note, showing, onShow, onSend, onReuse, onDelete }) {
  // The newest is what somebody is looking at, unless they have clicked another and it is still
  // there: a run that finishes while an older picture is up should not snatch the frame away.
  const pictures = state?.gallery ?? [];
  const picture = pictures.find((one) => one.file === showing) ?? pictures[0] ?? null;

  return html`
    <div className="output">
      <${Bar} progress=${progress} />
      ${note && html`<div className=${`note${note.bad ? " bad" : ""}`}>${note.said}</div>`}

      <div className="canvas">
        ${picture
          ? html`<img alt="" src=${`/picture/${picture.file}`} />`
          : html`<div className="nothing">Nothing drawn yet.</div>`}
      </div>

      ${picture &&
      html`
        <div className="actions">
          <a className="plain" href=${`/picture/${picture.file}`} download=${picture.file}>Save</a>
          <button className="plain" onClick=${onSend}>Send to img2img</button>
          <button className="plain" onClick=${() => onReuse(picture)}>Reuse these settings</button>
          ${/* Off to the side of the three that keep it, because it is the one that does not. */ ""}
          <button className="plain away last" onClick=${() => onDelete(picture)}>Delete</button>
        </div>
        <textarea
          className="parameters"
          rows="4"
          readOnly
          value=${`${picture.parameters}\nTime taken: ${picture.seconds.toFixed(2)}s -- written to ${picture.file}`}
        ></textarea>
      `}

      <div className="gallery">
        ${pictures.map(
          (one) => html`
            <img
              key=${one.file}
              src=${`/picture/${one.file}`}
              alt=${one.prompt}
              title=${`${one.file} -- ${one.seed}`}
              className=${one.file === (picture?.file ?? null) ? "on" : ""}
              onClick=${() => onShow(one.file)}
            />
          `,
        )}
      </div>
    </div>
  `;
}

// -- the whole of it ----------------------------------------------------------------------------

function App() {
  /** The last whole state, as the program described it. */
  const [state, setState] = useState(null);

  /** How far along a run is, which is read far more often than the rest. */
  const [progress, setProgress] = useState({ busy: false, drawing: false });

  /** What this page has to say about the last thing that was clicked, if it went wrong. */
  const [complaint, setComplaint] = useState(null);

  /** Which tab is on top. The only thing on this page the program does not know about: it decides
   *  what a run is asked for, not what the program is doing. */
  const [tab, setTab] = useState("txt2img");

  /** Whether the list of models is up over the page. */
  const [picking, setPicking] = useState(false);

  /** The picture in the big frame, which is the newest one until somebody clicks another. */
  const [showing, setShowing] = useState(null);

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
  });

  /** Which revision of the state this page has seen, so that a poll can tell news from quiet. */
  const seen = useRef(-1);

  /** The model whose numbers have been put in the boxes, so that they go in once rather than on
   *  every draw -- which would type over somebody mid-sentence. */
  const adopted = useRef(null);

  /** Which of the boxes have been moved by hand since. A chosen model is described twice -- from
   *  its name, and again from the package once a run has read it -- and the second description is
   *  worth taking, but not over the top of somebody's own numbers. */
  const byHand = useRef(new Set());

  const change = useCallback((what, value) => {
    byHand.current.add(what);
    setForm((form) => ({ ...form, [what]: value }));
  }, []);

  const readState = useCallback(async () => {
    const answer = await ask("GET", "/api/state");
    if (!answer.ok) return setComplaint(answer.error);

    setComplaint(null);
    seen.current = answer.said.revision;
    setState(answer.said);
  }, []);

  // Reads how far along things are, and the whole state again when something has happened.
  useEffect(() => {
    readState();
    const timer = setInterval(async () => {
      const answer = await ask("GET", "/api/progress");
      if (!answer.ok) return;

      setProgress(answer.said);
      if (answer.said.revision !== seen.current) readState();
    }, TICK);

    return () => clearInterval(timer);
  }, [readState]);

  // Puts the chosen model's own numbers in the boxes, and its card's suggestions in the prompts
  // -- which is what it asks to be drawn with, not a setting anybody chose.
  //
  // Twice for one model, in the ordinary case: once from its name when it is chosen, and again
  // when a run has read the package and the numbers in it turn out to be its own. The second time
  // goes only into boxes nobody has touched.
  const chosen = state?.model ?? null;
  const described = chosen && `${chosen.name} ${chosen.in_memory}`;
  useEffect(() => {
    if (!chosen) {
      adopted.current = null;
      return;
    }
    if (described === adopted.current) return;

    // The second description of one model waits for the run that produced it to finish. It only
    // happens to a model that was not on the disk when it was chosen -- what is here is read when
    // it is picked -- and numbers that change while the bar is counting the old ones read as a
    // screen that has lost track of what it is doing.
    const again = adopted.current?.startsWith(`${chosen.name} `);
    if (again && progress.busy) return;

    adopted.current = described;
    if (!again) byHand.current.clear();

    setForm((form) => {
      const keep = (what, value) => (again && byHand.current.has(what) ? form[what] : value);
      return {
        ...form,
        width: keep("width", chosen.width),
        height: keep("height", chosen.height),
        steps: keep("steps", chosen.steps),
        guidance: keep("guidance", chosen.guidance),
        strength: keep("strength", 0.8),
        // Only into a box nobody has typed in. A prompt that was being written when a model
        // finished loading is somebody's work, and a suggestion is not worth losing it over.
        prompt: form.prompt.trim() ? form.prompt : chosen.prompt ?? "",
        negative: form.negative.trim() ? form.negative : chosen.avoid ?? "",
      };
    });
  }, [chosen, described, progress.busy]);

  // A picture named on the command line opens the tab it is for: somebody who passed -i has said
  // which kind of run they came here to do. Once, on the first state this page ever reads.
  const arrived = useRef(false);
  useEffect(() => {
    if (arrived.current || !state) return;
    arrived.current = true;
    if (state.holding_a_picture && state.model?.draws_from_a_picture) setTab("img2img");
  }, [state]);

  // What this page has to say beats what the program had to say: a complaint is about the click
  // that was just made, and the note is about whatever happened last on the other side.
  const note = complaint ? { said: complaint, bad: true } : state?.note;

  // Nothing is chosen until the list has been to, and no run is asked for that the chosen model
  // will refuse: the reason is on the screen beside the button, so the button says no rather than
  // the wait does.
  const canDraw =
    !!chosen && !progress.busy && !(tab === "img2img" && chosen.no_picture_because);

  const generate = useCallback(async () => {
    const answer = await ask("POST", "/api/generate", {
      prompt: form.prompt,
      negative: form.negative,
      steps: Number(form.steps),
      guidance: Number(form.guidance),
      width: Number(form.width),
      height: Number(form.height),
      // As a string, because a seed is sixty-four bits and a JSON number is a double: the largest
      // ones there are would not survive the trip.
      seed: String(form.seed).trim(),
      strength: Number(form.strength),
      from_picture: tab === "img2img" && !!state?.holding_a_picture,
    });

    if (!answer.ok) return setComplaint(answer.error);
    // Straight away, rather than at the next poll: a button that stays live for half a second
    // after it is pressed is a button that gets pressed twice.
    setProgress({ busy: true, drawing: true, fraction: 0, doing: "starting", seconds: 0 });
  }, [form, tab, state]);

  // Ctrl-enter draws, from wherever the cursor is. The one keystroke every tool of this kind has.
  // Through a box rather than in the listener itself, so that the page is not listened to afresh
  // every time a letter is typed into the prompt.
  const draw = useRef(null);
  draw.current = canDraw ? generate : null;
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

  /** Says which model runs are of, or that none is. Reads nothing: the first run does that. */
  const chooseModel = useCallback(async (name) => {

    // Out of the way first: what was asked for is what the button that opened this said, and the
    // page behind it is the one the choice was made for.
    setPicking(false);
    const answer = await ask("POST", "/api/model", { model: name ?? null });
    if (!answer.ok) return setComplaint(answer.error);
    await readState();
  }, [readState]);

  /**
   * Opens the other kind of run, and un-chooses the model on the way.
   *
   * What is worth drawing with is a question about the run: the model that was picked to draw
   * from a prompt is not automatically the one to redraw a picture with, and one of them cannot
   * do the second at all. So the choice is made again for the run it is being made about.
   */
  const switchTask = useCallback(
    (which) => {
      setTab((tab) => {
        if (tab !== which) chooseModel(null);
        return which;
      });
    },
    [chooseModel],
  );

  const useDevice = useCallback(async (device) => {
    const answer = await ask("POST", "/api/device", { device });
    if (!answer.ok) setComplaint(answer.error);
  }, []);

  const forgetModel = useCallback(
    async (name) => {
      if (!name) return;
      if (!confirm(`Delete everything fetched of ${name}? It can be fetched again.`)) return;

      const answer = await ask("DELETE", "/api/model", { model: name });
      if (!answer.ok) return setComplaint(answer.error);
      await readState();
    },
    [readState],
  );

  /** Holds a picture to draw from, and shows it once the program has it. */
  const hold = useCallback(
    async (file) => {
      if (!file) return;

      const answer = await ask("POST", "/api/upload", file);
      if (!answer.ok) return setComplaint(answer.error);

      await readState();
      setTab("img2img");
    },
    [readState],
  );

  const clearPicture = useCallback(async () => {
    await ask("DELETE", "/api/upload");
    await readState();
  }, [readState]);

  /** Takes a picture that was drawn back round to the box it can be drawn from. */
  const sendToImg2Img = useCallback(async () => {
    const answer = await fetch(`/picture/${showing ?? state?.gallery[0]?.file}`);
    if (!answer.ok) return setComplaint("that picture can no longer be read");

    await hold(await answer.blob());
  }, [hold, showing, state]);

  /**
   * Deletes a picture, from the disk and from the page.
   *
   * Asked about first. What it costs to draw one again is a run, and what it costs to have
   * deleted the wrong one is that picture -- the file is written where the program was started
   * and there is no copy of it anywhere else.
   */
  const deletePicture = useCallback(
    async (picture) => {
      if (!confirm(`Delete ${picture.file}? The file itself goes, and nothing keeps a copy.`)) {
        return;
      }

      const answer = await ask("DELETE", "/api/picture", { file: picture.file });
      if (!answer.ok) return setComplaint(answer.error);
      await readState();
    },
    [readState],
  );

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
    const pictures = state?.gallery ?? [];
    const picture = pictures.find((one) => one.file === showing) ?? pictures[0];
    if (picture) change("seed", String(picture.seed));
  }, [change, showing, state]);

  return html`
    <${TopBar} state=${state} />

    <div className="below">
      <aside className="side">
        <${Nav} tab=${tab} onTab=${switchTask} />
      </aside>

      <main>
        <${Prompts}
          form=${form}
          change=${change}
          canDraw=${canDraw}
          off=${!chosen}
          drawing=${!!progress.drawing}
          onDraw=${generate}
          onInterrupt=${() => ask("POST", "/api/interrupt")}
        />

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
            onDevice=${useDevice}
            onModels=${() => setPicking(true)}
          />
          <${Output}
            state=${state}
            progress=${progress}
            note=${note}
            showing=${showing}
            onShow=${setShowing}
            onSend=${sendToImg2Img}
            onReuse=${reuse}
            onDelete=${deletePicture}
          />
        </section>
      </main>
    </div>

    ${picking &&
    html`<${ModelPicker}
      state=${state}
      progress=${progress}
      note=${note}
      onChoose=${chooseModel}
      onForget=${forgetModel}
      onRefresh=${readState}
      onClose=${() => setPicking(false)}
    />`}
  `;
}

ReactDOM.createRoot(document.getElementById("app")).render(html`<${App} />`);
