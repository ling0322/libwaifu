# Speech: the page before the model

`waifu draw` has a third tab. Type a sentence, press Speak, get a WAV -- and what says it is not a
speech model, because there is not one here yet.

That is deliberate, and it is the whole point of this document. What follows says what is real,
what is a stand-in, and exactly what a speech model has to implement to take the stand-in's place
without anything above it changing.

## What is here

| | |
| --- | --- |
| the page | a third tab: a box to type in, a recording to sound like, speed, temperature, a seed |
| the run | a command on the worker thread, a bar that moves, a button that stops it |
| the clip | `waifu-NNNN.wav` beside the pictures, played on the page, saved, deleted, said again |
| the voice | [`Voice`](../waifu/src/speech.rs), a trait with five methods |
| what implements it | `Tones`, which reads the text as pitched tones -- one to a syllable |

## What is not

`Tones` is not speech and does not claim to be. It is a tone every three letters of a word, a gap
between words, a longer one at a comma, pitched by the letters themselves so the same word is the
same little tune every time it appears. It uses the recording it is handed, honestly and narrowly:
it takes its pitch, by autocorrelation, and centres the tones there.

What it reproduces of speech is the rhythm and roughly the register. That is what can be had from
the letters alone, and it is enough to lay out a page around: a clip whose length tracks the text,
a bar that moves through a run, a player, a gallery, a stop button, a seed that reproduces a
reading.

The page says so itself. `Voice::not_a_voice_because` returns a sentence, the state carries it, and
the speech tab prints it in a card above every setting. A real model returns `None` from that
method and the card is gone -- which is how the warning disappears without anybody remembering to
delete it.

## The seam

```rust
pub trait Voice: Send {
    fn speak(
        &self,
        text: &str,
        like: Option<&Sound>,
        options: &SpeechOptions,
        report: &mut dyn FnMut(SpeechProgress) -> ControlFlow<()>,
    ) -> Result<Option<Sound>>;

    fn rate(&self) -> u32;
    fn defaults(&self) -> SpeechDefaults;
    fn name(&self) -> &'static str;
    fn no_likeness_because(&self) -> Option<&'static str> { None }
    fn not_a_voice_because(&self) -> Option<&'static str> { None }
}
```

`Option<Sound>` rather than `Sound`: `None` is a run that `report` asked to stop, which is not an
error and leaves nothing in the gallery. The same shape `generate_reporting` has, for the same
reason -- half a sentence is not a clip.

`SpeechProgress` has three stages, and the second one is the honest part:

```rust
Saying { done: i32, expected: i32 }
```

`expected` is an estimate. A model that decides what to say one token at a time does not know how
many there will be until it emits the one that ends the utterance, so the bar says "token 12 of
about 40" and holds at the end rather than running past it. That is written down in three places
-- the enum, the bar's arithmetic and the words under it -- because a bar that fills up and starts
again is the kind of thing that gets reported as a bug against the model.

### What a real model changes

Three functions in [`worker.rs`](../waifu/src/cli/webui/worker.rs), and nothing else above them:

- `look_at_voice()` takes a name and reads a manifest, the way `look_at` does for a picture model.
- `read_voice()` fetches and reads a package, reporting through the same `Doing::Fetching` and
  `Doing::Reading` the picture models already report through. The plumbing is already there; the
  function currently constructs a struct with two floats in it.
- The `voice` slot beside `model` becomes one slot the two take turns in. `Tones` holds nothing, so
  it can sit beside a picture model today. A real one cannot -- the card is not big enough for a
  voice and twelve billion parameters of Krea 2 -- and that swap is the same one `Command::Draw`
  already does between one picture model and the next.

`VOICE`, the constant naming the one voice there is, becomes a row in the hub's catalogue like
every other model, and the speech tab's voice card becomes the same button the picture tab has.

## Audio, and why there is no codec here

[`waifu::wav`](../waifu/src/wav.rs) reads and writes WAV, and that is the only audio format this
crate handles. Writing is one shape -- 16-bit mono -- and reading is wider, because a recording
someone drags onto the page has been through whatever wrote it: 8-, 16-, 24- and 32-bit integers,
32- and 64-bit floats, `WAVE_FORMAT_EXTENSIBLE`, however many channels, and chunks in between that
have to be walked past.

Everything else is a codec, which is several thousand lines and a licence to read. So the browser
decodes instead. `asWav` in [`app.js`](../waifu/src/cli/webui/assets/app.js) hands the dropped file
to `AudioContext.decodeAudioData`, mixes it to one channel, trims it to thirty seconds and encodes
a WAV -- which means the page takes mp3, m4a, ogg, flac and webm without this program knowing what
any of those are, and the far side only ever sees something it can read.

A file the browser cannot decode is said so by name, on the page, before anything is posted.

## The recording is checked at the door

`POST /api/voice` parses the WAV as it arrives and refuses what is not one. It is the only thing
posted to this server that is checked on arrival rather than at the run, and the reason is that the
check is a header rather than a decoder: it costs nothing, and it means somebody who dropped the
wrong file finds out now rather than after pressing Speak.

## Reading a clip back

Clips go in `Session::clips`, beside `Session::gallery` rather than mixed into it. One list of what
this session made, in the sense that matters -- one place decides which file names a request may
ask for, and `/clip/...` and `/picture/...` are two doors that cannot reach each other's files.
Two lists in the sense that also matters, which is that a clip has a length and a picture has a
shape and neither has the other.

## Testing it

Nothing here needs a model, a card or a network, so all of it runs in the fast suite:

```bash
LIBWAIFU_LIB_DIR="$PWD/build" cargo test --manifest-path waifu/Cargo.toml --features cli --lib
```

`--features cli` matters: without it, none of `waifu/src/cli` is compiled, so none of the routes,
the state or the worker is tested. The library half -- `speech` and `wav` -- is tested either way.
