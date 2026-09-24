# Speech: the page, and the model behind it

`waifu webui -voice indextts` has a third tab. Drop a few seconds of somebody speaking on it,
type a sentence, press Speak, and get the sentence back in that voice as a WAV -- said by
[IndexTTS-2.5](indextts.md).

Without `-voice` there is no third tab. What the page has then is `Tones`, the stand-in the tab was
built around before there was a model, and a tab whose whole content is an apology for not being
speech is not worth a place in the list.

This document is about the seam between the two: what the page asks of a voice, and what a voice
has to implement to answer it.

```bash
waifu webui -voice indextts
```

`indextts` is fetched off the hub like any picture model, into the same cache. Building the
package yourself instead -- `-voice models/indextts25.yaml` -- is [`indextts.md`](indextts.md).

## What is here

| | |
| --- | --- |
| the page | a third tab: a box to type in, a recording to sound like, speed, temperature, a seed |
| the run | a command on the worker thread, a bar that moves, a button that stops it |
| the clip | `waifu-NNNN.wav` beside the pictures, played on the page, saved, deleted, said again |
| the voice | [`Voice`](../waifu/src/speech.rs), a trait with five methods |
| what implements it | [`IndexTts`](../waifu/src/indextts/mod.rs), named by `-voice`; and `Tones`, the stand-in, when nothing is |

## The stand-in

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

### What a real model changed

The three things this section used to predict, in [`worker.rs`](../waifu/src/cli/webui/worker.rs):

- `look_at_voice(name)` describes a voice before any of it is read: `Tones` by asking it, and a
  package by what its kind says -- IndexTTS-2.5's rate and starting values, and that it is not in
  memory. The tab's boxes start from that.
- `read_voice()` fetches and reads a package at the first reading that wants it, reporting through
  the same `Doing::Fetching` and `Doing::Reading` the picture models report through.
- **The card is taken in turns.** IndexTTS-2.5 is five gigabytes and does not sit beside a twelve
  billion parameter picture model, so reading either puts the other down first -- the swap
  `Command::Draw` already did between one picture model and the next. `Tones` holds nothing and
  still sits beside anything. A stopped reading puts a voice with weights down and hands the
  memory back, as a stopped drawing does.

Two things the prediction did not cover:

- **`Voice` is no longer `Send`.** It was, while the only voice was two floats. A real one is made
  of tensors, and a tensor stays on the thread that made it -- the same reason the picture models
  are not `Send` either. Nothing moves a voice between threads.
- **A voice can refuse to speak without a recording.** IndexTTS-2.5 has no voice of its own; upstream
  takes the speaker's audio as a required argument. Asked to speak with none, it says so on the
  bar, in a sentence, rather than guessing at a voice.

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
