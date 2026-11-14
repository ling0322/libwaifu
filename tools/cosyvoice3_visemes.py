# The MIT License (MIT)
#
# Copyright (c) 2026 Xiaoyang Chen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""A mouth shape for every CosyVoice3 speech token: eight soft classes, trained on three languages.

    A  E  I  O  U  @  MBP  REST

`@` is the schwa, and the half-open, unspread mouth of Mandarin's e and en; MBP is the lips pressed
shut; REST is the relaxed, lightly closed mouth, not silence -- whether anybody is speaking is the
sound's energy, not a mouth shape.

The readings come from `waifu-tools`' `viseme_corpus`: sentences in several voices, and
S3Tokenizer's tokens of them, 25 a second. `align` labels each token's 40 ms:

- Qwen3-ForcedAligner puts each unit of the text on the audio: a character of Chinese, a mora of
  Japanese (the aligner is handed morae rather than its own words), a word of English.
- The unit is spelled into mouth shapes: pypinyin's initial and final for Chinese, a kana table for
  Japanese, CMUdict for English. Its span is shared among them by weight.
- Vowels, m/b/p and pauses are anchors with a shape of their own. Any other consonant has none: its
  frames run linearly from the anchor before to the anchor after, which is what a mouth does while
  it says one -- it is already on its way to the next vowel.
- A sound between two shapes is labelled as a mixture of them: English's cat is A 0.7 + E 0.3.

A 5 ms timeline is drawn of that and each token labelled with the mean of its eight frames.

`train` fits a softmax with one hidden layer on the flow's `input_embedding`, holding one voice
group (a recording and its pitch-shifted copy) out at a time, then fits on everything and writes a
package of its own -- `cosyvoice3-visemes.yaml` and `cosyvoice3-visemes.safetensors`, model type
`cosyvoice3_visemes` -- which a pipeline with CosyVoice3 loaded attaches:

- `cosyvoice3_visemes.{w1,b1,w2,b2}`: a token with a token either side, which is what runs.
- `cosyvoice3_visemes.embedding`: the flow's `input_embedding`, copied, so the package needs
  nothing of CosyVoice3's weights.
- `cosyvoice3_visemes.table`, `(6561, 8)`: each token's distribution alone, from a model of its own
  80 numbers, so that a token never seen in the readings has one too.

    PYTHONPATH=<pypinyin, nagisa>:<Qwen3-ASR> .venv-qwen3-asr/bin/python tools/cosyvoice3_visemes.py \\
        align <corpus_dir> -language Chinese|Japanese|English [-cmudict cmudict.dict]
    ... tools/cosyvoice3_visemes.py train <corpus_dir>... -package <cosyvoice3 package> \\
        -output <package directory>
"""

import argparse
import json
import struct
from pathlib import Path

import numpy as np

CLASSES = ["A", "E", "I", "O", "U", "@", "MBP", "REST"]
INDEX = {c: k for k, c in enumerate(CLASSES)}
TOKEN_SECONDS = 0.04
FRAME_SECONDS = 0.005
FRAMES_PER_TOKEN = 8
# The share of a Chinese character or a Japanese mora its onset consonant takes.
ONSET_SHARE = 0.3


def shape(**weights):
    """A distribution over the eight classes; `shape(A=0.7, E=0.3)`. `AT` stands for `@`."""
    out = np.zeros(len(CLASSES), dtype=np.float32)
    for name, weight in weights.items():
        out[INDEX["@" if name == "AT" else name]] = weight
    return out / out.sum()


# A unit of text is spelled as a list of (shape or None, weight). None is a consonant with no shape
# of its own, filled in from its neighbours.
CONSONANT = None
MBP = shape(MBP=1)
REST = shape(REST=1)


# ---- Chinese: pypinyin's initial and final


def mandarin_finals():
    a, e, i, o, u, at = shape(A=1), shape(E=1), shape(I=1), shape(O=1), shape(U=1), shape(AT=1)
    yu = shape(U=0.6, I=0.4)  # ü: an i with rounded lips
    ong = shape(U=0.6, O=0.4)  # [ʊŋ]
    return {
        "a": [(a, 1)], "o": [(u, 0.3), (o, 1)], "e": [(at, 1)], "ê": [(e, 1)], "er": [(at, 1)],
        "ai": [(a, 1), (i, 0.5)], "ei": [(e, 1), (i, 0.5)],
        "ao": [(a, 1), (o, 0.5)], "ou": [(o, 1), (u, 0.5)],
        "an": [(a, 1)], "en": [(at, 1)], "ang": [(a, 1)], "eng": [(at, 1)], "ong": [(ong, 1)],
        "i": [(i, 1)],
        "ia": [(i, 0.3), (a, 1)], "ie": [(i, 0.3), (e, 1)],
        "iao": [(i, 0.3), (a, 1), (o, 0.5)], "iou": [(i, 0.3), (o, 1), (u, 0.5)],
        "ian": [(i, 0.3), (e, 1)], "in": [(i, 1)], "iang": [(i, 0.3), (a, 1)],
        "ing": [(i, 1)], "iong": [(i, 0.3), (ong, 1)],
        "u": [(u, 1)], "ua": [(u, 0.3), (a, 1)], "uo": [(u, 0.3), (o, 1)],
        "uai": [(u, 0.3), (a, 1), (i, 0.5)], "uei": [(u, 0.3), (e, 1), (i, 0.5)],
        "uan": [(u, 0.3), (a, 1)], "uen": [(u, 0.3), (at, 1)],
        "uang": [(u, 0.3), (a, 1)], "ueng": [(u, 0.3), (at, 1)],
        "ü": [(yu, 1)], "üe": [(yu, 0.3), (e, 1)], "üan": [(yu, 0.3), (e, 1)], "ün": [(yu, 1)],
    }


FINALS = mandarin_finals()
# The apical vowel of zi ci si zhi chi shi ri: spread a little, barely open.
APICAL = shape(I=0.5, AT=0.5)
APICAL_INITIALS = {"z", "c", "s", "zh", "ch", "sh", "r"}


def with_onset(onset, rhyme):
    """`rhyme`, with `onset` in front of it taking ONSET_SHARE of the unit."""
    total = sum(weight for _, weight in rhyme)
    return [(onset, total * ONSET_SHARE / (1 - ONSET_SHARE))] + rhyme


def spell_chinese(units):
    from pypinyin import Style, pinyin

    finals = pinyin(units, style=Style.FINALS, strict=True)
    initials = pinyin(units, style=Style.INITIALS, strict=True)
    out = []
    for unit, final, initial in zip(units, finals, initials):
        final, initial = final[0].replace("v", "ü"), initial[0]
        rhyme = FINALS.get(final)
        if rhyme is None:
            out.append(None)
            continue
        if final == "i" and initial in APICAL_INITIALS:
            rhyme = [(APICAL, 1)]
        if initial:
            rhyme = with_onset(MBP if initial in {"b", "p", "m"} else CONSONANT, rhyme)
        out.append(rhyme)
    return out


# ---- Japanese: morae, through a kana table

ROWS = {
    "": "あいうえお", "k": "かきくけこ", "g": "がぎぐげご", "s": "さしすせそ", "z": "ざじずぜぞ",
    "t": "たちつてと", "d": "だぢづでど", "n": "なにぬねの", "h": "はひふへほ", "b": "ばびぶべぼ",
    "p": "ぱぴぷぺぽ", "m": "まみむめも", "y": "や\0ゆ\0よ", "r": "らりるれろ", "w": "わゐ\0ゑを",
    "v": "\0\0ゔ\0\0",
}
KANA = {kana: (consonant, "aiueo"[k]) for consonant, row in ROWS.items() for k, kana in enumerate(row)}
KANA["を"] = ("", "o")
SMALL_Y = {"ゃ": "a", "ゅ": "u", "ょ": "o"}
SMALL_V = {"ぁ": "a", "ぃ": "i", "ぅ": "u", "ぇ": "e", "ぉ": "o"}
JA_VOWEL = {"a": shape(A=1), "i": shape(I=1), "u": shape(U=1), "e": shape(E=1), "o": shape(O=1)}


def hiragana(text):
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in text)


def morae(text):
    """The morae of a kana text, as the strings they are written with; anything else dropped."""
    out = []
    for c in text:
        h = hiragana(c)
        if h in SMALL_Y or h in SMALL_V:
            if out:
                out[-1] += c
        elif h in KANA or h in "んっー":
            out.append(c)
    return out


def spell_japanese(units):
    out = []
    parsed = []
    for unit in units:
        h = hiragana(unit)
        if h[0] in KANA:
            consonant, vowel = KANA[h[0]]
            glide = None
            for small in h[1:]:
                if small in SMALL_Y:
                    glide, vowel = "y", SMALL_Y[small]
                elif small in SMALL_V:
                    vowel = SMALL_V[small]
            parsed.append(("mora", consonant, glide, vowel))
        else:
            parsed.append((h[0], None, None, None))
    for k, (kind, consonant, glide, vowel) in enumerate(parsed):
        following = parsed[k + 1] if k + 1 < len(parsed) else None
        before_lips = following is not None and following[0] == "mora" and following[1] in "mbp"
        if kind == "mora":
            rhyme = [(JA_VOWEL[vowel], 1)]
            if glide:
                rhyme = [(shape(I=1), 0.3)] + rhyme
            if consonant in {"m", "b", "p"}:
                rhyme = with_onset(MBP, rhyme)
            elif consonant == "w":
                rhyme = with_onset(shape(U=1), rhyme)
            elif consonant == "y":
                rhyme = with_onset(shape(I=1), rhyme)
            elif consonant:
                rhyme = with_onset(CONSONANT, rhyme)
            out.append(rhyme)
        elif kind == "ー":
            # The vowel before, held.
            previous = next((s for s in reversed(out) if s), None)
            out.append([(previous[-1][0], 1)] if previous else None)
        else:
            # ん and っ: the lips shut for the m, b or p after them, and otherwise follow.
            out.append([(MBP if before_lips else CONSONANT, 1)])
    return out


# ---- English: CMUdict

ARPABET = {
    "IY": [(shape(I=1), 1)], "IH": [(shape(I=0.7, E=0.3), 0.8)], "EH": [(shape(E=1), 1)],
    "AE": [(shape(A=0.7, E=0.3), 1)], "AA": [(shape(A=1), 1)], "AO": [(shape(O=1), 1)],
    "UH": [(shape(U=0.7, O=0.3), 0.8)], "UW": [(shape(U=1), 1)],
    "ER": [(shape(AT=0.6, U=0.4), 1)],
    "EY": [(shape(E=1), 0.6), (shape(I=1), 0.4)], "AY": [(shape(A=1), 0.6), (shape(I=1), 0.4)],
    "OY": [(shape(O=1), 0.6), (shape(I=1), 0.4)], "AW": [(shape(A=1), 0.6), (shape(U=1), 0.4)],
    "OW": [(shape(O=1), 0.6), (shape(U=1), 0.4)],
    "M": [(MBP, 0.5)], "B": [(MBP, 0.5)], "P": [(MBP, 0.5)],
    "W": [(shape(U=1), 0.4)], "Y": [(shape(I=1), 0.4)],
}


def read_cmudict(path):
    words = {}
    for line in open(path, encoding="utf-8"):
        word, *phones = line.split("#")[0].split()
        if "(" not in word and phones:
            words.setdefault(word, phones)
    return words


def spell_english(units, cmudict):
    out = []
    for unit in units:
        phones = cmudict.get(unit.lower())
        if phones is None:
            out.append(None)
            continue
        spelled = []
        for phone in phones:
            base, stress = phone.rstrip("012"), phone[len(phone.rstrip("012")) :]
            if base == "AH":
                # Unstressed it is the schwa; stressed, cup's ʌ.
                spelled.append((shape(AT=1), 0.6) if stress == "0" else (shape(A=0.6, AT=0.4), 1))
            elif base in ARPABET:
                spelled.extend(ARPABET[base])
            else:
                spelled.append((CONSONANT, 0.5))
        out.append(spelled)
    return out


# ---- the timeline


def timeline(spans, spelled, frames):
    """`(frames, 8)`: each 5 ms frame's distribution, consonants filled in from the anchors around."""
    out = np.full((frames, len(CLASSES)), np.nan, dtype=np.float32)
    out[:] = REST
    for (start, end), spelling in zip(spans, spelled):
        if spelling is None or end <= start:
            # A unit the spelling could not read: nothing is known about its frames.
            a, b = int(start / FRAME_SECONDS), int(end / FRAME_SECONDS)
            out[a:b] = np.nan
            continue
        total = sum(weight for _, weight in spelling)
        t = start
        for sound, weight in spelling:
            length = (end - start) * weight / total
            a, b = int(round(t / FRAME_SECONDS)), int(round((t + length) / FRAME_SECONDS))
            out[a:b] = np.nan if sound is None else sound
            t += length
    # Fill every run of unknown frames linearly between the anchors either side of it.
    unknown = np.isnan(out[:, 0])
    k = 0
    while k < frames:
        if not unknown[k]:
            k += 1
            continue
        end = k
        while end < frames and unknown[end]:
            end += 1
        before = out[k - 1] if k > 0 else (out[end] if end < frames else REST)
        after = out[end] if end < frames else before
        for j in range(k, end):
            w = (j - k + 1) / (end - k + 1)
            out[j] = (1 - w) * before + w * after
        k = end
    return out


def mbp_events(frames_mbp, tokens):
    """Token ranges `[first, last]` where the timeline's lips are shut."""
    events, k = [], 0
    while k < len(frames_mbp):
        if frames_mbp[k]:
            end = k
            while end < len(frames_mbp) and frames_mbp[end]:
                end += 1
            first, last = k // FRAMES_PER_TOKEN, min(tokens - 1, (end - 1) // FRAMES_PER_TOKEN)
            events.append([first, last])
            k = end
        else:
            k += 1
    return events


def align(corpus_dir, language, cmudict_path):
    import torch
    from qwen_asr import Qwen3ForcedAligner

    aligner = Qwen3ForcedAligner.from_pretrained(
        "Qwen/Qwen3-ForcedAligner-0.6B", dtype=torch.bfloat16, device_map="cuda:0"
    )
    if language == "Japanese":
        # Morae rather than nagisa's words: a time for every mora, as Chinese has one for every
        # character.
        processor = aligner.aligner_processor
        processor.tokenize_japanese = lambda text: morae(processor.clean_token(text))
    cmudict = read_cmudict(cmudict_path) if language == "English" else None

    readings = [json.loads(line) for line in open(corpus_dir / "corpus.jsonl")]
    with open(corpus_dir / "labels.jsonl", "w") as out:
        for batch in range(0, len(readings), 16):
            chunk = readings[batch : batch + 16]
            results = aligner.align(
                audio=[str(corpus_dir / r["wav"]) for r in chunk],
                text=[r["text"] for r in chunk],
                language=[language] * len(chunk),
            )
            for reading, result in zip(chunk, results):
                items = list(result)
                units = [item.text for item in items]
                spans = [(item.start_time, item.end_time) for item in items]
                if language == "Chinese":
                    spelled = spell_chinese(units)
                elif language == "Japanese":
                    spelled = spell_japanese(units)
                else:
                    spelled = spell_english(units, cmudict)
                count = len(reading["tokens"])
                frames = count * FRAMES_PER_TOKEN
                line = timeline(spans, spelled, frames)
                labels = line.reshape(count, FRAMES_PER_TOKEN, len(CLASSES)).mean(axis=1)
                reading["language"] = language
                reading["labels"] = np.round(labels, 4).tolist()
                reading["mbp"] = mbp_events(line[:, INDEX["MBP"]] > 0.99, count)
                reading["unread"] = sum(s is None for s in spelled)
                out.write(json.dumps(reading, ensure_ascii=False) + "\n")
            print(f"{language}: aligned {batch + len(chunk)} of {len(readings)}", flush=True)


# ---- training


def embedding(package):
    """The flow's `input_embedding`, `(6561, 80)`."""
    for path in sorted(Path(package).glob("*.safetensors")):
        with open(path, "rb") as f:
            size = struct.unpack("<Q", f.read(8))[0]
            entry = json.loads(f.read(size)).get("cosyvoice3.flow.input_embedding.weight")
            if entry is None:
                continue
            start, end = entry["data_offsets"]
            f.seek(8 + size + start)
            return np.frombuffer(f.read(end - start), dtype=np.float32).reshape(entry["shape"])
    raise SystemExit(f"no cosyvoice3.flow.input_embedding.weight under {package}")


def windowed(rows, width):
    padded = np.pad(rows, ((width, width), (0, 0)))
    return np.concatenate([padded[k : k + len(rows)] for k in range(2 * width + 1)], axis=1)


HIDDEN = 256


class Probe:
    """`softmax(W2 gelu(W1 x + b1) + b2)` over a window of embeddings, fitted to soft labels.

    One hidden layer of 256: against a linear softmax it lowers the held-out cross entropy from
    1.31 to 1.23 with a token either side, and E, O and @ gain the most.
    """

    def __init__(self, table, width):
        self.table, self.width = table, width

    def features(self, reading):
        return windowed(self.table[reading["tokens"]], self.width)

    def fit(self, readings, epochs=60):
        import torch

        torch.manual_seed(0)
        x = np.concatenate([self.features(r) for r in readings])
        y = np.concatenate([np.array(r["labels"], dtype=np.float32) for r in readings])
        self.mean, self.std = x.mean(0), x.std(0) + 1e-6
        xt = torch.tensor((x - self.mean) / self.std, dtype=torch.float32)
        yt = torch.tensor(y)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(x.shape[1], HIDDEN),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(HIDDEN, len(CLASSES)),
        )
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=2e-3, weight_decay=1e-2)
        for _ in range(epochs):
            self.net.train()
            for idx in torch.randperm(len(xt)).split(1024):
                optimizer.zero_grad()
                logp = torch.log_softmax(self.net(xt[idx]), dim=1)
                (-(yt[idx] * logp).sum(1).mean()).backward()
                optimizer.step()
        self.net.eval()
        return self

    def folded(self):
        """`(w1, b1, w2, b2)` with the input's normalization folded into the first layer."""
        w1 = self.net[0].weight.detach().numpy() / self.std[None, :]
        b1 = self.net[0].bias.detach().numpy() - w1 @ self.mean
        w2, b2 = self.net[3].weight.detach().numpy(), self.net[3].bias.detach().numpy()
        return w1, b1, w2, b2

    def predict_rows(self, x):
        import torch

        with torch.no_grad():
            z = torch.tensor((x - self.mean) / self.std, dtype=torch.float32)
            return torch.softmax(self.net(z), dim=1).numpy()

    def predict(self, reading):
        return self.predict_rows(self.features(reading))


def scores(readings, predict):
    """What a set of predictions gets right, against the labels."""
    p = np.concatenate([predict(r) for r in readings])
    y = np.concatenate([np.array(r["labels"]) for r in readings])
    ce = float(-(y * np.log(np.maximum(p, 1e-9))).sum(1).mean())
    sure = y.max(1) >= 0.8
    top = p.argmax(1) == y.argmax(1)
    per_class = {}
    for k, c in enumerate(CLASSES):
        mask = sure & (y.argmax(1) == k)
        per_class[c] = (float(top[mask].mean()) if mask.any() else float("nan"), int(mask.sum()))
    # Shut lips: did any token of each shut stretch, or the one either side, say MBP most? And how
    # often does the middle of a vowel say it?
    hits = events = 0
    for r in readings:
        guess = predict(r).argmax(1)
        for first, last in r["mbp"]:
            events += 1
            hits += bool((guess[max(0, first - 1) : last + 2] == INDEX["MBP"]).any())
    vowel = sure & (y.argmax(1) < INDEX["MBP"])
    closed = float((p.argmax(1)[vowel] == INDEX["MBP"]).mean())
    return {
        "cross_entropy": ce,
        "top1_sure": float(top[sure].mean()),
        "sure_share": float(sure.mean()),
        "per_class": per_class,
        "mbp_recall": hits / max(1, events),
        "mbp_events": events,
        "vowel_closed": closed,
    }


def table_predictor(table):
    return lambda r: table[r["tokens"]]


def sightings(readings, tokens):
    counts = np.zeros(tokens, dtype=np.float32)
    for r in readings:
        np.add.at(counts, r["tokens"], 1)
    return counts


def show(name, s):
    classes = " ".join(f"{c} {acc:.2f}" for c, (acc, n) in s["per_class"].items())
    print(
        f"  {name:28} CE {s['cross_entropy']:.3f}  top-1 {s['top1_sure']:.3f}  | {classes} | "
        f"MBP caught {s['mbp_recall']:.2f} of {s['mbp_events']}, vowels shut {s['vowel_closed']:.3f}"
    )


def train(corpus_dirs, package, output, hold_out):
    readings = []
    for d in corpus_dirs:
        readings += [json.loads(line) for line in open(Path(d) / "labels.jsonl")]
    table = embedding(package)
    groups = sorted({r["voice"] // 2 for r in readings})
    languages = sorted({r["language"] for r in readings})
    tokens = sum(len(r["tokens"]) for r in readings)
    print(f"{len(readings)} readings, {tokens} tokens, {len(groups)} voice groups, {languages}")
    print(f"distinct tokens: {len({t for r in readings for t in r['tokens']})} of {len(table)}")
    if hold_out:
        evaluate(readings, table, groups, languages)

    # Everything, for what is written. A token's own mean label is not mixed in where it was seen:
    # held out, that made the table worse (cross entropy 1.405 to 1.427), the model having already
    # learnt what the sightings say and the sightings bringing back their aligner's noise.
    alone = Probe(table, 0).fit(readings)
    context = Probe(table, 1).fit(readings)
    counts = sightings(readings, len(table))
    write(output, alone.predict_rows(table), table, context)
    print(f"\nwrote {output}: {len(table)} tokens, {int((counts > 0).sum())} seen at least once")


def evaluate(readings, table, groups, languages):
    print("top-1 is on tokens whose label is at least 0.8 one class ('sure');")
    print("each voice group (a recording and its pitch-shifted copy) held out in turn:")
    held_out = {}
    for held in groups:
        train_set = [r for r in readings if r["voice"] // 2 != held]
        test_set = [r for r in readings if r["voice"] // 2 == held]
        alone = Probe(table, 0).fit(train_set)
        context = Probe(table, 1).fit(train_set)
        held_out.setdefault("table", []).append(
            (test_set, table_predictor(alone.predict_rows(table)))
        )
        held_out.setdefault("context +-1", []).append((test_set, context.predict))

    for language in languages + [None]:
        print(f"\n== {language or 'all languages'}")
        for name, folds in held_out.items():
            subset = [
                (
                    [r for r in test if language is None or r["language"] == language],
                    predict,
                )
                for test, predict in folds
            ]
            merged = [r for test, _ in subset for r in test]
            predictors = {}
            for test, predict in subset:
                for r in test:
                    predictors[id(r)] = predict
            show(name, scores(merged, lambda r: predictors[id(r)](r)))


STEM = "cosyvoice3-visemes"
MODEL_TYPE = "cosyvoice3_visemes"

MANIFEST = f"""# A viseme model for CosyVoice3: attached to a pipeline that has CosyVoice3 loaded, with
# waifu_pipeline_attach_model(pipeline, WAIFU_ATTACHED_MODEL_TEXT_TO_SPEECH_VISEMES, NULL, model).
# Written by tools/cosyvoice3_visemes.py; docs/cosyvoice3.md#visemes says what it is.

weights:
  - {STEM}.safetensors

config:
  model:
    type: {MODEL_TYPE}
  {MODEL_TYPE}:
    speech_token_size: 6561
    visemes: [A, E, I, O, U, "@", MBP, REST]
"""


def write(output, token_table, embedding_table, context):
    """A package of its own in the directory `output`: the weights and their manifest."""
    import torch
    from safetensors.torch import save_file

    output.mkdir(parents=True, exist_ok=True)
    w1, b1, w2, b2 = context.folded()
    tensors = {
        # The flow's input_embedding, a copy: the package reads nothing of CosyVoice3's.
        f"{MODEL_TYPE}.embedding": embedding_table,
        # softmax(w2 @ gelu(w1 @ concat(E[t-1], E[t], E[t+1]) + b1) + b2), E the embedding, zeros
        # past either end of a sentence, and gelu the exact (erf) one.
        f"{MODEL_TYPE}.w1": w1,
        f"{MODEL_TYPE}.b1": b1,
        f"{MODEL_TYPE}.w2": w2,
        f"{MODEL_TYPE}.b2": b2,
        # Each token's distribution alone, for a caller with no neighbours to hand.
        f"{MODEL_TYPE}.table": token_table,
    }
    save_file(
        {k: torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32)) for k, v in tensors.items()},
        str(output / f"{STEM}.safetensors"),
        metadata={"classes": " ".join(CLASSES), "token_seconds": str(TOKEN_SECONDS)},
    )
    (output / f"{STEM}.yaml").write_text(MANIFEST)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("align")
    a.add_argument("corpus_dir", type=Path)
    a.add_argument("-language", required=True, choices=["Chinese", "Japanese", "English"])
    a.add_argument("-cmudict", type=Path)
    t = sub.add_parser("train")
    t.add_argument("corpus_dirs", type=Path, nargs="+")
    t.add_argument("-package", type=Path, required=True)
    t.add_argument("-output", type=Path, required=True)
    t.add_argument("-no-hold-out", action="store_true", help="skip the held-out scores")
    args = parser.parse_args()
    if args.command == "align":
        align(args.corpus_dir, args.language, args.cmudict)
    else:
        train(args.corpus_dirs, args.package, args.output, not args.no_hold_out)


if __name__ == "__main__":
    main()
