# The waifu2d package: `.waifu2d`

A drawing in layers as waifu2d keeps it: what img2layers writes and what waifu2d rigs. A zip of
PNGs and an index. "Package" here means this file. It has nothing to do with libwaifu's model
packages (safetensors and a manifest).

It carries everything a Photoshop document from img2layers has. It also carries what a document
cannot say:

- each layer's median depth;
- which layers are the near and far halves of one part;
- where the picture is on the canvas;
- the picture itself;
- what made it.

Any zip tool opens it, and any picture viewer opens what is inside.

The code is `waifu2d::package` (`waifu2d/src/package.rs`). The tools are `img2layers`,
`see_through_stratify` and `waifu2d_convert` in `waifu-tools`.

## The archive

```
girl.waifu2d
├── mimetype         application/vnd.libwaifu.waifu2d+zip
├── index.yaml
├── preview.png      the shown layers laid over each other
├── source.png       the picture they were taken from (optional)
├── layers/000.png   one per layer
└── depth/000.png    one per layer that has a depth map
```

- **`mimetype` comes first**, as EPUB's does:
  - it is stored, not deflated;
  - it has no extra field;
  - it holds exactly `application/vnd.libwaifu.waifu2d+zip`, ASCII, with no newline.

  The type therefore sits at byte 38 of every package, and `PK\3\4` then `mimetype` at byte 30
  says what the file is without unzipping it (`Package::is_package`). A reader refuses an archive
  whose first entry is not this one. `+zip` is the RFC 6839 suffix: it tells a tool that does not
  know the type that the file is a zip.
- **Entries are stored** by waifu2d: PNG is deflated already. A reader also takes deflated
  entries, which any zip tool writes when somebody unpacks a package, edits a layer and zips it
  again. Every entry's CRC is checked.
- **Entry names are numbers, never layer names.** A name can be `前髪２`, can hold spaces or a
  `/`, and zip tools disagree about a name's encoding. The index maps a layer to its files.
  - Writers number from `000`, bottom first; past 999 the number just grows wider.
  - A reader goes by what the index says, never by the numbering.
- **waifu2d writes the same bytes for the same package.** Every entry is dated 1980-01-01, and
  the PNG encoder is deterministic.

## index.yaml

```yaml
format: "waifu2d"
version: 1
canvas: [1280, 1280]
picture: [0, 140, 1280, 1000]
preview: "preview.png"
source: "source.png"
generator:
  "model": "see-through"
  "name": "img2layers"
  "seed": "7"
  "split_pairs": "true"
layers:
  - name: "back hair"
    image: "layers/000.png"
    at: [312, 80]
    depth: 0.83
    depth_map: "depth/000.png"
  - name: "topwear-1"
    image: "layers/007.png"
    at: [402, 610]
    kind: far
    of: "topwear"
    depth: 0.41
    depth_map: "depth/007.png"
  - name: "前髪"
    image: "layers/012.png"
    at: [300, 60]
    opacity: 0.5
    visible: false
```

| key | | what |
|---|---|---|
| `format` | required | `waifu2d` |
| `version` | required | `1`. A reader refuses a version newer than its own |
| `canvas` | required | `[width, height]`, each 1 to 16384 |
| `picture` | | `[x, y, width, height]`: where the source picture is on the canvas |
| `preview` | | the preview's entry. Readers need not read it: it is the layers again |
| `source` | | the source picture's entry |
| `generator` | | what made it, one value per key, as its maker says |
| `layers` | required | bottom first: the order they are drawn in |

Each layer:

| key | | what |
|---|---|---|
| `name` | required | what the layer is: `face`, `back hair`, `topwear-0`. waifu2d rigs by it |
| `image` | required | its picture's entry |
| `at` | required | `[left, top]`: where its top left corner is on the canvas. Either may be negative |
| `opacity` | 1 | 0 to 1 |
| `visible` | true | `true` or `false` |
| `kind` | `part` | `part`, as the model drew it; `near` or `far`, a half a part was cut into by its depth (docs/see_through.md, "Stratification") |
| `of` | | a half's part: `topwear` for `topwear-0` |
| `depth` | | the median depth of what it draws: 0 nearest, 1 farthest |
| `depth_map` | | its depth map's entry |

### The rules for the YAML

The index stays within what `waifu2d/src/yaml.rs` reads, which is a copy of `waifu/src/yaml.rs`
with list items that are mappings added. That means:

- block mappings, block sequences, `[a, b]` flow sequences, and quoted or plain scalars, indented
  with spaces;
- no flow mappings (`{x: 1}`), anchors, tags or block scalars.

The rules:

- **A writer quotes every string**, in double quotes, escaping `"` and `\`. Unquoted, YAML reads
  `no`, `on` or `1e3` as something else, and a name with ` #` in it as a comment. Numbers and
  `true`/`false` are written plain.
- **A reader ignores a key it does not know**, at any level. A writer adds keys without changing
  `version`. `version` goes up only for a change that a reader of the last version would get
  wrong.
- **Sizes are not in the index.** A PNG's header has them, and two places would one day disagree.

## The pictures

- **A layer** is an 8-bit RGBA PNG, not premultiplied, in sRGB, cut to what it draws. A reader
  also takes gray, gray with alpha, RGB, a palette and 16 bits, all made 8-bit RGBA.
- **A depth map** is an 8-bit gray PNG the size of its layer: 0 nearest, 255 farthest.
  - It has no alpha: what is drawn is the layer's alpha.
  - Its median under that alpha is the layer's `depth`, and a writer writes both.
  - A reader also takes 16 bits, keeping the high byte, and colour, taking red, which is how a
    depth from a Photoshop document's group arrives.
- **The preview** is the shown layers laid over each other at their opacity, on the canvas.
- **The source** is the picture as it was handed to img2layers, at its own size.

## Reading one safely

A package comes from anywhere, so a reader:

- refuses an entry that unpacks to more than a 16384 square of RGBA (1 GiB), or to more than
  its directory says;
- refuses a picture larger than 16384 on a side;
- refuses an entry whose CRC is wrong.

waifu2d never writes an entry to disk. A reader that unpacks one must refuse a name holding `..`
or one that is absolute.

## The tools

```bash
cargo run --release -p waifu-tools --bin img2layers -- models/see-through.yaml girl.png girl.waifu2d
cargo run --release -p waifu-tools --bin see_through_stratify -- girl.waifu2d girl-stratified.waifu2d
cargo run --release -p waifu-tools --bin waifu2d_convert -- old.psd old.waifu2d
```

- Each writes what its output's name says, `.waifu2d` or `.psd`, and refuses any other name.
- A tool that reads a drawing reads either format, told apart by how the file starts rather than
  by its name.
- `waifu2d_convert` turns a document's hidden `depth` group into depth maps and medians, and its
  `<part>-0`/`<part>-1` into marked halves.
- The other way round, it says which layers a document from here cannot keep as they were: a
  hidden layer, or one under full opacity.

In Rust, `Rig::read_package` and `Rig::from_package` rig a package, and `Rig::read` takes either
format. A package and a Photoshop document of the same layers make the same rig
(`waifu2d/tests/package.rs`).
