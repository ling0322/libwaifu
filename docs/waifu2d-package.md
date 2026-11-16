# The waifu2d package: `.waifu2d`

A character as waifu2d keeps it: a zip of PNGs and one index, `index.yaml`, which says
everything about the character. Today that is its name and its appearance: the drawing in layers
that img2layers writes and waifu2d rigs. What else a character is will be sections of the same
index beside `appearance`.

"Package" here means this file. It has nothing to do with libwaifu's model packages
(safetensors and a manifest).

The appearance carries everything a Photoshop document from img2layers has. It also carries what
a document cannot say:

- each layer's median depth;
- which layers are the near and far halves of one part, and the part they were cut from;
- the picture the layers were taken from;
- what made it.

Any zip tool opens a package, and any picture viewer opens what is inside.

The code is `waifu2d::package` (`waifu2d/src/package.rs`). The tools are `img2layers`,
`see_through_stratify` and `waifu2d_convert` in `waifu-tools`.

## The archive

```
miku.waifu2d
├── mimetype         application/vnd.libwaifu.waifu2d+zip
├── index.yaml       everything about the character
├── preview.png      the shown layers laid over each other
├── source.png       the picture the layers were taken from (optional)
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
- **Paths in the index are from the package's root**, and never hold `..`.
- **waifu2d writes the same bytes for the same package.** Every entry is dated 1980-01-01, and
  the PNG encoder is deterministic.

## index.yaml

A stratified package, `topwear` cut in two:

```yaml
format: "waifu2d"
version: 1
name: "初音ミク"
appearance:
  version: 1
  canvas: [1280, 1280]
  preview: "preview.png"
  source: "source.png"
  generator:
    "model": "see-through"
    "name": "img2layers"
    "seed": "7"
    "split_pairs": "true"
    "stratified_by": "see_through_stratify"
  layers:
    - name: "back hair"
      image: "layers/000.png"
      at: [551, 17]
      depth: 0.83
      depth_map: "depth/000.png"
    - name: "topwear-1"
      image: "layers/001.png"
      at: [402, 610]
      kind: stratified_far
      of: "topwear"
      depth: 0.47
      depth_map: "depth/001.png"
    - name: "face"
      image: "layers/002.png"
      at: [500, 200]
      depth: 0.40
      depth_map: "depth/002.png"
    - name: "topwear"
      image: "layers/003.png"
      at: [402, 610]
      visible: false
      depth: 0.41
      depth_map: "depth/003.png"
    - name: "topwear-0"
      image: "layers/004.png"
      at: [430, 640]
      kind: stratified_near
      of: "topwear"
      depth: 0.41
      depth_map: "depth/004.png"
```

The top level:

| key | | what |
|---|---|---|
| `format` | required | `waifu2d` |
| `version` | required | `1`: the version of this top level's layout. A reader refuses a newer one |
| `name` | | the character's name, to show |
| `appearance` | required | what the character looks like |

`appearance`:

| key | | what |
|---|---|---|
| `version` | required | `1`. A reader refuses a newer one, and says it is the appearance's |
| `canvas` | required | `[width, height]`, each 1 to 16384 |
| `preview` | | the preview's entry. Readers need not read it: it is the layers again |
| `source` | | the entry of the picture the layers were taken from |
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
| `kind` | `part` | `part`, `stratified_near` or `stratified_far`: the C API's `WaifuLayerType`, less its depth, which a layer here carries |
| `of` | | a half's part: `topwear` for `topwear-0` |
| `depth` | | the median depth of what it draws: 0 nearest, 1 farthest |
| `depth_map` | | its depth map's entry |

### Stratified layers

A part cut into a near and a far half by its depth (docs/see_through.md, "Stratification")
keeps all three:

- **The halves** are `stratified_near` and `stratified_far`, shown. Each names its part in `of`.
- **The part** stays a `part`, whole and hidden, right under its near half, which is its own
  place in the stack. That a half names it is what makes it a split part, as `derived_from_id`
  does in the C API.

Drawing every shown layer draws the stratified character, which is what waifu2d does. Showing the
split parts and hiding the halves draws it as it was before, in its old order, without moving
anything. A package img2layers wrote has only parts.

### The rules for the YAML

The index stays within what `waifu2d/src/yaml.rs` reads, which is a copy of `waifu/src/yaml.rs`
with list items that are mappings added. That means:

- block mappings, block sequences, `[a, b]` flow sequences, and quoted or plain scalars, indented
  with spaces;
- no flow mappings (`{x: 1}`), anchors, tags or block scalars.

The rules:

- **A writer quotes every string**, in double quotes, escaping `"` and `\`. Unquoted, YAML reads
  `no`, `on` or `1e3` as something else, and a name with ` #` in it as a comment. Numbers, `kind`
  and `true`/`false` are written plain.
- **A reader ignores a key or a section it does not know**, at any level. A writer adds keys
  without changing a `version`.
- **Each section has its own `version`**, which goes up only for a change that a reader of the
  last one would get wrong. A newer section that a reader does not need stops nothing: a reader
  that only draws reads `appearance` and nothing else.
- **Sizes are not in the index.** A PNG's header has them, and two places would one day disagree.
  Neither is where the picture sits on the canvas: img2layers centres it, scaled to fit, so its
  size and the canvas's say where.

## The pictures

- **A layer** is an 8-bit RGBA PNG, not premultiplied, in sRGB, cut to what it draws. A reader
  also takes gray, gray with alpha, RGB, a palette and 16 bits, all made 8-bit RGBA.
- **A depth map** is an 8-bit gray PNG the size of its layer: 0 nearest, 255 farthest.
  - It has no alpha: what is drawn is the layer's alpha.
  - Its median under that alpha is the layer's `depth`, and a writer writes both.
  - A reader also takes 16 bits, keeping the high byte, and colour, taking red, which is how a
    depth from a Photoshop document's group arrives.
- **The preview** is the shown layers laid over each other at their opacity, on the canvas.
  - It is at most 2048 on its longer side. A larger canvas's preview is scaled down to that, each
    pixel the canvas pixel under its middle.
  - It is a thumbnail, so writing one never needs memory the size of a 16384 canvas.
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
- `see_through_stratify` keeps the package's name, picture and generator, and each part it
  splits, hidden.

### To and from a Photoshop document

A document becomes a package losing nothing:

- its hidden `depth` group becomes depth maps and medians;
- its `<part>-0`/`<part>-1` become marked halves.

A package becomes a document losing what a document from here has no place for:

| | in the document | back as a package |
|---|---|---|
| shown layers, their pixels and places | kept | kept |
| depth maps | the hidden group `depth` | kept |
| halves (`kind`, `of`) | only their names, `-0`/`-1` | read again from the names |
| split parts | left out: the document is the stratified drawing | lost |
| other hidden layers, opacity | left out, or drawn at full opacity; the tool says which | lost |
| `name`, `generator`, `source` | nowhere | lost |
| `depth` | not kept | worked out again from the 8-bit depth map |

In Rust, `Rig::read_package` and `Rig::from_package` rig a package, and `Rig::read` takes either
format. A package and a Photoshop document of the same layers make the same rig
(`waifu2d/tests/package.rs`).
