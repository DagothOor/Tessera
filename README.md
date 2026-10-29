# Texelith

Texture maps from finished textures — normal, height, parallax (`_nh`), ambient
occlusion, specular/roughness and foliage translucency — built for old games
whose only asset is a painted diffuse. Made for Morrowind/OpenMW modding, but
nothing in it is tied to one engine.

Up to build 22 the program was called Tessera. Settings, downloaded models and `.tessera` projects are picked up automatically.

Русское описание — в [README.ru.md](README.ru.md).

## Why another map generator

Most tools that make maps from a picture read brightness as height. That holds
only for a single-coloured surface with no cast shadows — never the case for a
texture from 2002 with light painted in by hand. Texelith treats the source as
what it is: a photograph-like image with paint, baked light, compression damage
and seams.

What that means in practice:

- height is mixed from six layers, brightness being only one of them; another
  one integrates the painted lighting (shape from shading) after removing what
  colour explains as paint and what looks like cast shadow;
- depth and normals can come from neural networks (MiDaS, Depth Anything V2,
  ESRGAN) and be mixed with the rest;
- everything is seamless by construction — the maths runs on a torus, so a
  tiling texture stays tiling;
- damaged source areas (blown or crushed by compression) are repaired first;
- existing normal maps can be analysed, repaired and turned into height;
- foliage cards get a normal per leaf from the alpha silhouette;
- parallax for OpenMW is set in real centimetres;
- the main window is a node editor (like Substance Designer or Blender's
  nodes) with a thumbnail on every node; projects are saved, new ones start from
  templates, and a batch runs a project over a whole folder tree — also from the
  command line — skipping textures already done;
- a normal and a height from different sources are balanced into one
  consistent pair instead of one being thrown away.

## Features

- **Material tab** — damaged-area repair, removing painted light, height from six
  layers (band, multi-scale, brightness, edge-preserving, shape from shading,
  external or neural depth), normal (from height, from the lighting as R and G,
  from an ESRGAN network, or matched to parallax depth), AO, bent normal,
  specular/roughness, relief blended into the diffuse (overlay, soft light,
  multiply, linear burn), second material by mask, tiling repair.
- **Parallax for OpenMW** — packs height into the normal map's alpha around
  alpha 128 so the relief comes out at the depth you set in centimetres, given
  how many units one texture repeat covers. The 3D preview reproduces OpenMW's
  parallax formula exactly.
- **Foliage tabs** — per-leaf normals from the alpha silhouette, veins or tufts;
  light removal, crown occlusion, translucency; full-size saving with coverage-
  preserving mipmaps.
- **Normal map repair** — analysis, renormalisation, Z reconstruction, green
  flip, normal → height (Frankot–Chellappa) and height → normal (16-bit input).
- **Node workspace** — the main window: graph with thumbnails, 2D and 3D
  views, inspector with readouts, undo, projects (`.texelith`) and templates,
  batch over a folder tree (`python project.py project.texelith in out`).
- **Balance normal ↔ height** — one height closest to both the normal's slopes
  and a given height (detail from the normal, large form from the height), and
  the normal exactly from it.
- **DeepBump** normal network; **SuperMat** PBR network (base colour,
  roughness, metalness) run from its own environment, see
  [SUPERMAT_INSTALL.md](SUPERMAT_INSTALL.md).
- **Materials** — 34 real-world materials (f0 from the refractive index,
  typical roughness ranges), picked from a palette of preview spheres and laid
  along a shininess scale over the histogram of the roughness map; any number
  of materials, the dullest/shiniest pixels can be cut off before fitting, a
  pipette shows where a pixel of the texture falls on the scale. Materials can
  also be read from the texture name and kept the same across PBR nodes.
- **PBR outputs** — **PBR → OpenMW** (diffuse and `_spec` for plain OpenMW), a
  second set for the OpenMW-PBR shader (`_spec` as metal/roughness/AO/1−SSS),
  **PBR (any engine)** (base colour, roughness, metalness, AO, ORM, smoothness,
  spec-gloss), an **SSS** map from material physics, **Output by channels**
  (pick what goes into R, G, B and optional A).
- **De-lighting tools** — remove light, remove shading with automatic
  strength and no light halos, screen-space shadows of the relief, match a
  generated albedo back to the source.
- **Filters and adjustments** as in Photoshop/GIMP, all seamless, plus
  anti-aliasing with a choice of algorithm.
- **3D preview** — tilted plane with parallax (as in OpenMW, occlusion mapping,
  offset), mipmapped sampling, flat or hemispherical ambient.
- Every preview shows what will be saved: preview copies are computed at half
  size with parameters rescaled so that the numbers match the full-size file.

## Running

Python 3.10 or newer. On Windows double-click `Texelith.bat`; elsewhere run
`python Texelith.py`. Missing packages (numpy, scipy, scikit-image, Pillow) are
offered for installation on the first start. Keep all `.py` files together.

The neural features install `onnxruntime` (and, for ESRGAN, `onnx`) on first
use. Models are downloaded on demand into `~/.texelith/models`; a graphics card
is used when onnxruntime finds one.

The node graph itself can run on an NVIDIA graphics card through CuPy
(optional: `pip install cupy-cuda13x[ctk]`); without it everything runs on the
processor with the same results. `GPU-test.bat` checks that both give the same
maps. DDS files are compressed with quicktex when it is installed
(`pip install quicktex`; otherwise Pillow's encoder), using all processor cores.

## Separate batch tools

Small programs for textures that are already made, each with a window and a
command line. They use Texelith's modules, so keep them in their places in the
folder.

- **Normal Flip** (`Normal-Flip.bat`, `normal_flip.py`) — turns `_n`/`_nh`
  normal maps made for OpenGL into the DirectX layout OpenMW reads. Each map is
  checked and only OpenGL ones are flipped; DDS files are changed right in the
  compressed blocks, so mips and the height in the alpha stay as they were.
- **AO Strength** (`AO-Strength/`) — changes how strongly ambient occlusion
  darkens ready diffuse textures; the occlusion is read from the `_orm` map.
- **Albedo Fix** (`Albedo-Fix/`) — brings back the large-scale light and dark
  variation and the fine detail that unshading and network albedo lose, by
  comparing with the source texture, and writes a new diffuse and `_fixed`.

All of them write into a separate output folder and never change the source.

## Neural models and their licences

The models are **not** part of Texelith and are downloaded separately under
their own licences:

| Model | Used for | Licence |
|---|---|---|
| MiDaS v2.1 Small (Intel ISL) | depth | MIT |
| Depth Anything V2 Small (Yang et al.) | depth | Apache-2.0 |
| 1x NormalMapGenerator-CX-Lite (Joey) | normal map | CC-BY-NC-SA-4.0 — non-commercial |
| 1x FrankenMapGenerator-CX-Lite (Joey) | height, roughness | CC-BY-NC-SA-4.0 — non-commercial |
| DeepBump (Hugo Tini) | normal map | GPL-3.0 |
| SuperMat (Hong et al.) — installed by the user with `install_supermat_portable.bat` | base colour, roughness, metalness | MIT; its base model Stable Diffusion 2.1 — CreativeML Open RAIL++-M |

See [NOTICE](NOTICE) for attributions.

## Documentation

- [docs/NODES.md](docs/NODES.md) — every node, its ports and parameters
  (generated from the program by `docs/make_node_docs.py`).
- [docs/OPENMW_PACKAGING.md](docs/OPENMW_PACKAGING.md) — how to pack the maps
  into a mod for OpenMW and for the OpenMW-PBR shader.
- `guide.html` — the built-in guide (Help menu).
- [ARCHITECTURE.md](ARCHITECTURE.md) — how the code is organised.

## Written with an AI assistant

The code of Texelith was written with an AI assistant (Claude, by Anthropic).
The problem, the choice and combination of methods, the requirements and the
checking of results against real textures are the project author's. Commits
made with the assistant carry an `Assisted-by:` tag.

## Licence

Code: MIT, see [LICENSE](LICENSE). Forks may re-license their changes as they
see fit. Models: see the table above.

## For developers

Read [ARCHITECTURE.md](ARCHITECTURE.md) and [CONTRIBUTING.md](CONTRIBUTING.md).
`_check.py` is the regression suite.
