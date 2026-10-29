# Architecture

Texelith is a set of plain Python modules that sit side by side and are run with
a stock Python — no package, no build step. Everything numeric is numpy/scipy;
the UI is Tkinter/ttk.

## Modules

| File | Role |
|---|---|
| `Texelith.py` | The window: shared widgets, the classic tabs (behind View → Classic tabs), their batch. One `App` class. |
| `editor.py` | The node workspace — the main window since build 22: graph canvas with thumbnails, inspector, 2D/3D views, undo, projects, batch dialog, menu and shortcuts. |
| `project.py` | Project files (`.texelith`), templates, batch over a folder tree; also a command-line entry point. |
| `texio.py` | Reading and writing textures (DDS with mips, PNG 16-bit…), torus resize. Used by the window and the headless batch. |
| `deepbump.py` | DeepBump normal network through onnxruntime, windows laid out on the torus. |
| `supermat_bridge.py`, `supermat_worker.py` | SuperMat run as a separate process in its own environment (portable Python with PyTorch). |
| `matgen.py` | Opaque materials: de-lighting, six height layers and their mix, normal (from height, from the lighting, matched to parallax depth), AO, bent normal, spec/roughness, seamless tiling, shape from shading, OpenMW parallax alpha, relief into diffuse. `MaterialJob` caches the stages. |
| `foliage.py` | Foliage cards: leaf segmentation from alpha, per-leaf normals, crown depth, AO, translucency, DDS writer with coverage-preserving mips. |
| `repair.py` | Finding blown/crushed areas and repairing them (tone, smooth, patch). |
| `normalfix.py` | Analysing and repairing finished normal maps. |
| `normal_flip.py` | Separate tool (`Normal-Flip.bat`): flips the green of ready `_n`/`_nh` maps from OpenGL to DirectX for a whole folder, inside the DDS blocks. |
| `AO-Strength/ao_strength.py` | Separate tool (`AO-Strength/AO-Strength.bat`): changes the strength of the AO baked into ready diffuse textures, a folder at a time, taking AO from the R channel of the matching `_orm`/`_arm` map. |
| `Albedo-Fix/albedo_fix.py` | Separate tool (`Albedo-Fix/Albedo-Fix.bat`): finishes ready batch results without running the networks again — brings back the large-scale lightness spread and the fine texture SuperMat did not see, masks the source's black defects and bakes AO into the vanilla diffuse again. |
| `depthnet.py` | Monocular depth networks (MiDaS, Depth Anything V2) through onnxruntime; model download; seamless windowed inference; band shaping. |
| `esrgan.py` | ESRGAN NormalMapGenerator / FrankenMapGenerator: reads PyTorch `.pth` without PyTorch, builds an ONNX graph, seamless tiled inference. |
| `nodes.py` | Node graph: registry of node types, typed ports, cycle check, cached evaluation, JSON. |
| `imgfilters.py` | Filters and adjustments as in Photoshop / GIMP (blurs, unsharp mask, noise, max/min, emboss, edges, levels, curves, hue/saturation…), all on the torus; OKLab and the albedo-to-source match. |
| `materials.py` | Real-world materials (f0 from the refractive index, typical roughness ranges, file-name words) and laying roughness/metalness maps into the range from the dullest to the shiniest material. |
| `view3d.py` | Software 3D preview of a tilted plane with parallax (exact OpenMW mode, occlusion mapping, offset), mipmapped bilinear sampling. |
| `i18n.py` | Every UI string in Russian and English. |
| `guide.html` | The built-in guide, both languages. |
| `_check.py` | Regression suite (needs an X display; `DISPLAY=:99` under Xvfb). |
| `docs/make_node_docs.py` | Writes the node reference `docs/NODES.md` / `NODES.ru.md` from the registry and `i18n.py`. Re-run it after adding or changing a node. |

## Pipeline of the material tab

```
source RGB (sRGB 0…1)
  │ repair.repaired: auto/hand damage mask → tone | smooth | patch fill
  ▼
MaterialJob
  ├─ delit(): remove smooth light gradient and hard shadows → diffuse
  ├─ height():  six layers → mix_heights (weights, tone, tilt limit, hints)
  │              band · multi-scale · brightness · edge-preserving
  │              shape from shading (sfs_signal → sfs_layer)
  │              external depth (file / painted / neural) · ESRGAN franken B
  ├─ normal:    normal_from_height
  │              or matched to parallax depth (plx_normal)
  │              + ESRGAN normal (esr_nmix) + lighting normal R/G (n_direct_mix)
  │              [+ make_integrable fit] + bent normal
  ├─ ao():      cavity / horizon AO + AO from the normal
  ├─ material_maps: spec, shininess, roughness, metalness (+ ESRGAN roughness)
  ├─ relief_into_diffuse (overlay / soft / multiply / burn)
  ├─ make_seamless (optional band)
  └─ parallax_alpha → nh_alpha (_nh)
```

## Invariants

These are what the results rest on. A change that breaks one of them needs a
very good reason.

**One build.** Every module has `BUILD = N`; `check_build()` refuses to start if
they differ. Raise `BUILD` in all modules when a signature used across modules
changes.

**Seamless by construction.** Textures tile, so every filter, gradient and
integral runs on a torus: `mode='wrap'`, `np.roll`, FFT with periodic
boundaries. No `mode='reflect'`/`'nearest'` on anything that ends up in a map.
Neural networks run on wrap-padded windows; repair rolls the texture so a
patch crossing the edge is one patch. The suite checks shift-equivariance.

**Worker threads never touch Tk.** Long work (networks, batch, full-size 3D
maps, graph evaluation) runs in `threading.Thread`. Workers only write plain
attributes (`_net_msg`, `_bat_prog`, `_g_out`…); the main thread polls them
with `after()`. Everything that reads a widget or a Tk variable is gathered in
the main thread before the worker starts.

**What you see is what is saved.** The material preview is computed on a
half-size copy. Parameters are set for the full size and rescaled for the copy
by `matgen.at_scale` (radii and height range ×0.5, only the fine normal term
×0.5 because the medium and coarse terms are already multiplied by their
radius). The suite checks that preview and file agree.

**Colour spaces.** Input is sRGB. Anything physical — dividing out light,
multiplying by occlusion, averaging light — is done in linear. Photoshop-style
blend modes (overlay, soft light, linear burn) are done in sRGB on purpose, to
match what artists get in Photoshop.

**Normal convention.** Normal vectors are kept in the layout chosen by the
`directx` flag — DirectX (green down) by default, which is what OpenMW
expects. `slope_curl` and a few helpers assume DirectX and callers flip green
for OpenGL. Encoded files are `n·0.5+0.5`; mipmaps of normal maps are
renormalised per level.

**Parallax for OpenMW.** OpenMW computes
`offset = eyeDir.xy · (alpha · 0.04 − 0.02)`: no shift at alpha 128, full range
4 % of one texture repeat, no engine setting for depth. One game unit is
1.379 cm (22.1 units per foot). `parallax_alpha` maps the height into that
window for a requested depth in centimetres.

**Every visible string goes through `t()`.** Both languages must be complete;
the suite checks it.

**Every new function shows its effect.** The owner's rule: a feature whose
result is not visible (a view, a number under the sliders, a 3D mode) is not
finished.

## Caches

- `MaterialJob` keys each stage (`_dl_key`, `_vkey`/`_h_key`, `_g_key`,
  `_nd_key`, `_ao_key`, `_bent_key`) by every parameter it depends on. When
  adding a parameter, add it to the key of the stage that uses it — a missing
  key means a stale map, not an error.
- `App.cached(name, key, fn)` stores foliage stages as `_c_*` attributes;
  `drop_cache()` clears them.
- `nodes.Graph` caches node outputs by a key built from the node's params and
  everything upstream, plus the context key (texture identity).
- `view3d` caches mip pyramids by array identity.
- `nodes.disk_cached` keeps network results in `~/.texelith/cache` keyed by a
  SHA-1 of the input pixels and settings (LRU, 2 GB).

## Node workspace

`editor.Editor` keeps everything that must survive a window rebuild (language
change) in `app.ed`: the graph, project path, texture, batch settings, view.
A change schedules `evaluate()`: the graph is copied (sharing the content-keyed
cache) and every node is evaluated in `Graph.order()` in a worker thread, which
fills `results`, `errors` and PIL thumbnails; `poll()` in the main thread draws
them. Textures above `PREVIEW_MAX` are previewed on a reduced copy with
`ctx['px']` < 1, and every node multiplies its pixel radii by `nodes._px(ctx)`.
Nodes may return readouts under `'_info'` as (key, value) pairs; labels are
`ni.<key>` strings.

## Batch processing

`App.batch(kind)` collects all settings in the main thread and starts `_batch`
in a worker. For materials it re-runs the depth network (if the external depth
came from it) and the ESRGAN models (if their sliders are used) on every
texture, repeats automatic damage detection, and warns up front about what it
cannot reproduce (depth from a file or brush, hand-painted hints, second zone).
The node workspace uses `project.run_batch`: a copy of the graph for every
file of a folder tree, outputs mirrored under the output folder, finished files
skipped, a log written. A 'diffuse' output (the source's own name) is never
written into the source folder. Finished maps (`_n`, `_nh`, `_ao`, …) in the folder are never taken as
sources.

## Adding things

**A node type** — in `nodes.py`:

```python
@node('my_node', [('h', GRAY)], [('h', GRAY)], [('amount', 'f', 1.0, 0.0, 2.0)])
def _n_my_node(i, p, ctx):
    """…"""
    return {'h': something(i['h'], p['amount'])}
```

then add `nd.my_node`, `nd.my_node.info` and every new `np.<param>`/`np.<port>`
to `i18n.py`. The suite evaluates every registered node.

**A height layer** — compute it in `matgen.height_variants` (or
`MaterialJob.height` for layers that need job state), add its weight to the
`w` dict and to the `vkey`, add a slider and a string.

**A material map** — produce it in `MaterialJob.all_maps`, add it to
`App.material_outputs`, to the save flags, and a view in `MAT_VIEWS` so it can
be seen.
