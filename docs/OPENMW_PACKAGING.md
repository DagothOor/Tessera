# Packing the maps into an OpenMW mod

Русская версия: [OPENMW_PACKAGING.ru.md](OPENMW_PACKAGING.ru.md).

## 1. Which files

OpenMW finds extra maps by the end of the file name: the map lies next to the diffuse, with the same name plus a suffix.

| File | Contents | Texelith node |
|---|---|---|
| `tx_name.dds` | diffuse (replaces the game's) | Output, name "diffuse" |
| `tx_name_n.dds` | normal, DirectX layout (green down) | Output, name `_n` |
| `tx_name_nh.dds` | normal in RGB, parallax height in alpha | Parallax for OpenMW (_nh) → Output, name `_nh` |
| `tx_name_spec.dds` | RGB is the specular colour; alpha × 255 is the shininess exponent | PBR → OpenMW → Output, name `_spec` |

Ship one normal map per texture: `_n` or `_nh`, not both.

In the `_nh` alpha, 128 means no shift. The whole parallax range is 4 % of one texture repeat. That is why the node takes the depth in centimetres and converts it to alpha itself.

## 2. Folder layout

```
MyPack/
  Textures/
    tx_name.dds
    tx_name_nh.dds
    tx_name_spec.dds
```

In the Batch dialogue, set "Map layout" to "all together (next to the source)" and make `MyPack/Textures` the output folder. The other layouts (folder per kind of map, folder per texture) are only for viewing; the game will not find maps laid out that way.

## 3. Enabling it in OpenMW

1. Add the mod folder to the data list: in the launcher's Data Files tab, or as a line in `openmw.cfg`:

   ```
   data="D:/mods/MyPack"
   ```

   Lower lines win, so the mod must come after the game's own data.

2. In `settings.cfg`, section `[Shaders]`, turn on map detection. It is off by default.

   ```
   [Shaders]
   auto use object normal maps = true
   auto use object specular maps = true
   auto use terrain normal maps = true
   ```

   Without `auto use object specular maps`, `_spec` maps are not used on `.nif` models.

3. The name patterns (`normal map pattern = _n`, `normal height map pattern = _nh`, `specular map pattern = _spec`) already default to these values.

## 4. Terrain

Terrain normal maps work the same way: `_n` or `_nh`, enabled by `auto use terrain normal maps`.

Terrain specular is a different file, `_diffusespec`: RGB is the diffuse and alpha the specular strength. It is enabled by `auto use terrain specular maps`. Texelith has no `_diffusespec` file ending yet.

## 5. The OpenMW-PBR shader (wareya)

That shader reads `_spec` differently:

- R is metalness;
- G is roughness;
- B is AO;
- A is 1 − SSS.

Plain OpenMW cannot use such a file, so Texelith writes PBR versions as a separate set:

- set "set" to `pbr` on the Output or Output by channels node;
- the files go into a folder next to the output folder, named after it with `_PBR` added (`MyPack/Textures` → `MyPack/Textures_PBR`).

Two mods work well:

```
MyPack/Textures/...       — shared: _nh, _n and everything for plain OpenMW
MyPack_PBR/Textures/...   — only _spec and the diffuse for the PBR shader
```

Load `MyPack_PBR` after `MyPack` in `openmw.cfg`. Its `_spec` and diffuse then override the plain ones, while `_nh` still comes from the shared mod. Move `Textures_PBR` into `MyPack_PBR/Textures` by hand.

To install the shader itself (from the wareya repository README):

1. Copy OpenMW's shader folder (`resources/shaders`) as a backup. Copy it; do not rename it.
2. Copy the repository's shaders over the folder. Do not install it over another core shader mod.
3. Set these in `settings.cfg`:

   ```
   force shaders = true
   clamp lighting = false
   force per pixel lighting = true
   ```

4. The main branch targets OpenMW development builds. Stable OpenMW releases use the shader from the repository's Releases page.

## 6. Environment map reflection (Morrowind)

Models with an environment map (shiny armour, glass) reflect it equally over the whole surface. Where and how much to reflect is set by a gloss map: its RGB multiplies the reflection.

- In Texelith this is the «reflection mask (gloss)» output of the PBR → OpenMW node: metal reflects its own colour, the rest about 4 %, crevices are dimmed by AO, rough parts reflect less («Roughness dims the reflection»).
- Write it with an Output node named `_gloss`.
- The engine does not find it by file name: in NifSkope assign the texture to the Gloss slot of the model's `NiTexturingProperty`. The model must have an environment map (`NiTextureEffect`), or there is nothing for the mask to multiply.
- OpenMW reads this slot (`files/shaders/compatibility/objects.frag`: `envEffect *= texture2D(glossMap, …)`). Plain Morrowind disables it; the Morrowind Code Patch has an option that turns it on.

## 7. MGE XE

The map format of the MGE XE PBR prototype is not covered here, because its documentation was not available. This document does not cover plain `_n`/`_spec` maps for MGE XE either.

## Sources

- OpenMW Shaders settings: https://openmw.readthedocs.io/en/latest/reference/modding/settings/shaders.html
- OpenMW-PBR (wareya): https://github.com/wareya/OpenMW-PBR
- OpenMW gloss maps: https://gitlab.com/OpenMW/openmw/-/issues/6541
- OpenMW parallax formula: `files/shaders/lib/material/parallax.glsl` in the OpenMW sources.
