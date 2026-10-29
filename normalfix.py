# -*- coding: utf-8 -*-
"""
Analysis and repair of existing (already made) normal maps.

Three typical defects:
  * the vector length is not 1 — the engine reads that as "the surface is
    lit brighter or dimmer for no reason", and blotches of brightness run
    across the map;
  * there are pixels with negative Z, i.e. facing away from the viewer.
    Physically that cannot happen, and they turn into black dots;
  * the blue channel is empty or constant — the map is effectively
    two-channel, and the third coordinate has to be reconstructed.

Conventions. Maps come in as float RGB in 0…1 (H, W, 3); vectors are
decoded as n = 2·rgb − 1 without any axis flips. The project keeps normals
in the OpenGL convention internally (+Y up the image, "green up") and
converts to DirectX (+Y down) only when writing; in this module the only
convention-dependent step is ``fix(flip_green=True)``, which negates Y to
switch a map between the two.

Разбор и починка уже готовых карт нормалей.

Три типовые беды:
  * длина вектора не единичная — движок читает это как «поверхность светится
    сильнее или слабее просто так», и по карте идут пятна яркости;
  * есть пиксели с отрицательным Z, то есть отвёрнутые от зрителя. Физически
    так не бывает, и на них получаются чёрные точки;
  * синий канал пустой или постоянный — карта фактически двухканальная,
    третью координату надо восстановить.
"""

import math

import numpy as np

import xpu as X

BUILD = 25          # версия набора файлов, проверяется при запуске


def decode(rgb01):
    """Decode stored values 0…1 into vectors −1…1.

    rgb01: float array (H, W, 3), 0…1. Returns float32 (H, W, 3), −1…1,
    computed as 2·rgb − 1 with no renormalisation and no axis flips. On the
    card or the processor (the module of rgb01).
    """
    xp = X.xp(rgb01)
    return rgb01.astype(xp.float32) * 2.0 - 1.0


def encode(n):
    """Encode vectors −1…1 back to storable 0…1 (n·0.5 + 0.5, clipped to 0…1)."""
    return X.xp(n).clip(n * 0.5 + 0.5, 0.0, 1.0)


def analyze(rgb01):
    """Inspect a normal map without changing anything.

    rgb01: float array (H, W, 3), 0…1, as stored in the file.

    Returns (stats, problems): both are lists of (key, value) pairs. The keys
    are canonical and the interface translates them, so this module does not
    depend on the language.

    stats (values are preformatted strings):
      size    — "WxH";
      length  — mean vector length and its min … max;
      outside — share of pixels whose length is outside 0.9…1.1, %;
      negz    — share and count of pixels with Z < 0 (facing away);
      lowz    — share of pixels with Z < 0.2 (tilted steeper than ~78°), %;
      tilt    — mean tilt from +Z in degrees and its 99.9th percentile
                (computed from Z / length, so length errors do not skew it);
      uniq    — number of distinct 8-bit levels in R / G / B.

    problems (value is None unless noted):
      p.blue    — blue has ≤ 2 distinct levels or std < 0.01: the map is
                  effectively two-channel and Z should be rebuilt;
      p.negz    — more than 0.1 % of pixels have Z < 0;
      p.length  — more than 1 % of pixels have length outside 0.9…1.1;
      p.banding — some channel has fewer than 40 distinct levels (value: the
                  smallest count), i.e. visible banding;
      p.ok      — none of the above.
    """
    n = decode(rgb01)
    L = np.sqrt((n ** 2).sum(-1))
    z = n[..., 2]
    uniq = [int(len(np.unique((rgb01[..., i] * 255 + 0.5).astype(np.uint8)))) for i in range(3)]
    blue = rgb01[..., 2]
    tilt = np.degrees(np.arccos(np.clip(z / np.maximum(L, 1e-6), -1, 1)))

    r = [
        ('size', '%dx%d' % (rgb01.shape[1], rgb01.shape[0])),
        ('length', '%.3f  (%.3f … %.3f)' % (L.mean(), L.min(), L.max())),
        ('outside', '%.2f %%' % (100.0 * ((L < 0.9) | (L > 1.1)).mean())),
        ('negz', '%.3f %% (%d px)' % (100.0 * (z < 0).mean(), int((z < 0).sum()))),
        ('lowz', '%.2f %%' % (100.0 * (z < 0.2).mean())),
        ('tilt', '%.1f°  (99.9 %% < %.1f°)' % (tilt.mean(), np.percentile(tilt, 99.9))),
        ('uniq', '%d / %d / %d' % tuple(uniq)),
    ]

    problems = []
    if uniq[2] <= 2 or float(blue.std()) < 0.01:
        problems.append(('p.blue', None))
    if (z < 0).mean() > 0.001:
        problems.append(('p.negz', None))
    if ((L < 0.9) | (L > 1.1)).mean() > 0.01:
        problems.append(('p.length', None))
    if min(uniq) < 40:
        problems.append(('p.banding', min(uniq)))
    if not problems:
        problems.append(('p.ok', None))
    return r, problems


def fix(rgb01, rebuild_z=False, renormalize=True, min_z=0.0, flip_green=False):
    """Repair a normal map; returns float array (H, W, 3), 0…1, ready to store.

    rgb01: float array (H, W, 3), 0…1.
    rebuild_z   — recompute Z from X and Y from scratch, Z = √(1 − X² − Y²)
                  (for two-channel maps).
    renormalize — bring the length to one.
    min_z       — squeeze XY so that Z is not below this value (0…1; 0 turns
                  it off). At 0.2 the maximum tilt comes out around 78°, and
                  no pixels facing away remain at all. With min_z > 0 the
                  output is always unit length and Z is always recomputed
                  from XY, so pixels with negative Z are mirrored to the front.
    flip_green  — switch the green-channel format (OpenGL ↔ DirectX) by
                  negating Y. It is applied first, before everything else.

    Order matters for min_z: the vector is renormalised BEFORE Z is clamped
    (when renormalize is on and Z was not rebuilt). Otherwise, for a vector
    whose length is not 1, Z would be recomputed from "foreign" XY and the
    tilt would change: at length 0.8 a normal tilted 36.9° came out at 28.7°.
    Note that renormalize alone (min_z = 0) does not remove negative Z; only
    min_z (or rebuild_z) does. On the card or the processor (the module of
    rgb01).
    """
    xp = X.xp(rgb01)
    n = decode(rgb01).copy()
    if flip_green:
        n[..., 1] = -n[..., 1]
    if rebuild_z:
        r2 = n[..., 0] ** 2 + n[..., 1] ** 2
        n[..., 2] = xp.sqrt(xp.clip(1.0 - r2, 0.0, 1.0))
    if min_z > 0:
        if renormalize and not rebuild_z:
            # сначала единичная длина, потом прижим: иначе у вектора длиной не
            # 1 Z пересчитывался бы из «чужого» XY и наклон менялся (при длине
            # 0.8 нормаль в 36.9° выходила 28.7°)
            n = n / xp.maximum(xp.sqrt((n ** 2).sum(-1, keepdims=True)), 1e-6)
        # длину XY режем так, чтобы на неё точно хватило места под нужный Z
        r = xp.sqrt(n[..., 0] ** 2 + n[..., 1] ** 2)
        rmax = math.sqrt(max(1.0 - min_z * min_z, 1e-6))
        k = xp.where(r > rmax, rmax / xp.maximum(r, 1e-6), 1.0)
        n[..., 0] *= k
        n[..., 1] *= k
        n[..., 2] = xp.sqrt(xp.clip(1.0 - n[..., 0] ** 2 - n[..., 1] ** 2, 0.0, 1.0))
    elif renormalize:
        L = xp.sqrt((n ** 2).sum(-1, keepdims=True))
        n = n / xp.maximum(L, 1e-6)
        n[..., 2] = xp.abs(n[..., 2]) if rebuild_z else n[..., 2]
    return encode(n)
