# -*- coding: utf-8 -*-
"""Engine for textures with an alpha channel: foliage, grass, branches.
Shape is taken from the silhouette or from the veins, not from brightness.

What lives here:
  * DDS writer (_write_dds) with mip chains that renormalise normals
    (save_normal_dds) and keep alpha-tested coverage (save_diffuse_dds);
  * Source — the loaded card and its masks; Normals — per-leaf normals:
    the silhouette is cut into leaves (h-maxima markers + watershed), each
    piece gets a cross-section profile and a random tilt, plus clumps, a
    whole-card dome and a bend;
  * crown volume: an invented depth (leaf_depth), horizon-based AO on it
    (hbao, crown_occlusion, ao_map), bent normals, leaves lying behind;
  * translucency and a back-lit preview.

Conventions. Images are float 0…1 (H, W, C). Foliage cards do not tile, so
unlike the material path nothing here wraps round the edges ('nearest'
filters, shifts that fill with a constant). Normals are float (H, W, 3) unit
vectors in image axes: x to the right, y DOWN the image (the DirectX green
layout), z towards the viewer; save_normal_dds writes them as they are.

Движок для текстур с альфа-каналом: листва, трава, ветки.
Форма берётся из силуэта или из прожилок, а не из яркости."""

import io
import os
import struct

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import h_maxima
from skimage.segmentation import find_boundaries, watershed

import xpu as X

BUILD = 25          # версия набора файлов, проверяется при запуске


# ============================================================== запись DDS ==
def _dds_header(w, h, nlevels, compress):
    """Build the 128-byte DDS header ("DDS " + DDS_HEADER) for a mip chain.

    w, h: top level size in px; nlevels: number of mip levels in the file.
    compress: True — DXT5 FourCC with the top level's linear size; False —
    uncompressed 32-bit BGRA (A8R8G8B8 masks) with the row pitch. Caps are
    COMPLEX | MIPMAP | TEXTURE. Returns bytes.
    """
    if compress:
        flags = 0x1 | 0x2 | 0x4 | 0x1000 | 0x20000 | 0x80000
        linear = max(1, w // 4) * max(1, h // 4) * 16
        pf = struct.pack('<2I4s5I', 32, 0x4, b'DXT5', 0, 0, 0, 0, 0)
    else:
        flags = 0x1 | 0x2 | 0x4 | 0x8 | 0x1000 | 0x20000
        linear = w * 4
        pf = struct.pack('<2I4s5I', 32, 0x41, b'\0' * 4, 32,
                         0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000)
    return (b'DDS ' + struct.pack('<7I', 124, flags, h, w, linear, 0, nlevels)
            + b'\0' * 44 + pf + struct.pack('<5I', 0x401008, 0, 0, 0, 0))


# Чем сжимать DXT5: ('quicktex', уровень 0…18) или ('pillow', 0). Это то, что
# значит compress=True; редактор ставит сюда выбор из окна пакета.
DXT = {'algo': 'quicktex', 'level': 14}


def have_quicktex():
    """True if the quicktex encoder can be imported."""
    try:
        import quicktex.s3tc.bc3  # noqa: F401
        return True
    except ImportError:
        return False


def _bc3(rgba, spec=None):
    """DXT5 (BC3) blocks of one RGBA uint8 level, without a header.

    quicktex (Apache-2.0, the rgbcx encoder), when it is installed: it uses
    the in-between colours of each 4×4 block, so smooth gradients (skin, sky)
    keep about twice as many steps. Pillow's encoder, used otherwise, takes
    coarse block end colours, and a smooth dark texture came out in bands: on
    a test gradient 9 steps in red instead of 31, error up to 6 levels of 255
    (quicktex: 18 steps, error up to 2).

    spec: (algorithm, level) — ('quicktex', 0…18) or ('pillow', …); anything
    else (True) means the default DXT. Level 14 is the default: on a smooth
    skin texture and on a rock photo it gives the same error as 18 (to 0.01
    of a level), but 4…5 times faster — 18 took 4…6 s per 1024×1024 map;
    below 14 the bands on smooth gradients come back (12: error twice as
    large). If quicktex is not installed, Pillow is used."""
    algo, level = spec if isinstance(spec, tuple) else (DXT['algo'], DXT['level'])
    try:
        if algo != 'quicktex':
            raise ImportError
        from quicktex import RawTexture
        from quicktex.s3tc.bc3 import BC3Encoder
    except ImportError:
        buf = io.BytesIO()
        Image.fromarray(rgba).save(buf, format='DDS', pixel_format='DXT5')
        return buf.getvalue()[128:]
    level = min(max(int(level), 0), 18)
    h, w = rgba.shape[:2]
    pool = _bc3_pool() if h * w >= 512 * 512 and h % 4 == 0 else None
    if pool is not None:
        # блоки 4×4 независимы, а в файле идут строками блоков: полосы по
        # высоте, кратные 4, кодируются по отдельности и склеиваются — байт
        # в байт как целиком (проверено), только на всех ядрах сразу
        n = _BC3_POOL['n'] * 2
        step = max(64, (-(-h // n) + 3) // 4 * 4)
        strips = [np.ascontiguousarray(rgba[y:y + step], np.uint8) for y in range(0, h, step)]
        try:
            return b''.join(pool.map(_bc3_strip, strips, [level] * len(strips)))
        except Exception:                    # noqa: BLE001 — пул сломался: по-старому
            _BC3_POOL['pool'] = None
            _BC3_POOL['n'] = 0
    return _bc3_strip(rgba, level)


def _bc3_strip(rgba, level):
    """quicktex DXT5 of one RGBA uint8 piece (height a multiple of 4)."""
    from quicktex import RawTexture
    from quicktex.s3tc.bc3 import BC3Encoder
    h, w = rgba.shape[:2]
    raw = RawTexture.frombytes(np.ascontiguousarray(rgba, np.uint8).tobytes(), w, h)
    return BC3Encoder(level).encode(raw).tobytes()


# quicktex держит один поток (GIL не отпускает), поэтому большие уровни
# кодируются в отдельных процессах. Пул создаётся при первом большом файле
# и живёт до выхода программы; ядер берётся на одно меньше, чем есть
# (окну тоже нужно), но не больше 8 — каждый процесс заново импортирует
# запущенную программу и стоит памяти.
_BC3_POOL = {'pool': None, 'n': None}


def _bc3_pool():
    """The shared process pool for DXT5 encoding, or None (one core, or the
    pool could not start — then encoding stays in this process)."""
    if _BC3_POOL['n'] == 0:
        return None
    if _BC3_POOL['pool'] is None:
        import multiprocessing as mp
        n = min(8, (os.cpu_count() or 1) - 1)
        if n < 2 or mp.current_process().name != 'MainProcess':
            _BC3_POOL['n'] = 0
            return None
        try:
            from concurrent.futures import ProcessPoolExecutor
            _BC3_POOL['pool'] = ProcessPoolExecutor(n, mp_context=mp.get_context('spawn'))
            _BC3_POOL['n'] = n
        except Exception:                    # noqa: BLE001
            _BC3_POOL['n'] = 0
            return None
    return _BC3_POOL['pool']


def _write_dds(levels, path, compress):
    """Write a ready mip chain to a DDS file.

    levels: list of uint8 RGBA arrays (h, w, 4), top level first, each half
    the size of the previous. compress: True or an (algorithm, level) pair —
    each level is DXT5-encoded (_bc3); levels smaller
    than 4 px on a side are nearest-upscaled to one 4×4 block. False — raw
    BGRA bytes. Also used by Texelith.save_rgba for the material maps.
    """
    payload = b''
    for lv in levels:
        if compress:
            # блок DXT всегда 4x4; уровни мельче кодируем как один блок,
            # растянув их до 4x4 — так делает и DirectXTex
            h, w = lv.shape[:2]
            src = lv
            if h < 4 or w < 4:
                src = np.array(Image.fromarray(lv).resize((max(w, 4), max(h, 4)),
                                                          Image.NEAREST))
            payload += _bc3(src, compress)
        else:
            payload += np.concatenate(
                [lv[..., 2:3], lv[..., 1:2], lv[..., 0:1], lv[..., 3:4]], -1).tobytes()
    with open(path, 'wb') as f:
        f.write(_dds_header(levels[0].shape[1], levels[0].shape[0], len(levels), compress))
        f.write(payload)


def save_normal_dds(n, path, compress=False):
    """Write a normal map as DDS with a full mip chain.

    The mips average the vectors themselves and normalise them again,
    otherwise the relief flattens out at a distance.

    n: float (H, W, 3) normal vectors (any length; every level, the top one
    included, is normalised before encoding to 0…255), in the axes they come
    in — no green flip is done here. Each level is a 2×2 box average of the
    previous unnormalised one (odd sizes drop the last row/column); the chain
    stops when the shorter side reaches 1 px. Alpha is 255. compress: DXT5
    (default off).
    """
    levels, cur = [], n.copy()
    while True:
        L = np.sqrt((cur ** 2).sum(-1, keepdims=True))
        rgb = np.clip((cur / np.maximum(L, 1e-6) * 0.5 + 0.5) * 255 + 0.5, 0, 255).astype(np.uint8)
        levels.append(np.concatenate([rgb, np.full(rgb.shape[:2] + (1,), 255, np.uint8)], -1))
        if min(cur.shape[0], cur.shape[1]) <= 1:
            break
        hh, ww = max(1, cur.shape[0] // 2), max(1, cur.shape[1] // 2)
        cur = cur[:hh * 2, :ww * 2].reshape(hh, 2, ww, 2, 3).mean(axis=(1, 3))
    _write_dds(levels, path, compress)


def save_diffuse_dds(rgba, path, compress=True, keep_coverage=True, thr=0.5):
    """Write a foliage diffuse (RGBA) as DDS with a coverage-preserving mip chain.

    The mips keep the foliage area constant: with ordinary averaging alpha
    spreads out, the alpha test cuts away more pixels, and the crown goes
    bald at a distance.

    rgba: uint8 (H, W, 4), 0…255. thr: the alpha-test threshold, 0…1.
    keep_coverage: on every level below the top, alpha is scaled by a factor
    found by bisection (20 steps over 0.05…20) so that the share of pixels
    with alpha > thr equals that of the top level. The scale is applied only
    to the stored level; the next level is averaged from the unscaled alpha,
    so errors do not compound. Same idea as Ignacio Castaño, "Computing Alpha
    Mipmaps" (2010).

    Colour is averaged with alpha as the weight, so the background behind a
    leaf (black or anything else) does not bleed into its edge and no dark
    fringe appears at a distance; fully transparent 2×2 blocks fall back to a
    plain mean. The average is taken on the stored sRGB values, not in
    linear light. Chain down to a 1 px shorter side; compress: DXT5.
    """
    levels, cur = [], rgba.astype(np.float32) / 255.0
    target = float((cur[..., 3] > thr).mean())
    while True:
        a = cur[..., 3]
        if keep_coverage and levels and target > 0:
            lo, hi = 0.05, 20.0
            for _ in range(20):
                m = (lo + hi) / 2
                if (np.clip(a * m, 0, 1) > thr).mean() < target:
                    lo = m
                else:
                    hi = m
            a = np.clip(a * (lo + hi) / 2, 0, 1)
        levels.append(np.clip(np.concatenate([cur[..., :3], a[..., None]], -1) * 255 + 0.5,
                              0, 255).astype(np.uint8))
        if min(cur.shape[0], cur.shape[1]) <= 1:
            break
        hh, ww = max(1, cur.shape[0] // 2), max(1, cur.shape[1] // 2)
        c = cur[:hh * 2, :ww * 2].reshape(hh, 2, ww, 2, 4)
        # цвет усредняется с весом по альфе: иначе чёрный (или любой) фон за
        # листом затекает в край, и вдали у листвы появляется тёмная кайма
        w = c[..., 3:4]
        sw = w.sum(axis=(1, 3))
        rgb = np.where(sw > 1e-6, (c[..., :3] * w).sum(axis=(1, 3)) / np.maximum(sw, 1e-6),
                       c[..., :3].mean(axis=(1, 3)))
        cur = np.concatenate([rgb, c[..., 3:4].mean(axis=(1, 3))], -1)
    _write_dds(levels, path, compress)


# ================================================================= расчёты ==
def srgb_to_lin(c):
    """sRGB → linear by the gamma-2.2 approximation (not the exact curve), clipped."""
    return np.power(np.clip(c, 0, 1), 2.2)


def lin_to_srgb(c):
    """Linear → sRGB by the gamma-2.2 approximation, clipped to 0…1."""
    return np.power(np.clip(c, 0, 1), 1 / 2.2)


def fast_blur(x, sigma):
    """Gaussian blur with sigma in px; three box passes instead of a Gaussian
    at large radii.

    Below sigma 6 a true Gaussian is used; above it three uniform filters of
    odd width w ≈ √(12σ²/3 + 1), whose combined variance matches σ². Edges
    are 'nearest' (cards do not tile). Returns an array of the same shape.
    """
    if sigma < 6:
        return ndi.gaussian_filter(x, sigma, mode='nearest')
    w = max(3, int(round(np.sqrt(12.0 * sigma * sigma / 3.0 + 1.0))))
    if w % 2 == 0:
        w += 1
    y = x
    for _ in range(3):
        y = ndi.uniform_filter(y, size=w, mode='nearest')
    return y


def masked_blur(x, mask, r):
    """Blur that does not drag the empty background into the leaf.

    Normalised convolution: blur(x·mask) / blur(mask), Gaussian sigma r px
    (scipy's default 'reflect' edges). Outside the mask the result is
    extrapolated from nearby leaf pixels. Returns float (H, W).
    """
    return ndi.gaussian_filter(x * mask, r) / np.maximum(
        ndi.gaussian_filter(mask.astype(np.float32), r), 1e-4)


def alpha_from_background(rgba, tol=0.12, feather=1.0):
    """Make alpha from the background: the background colour is taken from
    the frame border, then pixels are cut by their distance to it. Needed for
    textures whose alpha was not saved while the foliage is drawn on a
    uniform background.

    rgba: float (H, W, 4), 0…1. The background colour is the median of the
    outermost rows and columns; distance is the RMS difference per channel.
    tol: distance (0…1) at which alpha starts to rise from 0; feather: width
    of the ramp relative to tol (alpha reaches 1 at tol·(1 + feather)).
    Returns a copy of rgba with the alpha replaced.
    """
    rgb = rgba[..., :3]
    h, w = rgb.shape[:2]
    edge = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]], 0)
    bg = np.median(edge, axis=0)
    d = np.sqrt(((rgb - bg) ** 2).sum(-1) / 3.0)
    lo = max(tol, 1e-3)
    hi = lo * (1.0 + max(feather, 0.05))
    a = np.clip((d - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    out = rgba.copy()
    out[..., 3] = a
    return out


class Source:
    """The loaded foliage card and the masks derived from it.

    Attributes: path; rgba float32 (H, W, 4), 0…1; alpha (H, W); mask bool,
    alpha > 0.5; cover, the opaque share; maskf, mask as float32; box, the
    bounding box of the opaque pixels (x0, x1, y0, y1) in px, for the
    whole-card shape; lum (H, W), luma 0.3R + 0.6G + 0.1B inside the mask
    and its masked blur (sigma 4 px) outside it, so filters do not see a
    hard step to the background.
    """
    def __init__(self, path, rgba=None):
        """path — a file, or else a ready RGBA array in the second argument
        (needed for the half-size preview).

        rgba: float (H, W, 4), 0…1; if None, the file is read and converted.
        Raises ValueError if the alpha is empty: constant (std < 1e-4) or
        more than 99.95 % opaque.
        """
        if rgba is None:
            im = Image.open(path)
            im.load()
            rgba = np.array(im.convert('RGBA')).astype(np.float32) / 255.0
        self.path = path
        self.rgba = rgba.astype(np.float32)
        self.alpha = self.rgba[..., 3]
        self.mask = self.alpha > 0.5
        # альфа считается настоящей, если она вообще меняется по картинке.
        # Прежний порог «непрозрачно больше 99.5 %» отбраковывал плотные
        # текстуры, у которых альфа есть, но дырок в ней мало
        self.cover = float(self.mask.mean())
        if float(self.alpha.std()) < 1e-4 or self.cover > 0.9995:
            raise ValueError('альфа-канал пустой: непрозрачно %.1f %% при разбросе %.4f'
                             % (100.0 * self.cover, float(self.alpha.std())))
        self.maskf = self.mask.astype(np.float32)
        ys, xs = np.nonzero(self.mask)          # габариты кроны для общей формы
        self.box = (float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max()))
        self.lum = np.where(self.mask,
                            0.3 * self.rgba[..., 0] + 0.6 * self.rgba[..., 1] + 0.1 * self.rgba[..., 2],
                            masked_blur(0.3 * self.rgba[..., 0] + 0.6 * self.rgba[..., 1]
                                        + 0.1 * self.rgba[..., 2], self.maskf, 4))


class Normals:
    """Per-leaf normal map of a foliage card.

    Computed in stages and remembers the intermediate results so the sliders
    respond immediately: segment (cut the silhouette into pieces), shape
    (cross-section profile), scatter (random per-piece tilt), combine (sum it
    all into a normal). run() calls all four; each stage keeps a key of its
    own parameters and returns at once if they did not change. Normals come
    out in image axes, y down the image (see the module docstring).
    """

    def __init__(self, src):
        """Precompute what does not depend on any slider.

        src: Source. Stores the distance to the silhouette edge silh (px) and
        its 1.2 px blur, card coordinates U (1, W) and V (H, 1) in −1…1
        relative to the opaque bounding box (for the dome), and gradients
        hx, hy of the high-passed luminance (minus a 3 px blur, then a 0.6 px
        blur) for the painted-detail term.
        """
        self.s = src
        m = src.mask
        self.silh = ndi.distance_transform_edt(m).astype(np.float32)
        self.silh_s = ndi.gaussian_filter(self.silh, 1.2)
        self._near = None

        x0, x1, y0, y1 = src.box
        H, W = m.shape
        U = (np.arange(W, dtype=np.float32) - (x0 + x1) / 2) / max((x1 - x0) / 2, 1.0)
        V = (np.arange(H, dtype=np.float32) - (y0 + y1) / 2) / max((y1 - y0) / 2, 1.0)
        self.U = np.clip(U, -1, 1)[None, :]
        self.V = np.clip(V, -1, 1)[:, None]

        hp = src.lum - ndi.gaussian_filter(src.lum, 3.0)
        self.hy, self.hx = np.gradient(ndi.gaussian_filter(hp, 0.6))
        self.ridge = None
        self._key = self._seed = None

    def _ridge_map(self):
        """Veins: long light ridges. They are used to cut the dense mass,
        where the silhouette has stuck together into a solid blob with no
        leaf boundaries in it.

        Sato tubeness filter for bright ridges, sigmas 1, 2, 3 px (Sato et
        al., "Three-dimensional multi-scale line filter for segmentation and
        visualization of curvilinear structures in medical images", 1998;
        skimage.filters.sato), masked, scaled to 0…1 and blurred 1 px.
        Cached. Returns float (H, W), 0…1.
        """
        if self.ridge is None:
            # на видеокарте, если она включена (_sato), результат — на процессор
            L = X.to_dev(np.ascontiguousarray(self.s.lum))
            r = _sato(L, (1, 2, 3), black_ridges=False)
            r = X.xp(r).where(X.to_dev(np.ascontiguousarray(self.s.mask)), r, 0.0)
            self.ridge = X.to_host(X.ndi(r).gaussian_filter(r / max(float(r.max()), 1e-9), 1.0))
        return self.ridge

    def _lum_map(self):
        """Light bumps. Moss, lichen and small cushion plants have neither a
        clear silhouette nor long veins: every tussock is simply a small
        light spot, and the cutting has to follow those.

        Luminance blurred 1.5 px, masked, stretched from its 2nd…98th
        percentile inside the mask to 0…1. Cached. Returns float32 (H, W).
        """
        if getattr(self, '_lm', None) is None:
            L = ndi.gaussian_filter(self.s.lum, 1.5)
            L = np.where(self.s.mask, L, 0.0)
            lo, hi = np.percentile(L[self.s.mask], [2, 98]) if self.s.mask.any() else (0.0, 1.0)
            self._lm = np.clip((L - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)
        return self._lm

    def segment(self, mode, split, force=None, min_area=0):
        """Cut the silhouette into pieces (leaves). Hand-marked areas are cut
        separately: a branch must stay one piece, whereas the watershed tears
        it apart at every thickening where leaves join it.

        Marker-controlled watershed: markers are the h-maxima of a field
        (skimage h_maxima), the flooding runs on −field inside the mask
        (skimage watershed; Beucher & Meyer; Vincent & Soille, 1991).

        mode: 'dense' — the vein ridges (_ridge_map), for a solid mass;
              'lum'   — the light bumps (_lum_map), for moss and lichen;
              anything else — the silhouette: distance to the transparent
              background, for separate leaves.
        split: 0…1, sets h (the minimum height of a marker peak); a higher
              split gives fewer, larger pieces.
        force: bool (H, W) hand-painted mask or None; each connected painted
              area becomes one whole label, appended after the automatic
              ones and flagged in self.hand (such pieces get no tilt).
        min_area: pieces smaller than this many px are not kept separately
              but handed to their neighbours by a second watershed from the
              big pieces only; otherwise a branch falls apart into crumbs and
              every crumb gets its own tilt.

        Sets: lab int (H, W), 0 = background; nlab; hand, bool per label;
        t (H, W), 0 at the edge of the piece … 1 on its axis (distance to the
        silhouette, or to the piece boundaries in 'dense' mode, divided by
        the piece maximum, blurred 0.8 px); dirx, diry, the unit direction
        from axis to edge; and per-piece moments: axis (nlab + 1, 2), the
        principal direction; elong, √(λ1/λ2); cx, cy, the centroid; half,
        2·√λ1 in px. Pieces under 12 px get axis (0, 0) and elong 1.

        The cache key holds a fingerprint of the force mask itself (not just
        its pixel count), so any change of the painting triggers a recompute.
        """
        fid = 0 if force is None else hash(np.packbits(force > 0.5).tobytes())
        key = (mode, round(split, 4), fid, int(min_area))
        if self._key == key:
            return
        m = self.s.mask
        if mode == 'dense':
            field = self._ridge_map()
            h = 0.015 + split * 0.15
        elif mode == 'lum':
            field = self._lum_map()
            h = 0.01 + split * 0.20
        else:
            field = self.silh_s
            h = 0.4 + split * 2.6
        free = m if force is None else (m & ~force)
        markers, _ = ndi.label(h_maxima(field, h) * free)
        lab = watershed(-field, markers, mask=free)
        if min_area > 1:
            # мелочь не режем отдельно, а раздаём соседям: иначе ветка
            # рассыпается в крошку, и каждая крошка получает свой наклон
            area = np.bincount(lab.ravel())
            big = np.zeros(len(area), bool)
            big[area >= min_area] = True
            big[0] = False
            if big.any():
                keep = np.where(big[lab], lab, 0)
                lab = watershed(-field, keep, mask=free)
        self.lab = lab
        self.nlab = int(self.lab.max())
        if force is not None and force.any():
            # каждый закрашенный кусок — одна цельная метка, без дробления
            fl, fn = ndi.label(force & m)
            self.lab = np.where(fl > 0, fl + self.nlab, self.lab)
            self.hand = np.zeros(self.nlab + fn + 1, bool)
            self.hand[self.nlab + 1:] = True
            self.nlab += fn
        else:
            self.hand = np.zeros(self.nlab + 1, bool)

        # «крыша» внутри каждого куска: по силуэту — если резали по силуэту,
        # по границам кусков — если резали по прожилкам
        base = self.silh if mode != 'dense' else ndi.distance_transform_edt(
            m & ~find_boundaries(self.lab, mode='outer')).astype(np.float32)
        # t — насколько глубоко точка внутри своего куска: 0 у края, 1 на оси.
        # Направление «поперёк листа» берём как направление спуска от оси к краю;
        # оно единично, поэтому крутизна дальше задаётся профилем, а не тем,
        # сколько пикселей в ширину у этого конкретного листа.
        dmax = ndi.maximum(base, self.lab, index=np.arange(1, self.nlab + 1))
        dmax = np.concatenate([[1.0], np.maximum(dmax, 1e-3)])
        self.t = ndi.gaussian_filter(np.where(m, base / dmax[self.lab], 0.0), 0.8)
        gy, gx = np.gradient(self.t)
        g = np.maximum(np.sqrt(gx * gx + gy * gy), 1e-6)
        self.dirx, self.diry = (-gx / g).astype(np.float32), (-gy / g).astype(np.float32)
        self._prof = None

        # моменты каждого куска считаем разом через bincount: цикл по
        # сотням меток на питоне занимал больше, чем весь остальной расчёт
        ys, xs = np.nonzero(m)
        ls = self.lab[ys, xs].astype(np.intp)
        nb = self.nlab + 1
        x = xs.astype(np.float64)
        y = ys.astype(np.float64)
        cnt = np.bincount(ls, minlength=nb)
        sx = np.bincount(ls, x, minlength=nb)
        sy = np.bincount(ls, y, minlength=nb)
        sxx = np.bincount(ls, x * x, minlength=nb)
        syy = np.bincount(ls, y * y, minlength=nb)
        sxy = np.bincount(ls, x * y, minlength=nb)
        good = cnt >= 12
        c = np.maximum(cnt, 1.0)
        mx, my = sx / c, sy / c
        a = sxx / c - mx * mx          # ковариация 2x2 на каждый кусок
        b = sxy / c - mx * my
        d = syy / c - my * my
        half = 0.5 * (a + d)
        rad = np.sqrt(np.maximum(((a - d) * 0.5) ** 2 + b * b, 0.0))
        l1, l2 = half + rad, np.maximum(half - rad, 1e-9)
        vx = np.where(np.abs(b) > 1e-9, l1 - d, 1.0)
        vy = np.where(np.abs(b) > 1e-9, b, 0.0)
        nrm = np.sqrt(vx * vx + vy * vy)
        nrm = np.where(nrm > 1e-9, nrm, 1.0)
        self.axis = np.zeros((nb, 2), np.float32)
        self.axis[good, 0] = (vx / nrm)[good]
        self.axis[good, 1] = (vy / nrm)[good]
        self.elong = np.ones(nb, np.float32)
        self.elong[good] = np.sqrt(l1 / l2)[good]
        self.cx = mx.astype(np.float32)
        self.cy = my.astype(np.float32)
        self.half = np.maximum(2.0 * np.sqrt(np.maximum(l1, 1e-6)), 1.0).astype(np.float32)
        self._bendmap = None
        self._key = key
        self._seed = None

    def shape(self, profile):
        """Cross-section profile of the leaf. 1 — a gable (roof), 2 — exactly
        round (a cylinder or, for a roundish piece, a hemisphere), more — a
        flat top with a sharp fall-off at the edge.

        The profile is the superellipse h = (1 − uⁿ)^(1/n) with u = 1 − t
        (0 on the axis, 1 at the edge, capped at 0.995) and n = profile ≥ 1;
        its slope magnitude u^(n−1)·(1 − uⁿ)^(1/n − 1), clipped to 4, is laid
        along dirx, diry into FX, FY. Needs segment() first.
        """
        if self._prof == profile:
            return
        n = max(1.0, float(profile))
        u = np.clip(1.0 - self.t, 0.0, 0.995)          # 0 на оси, 1 у края
        mag = np.clip(u ** (n - 1) * (1.0 - u ** n) ** (1.0 / n - 1.0), 0.0, 4.0)
        self.FX = (mag * self.dirx).astype(np.float32)
        self.FY = (mag * self.diry).astype(np.float32)
        self._prof = profile

    def along(self):
        """Coordinate along the piece's own axis: −1 at one end, +1 at the
        other. The bend is made along it — a grass blade and a branch are not
        straight.

        Returns (t, ax, ay): t float32 (H, W), the projection on the principal
        axis divided by the half-length, clipped to −1…1; ax, ay (H, W), the
        axis of each pixel's piece. Cached until the next segment().
        """
        if getattr(self, '_bendmap', None) is None:
            H, W = self.lab.shape
            X = np.arange(W, dtype=np.float32)[None, :]
            Y = np.arange(H, dtype=np.float32)[:, None]
            l = self.lab
            ax, ay = self.axis[:, 0][l], self.axis[:, 1][l]
            t = ((X - self.cx[l]) * ax + (Y - self.cy[l]) * ay) / self.half[l]
            self._bendmap = (np.clip(t, -1, 1).astype(np.float32), ax, ay)
        return self._bendmap

    def scatter(self, seed, thin, force=None, blade=0.0):   # force учтён уже при разрезании
        """Give every piece a random tilt (the leaves of a branch do not all
        face the viewer).

        seed  — RNG seed; the same seed gives the same tilts, and leaf_depth
                uses the same seed so depth and normals describe one branch.
        thin  — from what elongation a piece counts as a branch or a needle
                and no longer gets a tilt (rounding only). Leafiness is
                1 / (1 + (elong / thin)⁴), thin ≥ 0.5.
        force — the hand-marked mask: a piece more than half of which falls
                into it counts as a branch regardless of elongation. In the
                current code the argument itself is not used (see the
                comment on the def line): painted areas are already separate
                labels after segment(), flagged in self.hand, and those get
                leafiness 0. It only enters the cache key.
        blade — 0…1: a grass blade is as elongated as a branch but needs a
                tilt, because it is a separate leaf, not a twig everything is
                strung on; blade gives elongated pieces their leafiness back.

        The tilt per piece is drawn along the axis ~N(0, 0.7) and across it
        ~N(0, 0.45), times leafiness, turned into image axes and blurred
        1 px. Sets TX, TY (H, W) and leafy_used (nlab + 1,).
        """
        fid = 0 if force is None else hash(np.packbits(force > 0.5).tobytes())
        key = (seed, round(thin, 3), fid, round(blade, 3))
        if self._seed == key:
            return
        leafy = 1.0 / (1.0 + (self.elong / max(thin, 0.5)) ** 4)
        if blade > 0:
            # травинка вытянута так же, как ветка, но наклон ей нужен:
            # это отдельный лист, а не сучок, на который всё нанизано
            leafy = leafy + blade * (1.0 - leafy)
        hand = getattr(self, 'hand', None)
        if hand is not None and hand.any():
            leafy = np.where(hand[:len(leafy)], 0.0, leafy)
        self.leafy_used = leafy
        rng = np.random.default_rng(int(seed))
        s = (rng.normal(0, 0.7, self.nlab + 1) * leafy).astype(np.float32)
        p = (rng.normal(0, 0.45, self.nlab + 1) * leafy).astype(np.float32)
        ax, ay = self.axis[:, 0], self.axis[:, 1]
        tx = (ax * s - ay * p).astype(np.float32)
        ty = (ay * s + ax * p).astype(np.float32)
        tx[0] = ty[0] = 0.0
        m = self.s.mask
        self.TX = ndi.gaussian_filter(np.where(m, tx[self.lab], 0.0), 1.0)
        self.TY = ndi.gaussian_filter(np.where(m, ty[self.lab], 0.0), 1.0)
        self._seed = key

    def clumps(self, scale):
        """Foliage density at two sizes. It works like height: where the mass
        is denser it is higher, falling off towards the edge of the clump.
        This gives the convexity of groups, intermediate between a single
        leaf and the whole card.

        scale: px; alpha is blurred with sigma max(2, scale) and
        max(8, 4·scale). Returns ((c1x, c1y), (c2x, c2y)), the gradients of
        the two densities multiplied by scale and 4·scale so the slope does
        not fade with size. Cached by scale.
        """
        key = round(scale, 2)
        if getattr(self, '_clump_key', None) != key:
            a = self.s.alpha.astype(np.float32)
            d1 = fast_blur(a, max(2.0, scale))
            d2 = fast_blur(a, max(8.0, scale * 4.0))
            g1y, g1x = np.gradient(d1)
            g2y, g2x = np.gradient(d2)
            self._c1 = (g1x * scale, g1y * scale)
            self._c2 = (g2x * scale * 4.0, g2y * scale * 4.0)
            self._clump_key = key
        return self._c1, self._c2

    def combine(self, tilt, fold, detail, pad=8.0, dome_w=0.0, dome_h=0.0,
                clump_mid=0.0, clump_big=0.0, clump_scale=24.0, bend=0.0):
        """Sum the stages into a unit normal map; needs segment, shape, scatter.

        tilt       — weight of the random per-piece tilt (scatter);
        fold       — weight of the cross-section profile (shape);
        detail     — weight of the painted detail (high-passed luminance
                     gradient ×40);
        pad        — px beyond the silhouette over which edge normals are
                     carried outwards (from the nearest opaque pixel, fading
                     linearly to flat), so filtering and mips at the leaf edge
                     do not pull in the flat background (0 = off);
        dome_w/h   — whole-card dome along x / y from the bounding box: both
                     equal — a sphere, one — a cylinder along the other axis;
        clump_mid/big, clump_scale — convexity of groups at two sizes
                     (clumps), weights ×1.6, scale in px;
        bend       — curvature of pieces along their own axis (along).

        Returns float (H, W, 3) unit vectors, image axes with y down; (0, 0, 1)
        on the background beyond pad.
        """
        m = self.s.mask
        nx = np.where(m, self.FX * fold + self.TX * tilt - self.hx * detail * 40.0, 0.0)
        ny = np.where(m, self.FY * fold + self.TY * tilt - self.hy * detail * 40.0, 0.0)
        if bend:
            t, ax, ay = self.along()
            nx = nx + np.where(m, bend * ax * t, 0.0)
            ny = ny + np.where(m, bend * ay * t, 0.0)
        if pad > 0:
            if self._near is None:      # дорогая штука, считаем только если нужна
                self.outdist, self._near = ndi.distance_transform_edt(
                    ~self.s.mask, return_indices=True)
            iy, ix = self._near
            k = np.clip(1.0 - self.outdist / pad, 0.0, 1.0).astype(np.float32)
            nx, ny = nx[iy, ix] * k, ny[iy, ix] * k
        # выпуклость групп листьев — между листом и всей карточкой
        if clump_mid or clump_big:
            (c1x, c1y), (c2x, c2y) = self.clumps(clump_scale)
            nx = nx - (c1x * clump_mid + c2x * clump_big) * 1.6
            ny = ny - (c1y * clump_mid + c2y * clump_big) * 1.6

        # общая форма всей карточки: купол по габаритам силуэта.
        # обе ручки поровну — шар, одна — цилиндр вдоль другой оси
        if dome_w or dome_h:
            nx = nx + dome_w * self.U
            ny = ny + dome_h * self.V
        nz = np.ones_like(nx)
        L = np.sqrt(nx * nx + ny * ny + nz * nz)
        return np.stack([nx / L, ny / L, nz / L], -1)

    def run(self, mode, split, seed, thin, tilt, fold, detail, profile=2.0, pad=8.0,
            dome_w=0.0, dome_h=0.0, force=None,
            clump_mid=0.0, clump_big=0.0, clump_scale=24.0,
            min_area=0, blade=0.0, bend=0.0):
        """Run segment → shape → scatter → combine; returns combine's normals.

        Each stage recomputes only if its own parameters changed. Parameters
        are those of the stages: mode, split, force, min_area (segment);
        profile (shape); seed, thin, blade (scatter); the rest (combine).
        """
        self.segment(mode, split, force, min_area)
        self.shape(profile)
        self.scatter(seed, thin, force, blade)
        return self.combine(tilt, fold, detail, pad, dome_w, dome_h,
                            clump_mid, clump_big, clump_scale, bend)

    def branch_picture(self):
        """Who counts as what: green — a leaf (gets the full tilt), red — a
        branch or needle (rounding only); mixed colours in between.

        Returns float (H, W, 3), 0…1, background 0.13 grey. Before scatter()
        everything shows as leaf.
        """
        lf = getattr(self, 'leafy_used', None)
        if lf is None:
            lf = np.ones(self.nlab + 1, np.float32)
        col = np.stack([1.0 - lf, lf * 0.85, np.full_like(lf, 0.15)], -1)
        col[0] = 0.13
        return col[self.lab]

    def branch_share(self):
        """Share (0…1) of pieces counted as branches (leafiness < 0.5).

        0 before scatter() or when there are no pieces.
        """
        lf = getattr(self, 'leafy_used', None)
        if lf is None or self.nlab == 0:
            return 0.0
        return float((lf[1:] < 0.5).mean())

    def segments_picture(self):
        """Random colour per piece (fixed seed), background 0.13; float (H, W, 3)."""
        rng = np.random.default_rng(0)
        col = rng.random((self.nlab + 1, 3)) * 0.75 + 0.25
        col[0] = 0.13
        return col[self.lab]


def height_map(nrm):
    """Foliage height — that very "how deep inside the piece" field the leaf
    shape is built from. Also usable as displacement.

    nrm: Normals after segment(). Returns nrm.t clipped to 0…1, float (H, W):
    0 at the edge of every piece, 1 on its axis, 0 on the background.
    """
    return np.clip(nrm.t, 0.0, 1.0)


def delight(src, unlight=0.8, unlight_radius=40.0, ao=0.5, ao_radius=24.0, lift=1.0):
    """Remove painted directional light and add shading from density.

    Works in linear light (gamma-2.2 approximation). unlight 0…1: the colour
    is divided by (large-scale masked luminance / its mean over the leaves),
    clipped to 0.35…3, raised to unlight; unlight_radius: Gaussian sigma in
    px of that large scale. ao 0…1: multiplies by 1 − ao·(alpha blurred with
    sigma ao_radius px, relative to its maximum), darkening where the foliage
    is dense. lift: overall brightness factor. Returns float (H, W, 4) RGBA,
    0…1; transparent pixels keep their original colour, alpha is unchanged.
    """
    lin = srgb_to_lin(src.rgba[..., :3])
    m, mf = src.mask, src.maskf
    out = lin
    if unlight > 0:
        L = 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]
        big = masked_blur(L, mf, unlight_radius)
        ref = float((big * mf).sum() / max(mf.sum(), 1e-6))
        out = out / (np.clip(big / max(ref, 1e-6), 0.35, 3.0) ** unlight)[..., None]
    if ao > 0:
        cov = ndi.gaussian_filter(src.alpha, ao_radius)
        out = out * (1.0 - ao * cov / max(float(cov.max()), 1e-6))[..., None]
    if lift != 1.0:
        out = out * lift
    rgb = np.where(m[..., None], lin_to_srgb(out), src.rgba[..., :3])
    return np.concatenate([rgb, src.alpha[..., None]], -1)




# ============================================================ объём кроны ==
def crown_height(src, nrm=None, leaf=0.5, clump=1.0, scale=24.0):
    """Approximate shape of the crown as a relief: where there is more
    foliage it is denser and higher, plus each leaf's own little hump.

    Needed for shading. Without such a surface "shading" on a card comes out
    not as shading but as an outline of the silhouette: a point can only be
    occluded by what lies ABOVE it, and a flat card has no height.

    src: Source; nrm: Normals (optional, for the per-leaf hump t·0.5 weighted
    by leaf); clump: weight of the density 0.6·blur(alpha, scale) +
    0.4·blur(alpha, 3·scale), scale in px. Returns float32 (H, W),
    min-max normalised to 0…1. (The AO path now uses leaf_depth instead.)
    """
    a = src.alpha.astype(np.float32)
    d1 = fast_blur(a, max(3.0, scale))
    d2 = fast_blur(a, max(10.0, scale * 3.0))
    h = clump * (0.6 * d1 + 0.4 * d2)
    t = getattr(nrm, 't', None)
    if t is not None and leaf > 0:
        h = h + leaf * np.clip(t, 0, 1) * 0.5
    lo, hi = float(h.min()), float(h.max())
    return ((h - lo) / max(hi - lo, 1e-6)).astype(np.float32)


def _shift(a, oy, ox, fill):
    """Shift without wrapping: a foliage card does not tile, and dragging a
    piece of the crown in from the opposite edge would be a plain error.

    a: (H, W); oy, ox: integer offset in px; out[p] = a[p − o], the uncovered
    part is filled with fill. Returns an array like a (on the video card if
    a is there).
    """
    out = X.xp(a).full_like(a, fill)
    H, W = a.shape
    ys0, ys1 = max(0, -oy), min(H, H - oy)
    xs0, xs1 = max(0, -ox), min(W, W - ox)
    if ys1 <= ys0 or xs1 <= xs0:
        return out
    out[ys0 + oy:ys1 + oy, xs0 + ox:xs1 + ox] = a[ys0:ys1, xs0:xs1]
    return out


def leaf_by_label(lab, nlab, value):
    """Mean value of a map over each piece of the segmentation, one value per
    label (the caller lays it back onto the pixels with m[lab]). Computed
    through bincount: a single pass over the array instead of a loop over a
    thousand leaves.

    lab: int (H, W) labels, 0 = background; nlab: highest label; value:
    (H, W) map, e.g. the network depth (larger = closer). Returns float32
    (nlab + 1,): m[0] = 0, the rest min-max normalised to 0…1 across the
    pieces (left as raw means if all are equal). The normalisation puts the
    network order on the same 0…1 scale as the uniform random draw it is
    mixed with in leaf_depth.
    """
    n = int(nlab) + 1
    flat = lab.ravel()
    s = np.bincount(flat, weights=np.asarray(value, np.float64).ravel(), minlength=n)
    c = np.bincount(flat, minlength=n)
    m = (s / np.maximum(c, 1)).astype(np.float32)
    m[0] = 0.0
    if n > 1:
        v = m[1:]
        lo, hi = float(v.min()), float(v.max())
        if hi - lo > 1e-6:
            m[1:] = (v - lo) / (hi - lo)
    return m


def leaf_depth(nrm, mask, spread=12.0, bulge=6.0, seed=3, order=0.0, alpha=None,
               net=None, net_w=0.0):
    """A plausible crown depth, which the card does not have.

    That was the whole trouble with shading: the card is flat, and one thing
    shades another only when one LIES ABOVE the other. Without depth any
    calculation degenerates into a derivative of the silhouette — now an
    outline, now a black middle.

    The depth need not be guessed, it can be invented, and that is enough.
    In a real branch the leaves sit at different depths more or less at
    random, therefore:

    spread — every leaf of the segmentation gets its own random depth offset
             (uniform 0…1 times spread, px of height). The seed is the same
             as for the tilts on the normals tab, so depth and normal
             describe one and the same branch;
    bulge  — the leaf's own convexity from edge to middle (bulge·t, px);
    order  — optional slope: leaves nearer the edge of the crown are brought
             forward, those deep inside go back (order·spread·coverage,
             coverage = alpha blurred 32 px relative to its maximum; needs
             alpha).

    net, net_w — network depth (H, W), larger = closer, and its weight 0…1.
    The leaf order is then not invented but taken from the network: each
    piece gets its own mean depth (leaf_by_label). The random draw is not
    thrown away but mixed in, (1 − net_w)·random + net_w·network: the
    network confidently says which branch is closer, but inside one branch
    neighbouring leaves look the same to it — flat leaves get equal depth —
    and without the scatter they would stick together in one plane again.

    The steps at leaf boundaries are NOT smoothed: they are exactly what
    gives the contact shadow where one leaf goes under another.

    nrm: Normals after segment(); mask: bool (H, W). Returns float32 (H, W)
    height in px (larger = closer to the viewer), 0 on the background.
    """
    rng = np.random.default_rng(int(seed))
    off = rng.random(nrm.nlab + 1).astype(np.float32)
    off[0] = 0.0
    if net is not None and net_w > 0:
        # порядок листьев не выдумывается, а берётся у сети: каждому куску
        # разбиения — его собственная средняя глубина. Случайность при этом
        # не выбрасывается, а подмешивается: сеть уверенно говорит, какая
        # ветка ближе, но внутри одной ветки соседние листья ей одинаковы,
        # и без разброса они снова лягут в одну плоскость
        w = float(np.clip(net_w, 0, 1))
        off = (1.0 - w) * off + w * leaf_by_label(nrm.lab, nrm.nlab, net)
        off[0] = 0.0
    z = off[nrm.lab] * float(spread)
    t = getattr(nrm, 't', None)
    if t is not None and bulge:
        z = z + float(bulge) * np.clip(t, 0, 1)
    if order and alpha is not None:
        cov = fast_blur(alpha, 32.0)
        z = z - float(order) * spread * (cov / max(float(cov.max()), 1e-6))
    return np.where(mask, z, 0.0).astype(np.float32)


def hbao(z, mask, normal=None, radius=24.0, dirs=10, strength=1.0, tilt=0.4,
         bias=0.05, want_bent=False):
    """Horizon-based ambient occlusion (HBAO) on this depth.

    From a point we look around in a circle and find how high the
    neighbouring foliage rises above it. The share of closed sky in a
    direction is the sine of the horizon angle. Method after Bavoil, Sainz &
    Dimitrov, "Image-Space Horizon-Based Ambient Occlusion" (SIGGRAPH 2008).
    Two corrections against the usual calculation on a solid texture:

    * empty alpha is SKY, not a low surface. Through a hole in the crown the
      light passes entirely, so a leaf next to a gap stays light;
    * the count is made not from the plane of the card but from the plane
      of the leaf itself, given by the normal map: on a tilted leaf part of
      the sky is closed by the leaf itself. This is the honest form of
      "shading by tilt" — as a separate map it was just the blue channel of
      the normal, while inside HBAO it sits in its proper place.

    z: float (H, W) height in px (leaf_depth); mask: bool (H, W).
    normal: (H, W, 3) leaf normals (image axes, y down) or None; used only
    for the tilt correction, with weight tilt 0…1 (n_z floored at 0.15).
    radius: px, the farthest sample; per direction samples are taken at
    1, 2, 3, 5, 8, 13, 21, 34, 55 px up to radius (sparse, so very thin
    occluders can be missed). dirs: number of directions. strength: factor
    on the result. bias: horizon slope (tan) ignored, against self-occlusion
    by small steps.

    Returns occ float32 (H, W), 0 (open sky) … 1, 0 on the background; with
    want_bent also bent, float32 (H, W, 3) unit vectors towards the middle
    of the open sky (image axes; (0, 0, 1) where undefined). The bent normal
    is taken before the tilt correction.

    Runs where z is: on the video card if crown_occlusion moved it there.
    """
    xp = X.xp(z)
    steps = [r for r in (1, 2, 3, 5, 8, 13, 21, 34, 55) if r <= max(radius, 1)]
    if not steps:
        steps = [1]
    mf = mask.astype(xp.float32)
    occ = xp.zeros_like(z)
    bx = by = bz = None
    if want_bent:
        bx, by, bz = (xp.zeros_like(z), xp.zeros_like(z), xp.zeros_like(z))
    nx = ny = nz = None
    if normal is not None and tilt > 0:
        nx, ny = normal[..., 0], normal[..., 1]
        nz = xp.maximum(normal[..., 2], 0.15)
    for k in range(int(dirs)):
        ang = 2.0 * np.pi * k / dirs
        dx, dy = np.cos(ang), np.sin(ang)
        best = xp.full_like(z, -1.0)
        for r in steps:
            ox, oy = int(round(dx * r)), int(round(dy * r))
            if ox == 0 and oy == 0:
                continue
            zs = _shift(z, oy, ox, 0.0)
            ms = _shift(mf, oy, ox, 0.0)
            e = (zs - z) / np.hypot(ox, oy)
            best = xp.maximum(best, xp.where(ms > 0.5, e, -1.0))
        best = xp.maximum(best - bias, 0.0)
        sin_h = best / xp.sqrt(1.0 + best * best)
        if want_bent:
            # середина открытой части неба в этом направлении: от горизонта
            # до зенита. Вес — сколько неба в этом направлении видно
            half = 0.5 * (xp.arcsin(xp.clip(sin_h, 0, 1)) + np.pi * 0.5)
            w = 1.0 - sin_h
            # знак обратный: _shift берёт значение из точки p − o, то есть
            # горизонт в этом проходе щупается в направлении МИНУС (dx, dy).
            # Для самой окклюзии это незаметно — направления обходятся по
            # кругу и сумма симметрична, — а вот направление bent normal
            # без этого разворачивается ровно в сторону затеняющего листа
            c = xp.cos(half) * w
            bx -= c * dx
            by -= c * dy
            bz += xp.sin(half) * w
        if nx is not None:
            slope = -(nx * dx + ny * dy) / nz          # уклон самого листа
            sin_t = slope / xp.sqrt(1.0 + slope * slope)
            sin_h = xp.maximum(sin_h - tilt * sin_t, 0.0)
        occ += sin_h
    occ = xp.clip(occ / max(int(dirs), 1) * strength, 0.0, 1.0)
    occ = xp.where(mask, occ, 0.0).astype(xp.float32)
    if not want_bent:
        return occ
    b = xp.stack([bx, by, bz], -1)
    L = xp.sqrt((b * b).sum(-1, keepdims=True))
    b = xp.where(L > 1e-6, b / xp.maximum(L, 1e-6),
                 xp.asarray(np.array([0.0, 0.0, 1.0], np.float32)))
    return occ, b.astype(xp.float32)


def bend_normal(n, bent, amount=1.0):
    """Mix into the normal the direction the light actually reaches the point from.

    An ordinary normal looks where the surface is turned, and does not care
    that half the sky above it is closed by a neighbouring leaf. A bent
    normal looks at the middle of the part of the sky that stayed open. For
    diffuse light that is the right direction, so in gaps and under an
    overhanging leaf the lighting stops being flat — and nothing has to be
    done in the engine, everything already lies in the normal map.

    Careful: after this the map is no longer purely geometric. For parallax
    and for anything that reconstructs relief from the normal it is better
    to keep a separate ordinary copy.

    n, bent: float (H, W, 3) unit vectors (bent from crown_occlusion with
    want_bent). amount: 0…1, linear blend weight, then renormalised.
    Returns float32 (H, W, 3); n itself if amount ≤ 0 or bent is None.
    """
    if amount <= 0 or bent is None:
        return n
    a = float(np.clip(amount, 0, 1))
    out = n * (1.0 - a) + bent * a
    L = np.sqrt((out * out).sum(-1, keepdims=True))
    return (out / np.maximum(L, 1e-6)).astype(np.float32)


def behind_leaves(z, mask, reach=14.0, spread=12.0):
    """Silhouettes of the leaves that lie BEHIND this leaf.

    Against the light this is the main thing you see: through the upper leaf
    a dark spot shows in the shape of another leaf — the one behind it. It
    is not a shadow cast on a neighbour: it is a leaf that blocked the sun,
    and its silhouette is seen through the blade of the front leaf.

    On the card the leaf that goes behind the front one is not drawn — it
    was covered. But we already have depth, and continuing it under the
    neighbour is easy: if within reach there is a leaf noticeably lower,
    then it passes under us. The lower it is, the darker.

    Computed without a light direction, so unlike a cast shadow it bakes
    into a map perfectly: what lies behind a leaf lies behind it under any
    sun.

    z: float (H, W) height in px (leaf_depth); mask: bool (H, W); reach: px,
    radius of the (circular, 12 directions × 4 distances) neighbourhood
    searched for the lowest opaque point; spread: px of height that counts
    as fully behind (use the leaf_depth spread). Returns float32 (H, W),
    0…1: (z − lowest z nearby) / spread, 0 on the background.
    """
    big = float(z.max()) + 1.0
    zz = np.where(mask, z, big)
    # круглая окрестность, а не квадратная: квадратное окно оставляет на
    # карте прямоугольные заплаты, которых в листве не бывает
    zmin = zz.copy()
    R = max(float(reach), 1.0)
    for k in range(12):
        ang = 2.0 * np.pi * k / 12
        dx, dy = np.cos(ang), np.sin(ang)
        for q in (0.35, 0.6, 0.82, 1.0):
            ox, oy = int(round(dx * R * q)), int(round(dy * R * q))
            if ox or oy:
                np.minimum(zmin, _shift(zz, oy, ox, big), out=zmin)
    d = (z - zmin) / max(float(spread), 1e-3)
    return np.where(mask, np.clip(d, 0.0, 1.0), 0.0).astype(np.float32)


def crown_occlusion(src, nrm=None, normal=None, spread=12.0, bulge=6.0,
                    radius=24.0, seed=3, tilt=0.4, order=0.0, dirs=10,
                    quality=1024, want_bent=False, net=None, net_w=0.0):
    """Raw crown occlusion 0…1: 0 — the point sees the whole sky, 1 — walled in.
    Both the AO map and the weakening of translucency are made from it: both
    are one and the same question, "how much foliage lies over this point".

    It builds the invented depth (leaf_depth) and runs hbao on it. A leaf
    that lies lower — behind others — has part of its sky covered by the
    leaves above it, receives less ambient light and so must come out
    darker; on a flat card without depth there would be nothing to cover it.

    On big textures the horizon walk runs on a reduced copy: a hundred
    shifts of 4 megapixels is seconds, while a contact shadow a few pixels
    wide does not suffer from half size. quality: largest working side in px
    (min 128); a bigger texture uses every k-th pixel, k = ⌈side / quality⌉,
    with z and radius divided by k, and is scaled back up bilinearly through
    an 8-bit image (occlusion quantised to 1/255, bent vectors to ~1/127).

    src: Source; nrm: Normals after segment() (if None or not segmented,
    returns zeros and a flat bent map). normal: leaf normals for the tilt
    correction, or None. spread, bulge, seed, order, net, net_w: as in
    leaf_depth; radius (px), dirs, tilt: as in hbao. Returns occ float32
    (H, W), 0…1, or (occ, bent (H, W, 3)) with want_bent.
    """
    m = src.mask
    if nrm is None or getattr(nrm, 'lab', None) is None:
        z0 = np.zeros(m.shape, np.float32)
        return (z0, np.dstack([z0, z0, np.ones_like(z0)])) if want_bent else z0
    h0, w0 = m.shape
    z = leaf_depth(nrm, m, spread, bulge, seed, order, src.alpha, net, net_w)
    k = max(1, int(np.ceil(max(h0, w0) / max(quality, 128))))

    def walk(zz, mm, nn, rad):
        """hbao on the video card when it is on, the result on the processor."""
        dn = None if nn is None else X.to_dev(np.ascontiguousarray(nn))
        r = hbao(X.to_dev(np.ascontiguousarray(zz)), X.to_dev(np.ascontiguousarray(mm)),
                 dn, rad, dirs, 1.0, tilt, want_bent=want_bent)
        return tuple(X.to_host(v) for v in r) if want_bent else X.to_host(r)
    if k == 1:
        return walk(z, m, normal, radius)
    zk, mk = z[::k, ::k] / k, m[::k, ::k]
    nk = normal[::k, ::k] if normal is not None else None
    r = walk(zk, mk, nk, radius / k)
    occ, bent = r if want_bent else (r, None)

    def up(a):
        """Scale a 0…1 (h, w) map up to (h0, w0), bilinear, via an 8-bit image."""
        im = Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8))
        return np.asarray(im.resize((w0, h0), Image.BILINEAR)).astype(np.float32) / 255.0

    occ = np.where(m, up(occ), 0.0).astype(np.float32)
    if not want_bent:
        return occ
    b = np.stack([up(bent[..., i] * 0.5 + 0.5) * 2.0 - 1.0 for i in range(3)], -1)
    L = np.sqrt((b * b).sum(-1, keepdims=True))
    return occ, (b / np.maximum(L, 1e-6)).astype(np.float32)


def ao_map(src, strength=1.0, gamma=1.0, contrast=1.0, occ=None, **kw):
    """Foliage ambient occlusion: how much sky a point sees.

    Computed on the invented crown depth by a horizon walk. The former three
    terms — alpha density, leaf tilt and the split into leaves — were not
    occlusion either separately or together: the first outlined the
    silhouette, the second was literally the blue channel of the normal map,
    the third turned every leaf into a roof tile. The share of closed sky
    can only be computed from depth, so depth appears here first.

    occ: a ready crown_occlusion result, or None to compute it with **kw.
    Then a = 1 − strength·occ, contrast around 0.5, gamma as a^(1/gamma).
    Returns float32 (H, W), 0…1, 1 = fully open (also on the background).
    """
    m = src.mask
    if occ is None:
        occ = crown_occlusion(src, **kw)
    a = np.clip(1.0 - strength * occ, 0.0, 1.0)
    if contrast != 1.0:
        a = np.clip((a - 0.5) * contrast + 0.5, 0.0, 1.0)
    if abs(gamma - 1.0) > 1e-3:
        a = np.power(np.clip(a, 0, 1), 1.0 / max(gamma, 0.05))
    return np.where(m, a, 1.0).astype(np.float32)


# ================================================================ просвет ==
def vein_map(src):
    """Veins and rims of the leaf — everything elongated that differs from
    the blade. Ridges of both polarities are taken: on some textures veins
    are drawn lighter than the blade, on others darker, and against the
    light both are equally opaque because they are thickenings.

    Sato tubeness filter (Sato et al., 1998; skimage.filters.sato), sigmas
    1, 2 px, bright and dark ridges each scaled to its maximum, the larger
    taken, masked, blurred 0.8 px. Returns float32 (H, W), 0…1.

    On the video card when it is on (_sato, the same formulas as skimage);
    the result comes back to the processor.
    """
    L = X.to_dev(np.ascontiguousarray(src.lum, np.float32))
    xp, nd = X.xp(L), X.ndi(L)
    r = _sato(L, (1, 2), black_ridges=False)
    d = _sato(L, (1, 2), black_ridges=True)
    v = xp.maximum(r / max(float(r.max()), 1e-9), d / max(float(d.max()), 1e-9))
    v = xp.where(X.to_dev(np.ascontiguousarray(src.mask)), v, 0.0)
    return X.to_host(nd.gaussian_filter(v, 0.8).astype(xp.float32))


def _sato(img, sigmas, black_ridges):
    """skimage.filters.sato for a 2-D picture, written with array
    operations so that it also runs on the video card.

    The Hessian is taken with Gaussian derivatives as skimage does
    (_hessian_matrix_with_gaussian): two first-derivative passes with
    σ/√2 each, truncate 100 when σ ≤ 1 (8 otherwise), 'reflect' edges.
    In 2-D the tubeness is σ² · max(λ₁, 0), λ₁ the larger eigenvalue of
    the Hessian; the result is the largest over sigmas. black_ridges False
    looks for light ridges (the picture is negated first, as in skimage).
    """
    xp, nd = X.xp(img), X.ndi(img)
    im = xp.asarray(img, xp.float32)
    if not black_ridges:
        im = -im
    best = xp.zeros_like(im)
    for sg in sigmas:
        s2 = float(sg) / np.sqrt(2.0)
        tr = 8 if float(sg) > 1 else 100

        def g(a, o):
            """Gaussian with derivative order o along (rows, columns)."""
            return nd.gaussian_filter(a, s2, order=o, mode='reflect', truncate=tr)
        gr, gc = g(im, (1, 0)), g(im, (0, 1))
        hrr, hrc, hcc = g(gr, (1, 0)), g(gr, (0, 1)), g(gc, (0, 1))
        lam = (hrr + hcc) / 2 + xp.sqrt(hrc ** 2 + ((hrr - hcc) / 2) ** 2)
        best = xp.maximum(best, float(sg) ** 2 * xp.maximum(lam, 0))
    return best


def trans_color(src, tint=0.35):
    """Colour of the light filter the leaf acts as. Computed separately from
    the translucency strength: it depends on a single knob only, yet costs
    more than all the rest put together.

    The filter is the pixel colour divided by its largest channel (hue at
    full brightness), raised to 1 + 2·tint to deepen it; tint 0…1.
    Returns float32 (H, W, 3), 0…1.
    """
    rgb = src.rgba[..., :3]
    mx = np.maximum(rgb.max(-1), 1e-4)
    filt = np.clip(rgb / mx[..., None], 0.0, 1.0)
    if tint > 0:
        filt = np.power(filt, 1.0 + 2.0 * np.clip(tint, 0, 1))
    return filt.astype(np.float32)


def translucency(src, nrm=None, strength=1.0, base=0.15, veins=0.9,
                 layers=0.5, pigment=0.7, edge=0.15, branch=1.0, tint=0.35,
                 gamma=1.0, contrast=1.0, color=None, vein=None, occ=None,
                 behind=1.0, back=None):
    """How much light passes right through the leaf — and of what colour.

    You have to look at a real crown against the sun. There the WHOLE blade
    of the leaf glows, evenly and brightly, and what comes out dark on it is
    something else: veins and stalks — what is thicker than the blade — and
    those leaves behind which lies another leaf. There is no glowing outline
    along the edge in nature: at the edge the blade is exactly as thin as in
    the middle.

    So the thickness here is made up not from the distance to the edge of
    the silhouette but from these things (weights, 0 turns a term off):

    base    — the blade itself, the same over the whole leaf;
    veins   — veins and rims: thickenings, light does not pass through them
              (vein map ×2; vein: a precomputed vein_map or None);
    layers  — how much foliage lies on top, from the same crown occlusion as
              for AO: the overall depth in the crown (occ ×1.2; occ: the
              crown_occlusion result or None);
    behind  — leaves lying BEHIND this leaf. Against the light this is the
              main thing you see: through the front blade the dark
              silhouette of what blocked the sun shows (back ×2.2; back: the
              behind_leaves result or None);
    pigment — how dark the leaf is: a dark leaf is packed more densely with
              pigment, and that is also exactly how background leaves are
              drawn on the texture (1 − luminance stretched 3rd…97th
              percentile);
    edge    — a small addition at the very edge, where the leaf really is
              thinner (edge·t: thickness added towards the middle).

    Transmission is T = exp(−1.3·thickness) (Beer–Lambert), divided by its
    98th percentile inside the mask (at least 0.35). branch 0…1: pieces
    counted as branches (Normals.leafy_used) do not transmit, T ×(1 −
    branch·(1 − leafiness)). Then strength, contrast around 0.5 and gamma
    (T^(1/gamma)). tint: see trans_color; color: its precomputed result.

    Returns (filt, T): filt float32 (H, W, 3) colour filter 0…1, T float32
    (H, W) transmission 0…1, 0 on the background.
    """
    A = src.alpha
    m = src.mask
    depth = np.full(A.shape, float(base), np.float32)

    if veins > 0:
        v = vein_map(src) if vein is None else vein
        depth = depth + veins * 2.0 * v

    if layers > 0 and occ is not None:
        depth = depth + layers * 1.2 * occ

    if behind > 0 and back is not None:
        depth = depth + behind * 2.2 * back

    if pigment > 0:
        L = src.lum
        lo, hi = np.percentile(L[m], [3, 97]) if m.any() else (0.0, 1.0)
        Ln = np.clip((L - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        depth = depth + pigment * (1.0 - Ln)

    t = getattr(nrm, 't', None)
    if t is not None and edge != 0:
        depth = depth + edge * np.clip(t, 0, 1)

    T = np.exp(-1.3 * depth)
    # тянем к рабочему диапазону: у тёмной текстуры просвет иначе выходит
    # почти чёрным, хотя относительный рисунок в нём правильный
    if m.any():
        T = T / max(float(np.percentile(T[m], 98)), 0.35)

    lf = getattr(nrm, 'leafy_used', None)      # ветки и черешки не просвечивают
    if lf is not None and branch > 0:
        leafy = np.clip(lf[nrm.lab], 0.0, 1.0)
        T = T * (1.0 - branch * (1.0 - leafy))

    T = np.where(m, T, 0.0) * strength
    if contrast != 1.0:
        T = np.clip((T - 0.5) * contrast + 0.5, 0.0, 1.0)
    T = np.clip(T, 0, 1)
    if abs(gamma - 1.0) > 1e-3:
        T = np.power(T, 1.0 / max(gamma, 0.05))

    filt = trans_color(src, tint) if color is None else color
    return filt, np.clip(T, 0, 1).astype(np.float32)


def backlit(rgba, normal, trans_rgb, trans_a, light, view=(0.0, 0.0, 1.0),
            power=3.0, ambient=0.10, front=1.0):
    """How the crown looks against the light (preview render).

    Light that has passed right through goes mostly forward, along the ray,
    not in all directions: so the translucency is visible only when the
    source is roughly behind the leaf. This is the standard approximation
    games compute — the share of transmitted light is taken as
    (view · −light) to some power (cf. Barré-Brisebois & Bouchard, GDC 2011).

    rgba: float (H, W, 4), 0…1; normal: (H, W, 3); trans_rgb, trans_a: the
    translucency result; light: unit vector from the surface towards the
    light, view: towards the viewer (same axes as the normals); power:
    sharpness of the forward lobe; ambient, front: front lighting
    C·(ambient + front·max(n·light, 0)). Returns float (H, W, 3), 0…1,
    composited by alpha over a dark grey-green background.
    """
    L = np.asarray(light, np.float32)
    V = np.asarray(view, np.float32)
    n = normal
    d = np.clip((n * L).sum(-1, keepdims=True), 0, 1)
    back = np.clip(float(np.dot(V, -L)), 0.0, 1.0) ** max(power, 0.1)
    C = np.clip(rgba[..., :3], 0, 1)
    lit = C * (ambient + front * d)
    glow = trans_rgb * (trans_a[..., None] * back)
    a = rgba[..., 3:4]
    return np.clip((lit + glow) * a + np.array([0.09, 0.10, 0.09], np.float32) * (1 - a), 0, 1)
