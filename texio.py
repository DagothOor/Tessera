# -*- coding: utf-8 -*-
"""
Texelith — reading and writing texture files.

Kept apart from the window (Texelith.py) so that batch processing can run
without it: project.py processes a whole folder from the command line with
the same readers and writers the window uses.

Texelith — чтение и запись файлов текстур, отдельно от окна: пакетная
обработка проекта из командной строки пользуется теми же функциями.
"""

import os

import numpy as np
from PIL import Image

import foliage as FO

BUILD = 25          # версия набора файлов, проверяется при запуске

# окончания готовых карт: пакет не берёт такие файлы как исходники
MAP_SUFFIXES = ('_n', '_nh', '_spec', '_ao', '_h', '_rough', '_metal', '_fixed', '_fix',
                '_sss', '_basecolor', '_orm', '_smooth', '_ms', '_specular', '_bent',
                '_gloss')
TEX_EXT = ('.dds', '.png', '.tga', '.bmp', '.jpg', '.jpeg', '.tif', '.tiff')


# ------------------------------------------------------------ запись карт --
def gray_rgba(x, alpha=None):
    """Grey map (H, W) in 0…1 to uint8 RGBA (H, W, 4), grey in RGB.

    alpha: optional (H, W) 0…1; opaque if None.
    """
    g = np.clip(x, 0, 1)
    rgb = np.stack([g] * 3, -1)
    a = np.ones_like(g) if alpha is None else np.clip(alpha, 0, 1)
    return (np.concatenate([rgb, a[..., None]], -1) * 255 + 0.5).astype(np.uint8)


def rgb_rgba(x, alpha=None):
    """Colour image (H, W, 3) in 0…1 to uint8 RGBA (H, W, 4).

    alpha: optional (H, W) 0…1; opaque if None.
    """
    rgb = np.clip(x, 0, 1)
    a = np.ones(rgb.shape[:2], np.float32) if alpha is None else np.clip(alpha, 0, 1)
    return (np.concatenate([rgb, a[..., None]], -1) * 255 + 0.5).astype(np.uint8)


PLAIN_EXT = ('.png', '.tga', '.tif', '.tiff', '.bmp', '.jpg', '.jpeg')


def dxt(opts):
    """The ``compress`` value for the writers from batch settings: False
    without compression, else (algorithm, level) — see foliage._bc3."""
    if not opts.get('compress', True):
        return False
    return (opts.get('dxt_algo', 'quicktex'), int(opts.get('dxt_level', 14)))


def save_rgba(rgba, path, compress, normal=False):
    """Write an RGBA image as DDS with mipmaps or as a plain picture.

    rgba: uint8 (or 0…255 numeric) array (H, W, 4). path: output file.
    compress: DDS block compression on/off (passed to foliage._write_dds).
    normal: RGB holds a normal map (see below).

    The format is chosen by the extension: .dds is DDS with mipmaps, the rest
    (PLAIN_EXT) are ordinary pictures; JPEG and BMP drop the alpha. Previously
    DDS was silently written into a .png file.

    The mip chain is taken down to 1x1 even with compression: many viewers and
    file-manager shells consider a file with a truncated chain broken and make
    no thumbnail. Mips are 2×2 box averages.

    normal: RGB contains a normal map, so on every level the vectors are
    renormalised to unit length. Plain averaging shortens them, and the relief
    flattens out at a distance (the same is done in foliage.save_normal_dds).
    """
    if os.path.splitext(path)[1].lower() in PLAIN_EXT:
        im = Image.fromarray(np.ascontiguousarray(rgba.astype(np.uint8)), 'RGBA')
        if os.path.splitext(path)[1].lower() in ('.jpg', '.jpeg', '.bmp'):
            im = im.convert('RGB')
        im.save(path)
        return
    levels, cur = [], rgba.astype(np.float32)
    while True:
        levels.append(np.clip(cur + 0.5, 0, 255).astype(np.uint8))
        if min(cur.shape[0], cur.shape[1]) <= 1:
            break
        hh, ww = max(1, cur.shape[0] // 2), max(1, cur.shape[1] // 2)
        cur = cur[:hh * 2, :ww * 2].reshape(hh, 2, ww, 2, 4).mean(axis=(1, 3))
        if normal:
            v = cur[..., :3] / 127.5 - 1.0
            v = v / np.maximum(np.sqrt((v * v).sum(-1, keepdims=True)), 1e-6)
            cur = np.concatenate([(v + 1.0) * 127.5, cur[..., 3:4]], -1)
    FO._write_dds(levels, path, compress)


def save_gray16(x, path, compress=False):
    """Write a greyscale map x (H, W), values 0…1.

    PNG is written as 16 bit: 256 levels are too few for a height map, and
    steps become visible in a normal map derived from it. DDS (and other
    formats) are 8 bit, like every other map, via save_rgba.
    """
    if os.path.splitext(path)[1].lower() == '.png':
        a = (np.clip(x, 0, 1) * 65535 + 0.5).astype(np.uint16)
        Image.fromarray(a).save(path)
    else:
        save_rgba(gray_rgba(x), path, compress)


def resize_to3(a, shape):
    """Resize a colour image to the given size.

    a: float (H, W, 3) in 0…1, or None. shape: (height, width). Returns float32
    in 0…1 (Lanczos, via 8-bit), or a itself if the size already matches.
    """
    if a is None or a.shape[:2] == tuple(shape):
        return a
    im = Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8))
    return np.asarray(im.resize((shape[1], shape[0]), Image.LANCZOS)).astype(np.float32) / 255


def read_rgba(path):
    """Read an image file into float32 RGBA (H, W, 4) in 0…1.

    16-bit and floating-point images (usually height maps) are converted
    properly: convert('RGBA') would clip everything above 255 to white. They
    come back as grey RGB with opaque alpha.
    """
    im = Image.open(path)
    im.load()
    if im.mode in ('I;16', 'I;16B', 'I;16L', 'I', 'F'):
        g = np.asarray(im).astype(np.float32)
        hi = float(g.max()) if g.size else 1.0
        g = g / (65535.0 if im.mode.startswith('I;16') or hi > 255.0
                 else (255.0 if im.mode == 'I' or hi > 1.0 else 1.0))
        g = np.clip(g, 0, 1)
        return np.stack([g, g, g, np.ones_like(g)], -1)
    return np.array(im.convert('RGBA')).astype(np.float32) / 255


def resize_signed(a, shape):
    """Resize a signed map (−1…1, the height hint) to shape (height, width).

    resize_to clips to 0…1, and everything negative would be lost, so the map
    is mapped to 0…1 first and back afterwards. Returns float32 in −1…1 (8-bit
    precision), or a itself if the size already matches.
    """
    if a is None or a.shape[:2] == tuple(shape):
        return a
    return resize_to(np.clip(a, -1, 1) * 0.5 + 0.5, shape) * 2.0 - 1.0


def resize_to(a, shape, nearest=False):
    """Resize a grey or boolean map to shape (height, width).

    Grey maps (0…1) are resized with Lanczos through 8 bit and returned as
    float32 in 0…1. Boolean masks (or nearest=True) use nearest-neighbour;
    boolean input comes back boolean. Returns a unchanged if the size matches.
    """
    if a is None or a.shape[:2] == tuple(shape):
        return a
    was_bool = a.dtype == bool
    im = Image.fromarray((np.clip(a.astype(np.float32), 0, 1) * 255 + 0.5).astype(np.uint8))
    im = im.resize((shape[1], shape[0]), Image.NEAREST if (nearest or was_bool) else Image.LANCZOS)
    out = np.asarray(im).astype(np.float32) / 255.0
    return out > 0.5 if was_bool else out


def resize_wrap(a, h, w):
    """Resize H×W or H×W×C to h×w treating the picture as a tile.

    The picture is padded by wrapping before the bilinear resample and the
    margin is cut off after, so pixels at the edge are interpolated from
    their neighbours on the other side and the result still tiles."""
    a = np.asarray(a, np.float32)
    H, W = a.shape[:2]
    if (H, W) == (h, w):
        return a.copy()
    m = 4
    pad = ((m, m), (m, m)) + ((0, 0),) * (a.ndim - 2)
    big = np.pad(a, pad, mode='wrap')
    sy, sx = h / float(H), w / float(W)
    th, tw = int(round((H + 2 * m) * sy)), int(round((W + 2 * m) * sx))
    chans = [big] if a.ndim == 2 else [big[..., c] for c in range(a.shape[2])]
    out = []
    for c in chans:
        im = Image.fromarray(np.ascontiguousarray(c), 'F').resize((tw, th), Image.BILINEAR)
        r = np.asarray(im, np.float32)
        oy, ox = int(round(m * sy)), int(round(m * sx))
        out.append(r[oy:oy + h, ox:ox + w])
    return out[0] if a.ndim == 2 else np.stack(out, -1)


def write_value(v, p, compress, alpha=None, as_normal=None):
    """Write the value of a graph output to a file: _nh, normal, colour or grey.

    v: float32 array. H×W×4 is written as RGBA 0…1 — the _nh map (normal and
    height in alpha, mips renormalised) or a _spec map (colour and exponent);
    it is told apart by being a normal only if its RGB decodes to unit
    vectors. H×W×3 with values below −0.01 is taken as a signed normal in
    −1…1 and written by foliage.save_normal_dds; other H×W×3 as colour 0…1
    (with ``alpha`` if given — a replacement diffuse keeps the source alpha);
    H×W as grey 0…1. p: output path (.dds or a plain picture format);
    compress: DDS block compression. as_normal: for H×W×4 — True/False says
    whether it is a normal map (mips renormalised), None — tell it by the
    values as above (a packed metal/roughness/AO map could pass for one).
    """
    v = np.asarray(v, np.float32)
    if alpha is not None and v.ndim == 3 and v.shape[2] == 4:
        v = v[..., :3]          # диффуз с альфой: альфа — та, что передана, мипы по ней
    if v.ndim == 3 and v.shape[2] == 4:
        n = v[..., :3] * 2.0 - 1.0
        is_n = (abs(float(np.sqrt((n * n).sum(-1)).mean()) - 1.0) < 0.08
                if as_normal is None else bool(as_normal))
        save_rgba((np.clip(v, 0, 1) * 255 + 0.5).astype(np.uint8), p, compress,
                  normal=is_n)
    elif v.ndim == 3 and float(v.min()) < -0.01:
        if os.path.splitext(p)[1].lower() == '.dds':
            FO.save_normal_dds(v, p, compress)
        else:
            save_rgba(rgb_rgba(np.clip(v * 0.5 + 0.5, 0, 1)), p, compress)
    elif v.ndim == 3:
        rgba = rgb_rgba(np.clip(v[..., :3], 0, 1), alpha)
        if (alpha is not None and os.path.splitext(p)[1].lower() == '.dds'
                and float(np.min(alpha)) < 0.5):
            # диффуз с дырками в альфе (листва, решётки): мипы, сохраняющие
            # площадь непрозрачного, иначе вдали крона лысеет
            FO.save_diffuse_dds(rgba, p, compress, keep_coverage=True)
        else:
            save_rgba(rgba, p, compress)
    else:
        save_rgba(gray_rgba(np.clip(v, 0, 1)), p, compress)
