# -*- coding: utf-8 -*-
"""Classic image filters and adjustments (as in Photoshop and GIMP) for the
node editor.

Every function here takes one grey plane H×W (float32, 0…1 for maps) and
returns a plane of the same size; colour pictures are handled by the caller
(nodes._per_channel runs a filter on each channel, the colour adjustments
take H×W×3 themselves). All work goes through xpu, so the same code runs on
the processor (NumPy/SciPy) and on the graphics card (CuPy).

The texture is a tile: every neighbourhood wraps round the edges (mode
'wrap', FFT), so a seamless texture stays seamless. Sizes are in pixels of
the array passed in; the nodes scale them for the reduced preview copy.

Классические фильтры и коррекции (как в Photoshop и GIMP). Всё на торе:
соседи берутся с другой стороны текстуры, шов не появляется.
"""

import math

import numpy as np

import xpu as X

BUILD = 25          # версия набора файлов, проверяется при запуске


# ----------------------------------------------------------- служебное ---
def _disk(r, like=None):
    """Boolean round footprint of radius r px (at least 3×3); on the card if
    ``like`` lives there (CuPy's filters want their footprint on the card)."""
    r = max(1, int(round(r)))
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    fp = (x * x + y * y) <= r * r + r       # +r: круг без «зубцов» на осях
    return X.to_dev(fp) if like is not None and X.is_dev(like) else fp


def noise_sigma(a):
    """Noise level of a plane: median absolute difference of neighbouring
    pixels, scaled to a Gaussian σ (robust: edges barely move the median)."""
    xp = X.xp(a)
    d = xp.abs(xp.diff(a, axis=1)).ravel()
    return float(xp.median(d)) / (0.6745 * math.sqrt(2.0))


def fft_shift(h, dy, dx):
    """Shift a plane by (dy, dx) pixels on the torus, sub-pixel exact
    (multiplication by a phase ramp in the frequency domain)."""
    xp = X.xp(h)
    H, W = h.shape
    fy = xp.fft.fftfreq(H).astype(xp.float32)[:, None]
    fx = xp.fft.rfftfreq(W).astype(xp.float32)[None, :]
    ph = xp.exp(-2j * math.pi * (fy * float(dy) + fx * float(dx)))
    return xp.fft.irfft2(xp.fft.rfft2(h) * ph, s=(H, W)).astype(xp.float32)


def luma(rgb):
    """Brightness of an sRGB picture as Photoshop takes it for grey
    (Rec. 709 weights on the stored values)."""
    return (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2])


# ---------------------------------------------------------- размытие -----
def box_blur(h, r):
    """Box Blur: the mean over a (2r+1)² square."""
    nd = X.ndi(h)
    return nd.uniform_filter(h, size=2 * max(1, int(round(r))) + 1, mode='wrap')


def motion_blur(h, angle, dist):
    """Motion Blur: the mean along a line of length ``dist`` px through the
    pixel, at ``angle`` degrees (0 — horizontal, counter-clockwise). Done as
    a convolution in the frequency domain with the line drawn into a
    texture-sized kernel, so the cost does not grow with the length."""
    xp = X.xp(h)
    H, W = h.shape
    dist = float(dist)
    if dist < 1.0:
        return h
    n = max(2, int(math.ceil(dist)) * 2 + 1)
    t = np.linspace(-dist / 2.0, dist / 2.0, n)
    a = math.radians(float(angle))
    ys, xs = -t * math.sin(a), t * math.cos(a)
    k = np.zeros((H, W), np.float32)
    # билинейно разносим точки линии по четырём соседям — линия без ступенек
    for y, x in zip(ys, xs):
        y0, x0 = math.floor(y), math.floor(x)
        fy, fx = y - y0, x - x0
        for yy, xx, w in ((y0, x0, (1 - fy) * (1 - fx)), (y0, x0 + 1, (1 - fy) * fx),
                          (y0 + 1, x0, fy * (1 - fx)), (y0 + 1, x0 + 1, fy * fx)):
            k[yy % H, xx % W] += w
    k /= k.sum()
    K = xp.fft.rfft2(X.to_dev(k) if X.is_dev(h) else k)
    return xp.fft.irfft2(xp.fft.rfft2(h) * K, s=(H, W)).astype(xp.float32)


def surface_blur(h, r, threshold):
    """Surface Blur: averages up to radius r but not across steps higher than
    ``threshold`` (0…1). Self-guided filter (He, Sun & Tang 2010) — one pass;
    a second pass lets the edge leak back."""
    import matgen as MG
    rb = max(1, int(round(float(r))))
    eps = max(float(threshold), 1e-3) ** 2
    return MG.guided_filter(h, h, rb, eps).astype(X.xp(h).float32)


# ---------------------------------------------------------- резкость -----
def unsharp_mask(h, amount, r, threshold):
    """Unsharp Mask: h + amount·(h − blur(h, r)) where the difference exceeds
    ``threshold`` (0…1); amount 1 = 100 %. Clipped to 0…1."""
    import matgen as MG
    xp = X.xp(h)
    d = h - MG.fast_blur(h, max(float(r), 0.3), mode='wrap')
    if threshold > 0:
        d = xp.where(xp.abs(d) >= float(threshold), d, 0.0)
    return xp.clip(h + float(amount) * d, 0, 1).astype(xp.float32)


# -------------------------------------------------------------- шум ------
def add_noise(h, amount, gaussian, seed, plane=0):
    """Add Noise: uniform (±amount) or Gaussian (σ = amount/2) noise, the same
    for every run with the same seed. ``plane`` makes channels differ when
    the noise is not monochrome."""
    rng = np.random.default_rng(int(seed) * 7 + int(plane))
    if gaussian:
        n = rng.normal(0.0, float(amount) / 2.0, h.shape).astype(np.float32)
    else:
        n = rng.uniform(-float(amount), float(amount), h.shape).astype(np.float32)
    xp = X.xp(h)
    if X.is_dev(h):
        n = X.to_dev(n)
    return xp.clip(h + n, 0, 1).astype(xp.float32)


def median(h, r):
    """Median over a round window of radius r px."""
    nd = X.ndi(h)
    return nd.median_filter(h, footprint=_disk(r, h), mode='wrap')


def dust_scratches(h, r, threshold):
    """Dust & Scratches: a pixel that differs from the median of radius r by
    more than ``threshold`` (0…1) is replaced by that median."""
    xp = X.xp(h)
    m = median(h, r)
    return xp.where(xp.abs(h - m) > float(threshold), m, h).astype(xp.float32)


def reduce_noise(h, strength, detail):
    """Reduce Noise: non-local means (Buades, Coll & Morel 2005), the fast
    variant of skimage.restoration.denoise_nl_means (Darbon et al. 2008)
    with patch_size 5, patch_distance 6, written with array operations so
    that it runs on the video card as well.

    The noise level σ is estimated from the picture (noise_sigma); strength
    scales the filtering against it, detail 0…1 keeps that share of the
    removed difference.

    Every pixel becomes a weighted mean of the pixels up to 6 px away
    (13×13 shifts). The weight of a shift t is exp(−D), where D is the sum,
    over a small window around the pixel, of (I(q) − I(q + t))² − 2σ²,
    clipped at zero and divided by (strength·σ)²·25; above 5 the weight is
    zero. The window is the one skimage uses: 4×4 pixels from one before
    to two after the pixel (its integral image takes the patch between
    corners ±2), normalised as if it were 5×5. Shifts wrap around the
    edges, so the result tiles. Against skimage on a wrap-padded picture
    the difference is within its fast approximate exp (mean 2e-4, at most
    1/255 apart).

    Without the video card skimage itself is used: on the processor it is
    about five times faster than this array version.
    """
    xp, nd = X.xp(h), X.ndi(h)
    a = xp.asarray(h, xp.float32)
    sigma = noise_sigma(a)
    if sigma <= 0:
        return a
    if not X.is_dev(a):
        return _reduce_noise_skimage(a, sigma, strength, detail)
    hh = (max(float(strength), 0.01) * sigma) ** 2 * 25.0
    var2 = 2.0 * sigma * sigma
    acc = xp.zeros_like(a)
    ws = xp.zeros_like(a)
    for dy in range(-6, 7):
        for dx in range(-6, 7):
            # sh[p] = a[p + t] — тот же снимок, сдвинутый на t (с переносом)
            sh = xp.roll(a, (-dy, -dx), (0, 1))
            # сумма по окну 4×4 (строки и столбцы p−1…p+2), как у skimage
            d = nd.uniform_filter((a - sh) ** 2 - var2, 4, mode='wrap', origin=-1) * 16.0
            d = xp.maximum(d, 0.0) / hh
            w = xp.where(d > 5.0, 0.0, xp.exp(-d)).astype(xp.float32)
            acc += w * sh
            ws += w
    # сдвиг (0, 0) всегда даёт вес 1, так что делить на ноль не придётся
    out = acc / ws
    out = out + float(detail) * (a - out)
    return xp.clip(out, 0, 1).astype(xp.float32)


def _reduce_noise_skimage(a, sigma, strength, detail):
    """reduce_noise on the processor: skimage's own fast non-local means on
    the picture padded by wrapping (so no seam appears at the edges)."""
    from skimage.restoration import denoise_nl_means
    # поиск похожих кусков у края идёт через край: дорисовываем 9 пикселей
    # с другой стороны текстуры и потом срезаем, чтобы шов не появился
    m = 9
    big = np.pad(a, m, mode='wrap')
    out = denoise_nl_means(big, h=max(float(strength), 0.01) * sigma, sigma=sigma,
                           patch_size=5, patch_distance=6, fast_mode=True,
                           preserve_range=True)[m:-m, m:-m].astype(np.float32)
    out = out + float(detail) * (a - out)
    return np.clip(out, 0, 1).astype(np.float32)


# ------------------------------------------------------------ другое -----
def _round_extreme(h, r, op):
    """Max or min over a round window of radius r. Up to 3 px — the exact
    disk; larger — r steps of 3×3, square and cross in turn, which grows an
    octagon (close to a circle) at a cost that does not grow with r²."""
    nd = X.ndi(h)
    f = nd.grey_dilation if op == 'max' else nd.grey_erosion
    r = max(1, int(round(r)))
    if r <= 3:
        return f(h, footprint=_disk(r, h), mode='wrap')
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
    cross = X.to_dev(cross) if X.is_dev(h) else cross
    for k in range(r):
        h = f(h, size=(3, 3), mode='wrap') if k % 2 == 0 else f(h, footprint=cross, mode='wrap')
    return h


def maximum(h, r):
    """Maximum (GIMP: Dilate): the largest value in a round window."""
    return _round_extreme(h, r, 'max')


def minimum(h, r):
    """Minimum (GIMP: Erode): the smallest value in a round window."""
    return _round_extreme(h, r, 'min')


def offset(h, fx, fy):
    """Offset with wrap-around: shift by fx·width and fy·height (fractions
    −0.5…0.5). Brings the tile edges to the middle to check the seam."""
    xp = X.xp(h)
    H, W = h.shape
    return xp.roll(h, (int(round(float(fy) * H)), int(round(float(fx) * W))),
                   (0, 1)).astype(xp.float32)


def emboss(h, angle, height, amount):
    """Emboss: 0.5 + amount·(h(p + d) − h(p − d)), d — ``height`` px along
    ``angle`` degrees (the light direction, counter-clockwise from the
    right)."""
    xp = X.xp(h)
    a = math.radians(float(angle))
    dy, dx = -math.sin(a) * float(height), math.cos(a) * float(height)
    d = fft_shift(h, -dy, -dx) - fft_shift(h, dy, dx)
    return xp.clip(0.5 + float(amount) * d, 0, 1).astype(xp.float32)


def find_edges(h, method, invert):
    """Find Edges / GIMP Edge Detect: gradient magnitude (Sobel, Prewitt,
    Scharr) or the absolute Laplacian, stretched so the 99.5th percentile is
    1. invert — dark lines on white, as Photoshop shows them."""
    xp = X.xp(h)
    nd = X.ndi(h)
    if method == 'laplace':
        g = xp.abs(nd.laplace(h, mode='wrap'))
    elif method == 'scharr':
        k = np.array([[3, 0, -3], [10, 0, -10], [3, 0, -3]], np.float32) / 32.0
        kd = X.to_dev(k) if X.is_dev(h) else k
        gx = nd.convolve(h, kd, mode='wrap')
        gy = nd.convolve(h, kd.T, mode='wrap')
        g = xp.sqrt(gx * gx + gy * gy)
    else:
        f = nd.prewitt if method == 'prewitt' else nd.sobel
        gx, gy = f(h, axis=1, mode='wrap'), f(h, axis=0, mode='wrap')
        g = xp.sqrt(gx * gx + gy * gy)
    s = max(float(xp.percentile(g, 99.5)), 1e-6)
    g = xp.clip(g / s, 0, 1)
    return (1.0 - g if invert else g).astype(xp.float32)


# ---------------------------------------------------------- коррекция ----
def levels(h, in_black, in_white, gamma, out_black, out_white):
    """Levels: input black/white points (0…1), midtone gamma (1 — neutral,
    above 1 lightens, as in Photoshop), output black/white points."""
    xp = X.xp(h)
    lo, hi = float(in_black), max(float(in_white), float(in_black) + 1e-4)
    v = xp.clip((h - lo) / (hi - lo), 0, 1)
    v = v ** (1.0 / max(float(gamma), 0.01))
    return (float(out_black) + v * (float(out_white) - float(out_black))).astype(xp.float32)


def curve_lut(points):
    """256-entry lookup table of a curve through (x, y) points 0…1: a natural
    cubic spline, like the Curves dialogue, clipped to 0…1."""
    from scipy.interpolate import CubicSpline
    pts = sorted(points)
    xs = np.array([p[0] for p in pts], np.float64)
    ys = np.array([p[1] for p in pts], np.float64)
    cs = CubicSpline(xs, ys, bc_type='natural')
    return np.clip(cs(np.linspace(0, 1, 256)), 0, 1).astype(np.float32)


def apply_lut(h, lut):
    """Map values 0…1 through a lookup table with linear interpolation."""
    xp = X.xp(h)
    grid = np.linspace(0, 1, len(lut)).astype(np.float32)
    if X.is_dev(h):
        grid, lut = X.to_dev(grid), X.to_dev(np.asarray(lut, np.float32))
    return xp.interp(xp.clip(h, 0, 1), grid, lut).astype(xp.float32)


def brightness_contrast(h, brightness, contrast):
    """Brightness/Contrast: brightness −1…1 is added, contrast −1…1 scales
    the spread around the mean (1 doubles it, −1 flattens to the mean)."""
    xp = X.xp(h)
    m = float(h.mean())
    k = 1.0 + float(contrast)
    return xp.clip((h - m) * k + m + float(brightness), 0, 1).astype(xp.float32)


def posterize(h, n):
    """Posterize: n levels per channel."""
    xp = X.xp(h)
    n = max(2, int(n))
    return (xp.round(xp.clip(h, 0, 1) * (n - 1)) / (n - 1)).astype(xp.float32)


def equalize(h, bins=1024):
    """Equalize: flatten the histogram (the cumulative distribution becomes a
    straight line)."""
    xp = X.xp(h)
    v = xp.clip(h, 0, 1)
    hist, edges = xp.histogram(v, bins=bins, range=(0.0, 1.0))
    cdf = xp.cumsum(hist).astype(xp.float32)
    cdf = (cdf - cdf[0]) / xp.maximum(cdf[-1] - cdf[0], 1.0)
    return xp.interp(v, edges[1:].astype(xp.float32), cdf).astype(xp.float32)


def auto_contrast(a, clip_lo, clip_hi):
    """Auto Contrast / GIMP Stretch Contrast: the clip_lo share of pixels
    becomes black and clip_hi white, the rest is stretched linearly. Applied
    to all channels together, so hues do not shift."""
    xp = X.xp(a)
    lo = float(xp.percentile(a, 100.0 * float(clip_lo)))
    hi = float(xp.percentile(a, 100.0 - 100.0 * float(clip_hi)))
    if hi - lo < 1e-6:
        return a
    return xp.clip((a - lo) / (hi - lo), 0, 1).astype(xp.float32)


def exposure(rgb, stops, offset_, gamma):
    """Exposure, as in Photoshop: in linear light ×2^stops, + offset, then a
    gamma correction (1 — neutral). Input and output sRGB 0…1."""
    import matgen as MG
    xp = X.xp(rgb)
    lin = MG.srgb_to_lin(xp.clip(rgb, 0, 1)) * (2.0 ** float(stops)) + float(offset_)
    lin = xp.clip(lin, 0, 1) ** (1.0 / max(float(gamma), 0.01))
    return MG.lin_to_srgb(lin).astype(xp.float32)


def rgb_to_hsl(rgb):
    """sRGB 0…1 → hue (0…1, a turn), saturation, lightness, as in HSL."""
    xp = X.xp(rgb)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = xp.maximum(xp.maximum(r, g), b)
    mn = xp.minimum(xp.minimum(r, g), b)
    l = (mx + mn) / 2.0
    d = mx - mn
    s = xp.where(d > 1e-6, d / xp.maximum(1.0 - xp.abs(2.0 * l - 1.0), 1e-6), 0.0)
    dd = xp.maximum(d, 1e-6)
    hr = ((g - b) / dd) % 6.0
    hg = (b - r) / dd + 2.0
    hb = (r - g) / dd + 4.0
    h = xp.where(mx == r, hr, xp.where(mx == g, hg, hb)) / 6.0
    h = xp.where(d > 1e-6, h, 0.0)
    return h, xp.clip(s, 0, 1), l


def hsl_to_rgb(h, s, l):
    """Hue 0…1, saturation, lightness → sRGB 0…1 (inverse of rgb_to_hsl)."""
    xp = X.xp(h)
    c = (1.0 - xp.abs(2.0 * l - 1.0)) * s
    hp = (h % 1.0) * 6.0
    x = c * (1.0 - xp.abs(hp % 2.0 - 1.0))
    z = xp.zeros_like(h)
    conds = [(hp < 1), (hp < 2), (hp < 3), (hp < 4), (hp < 5)]
    r = xp.where(conds[0], c, xp.where(conds[1], x, xp.where(conds[2], z, xp.where(
        conds[3], z, xp.where(conds[4], x, c)))))
    g = xp.where(conds[0], x, xp.where(conds[1], c, xp.where(conds[2], c, xp.where(
        conds[3], x, z))))
    b = xp.where(conds[0], z, xp.where(conds[1], z, xp.where(conds[2], x, xp.where(
        conds[3], c, xp.where(conds[4], c, x)))))
    m = l - c / 2.0
    return xp.clip(xp.stack([r + m, g + m, b + m], axis=-1), 0, 1).astype(xp.float32)


def hue_saturation(rgb, hue, sat, light):
    """Hue/Saturation: hue shift in degrees, saturation and lightness −1…1
    (as Photoshop's −100…100: saturation −1 gives grey, +1 doubles it;
    lightness −1 gives black, +1 white)."""
    xp = X.xp(rgb)
    h, s, l = rgb_to_hsl(xp.clip(rgb[..., :3], 0, 1))
    h = h + float(hue) / 360.0
    s = xp.clip(s * (1.0 + float(sat)), 0, 1)
    out = hsl_to_rgb(h, s, l)
    li = float(light)
    if li > 0:
        out = out + (1.0 - out) * li
    elif li < 0:
        out = out * (1.0 + li)
    return xp.clip(out, 0, 1).astype(xp.float32)


def desaturate(rgb, mode):
    """Desaturate to one grey plane: 'luma' (Rec. 709 weights), 'average',
    'lightness' ((max + min)/2), 'max', or one channel 'r', 'g', 'b'."""
    xp = X.xp(rgb)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    if mode == 'average':
        v = (r + g + b) / 3.0
    elif mode == 'lightness':
        v = (xp.maximum(xp.maximum(r, g), b) + xp.minimum(xp.minimum(r, g), b)) / 2.0
    elif mode == 'max':
        v = xp.maximum(xp.maximum(r, g), b)
    elif mode in ('r', 'g', 'b'):
        v = rgb[..., 'rgb'.index(mode)]
    else:
        v = luma(rgb)
    return xp.clip(v, 0, 1).astype(xp.float32)


def color_balance(rgb, tone, cr, mg, yb, keep_light):
    """Color Balance for one tonal range ('shadows', 'midtones',
    'highlights'): cr — cyan…red, mg — magenta…green, yb — yellow…blue, each
    −1…1, moving that channel by up to ±0.25 where the range is in force.
    keep_light — then the brightness (Rec. 709) is put back as it was."""
    xp = X.xp(rgb)
    c = xp.clip(rgb[..., :3], 0, 1)
    l = luma(c)
    if tone == 'shadows':
        w = xp.clip(1.0 - l / 0.5, 0, 1)
    elif tone == 'highlights':
        w = xp.clip((l - 0.5) / 0.5, 0, 1)
    else:
        w = 1.0 - xp.abs(2.0 * l - 1.0)
    d = xp.stack([float(cr) * w, float(mg) * w, float(yb) * w], axis=-1) * 0.25
    out = xp.clip(c + d, 0, 1)
    if keep_light:
        out = xp.clip(out + (l - luma(out))[..., None], 0, 1)
    return out.astype(xp.float32)


# ------------------------------------------------- подгонка альбедо -----
def to_oklab(rgb):
    """sRGB 0…1 → OKLab (Ottosson 2020): L lightness 0…1, a, b — the colour
    axes. Perceptually even: equal steps look equal, so shifting the mean of
    L or of (a, b) changes brightness or cast without touching the rest."""
    import matgen as MG
    xp = X.xp(rgb)
    c = MG.srgb_to_lin(xp.clip(rgb[..., :3], 0, 1))
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    l = xp.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    m = xp.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    s = xp.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    return (0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s)


def from_oklab(L, a, b):
    """OKLab → sRGB 0…1 (inverse of to_oklab), clipped."""
    import matgen as MG
    xp = X.xp(L)
    l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    rgb = xp.stack([4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
                    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
                    -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s], axis=-1)
    return xp.clip(MG.lin_to_srgb(xp.clip(rgb, 0, 1)), 0, 1).astype(xp.float32)


_LUMW = np.array([0.2126, 0.7152, 0.0722], np.float32)


def _grad_wrap(h):
    """Central differences of a tiling map: (d/dx along columns, d/dy along rows)."""
    xp = X.xp(h)
    return ((xp.roll(h, -1, 1) - xp.roll(h, 1, 1)) * 0.5,
            (xp.roll(h, -1, 0) - xp.roll(h, 1, 0)) * 0.5)


def height_scale(n, h):
    """Height h brought to pixel units so that it fits the normal n.

    The normal of a surface z = H(x, y) is (−Hx, −Hy, 1) normalised; so
    nx/nz = sx·(−k·hx), ny/nz = sy·(−k·hy) with H = k·h. k and the signs sx,
    sy (the layout of the normal: DirectX / OpenGL green, a flipped red) are
    found by least squares. Returns (H, sx, sy, fit) — fit is the share of
    the slopes the height explains (0…1, R²). On the card or the processor
    (the module of n)."""
    xp = X.xp(n, h)
    gx, gy = _grad_wrap(h)
    nz = xp.maximum(n[..., 2], 0.05)
    tx, ty = n[..., 0] / nz, n[..., 1] / nz
    ax = float((tx * gx).sum() / max(float((gx * gx).sum()), 1e-12))
    ay = float((ty * gy).sum() / max(float((gy * gy).sum()), 1e-12))
    sx, sy = (-1.0 if ax > 0 else 1.0), (-1.0 if ay > 0 else 1.0)
    k = 0.5 * (abs(ax) + abs(ay))
    res = ((tx + sx * k * gx) ** 2 + (ty + sy * k * gy) ** 2).sum()
    tot = max(float((tx * tx + ty * ty).sum()), 1e-12)
    return (h * k).astype(xp.float32), sx, sy, max(0.0, 1.0 - float(res) / tot)


def height_shadow(H, lg, max_side=512, max_steps=160):
    """See _height_shadow. If the card's ndimage lacks a mode it needs (old
    CuPy), the march is done on the processor instead of failing."""
    try:
        return _height_shadow(H, lg, max_side, max_steps)
    except (NotImplementedError, ValueError, TypeError):
        if not X.is_dev(H) or X._state['fake']:
            raise                       # строгая подмена должна видеть ошибку
        return X.to_dev(_height_shadow(X.to_host(H), lg, max_side, max_steps))


def _height_shadow(H, lg, max_side=512, max_steps=160):
    """Cast shadows of a height field: True where, going from a pixel towards
    the light, the relief rises above the ray (the same test as screen-space
    shadows, but on the height map). H — height in pixel units, tiling;
    lg — direction to the light (x along columns, y along rows, z up). The
    march runs on a copy of at most max_side px (shadows are coarse; the
    height is scaled with it) and the mask is enlarged back. On the card or
    the processor (the module of H)."""
    xp, nd = X.xp(H), X.ndi(H)
    h0, w0 = H.shape
    f = max(1.0, max(h0, w0) / float(max_side))
    Hs = H if f == 1.0 else nd.zoom(H, 1.0 / f, order=1, mode='grid-wrap') / f
    lxy = math.hypot(float(lg[0]), float(lg[1]))
    if lxy < 1e-3 or lg[2] <= 0:
        return xp.zeros((h0, w0), bool)
    dx, dy, dz = lg[0] / lxy, lg[1] / lxy, lg[2] / lxy       # подъём луча на шаг в пиксель
    span = float(Hs.max() - Hs.min())
    steps = int(min(max_steps, math.ceil(span / max(dz, 1e-3))))
    tol = max(0.25, 0.02 * span)
    rows, cols = xp.mgrid[0:Hs.shape[0], 0:Hs.shape[1]].astype(xp.float32)
    crd = xp.stack([rows, cols])
    sh = xp.zeros(Hs.shape, bool)
    for t in range(1, steps + 1):
        v = nd.map_coordinates(Hs, crd + xp.asarray(np.array([dy * t, dx * t], np.float32)
                                                    ).reshape(2, 1, 1),
                               order=1, mode='grid-wrap')
        sh |= v > Hs + dz * t + tol
    if f > 1.0:
        sh = nd.zoom(sh.astype(xp.float32), (h0 / sh.shape[0], w0 / sh.shape[1]), order=0,
                     mode='grid-wrap') > 0.5
    return sh


def _light_search(N, r, az, el):
    """Best light direction among candidates for r ≈ a + b·max(0, n·l):
    for each direction the a, b are found by least squares in closed form,
    the one explaining most of r (largest correlation, b > 0) wins.
    N — K×3 normals, r — K values; az, el — 1-D arrays of candidate angles
    (radians, all pairs; numpy). N, r on the card or the processor; the
    K × M products are where the time goes. Returns (l as numpy, corr)."""
    xp = X.xp(N)
    A_, E_ = np.meshgrid(az, el)
    L = np.stack([np.cos(E_) * np.cos(A_), np.cos(E_) * np.sin(A_), np.sin(E_)], -1).reshape(-1, 3)
    D = xp.maximum(N @ xp.asarray(L.T.astype(np.float32)), 0.0)    # K × M
    rc = r - r.mean()
    Dc = D - D.mean(0)
    cov = rc @ Dc
    var = (Dc * Dc).sum(0)
    corr = cov / xp.sqrt(xp.maximum(var * float((rc * rc).sum()), 1e-20))
    corr = xp.where(cov > 0, corr, -1.0)
    j = int(xp.argmax(corr))
    return L[j], float(corr[j])


def _fit_ab(D, r):
    """a, b of r ≈ a + b·D by least squares (closed form)."""
    dm, rm = float(D.mean()), float(r.mean())
    var = float(((D - dm) ** 2).sum())
    b = float(((D - dm) * (r - rm)).sum()) / var if var > 1e-12 else 0.0
    return rm - b * dm, b


def relief_fit(alb, ref, n, h=None):
    """How bright the albedo really is, from the relief: the baked lighting
    of the source is fitted to source / albedo as light = a + b·max(0, n·l)
    — an even light plus one directional light l, which does not reach the
    places turned away from it. l is searched over the hemisphere (coarse,
    then fine) on a sample of pixels; a, b by least squares, separately for
    R, G, B. The scale of albedo against light cannot be told apart (bright
    light on a dark surface looks the same as dim light on a light one), so
    the convention is: a flat place (normal straight out) keeps the source's
    own colour — the gain is a + b·max(0, l_z). Pixels much darker than the
    model are dropped (crevices, shadows), and with a height (h) the cast
    shadows of the found light are masked out by marching over the height
    (height_shadow), then the fit is repeated.

    alb, ref — sRGB H×W×3 of one size; n — signed normal H×W×3; on the
    card or the processor (the module of alb), the result the same.
    Returns (gain per channel (3,), shadow mask, readouts dict); the light
    azimuth is in picture terms (0° — from the right, 90° — from the top),
    the layout of the normal taken from the height, else DirectX."""
    import matgen as MG
    xp = X.xp(alb)
    lw = xp.asarray(_LUMW)
    A = MG.srgb_to_lin(xp.clip(alb[..., :3], 0, 1)).astype(xp.float32)
    I = MG.srgb_to_lin(xp.clip(ref[..., :3], 0, 1)).astype(xp.float32)
    n = xp.asarray(n[..., :3], xp.float32)
    n = n / xp.maximum(xp.linalg.norm(n, axis=-1, keepdims=True), 1e-6)
    AY, IY = A @ lw, I @ lw
    valid = ((AY > 0.01) & (I.max(-1) < 0.985) & (n[..., 2] > 0.05)).ravel()
    N = n.reshape(-1, 3)
    rY = (IY / xp.maximum(AY, 1e-3)).ravel()
    shadow = xp.zeros(AY.shape, bool)
    sx, sy = 1.0, -1.0          # раскладка нормали; без высоты считаем DirectX (как везде в программе)
    if int(valid.sum()) < 64:
        return np.ones(3, np.float32), shadow, {'fit_px': 0.0}
    rng = np.random.default_rng(0)
    keep = valid.copy()
    l = np.array([0.0, 0.0, 1.0], np.float32)
    H = None                    # высота в пикселях — считается при первой нужде
    for it in range(3):
        idx = xp.flatnonzero(keep)
        if idx.size > 40000:
            # выборка по номерам — на процессоре, одна и та же везде
            idx = idx[xp.asarray(rng.choice(int(idx.size), 40000, replace=False))]
        # направление: грубо по полусфере, потом точнее вокруг лучшего
        l, _ = _light_search(N[idx], rY[idx], np.radians(np.arange(0, 360, 10)),
                             np.radians(np.arange(3, 91, 6)))
        az0 = math.atan2(float(l[1]), float(l[0]))
        el0 = math.asin(max(-1.0, min(1.0, float(l[2]))))
        l, corr = _light_search(N[idx], rY[idx], az0 + np.radians(np.arange(-10, 10.5, 1.0)),
                                np.clip(el0 + np.radians(np.arange(-6, 6.5, 0.75)),
                                        math.radians(1), math.radians(90)))
        D = xp.maximum(N @ xp.asarray(l, xp.float32), 0.0)
        a, b = _fit_ab(D[keep], rY[keep])
        res = rY - (a + b * D)
        sd = float(xp.std(res[keep])) or 1e-6
        # тени — только от заметного направленного света (не меньше 20 % всего)
        if it < 2 and h is not None and b > 0.2 * (a + b):
            if H is None:
                # первая итерация могла пройти без теней (направленного света
                # тогда ещё не нашлось) — высоту считаем, когда она понадобилась
                H, sx, sy, _ = height_scale(n, h)
                # крупные «холмы» из интегрированной нормали теней на текстуре не
                # дают: их убираем, оставляя рельеф мельче 1/16 стороны
                H = H - MG.fast_blur(H, max(H.shape) / 16.0, mode='wrap')
            shadow = height_shadow(H, (sx * l[0], sy * l[1], l[2]))
            if float(shadow.mean()) > 0.5:
                shadow[:] = False          # полтекстуры в тени — высота не та, маске не верим
        k2 = valid & (res > -2.0 * sd) & ~shadow.ravel()
        if int(k2.sum()) < 64:
            break
        keep = k2
    D = xp.maximum(N @ xp.asarray(l, xp.float32), 0.0)
    gain = []
    for ch in range(3):
        r = (I[..., ch] / xp.maximum(A[..., ch], 1e-3)).ravel()
        a_c, b_c = _fit_ab(D[keep], r[keep])
        gain.append(max(a_c + max(b_c, 0.0) * max(float(l[2]), 0.0), 1e-3))
    a, b = _fit_ab(D[keep], rY[keep])
    info = dict(gain=a + max(b, 0.0) * max(float(l[2]), 0.0),
                light_az=math.degrees(math.atan2(-sy * float(l[1]), sx * float(l[0]))),
                light_el=math.degrees(math.asin(max(-1.0, min(1.0, float(l[2]))))),
                light_share=(max(b, 0.0) / max(a + max(b, 0.0), 1e-6)) * 100.0,
                shadow_pc=float(shadow.mean()) * 100.0,
                fit_px=float(keep.mean()) * 100.0)
    return np.asarray(gain, np.float32), shadow, info


def dark_defects(ref, cut):
    """Black defects of a source picture (dead pixels of an upscale, holes):
    sRGB luminance below ``cut``, grown by max(1, width/512) px on the torus.
    Returns a boolean H×W mask (all False when cut ≤ 0). Used so that such
    places neither count in the statistics nor give detail."""
    xp, nd = X.xp(ref), X.ndi(ref)
    H, W = ref.shape[:2]
    if cut <= 0:
        return xp.zeros((H, W), bool)
    lum = (ref[..., :3] * xp.asarray(_LUMW)).sum(-1)
    g = max(1, int(round(W / 512.0)))
    return nd.maximum_filter((lum < float(cut)).astype(xp.float32), size=2 * g + 1,
                             mode='wrap') > 0.5


def match_albedo(alb, ref, w_light, w_cast, w_sat, detail, detail_r, gain=None,
                 w_spread=0.0, dark_cut=0.0):
    """Bring a generated albedo back towards the source diffuse — overall
    only, so the baked lighting the albedo was freed of does not return.

    In OKLab: w_light — share of the difference in mean lightness L that is
    taken over; w_cast — share of the difference in the mean colour (a, b),
    i.e. the overall cast / hue; w_sat — how far the spread of colour round
    its mean (saturation) is scaled to the source's. detail — the source's
    fine lightness detail (L minus its blur of detail_r px) added on top, ×
    detail: sharpness for albedos that came out soft. All weights 0…1 (detail
    up to 2). Returns sRGB 0…1 and a dict of readouts.

    gain — per-channel linear gains from relief_fit: then lightness and cast
    are not matched by the means (they include the baked shadows and so make
    the albedo too dark) but taken from the fit: the albedo × gain^w_light
    in brightness and × (gain / its brightness)^w_cast in colour; saturation
    and detail as above.

    w_spread 0…1 — the spread of lightness on a large scale (blur of radius
    width/256: what is seen from afar). De-lighting takes large light and
    dark patches for shadows and evens them out, the means above do not
    bring them back, and the texture looks milky from a distance. The
    albedo's own large-scale variation is scaled by k (towards the source's
    spread, k^w_spread, kept within 0.5…3); only that number comes from the
    source, not its pattern, so its shading does not return.

    dark_cut — black defects of the source (dark_defects: luminance below
    it, with a margin) are left out of the spread measurement and give no
    detail. 0 — off."""
    xp = X.xp(alb)
    import matgen as MG
    if gain is not None:
        g = xp.asarray(gain, xp.float32)
        gY = float((gain * _LUMW).sum())
        mult = (gY ** float(w_light)) * (g / gY) ** float(w_cast)
        alb = MG.lin_to_srgb(xp.clip(MG.srgb_to_lin(xp.clip(alb, 0, 1)) * mult, 0, 1))
    L, a, b = to_oklab(alb)
    Lr, ar, br = to_oklab(ref)
    dL = float(Lr.mean()) - float(L.mean())
    ma, mb = float(a.mean()), float(b.mean())
    mar, mbr = float(ar.mean()), float(br.mean())
    if gain is not None:
        dL, mar, mbr = 0.0, ma, mb          # светлота и оттенок уже из подгонки по рельефу
    c = float(xp.sqrt(((a - ma) ** 2 + (b - mb) ** 2).mean()))
    cr = float(xp.sqrt(((ar - mar) ** 2 + (br - mbr) ** 2).mean()))
    k = (cr / c) ** float(w_sat) if c > 1e-5 and cr > 1e-5 else 1.0
    L2 = L + float(w_light) * dL
    a2 = ma + float(w_cast) * (mar - ma) + (a - ma) * k
    b2 = mb + float(w_cast) * (mbr - mb) + (b - mb) * k
    bad = dark_defects(ref, float(dark_cut))
    good = ~bad
    info = dict(light_diff=dL * 100.0, sat_ratio=(cr / c) if c > 1e-5 else 1.0)
    if w_spread > 0 and int(good.sum()) > 64:
        R = max(1.0, alb.shape[1] / 256.0)
        lo_r = MG.fast_blur(Lr, R, mode='wrap')
        lo_a = MG.fast_blur(L2, R, mode='wrap')
        sr, sa = float(lo_r[good].std()), float(lo_a[good].std())
        ks = (sr / sa) ** float(w_spread) if sa > 1e-6 and sr > 1e-6 else 1.0
        ks = min(max(ks, 0.5), 3.0)
        # своё крупное разнообразие альбедо — шире в ks раз вокруг его среднего
        L2 = L2 + (ks - 1.0) * (lo_a - float(lo_a[good].mean()))
        info['spread_k'] = ks
    if detail > 0:
        det = Lr - MG.fast_blur(Lr, max(float(detail_r), 0.3), mode='wrap')
        if bool(bad.any()):
            keep = MG.fast_blur((~bad).astype(xp.float32), 2.0, mode='wrap')
            det = det * keep
        L2 = L2 + float(detail) * det
    return from_oklab(xp.clip(L2, 0, 1), a2, b2), info


# ------------------------------------------------------ рисованная маска --
def paint_strokes(strokes, h, w, base=0.0):
    """Rasterise brush strokes into an H×W mask 0…1 (numpy, float32).

    strokes — list of {'pts': [[x, y], …] in fractions of the size,
    'r': brush radius as a fraction of the width, 'hard': 0…1, 'op': 0…1,
    'v': 1 (paint white) or 0 (paint black)}; base — the value before any
    stroke. A stroke is the band swept by the round brush along its points,
    with the edge fading out over the outer (1 − hard)·r; it is laid over what is there
    with opacity op as one shape, so where a stroke crosses itself it does
    not build up (as a brush stroke in Photoshop). Strokes wrap round the
    edges, so a mask painted over a tiling texture tiles too. Because all
    sizes are fractions, the same strokes fit the preview copy and the full
    size."""
    from PIL import Image, ImageDraw
    from scipy.ndimage import distance_transform_edt
    m = np.full((h, w), float(base), np.float32)
    for st in strokes or []:
        pts = st.get('pts') or []
        if not pts:
            continue
        r = max(0.5, float(st.get('r', 0.02)) * w)
        hard = min(max(float(st.get('hard', 0.7)), 0.0), 1.0)
        op = min(max(float(st.get('op', 1.0)), 0.0), 1.0)
        v = float(st.get('v', 1.0))
        # по расстоянию до линии мазка: до r − мягкость — полная сила, к r —
        # плавно до нуля (не меньше пикселя, чтобы край не был зубчатым)
        soft = max(1.0, (1.0 - hard) * r)
        P = np.array(pts, np.float64) * [w, h]
        pad = r + 2.0
        x0, y0 = int(math.floor(P[:, 0].min() - pad)), int(math.floor(P[:, 1].min() - pad))
        x1, y1 = int(math.ceil(P[:, 0].max() + pad)), int(math.ceil(P[:, 1].max() + pad))
        im = Image.new('L', (x1 - x0, y1 - y0), 0)
        dr = ImageDraw.Draw(im)
        Q = [(float(x - x0), float(y - y0)) for x, y in P]
        if len(Q) > 1:
            dr.line(Q, fill=255, width=1)
        for x, y in Q:
            dr.point((x, y), fill=255)
        d = distance_transform_edt(np.asarray(im) == 0)
        t = np.clip((r - d) / soft, 0.0, 1.0)
        a = (t * t * (3.0 - 2.0 * t) * op).astype(np.float32)
        if x1 - x0 <= w and y1 - y0 <= h:
            # по кругу: что вышло за край, ложится с другой стороны
            ys = np.arange(y0, y1) % h
            xs = np.arange(x0, x1) % w
            blk = m[np.ix_(ys, xs)]
            m[np.ix_(ys, xs)] = blk + (v - blk) * a
        else:
            # мазок шире самой текстуры — без переноса, по обрезанной рамке
            cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
            if cx1 > cx0 and cy1 > cy0:
                aa = a[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
                blk = m[cy0:cy1, cx0:cx1]
                m[cy0:cy1, cx0:cx1] = blk + (v - blk) * aa
    return np.clip(m, 0, 1).astype(np.float32)


# --------------------------------------------------------- заплатка ------
def polygon_mask(poly, h, w):
    """Boolean H×W mask of a polygon given in fractions of the size
    (list of [x, y], 0…1)."""
    from PIL import Image, ImageDraw
    im = Image.new('L', (w, h), 0)
    pts = [(float(x) * w, float(y) * h) for x, y in poly]
    if len(pts) >= 3:
        ImageDraw.Draw(im).polygon(pts, fill=255)
    return np.asarray(im) > 127


def _harmonic_fill(d, mask, push=None):
    """Membrane: the smooth (harmonic) function inside ``mask`` that equals
    ``d`` on the pixels round it. d: H×W×C, mask: H×W bool, not touching the
    array edge. One sparse factorisation, solved for every channel.

    push — optional H×W map: the membrane is then pushed by it (−∇²u = push,
    as a soap film under pressure), so a positive push bulges it into a dome
    and a negative one sags it into a dish."""
    import scipy.sparse as sps
    from scipy.sparse.linalg import splu
    H, W = mask.shape
    idx = -np.ones((H, W), np.int64)
    ys, xs = np.nonzero(mask)
    n = len(ys)
    idx[ys, xs] = np.arange(n)
    rows, cols, vals = [np.arange(n)], [np.arange(n)], [np.full(n, 4.0)]
    rhs = np.zeros((n, d.shape[2]), np.float64)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        qy, qx = ys + dy, xs + dx
        q = idx[qy, qx]
        inside = q >= 0
        rows.append(np.arange(n)[inside])
        cols.append(q[inside])
        vals.append(np.full(int(inside.sum()), -1.0))
        out = ~inside
        rhs[out] += d[qy[out], qx[out]]
    if push is not None:
        rhs += np.asarray(push, np.float64)[ys, xs][:, None]
    A = sps.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(n, n))
    lu = splu(A)
    res = np.zeros_like(d, dtype=np.float32)
    for c in range(d.shape[2]):
        res[ys, xs, c] = lu.solve(rhs[:, c])
    return res


def apply_patch(img, poly, dx, dy, seamless=True, feather=4.0, mask=None):
    """Patch tool, as in Photoshop («Source» mode): the area inside ``poly``
    is replaced by the area shifted by (dx, dy) (fractions of the size; the
    picture is a tile, the shift wraps).

    seamless — gradient-domain blend (Pérez, Gangnet & Blake 2003): the
    texture of the source is kept, and a smooth membrane brings its edge to
    the surroundings, so no seam and no colour step. Off — a plain copy with
    an edge feathered by ``feather`` px. img: H×W or H×W×C float; returns the
    same shape, float32. On the card or the processor (the module of img);
    the membrane itself (a sparse solve over the patch only) is solved on
    the processor."""
    xp, nd = X.xp(img), X.ndi(img)
    a = xp.asarray(img, xp.float32)
    two = a.ndim == 2
    if two:
        a = a[..., None]
    H, W = a.shape[:2]
    m = xp.asarray(polygon_mask(poly, H, W)) if mask is None else xp.asarray(mask, bool)
    if not bool(m.any()):
        return img
    oy, ox = int(round(float(dy) * H)), int(round(float(dx) * W))
    src = xp.roll(a, (-oy, -ox), (0, 1))          # src[p] = a[p + сдвиг]
    if not seamless:
        w = nd.gaussian_filter(m.astype(xp.float32), max(float(feather), 0.01), mode='wrap')
        w = xp.where(m, xp.maximum(w, 0.5), w)[..., None]
        out = a * (1 - w) + src * w
    else:
        # сдвигаем так, чтобы заплатка была в середине и не касалась края
        ys, xs = xp.nonzero(m)
        cy, cx = H // 2 - int(round(float(ys.mean()))), W // 2 - int(round(float(xs.mean())))
        ar, sr, mr = (xp.roll(v, (cy, cx), (0, 1)) for v in (a, src, m))
        ys, xs = xp.nonzero(mr)
        y0, y1 = max(int(ys.min()) - 1, 0), min(int(ys.max()) + 2, H)
        x0, x1 = max(int(xs.min()) - 1, 0), min(int(xs.max()) + 2, W)
        mb = X.to_host(mr[y0:y1, x0:x1]).copy()
        mb[0, :] = mb[-1, :] = False
        mb[:, 0] = mb[:, -1] = False
        d = X.to_host((ar - sr)[y0:y1, x0:x1]).astype(np.float64)
        fill = xp.asarray(_harmonic_fill(d, mb))
        mbd = xp.asarray(mb)
        box = ar[y0:y1, x0:x1].copy()
        box[mbd] = (sr[y0:y1, x0:x1] + fill)[mbd]
        ar = ar.copy()
        ar[y0:y1, x0:x1] = box
        out = xp.roll(ar, (-cy, -cx), (0, 1))
    if out.shape[2] == 3 and float(a.min()) < -0.01:
        # нормали: после смешивания — снова единичной длины
        out = out / xp.maximum(xp.sqrt((out * out).sum(-1, keepdims=True)), 1e-6)
    elif float(a.min()) >= 0.0 and float(a.max()) <= 1.0:
        # мембрана может слегка выйти за края диапазона — карта остаётся 0…1
        out = xp.clip(out, 0.0, 1.0)
    out = out.astype(xp.float32)
    return out[..., 0] if two else out


def find_source(img, poly, band=6, mask=None, avoid=None):
    """Where to take a patch from automatically (like Content-Aware patch):
    the shift whose surroundings best match the surroundings of the area.

    A ring ``band`` px wide round the polygon is compared with the same ring
    at every shift on the torus at once (sums of squared differences through
    FFT correlations: Σw(A(p+o) − A(p))² = ΣwA² − 2·corr(wA, A) + corr(w, A²)).
    Shifts at which the taken area would overlap the area itself are
    excluded, and so are shifts that would take from ``avoid`` (a mask of
    other damaged places). img: H×W or H×W×C; returns (dx, dy) in fractions
    of the size, or None if the polygon is empty or nothing fits. On the
    card or the processor (the module of img): the FFTs are what takes time."""
    xp, nd = X.xp(img), X.ndi(img)
    a = xp.asarray(img, xp.float64)
    if a.ndim == 2:
        a = a[..., None]
    H, W = a.shape[:2]
    m = xp.asarray(polygon_mask(poly, H, W)) if mask is None else xp.asarray(mask, bool)
    if not bool(m.any()):
        return None
    md = nd.binary_dilation(m, iterations=max(1, int(band)), brute_force=True)
    ring = (md & ~m).astype(xp.float64)
    F = xp.fft.rfft2
    Fi = lambda x: xp.fft.irfft2(x, s=(H, W))

    def corr(f, g):
        """corr(f, g)(o) = Σ_p f(p)·g(p + o), on the torus."""
        return Fi(xp.conj(F(f)) * F(g))
    cost = xp.zeros((H, W))
    Fr = F(ring)
    for c in range(a.shape[2]):
        A = a[..., c]
        cost += -2.0 * Fi(xp.conj(F(ring * A)) * F(A)) + Fi(xp.conj(Fr) * F(A * A))
    over = corr(md.astype(xp.float64), md.astype(xp.float64)) > 0.5
    if avoid is not None and bool(xp.asarray(avoid).any()):
        # и с других испорченных мест не берём: там то же самое
        over |= corr(md.astype(xp.float64), xp.asarray(avoid, xp.float64)) > 0.5
    cost[over] = np.inf
    if not bool(xp.isfinite(cost).any()):
        return None
    oy, ox = np.unravel_index(int(xp.argmin(cost)), cost.shape)
    oy = oy - H if oy > H // 2 else oy
    ox = ox - W if ox > W // 2 else ox
    return ox / W, oy / H


# ------------------------------------------- плоские пятна на нормали ----
def flat_spots(n, spread_deg=0.5, min_area=30, grow=3):
    """Where a normal map is cut flat: patches in which the normal does not
    change at all (a blown-out or crushed spot in the source picture gives a
    network nothing to see, and it answers with one constant vector).

    n: H×W×3 unit normals. A pixel is flat when, in its 5×5 neighbourhood, the
    direction varies by less than ``spread_deg`` degrees; patches smaller
    than ``min_area`` px are ignored (real relief has single flat pixels),
    the rest grown by ``grow`` px to take in the seam. Wraps round the tile.
    Returns an H×W bool mask (the module of n)."""
    xp, nd = X.xp(n), X.ndi(n)
    a = xp.asarray(n, xp.float32)
    s = xp.zeros(a.shape[:2], xp.float32)
    for c in range(3):
        ac = xp.ascontiguousarray(a[..., c])
        m1 = nd.uniform_filter(ac, 5, mode='wrap')
        m2 = nd.uniform_filter(ac * ac, 5, mode='wrap')
        s += xp.maximum(m2 - m1 * m1, 0)
    flat = xp.sqrt(s) < math.radians(float(spread_deg))
    return _big_patches(flat, min_area, grow)


def _big_patches(flat, min_area, grow):
    """Patches of a bool mask at least min_area px, grown by grow px."""
    xp, nd = X.xp(flat), X.ndi(flat)
    lab, k = nd.label(flat)
    k = int(k)
    if k == 0:
        return flat
    sizes = xp.bincount(lab.ravel(), minlength=k + 1)
    keep = sizes >= int(min_area)
    keep[0] = False
    m = keep[lab]
    if grow > 0:
        m = nd.binary_dilation(m, iterations=int(grow), brute_force=True)
    return m


def fill_harmonic(a, mask):
    """Fill the masked pixels of an H×W×C array with the smoothest surface
    that meets the surroundings (membrane: Laplace equation, the edge values
    fixed), on the torus. Returns a float32 copy in the module of a (the
    sparse solve runs on the processor)."""
    dev = X.is_dev(a)
    a = np.asarray(X.to_host(a), np.float32)
    mask = np.asarray(X.to_host(mask), bool)
    two = a.ndim == 2
    if two:
        a = a[..., None]
    p = 2
    ap = np.pad(a, ((p, p), (p, p), (0, 0)), mode='wrap')
    mp = np.pad(mask, p, mode='wrap')
    mp[:p, :] = mp[-p:, :] = False
    mp[:, :p] = mp[:, -p:] = False
    fill = _harmonic_fill(ap.astype(np.float64), mp)
    ap[mp] = fill[mp]
    out = ap[p:-p, p:-p]
    out = out[..., 0] if two else out
    return X.to_dev(np.ascontiguousarray(out)) if dev else out


def fix_flat_normals(n, spread_deg=0.5, min_area=30, grow=3, detail=True):
    """Find the flat-cut patches of a normal map (flat_spots) and fill them.

    detail — each patch gets a seamless patch of relief from the place whose
    surroundings match best (find_source + apply_patch, as the auto patch
    does), so the fill has the same fine relief as the rest. Without it, or
    if no place fits: a smooth continuation of the surrounding tilt
    (fill_harmonic on x and y, z rebuilt for unit length).
    Returns (normals, mask). On the card or the processor (the module of n)."""
    xp, nd = X.xp(n), X.ndi(n)
    m = flat_spots(n, spread_deg, min_area, grow)
    a = xp.asarray(n, xp.float32)
    if not bool(m.any()):
        return a, m
    if detail:
        out = a.copy()
        lab, k = nd.label(m)
        left = xp.zeros_like(m)
        for j in range(1, int(k) + 1):
            mj = lab == j
            off = find_source(out, None, mask=mj, avoid=m & ~mj)
            if off is None:
                left |= mj
                continue
            out = apply_patch(out, None, off[0], off[1], True, mask=mj)
        if not bool(left.any()):
            return out, m
        a, m_fill = out, left
    else:
        m_fill = m
    xy = fill_harmonic(a[..., :2], m_fill)
    z = xp.sqrt(xp.clip(1.0 - (xy * xy).sum(-1), 0.0, 1.0))
    out = xp.concatenate([xy, z[..., None]], -1)
    out = xp.where(m_fill[..., None], out, a)
    out /= xp.maximum(xp.sqrt((out * out).sum(-1, keepdims=True)), 1e-6)
    return out.astype(xp.float32), m


def flat_height(h, level=0.3 / 255.0, min_area=30, grow=3):
    """Plateaus of a height map: patches where the height does not change at
    all (spread over 5×5 below ``level``, 0…1 units), at least ``min_area``
    px, grown by ``grow`` px. A clipped stretch leaves exactly such tops and
    bottoms. Returns an H×W bool mask (the module of h)."""
    xp, nd = X.xp(h), X.ndi(h)
    a = xp.asarray(h, xp.float32)
    m1 = nd.uniform_filter(a, 5, mode='wrap')
    m2 = nd.uniform_filter(a * a, 5, mode='wrap')
    flat = xp.sqrt(xp.maximum(m2 - m1 * m1, 0)) < float(level)
    return _big_patches(flat, min_area, grow)


def fix_flat_height(h, level=0.3 / 255.0, min_area=30, grow=3, detail=True):
    """Fill the plateaus of a height map (flat_height).

    A plateau is a clipped top (or bottom): the real relief went on rising
    there. So each one is filled with a dome — a membrane pushed up (down for
    a bottom) just enough that its slope at the rim matches the mean slope
    round it: for a round patch of radius R and rim slope g the push is 2g/R.
    detail — then the fine relief of the best-matching place (its height
    minus a 2 px blur, find_source) is laid over the dome.

    The dome rises above the old top, so the map is stretched back into 0…1
    (0.5…99.5 percentiles, tails bent in by matgen.soft01).
    Returns (height, mask). On the card or the processor (the module of h);
    the dome (one sparse solve) on the processor."""
    import matgen as MG
    xp, nd = X.xp(h), X.ndi(h)
    m = flat_height(h, level, min_area, grow)
    a = xp.asarray(h, xp.float32)
    if not bool(m.any()):
        return a.copy(), m
    gy, gx = xp.gradient(xp.pad(a, 1, mode='wrap'))
    slope = xp.sqrt(gx * gx + gy * gy)[1:-1, 1:-1]
    med = float(xp.median(a))
    lab, k = nd.label(m)
    push = xp.zeros_like(a)
    fine = xp.zeros_like(a)
    det = a - MG.fast_blur(a, 2.0, mode='wrap')
    for j in range(1, int(k) + 1):
        mj = lab == j
        ring = nd.binary_dilation(mj, iterations=2, brute_force=True) & ~mj
        if not bool(ring.any()):
            continue
        g = float(slope[ring].mean())
        R = math.sqrt(float(mj.sum()) / math.pi)
        sign = 1.0 if float(a[ring].mean()) >= med else -1.0
        push[mj] = sign * 2.0 * g / max(R, 1.0)
        if detail:
            off = find_source(a, None, mask=mj, avoid=m & ~mj)
            if off is not None:
                H, W = a.shape
                oy, ox = int(round(off[1] * H)), int(round(off[0] * W))
                fine[mj] = xp.roll(det, (-oy, -ox), (0, 1))[mj]
    p = 2
    ah, mh, ph = X.to_host(a), X.to_host(m), X.to_host(push)
    ap = np.pad(ah, p, mode='wrap')[..., None].astype(np.float64)
    mp = np.pad(mh, p, mode='wrap')
    mp[:p, :] = mp[-p:, :] = False
    mp[:, :p] = mp[:, -p:] = False
    dome = _harmonic_fill(ap, mp, np.pad(ph, p, mode='wrap'))[..., 0][p:-p, p:-p]
    out = xp.where(m, xp.asarray(np.ascontiguousarray(dome)) + fine, a)
    lo, hi = float(xp.percentile(out, 0.5)), float(xp.percentile(out, 99.5))
    return xp.asarray(MG.soft01((out - lo) / max(hi - lo, 1e-6)), xp.float32), m


# ------------------------------------------- горизонт: тени и AO по свету -
def horizon(hpx, angles, radius, steps=12):
    """How high the horizon rises round each pixel, in given directions.

    hpx — height in pixels (height 0…1 × depth); angles — list of directions
    in radians, screen convention (0 — to the right, π/2 — up; the image y
    grows down); radius — how far to look, px; steps — samples per direction,
    spaced evenly. For each direction the sine of the highest elevation
    seen, max over t of atan((h(p + t·d) − h(p)) / t), clipped at 0. Wraps
    round the tile. Returns an array (len(angles), H, W)."""
    xp = X.xp(hpx)
    out = []
    for a in angles:
        dx, dy = math.cos(a), -math.sin(a)
        best = xp.zeros_like(hpx)
        for s in range(1, int(steps) + 1):
            t = float(radius) * s / steps
            ox, oy = int(round(dx * t)), int(round(dy * t))
            if ox == 0 and oy == 0:
                continue
            sh = xp.roll(hpx, (-oy, -ox), (0, 1))
            best = xp.maximum(best, (sh - hpx) / math.hypot(ox, oy))
        out.append(best / xp.sqrt(1.0 + best * best))
    return xp.stack(out, 0)


def light_shadow(hpx, azimuth, elevation, radius, softness=4.0, steps=24):
    """Cast shadows of the relief under a far light, in screen space.

    azimuth — where the light comes from (degrees, 0 — right, 90 — up),
    elevation — its height above the surface (degrees); softness — the
    light's angular size (degrees): the shadow edge fades over it. A pixel is
    lit by the share of the light disc above its horizon towards the light
    (looked up in three directions across the disc). Returns H×W 0…1,
    1 = fully lit."""
    xp = X.xp(hpx)
    a = math.radians(float(azimuth))
    s = math.radians(max(float(softness), 0.1))
    angs = [a - s / 2, a, a + s / 2]
    hz = horizon(hpx, angs, radius, steps)
    el = math.radians(float(elevation))
    hz_ang = xp.arcsin(xp.clip(hz, 0, 1))
    lit = xp.clip((el - hz_ang) / s + 0.5, 0, 1)
    return lit.mean(0).astype(xp.float32)


# Ядро видеокарты для _lsao_sweep: одна нить — одна линия, вся линия
# проходится внутри нити, без шагов из Python. Логика та же, что в numpy-
# версии ниже (она — запасная и эталон для проверки).
_LSAO_CU = r"""
extern "C" __global__
void lsao_sweep(const float* z, const float* fg, const int has_fg,
                const int* sgn, const float* mm, const int* lay, const int n,
                const int hh, const int ww, const int run, const int steps,
                const float radius, const int horizon,
                float* out, int* sj, float* sz, const int cap)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n * hh) return;
    int k = t / hh, i = t % hh;
    float m = mm[k];
    int sg = sgn[k];
    long long plane = (long long)hh * ww;
    long long lofs = (long long)lay[k] * plane;
    long long obase = horizon ? (long long)k * plane : lofs;
    float ln = sqrtf(1.0f + m * m);
    int* SJ = sj + (long long)t * cap;
    float* SZ = sz + (long long)t * cap;
    int top = -1;
    float zprev = 0.0f;
    for (int j = 0; j < steps; j++) {
        double jm = (double)j * (double)m;
        double shd = floor(jm);
        float f = (float)(jm - shd);
        int r = (int)(((long long)i + (long long)shd) % hh);
        if (r < 0) r += hh;
        int r1 = (r + 1) % hh;
        int x = (sg * (j - run)) % ww;
        if (x < 0) x += ww;
        long long i0 = (long long)r * ww + x, i1 = (long long)r1 * ww + x;
        float zp = (1.0f - f) * z[lofs + i0] + f * z[lofs + i1];
        while (top >= 1) {
            float z1 = SZ[top], z0 = SZ[top - 1];
            float j1 = (float)SJ[top], j0 = (float)SJ[top - 1];
            if ((z0 - zp) * ((float)j - j1) >= (z1 - zp) * ((float)j - j0)) top--;
            else break;
        }
        if (j >= run) {
            float val = 0.0f;
            float d = 1.0f, sv = -1.0f;
            if (top >= 0) {
                d = (float)(j - SJ[top]) * ln;
                float e = (SZ[top] - zp) / fmaxf(d, 1e-6f);
                sv = e / sqrtf(1.0f + e * e);
            }
            if (horizon) {
                if (top >= 0) {
                    float q = d / radius;
                    val = -1.0f + (sv + 1.0f) / (1.0f + q * q * q * q);
                } else {
                    val = -1.0f;
                }
            } else {
                float sh = top >= 0 ? fmaxf(sv, 0.0f) : 0.0f;
                float et = fmaxf((zprev - zp) / ln, 0.0f);
                float st = et / sqrtf(1.0f + et * et);
                float q = d / radius;
                val = fmaxf(sh - st, 0.0f) / (1.0f + q * q);
                if (has_fg) {
                    float fv = (1.0f - f) * fg[lofs + i0] + f * fg[lofs + i1];
                    if (fv < 0.5f) val = 0.0f;
                }
            }
            atomicAdd(&out[obase + i0], (1.0f - f) * val);
            atomicAdd(&out[obase + i1], f * val);
        }
        top++;
        SJ[top] = j;
        SZ[top] = zp;
        zprev = zp;
    }
}
"""


def _lsao_sweep_gpu(kern, z, fg, sgns, ms, lay, radius, horizon):
    """_lsao_sweep on the graphics card through the kernel _LSAO_CU: the
    same inputs and outputs, one GPU thread per line."""
    xp = X.xp(z)
    nlay, hh, ww = z.shape
    n = len(ms)
    run = min(ww, int(math.ceil(8.0 * radius)))
    steps = ww + run
    cap = steps + 1
    nl = n * hh
    zc = xp.ascontiguousarray(z, dtype=xp.float32)
    fc = xp.ascontiguousarray(fg, dtype=xp.float32) if fg is not None \
        else xp.zeros(1, xp.float32)
    out = xp.zeros((n if horizon else nlay) * hh * ww, xp.float32)
    sj = xp.empty(nl * cap, xp.int32)
    sz = xp.empty(nl * cap, xp.float32)
    blk = 128
    kern(((nl + blk - 1) // blk,), (blk,),
         (zc, fc, np.int32(fg is not None),
          xp.asarray(np.asarray(sgns, np.int32)), xp.asarray(np.asarray(ms, np.float32)),
          xp.asarray(np.asarray(lay, np.int32)), np.int32(n),
          np.int32(hh), np.int32(ww), np.int32(run), np.int32(steps),
          np.float32(radius), np.int32(1 if horizon else 0),
          out, sj, sz, np.int32(cap)))
    return out.reshape(n if horizon else nlay, hh, ww)


def _lsao_sweep(z, fg, sgns, ms, lay, radius, horizon=False):
    """Several directions of line_sweep_ao at once. For direction k the
    lines step along axis 1 by ``sgns[k]`` (±1) px, moving ``ms[k]`` px along
    axis 0 per step (|m| ≤ 1); all H lines of all the directions go side by
    side as one long vector, so the Python loop runs once per step for the
    whole group, not once per direction. At every step the lines of one
    direction stand in one column at rows k + f (one fraction f for all of
    them). z, fg: stacks L×H×W (the picture and, for a square tile, its
    transpose, so that both axes go in one loop); lay[k] — which layer
    direction k walks. Returns L×H×W: summed occlusion per layer.

    horizon=True (for GTAO): instead of occlusion, the sine of each
    direction's horizon elevation, with the falloff applied the GTAO way
    (lerp towards −1 — no horizon — by the weight 1 / (1 + (d/radius)⁴):
    near 1 within the radius, as GTAO's plateau, with a smooth tail);
    −1 where nothing lies ahead. Returns n×H×W, one map per direction."""
    kern = X.kernel(_LSAO_CU, 'lsao_sweep') if X.is_dev(z) else None
    if kern is not None:
        return _lsao_sweep_gpu(kern, z, fg, sgns, ms, lay, radius, horizon)
    xp = X.xp(z)
    nlay, hh, ww = z.shape
    n = len(ms)
    nl = n * hh                                  # всех линий в группе
    # разгон: дальше 8·radius вклад < 1/65, полный круг не нужен
    run = min(ww, int(math.ceil(8.0 * radius)))
    steps = ww + run
    cap = steps + 1
    # таблицы по шагам для каждого направления (маленькие, n чисел на шаг)
    jj = np.arange(steps, dtype=np.float64)[:, None]
    mm = np.asarray(ms, np.float64)[None, :]
    sh_t = np.floor(jj * mm)
    f_t = xp.asarray((jj * mm - sh_t).astype(np.float32))
    sh_t = xp.asarray(sh_t.astype(np.int64))
    x_t = xp.asarray(((np.asarray(sgns)[None, :] * (np.arange(steps)[:, None] - run)) % ww)
                     .astype(np.int64))
    ln = xp.asarray(np.repeat(np.sqrt(1.0 + np.asarray(ms, np.float64) ** 2), hh)
                    .astype(np.float32))      # длина шага у каждой линии, пикс.
    dl = xp.repeat(xp.arange(n), hh)          # направление каждой линии
    lofs = xp.repeat(xp.asarray(np.asarray(lay, np.int64) * hh * ww), hh)  # слой линии
    ar_h = xp.arange(hh)
    sj = xp.zeros((nl, cap), xp.float32)      # оболочка: номер шага точки
    sz = xp.zeros((nl, cap), xp.float32)      # и её высота
    top = xp.full(nl, -1, xp.int32)           # вершина стека у каждой линии
    ar = xp.arange(nl)
    base_s = ar * cap                         # начало стека линии в плоском массиве
    sjf, szf = sj.ravel(), sz.ravel()
    out = xp.zeros(n * hh * ww, xp.float32)   # своё полотно на направление
    zf = z.ravel()
    ff = None if fg is None else fg.ravel()
    zprev = None
    for j in range(steps):                    # сначала разгон, потом пишем
        r = ((ar_h[None, :] + sh_t[j][:, None]) % hh).ravel()
        r1 = (r + 1) % hh
        x = xp.repeat(x_t[j], hh)
        f = xp.repeat(f_t[j], hh)
        i0, i1 = lofs + r * ww + x, lofs + r1 * ww + x
        zp = (1.0 - f) * zf[i0] + f * zf[i1]
        # выпуклая оболочка: снимаем вершины, которые из новой точки
        # не видны как горизонт (ниже отрезка к предыдущей вершине)
        # дальше считаем только линии, где ещё что-то снимается
        act = ar[top >= 1]
        while act.size:
            ti = base_s[act] + top[act]
            zpa = zp[act]
            pop = ((szf[ti - 1] - zpa) * (j - sjf[ti]) >=
                   (szf[ti] - zpa) * (j - sjf[ti - 1]))
            act = act[pop]
            top[act] -= 1
            act = act[top[act] >= 1]
        if j >= run and horizon:
            has = top >= 0
            ti = base_s + xp.maximum(top, 0)
            d = (j - sjf[ti]) * ln
            e = (szf[ti] - zp) / xp.maximum(d, 1e-6)
            sv = xp.where(has, e / xp.sqrt(1.0 + e * e), -1.0)
            val = -1.0 + (sv + 1.0) / (1.0 + (d / radius) ** 4)
            base = dl * (hh * ww) - lofs
            out[base + i0] += (1.0 - f) * val
            out[base + i1] += f * val
        elif j >= run:
            has = top >= 0
            ti = base_s + xp.maximum(top, 0)
            d = (j - sjf[ti]) * ln
            e = (szf[ti] - zp) / xp.maximum(d, 1e-6)
            s_h = xp.where(has, xp.maximum(e / xp.sqrt(1.0 + e * e), 0.0), 0.0)
            # касательная к поверхности в эту сторону (по соседней точке);
            # вниз не считаем — иначе край у фона давал бы чёрную кайму
            et = xp.maximum((zprev - zp) / ln, 0.0)
            s_t = et / xp.sqrt(1.0 + et * et)
            occ = xp.maximum(s_h - s_t, 0.0) / (1.0 + (d / radius) ** 2)
            if ff is not None:
                occ = occ * (((1.0 - f) * ff[i0] + f * ff[i1]) >= 0.5)
            base = dl * (hh * ww) - lofs
            out[base + i0] += (1.0 - f) * occ
            out[base + i1] += f * occ
        top += 1
        ti = base_s + top
        sjf[ti] = j
        szf[ti] = zp
        zprev = zp
    out = out.reshape(n, hh, ww)
    if horizon:
        return out
    res = xp.zeros((nlay, hh, ww), xp.float32)
    for k in range(n):
        res[lay[k]] += out[k]
    return res


def _sweep_dirs(z, fg, n, r, horizon=False):
    """Run _lsao_sweep for n directions round the circle (direction k looks
    towards angle 2π(k + 0.5)/n, screen convention: 0 — right, π/2 — up).
    Returns the summed H×W occlusion, or with horizon=True the list of n
    horizon maps (H×W each, in the order of k)."""
    xp = X.xp(z)
    dirs_all = []                            # (sgn, m, слой, k)
    for k in range(n):
        a = 2.0 * math.pi * (k + 0.5) / n
        dx, dy = math.cos(a), -math.sin(a)   # куда смотрим (y экрана вниз)
        # идём навстречу взгляду: пройденные точки лежат по направлению d
        if abs(dx) >= abs(dy):
            dirs_all.append((-1 if dx > 0 else 1, -dy / abs(dx), 0, k))
        else:
            dirs_all.append((-1 if dy > 0 else 1, -dx / abs(dy), 1, k))
    # квадратная плитка: картинка и её транспонированная копия идут одной
    # пачкой (обе оси за один проход); иначе — каждая ось своей пачкой.
    # В пачке не больше направлений, чем влезает в ~128 МБ полотен.
    per = max(1, int(32e6 // max(z.size, 1)))
    zt = xp.ascontiguousarray(z.T)
    ft = None if fg is None else xp.ascontiguousarray(fg.T)
    if z.shape[0] == z.shape[1]:
        packs = [(xp.stack([z, zt]), None if fg is None else xp.stack([fg, ft]),
                  dirs_all, False)]
    else:
        packs = [(z[None], None if fg is None else fg[None],
                  [(a, b, 0, k) for a, b, L, k in dirs_all if L == 0], False),
                 (zt[None], None if ft is None else ft[None],
                  [(a, b, 0, k) for a, b, L, k in dirs_all if L == 1], True)]
    occ = xp.zeros_like(z)
    hz = [None] * n
    for zs, fs, grp, tr_pack in packs:
        for c in range(0, len(grp), per):
            part = grp[c:c + per]
            res = _lsao_sweep(zs, fs, [g[0] for g in part], [g[1] for g in part],
                              [g[2] for g in part], r, horizon)
            if horizon:
                for q, g in enumerate(part):
                    # слой 1 (или вторая пачка) — по транспонированной картинке
                    tr = tr_pack or g[2] == 1
                    hz[g[3]] = res[q].T if tr else res[q]
                continue
            for L in range(res.shape[0]):
                tr = tr_pack or L == 1
                occ = occ + (res[L].T if tr else res[L])
    return hz if horizon else occ


def gtao(hpx, radius, dirs=12, mask=None, normal=None):
    """Ground-truth ambient occlusion (Jimenez, Wu, Pesce, Jarabo,
    "Practical Real-Time Strategies for Accurate Indirect Occlusion", 2016),
    on a height map seen from straight above, on the torus.

    hpx — H×W height in pixels; radius — falloff distance, px (a horizon at
    distance d counts with weight 1 / (1 + (d/radius)⁴), lerped towards no
    horizon as in GTAO); dirs — directions round the circle, rounded up to
    an even number (they pair into dirs/2 slices); mask — None or H×W 0/1:
    the background never occludes and gets 1.

    For every slice the two horizons (one to each side) come from the
    line sweep of line_sweep_ao — exact, at any distance. The surface
    normal (from the height slopes, wrapping) is projected into the slice;
    the horizons are clamped to the hemisphere round it, and the
    cosine-weighted visible part is integrated in closed form:
    a = ¼·(−cos(2h₁ − γ) + cos γ + 2h₁ sin γ) + the same for h₂, weighted by
    the length of the projected normal. Unlike HBAO/LSAO this is the share
    of diffuse light that really reaches the surface (cosine law), so a flat
    open surface is exactly 1 and a slope is not darkened for being a slope.
    normal — None (the tilt comes from the slopes of hpx) or an H×W×3 unit
    normal in the same convention (x along columns, y along rows, as
    (−∂h/∂x, −∂h/∂y, 1) normalised), e.g. from a normal map with finer
    detail than the height.
    Returns H×W 0…1, 1 = open."""
    xp = X.xp(hpx)
    z = xp.asarray(hpx, xp.float32)
    fg = None
    if mask is not None:
        fg = (xp.asarray(mask) >= 0.5).astype(xp.float32)
        z = xp.where(fg > 0, z, float(z.min()) - 4.0 * float(radius) - 1e3)
    r = max(float(radius), 1.0)
    n = max(2, int(dirs) + (int(dirs) % 2))
    hz = _sweep_dirs(z, fg, n, r, horizon=True)
    if normal is not None:
        nx, ny, nz = normal[..., 0], normal[..., 1], normal[..., 2]
    else:
        # наклон поверхности (по исходной высоте, с заворотом)
        h0 = xp.asarray(hpx, xp.float32)
        gx = (xp.roll(h0, -1, 1) - xp.roll(h0, 1, 1)) * 0.5
        gy = (xp.roll(h0, -1, 0) - xp.roll(h0, 1, 0)) * 0.5   # y экрана вниз
        inv = 1.0 / xp.sqrt(gx * gx + gy * gy + 1.0)
        nx, ny, nz = -gx * inv, -gy * inv, inv
    vis = xp.zeros_like(z)
    half = n // 2
    for k in range(half):
        a = 2.0 * math.pi * (k + 0.5) / n
        dx, dy = math.cos(a), -math.sin(a)
        # горизонты от вертикали: в сторону d — плюс, в обратную — минус
        h2 = 0.5 * math.pi - xp.arcsin(xp.clip(hz[k], -1.0, 1.0))
        h1 = -(0.5 * math.pi - xp.arcsin(xp.clip(hz[k + half], -1.0, 1.0)))
        nd = nx * dx + ny * dy                 # нормаль вдоль среза
        plen = xp.sqrt(nd * nd + nz * nz)
        gam = xp.arctan2(nd, nz)
        h1 = gam + xp.maximum(h1 - gam, -0.5 * math.pi)
        h2 = gam + xp.minimum(h2 - gam, 0.5 * math.pi)
        sg, cg = xp.sin(gam), xp.cos(gam)
        a1 = -xp.cos(2.0 * h1 - gam) + cg + 2.0 * h1 * sg
        a2 = -xp.cos(2.0 * h2 - gam) + cg + 2.0 * h2 * sg
        vis = vis + plen * 0.25 * (a1 + a2)
    ao = xp.clip(vis / half, 0.0, 1.0)
    if fg is not None:
        ao = ao * fg + (1.0 - fg)
    return ao.astype(xp.float32)


def line_sweep_ao(hpx, radius, dirs=12, mask=None):
    """Line-sweep ambient obscurance (Timonen, "Line-Sweep Ambient
    Obscurance", EGSR 2013), on the torus.

    hpx — H×W height in pixels; radius — falloff distance, px: an occluder
    at distance d counts with weight 1 / (1 + (d/radius)²); dirs — number of
    directions round the circle; mask — None or H×W 0/1 (the object): the
    background never occludes and gets 1.

    For every direction all lines of the image in that direction are swept
    at once; along each line the upper convex hull of the heights already
    passed is kept on a stack, so the true horizon (the highest elevation at
    any distance, not only at sampled distances) is found in amortised
    constant time per pixel. Occlusion of a direction is the sine of the
    horizon above the surface tangent (as in HBAO), times the falloff of the
    horizon point's distance. Lines wrap round the tile, with a run-up of
    min(tile, 8·radius) px (farther points weigh under 1/65), so the result
    tiles. Lines at fractional slopes are sampled
    and written back with linear weights between the two nearest rows.
    Returns H×W 0…1, 1 = open."""
    xp = X.xp(hpx)
    z = xp.asarray(hpx, xp.float32)
    fg = None
    if mask is not None:
        fg = (xp.asarray(mask) >= 0.5).astype(xp.float32)
        # фон проваливаем вниз: он никогда не станет горизонтом
        z = xp.where(fg > 0, z, float(z.min()) - 4.0 * float(radius) - 1e3)
    r = max(float(radius), 1.0)
    n = max(int(dirs), 1)
    occ = _sweep_dirs(z, fg, n, r)
    ao = xp.clip(1.0 - occ / n, 0.0, 1.0)
    if fg is not None:
        ao = ao * fg + (1.0 - fg)
    return ao.astype(xp.float32)


def directional_occlusion(hpx, azimuth, radius, dirs=16, spread=60.0, steps=12):
    """Occlusion of sky light that comes mostly from one side: horizon sines
    weighted by a lobe cos^p round ``azimuth`` (degrees), p chosen so the
    lobe's half-width is ``spread`` degrees. Returns H×W 0…1 (1 = open)."""
    xp = X.xp(hpx)
    angs = [2 * math.pi * k / dirs for k in range(int(dirs))]
    a0 = math.radians(float(azimuth))
    half = math.radians(max(float(spread), 5.0))
    p = math.log(0.5) / math.log(max(math.cos(min(half, 1.5)), 1e-3))
    w = np.array([max(math.cos(a - a0), 0.0) ** p for a in angs])
    w = w / max(w.sum(), 1e-9)
    hz = horizon(hpx, angs, radius, steps)
    occ = xp.zeros_like(hpx)
    for k in range(len(angs)):
        if w[k] > 1e-4:
            occ = occ + float(w[k]) * hz[k]
    return xp.clip(1.0 - occ, 0, 1).astype(xp.float32)


def shade_strength(rgb, shade, floor=0.25, r0=4.0):
    """How strongly the picture really is darkened where ``shade`` says so —
    the strength for remove_shading, found from the data.

    Shading changes brightness, not chromaticity. So the fine detail
    (finer than r0 px) of log brightness is compared with that of
    log shade, by least squares: brightness ≈ … + s·shade. Near chromaticity
    edges (a change of material, not of light: the colour itself changes)
    a step in brightness says nothing about the shade, so a band of 4·r0
    px round them is left out. rgb sRGB H×W×3, shade H×W, on the card or
    the processor (the module of rgb).
    Returns (s in 0…3, share of pixels used 0…1), or (None, share) when
    too few are left."""
    import matgen as MG
    xp, nd = X.xp(rgb), X.ndi(rgb)
    lin = MG.srgb_to_lin(xp.clip(xp.asarray(rgb[..., :3], xp.float32), 0, 1))
    L = lin @ xp.asarray(_LUMW)
    lI = xp.log(xp.maximum(L, 1e-4)).astype(xp.float32)
    lS = xp.log(xp.maximum(xp.clip(xp.asarray(shade, xp.float32), 0, 1),
                           max(float(floor), 1e-3))).astype(xp.float32)
    ch = lin / xp.maximum(lin.sum(-1, keepdims=True), 1e-4)
    cg = xp.zeros(L.shape, xp.float32)
    for c in (0, 1):
        # цветность чуть сглажена: шум отдельных пикселей — не смена материала
        gx, gy = _grad_wrap(MG.fast_blur(ch[..., c], max(1.0, 0.5 * r0), mode='wrap'))
        cg += gx * gx + gy * gy
    cg = xp.sqrt(cg)
    edge = cg > 6.0 * float(xp.median(cg)) + 1e-6
    near = nd.maximum_filter(edge.astype(xp.uint8), size=int(4 * r0) + 1, mode='wrap') > 0
    w = ~near & (L > 0.003)
    used = float(w.mean())
    hI = lI - MG.fast_blur(lI, r0, mode='wrap')
    hS = lS - MG.fast_blur(lS, r0, mode='wrap')
    den = float((hS[w] * hS[w]).sum())
    if used < 0.01 or den < 1e-6:
        return None, used
    # не больше 3: сильнее картинка почти не бывает затенена, а выброс раздул бы тени
    return min(3.0, max(0.0, float((hI[w] * hS[w]).sum()) / den)), used


def halo_guard(src, out, radius, k=6):
    """Keep remove_shading from making halos: no pixel is made brighter
    than the lit places of its own colour nearby.

    Pixels are grouped by chromaticity (k-means, k groups); for each group
    the local level of lit is its mean brightness + one standard deviation
    in a window of ``radius`` px (blurs masked by the group). A pixel of
    ``out`` brighter than that level — and than the source pixel itself —
    is scaled down to it. src, out — sRGB H×W×3, on the card or the
    processor (the module of src). The k-means runs on a fixed sample of
    50 000 pixels (the same seed), so the groups do not depend on where it
    runs. Returns (out, share of pixels changed)."""
    import matgen as MG
    xp = X.xp(src)
    I = MG.srgb_to_lin(xp.clip(xp.asarray(src[..., :3], xp.float32), 0, 1))
    O = MG.srgb_to_lin(xp.clip(xp.asarray(out[..., :3], xp.float32), 0, 1))
    lw = xp.asarray(_LUMW)
    LI, LO = I @ lw, O @ lw
    ch = (I / xp.maximum(I.sum(-1, keepdims=True), 1e-4))[..., :2]
    f = ch.reshape(-1, 2)
    rng = np.random.default_rng(0)
    # выборка и центры — маленькие, считаются на процессоре одинаково везде
    smp = X.to_host(f[xp.asarray(rng.choice(f.shape[0], min(f.shape[0], 50000),
                                            replace=False))])
    cen = [smp[0]]                              # k-means++: центры подальше друг от друга
    for _ in range(1, k):
        d2 = np.min(((smp[:, None, :] - np.array(cen)[None]) ** 2).sum(-1), 1)
        cen.append(smp[int(rng.choice(len(smp), p=d2 / max(d2.sum(), 1e-12)))])
    cen = np.array(cen)
    for _ in range(12):
        lab = np.argmin(((smp[:, None, :] - cen[None]) ** 2).sum(-1), 1)
        cen = np.array([smp[lab == j].mean(0) if (lab == j).any() else cen[j]
                        for j in range(k)])
    cd = xp.asarray(cen.astype(np.float32))
    lab = xp.argmin(((f[:, None, :] - cd[None]) ** 2).sum(-1), 1).reshape(LI.shape)
    ref = xp.zeros_like(LI)
    for j in range(k):
        m = (lab == j).astype(xp.float32)
        if not bool(m.any()):
            continue
        den = xp.maximum(MG.fast_blur(m, radius, mode='wrap'), 1e-4)
        mu = MG.fast_blur(m * LI, radius, mode='wrap') / den
        sq = MG.fast_blur(m * LI * LI, radius, mode='wrap') / den
        ref = xp.where(lab == j, mu + xp.sqrt(xp.maximum(sq - mu * mu, 0.0)), ref)
    cap = xp.maximum(ref, LI)
    k_ = xp.minimum(1.0, cap / xp.maximum(LO, 1e-6))
    res = MG.lin_to_srgb(xp.clip(O * k_[..., None], 0, 1)).astype(xp.float32)
    return res, float((k_ < 0.999).mean())


def remove_shading(rgb, shade, strength=1.0, floor=0.25):
    """Take baked shading (AO, shadows) out of a picture: in linear light the
    colour is divided by shade^strength, the divisor kept at least ``floor``
    so deep black does not blow up into noise. shade: H×W, 1 = lit."""
    import matgen as MG
    xp = X.xp(rgb)
    lin = MG.srgb_to_lin(xp.clip(rgb[..., :3], 0, 1))
    d = xp.maximum(xp.clip(shade, 0, 1), float(floor)) ** float(strength)
    return xp.clip(MG.lin_to_srgb(xp.clip(lin / d[..., None], 0, 1)), 0, 1).astype(xp.float32)


# ------------------------------------------------------------ сглаживание лесенки --
def _sample_wrap(img, ys, xs):
    """Bilinear samples of a tiling H×W or H×W×C map at fractional (ys, xs),
    on the card or the processor (the module of img)."""
    xp, nd = X.xp(img), X.ndi(img)
    crd = xp.stack([xp.asarray(ys, xp.float32), xp.asarray(xs, xp.float32)])
    if img.ndim == 2:
        return nd.map_coordinates(img, crd, order=1, mode='grid-wrap')
    return xp.stack([nd.map_coordinates(xp.ascontiguousarray(img[..., c]), crd, order=1,
                                        mode='grid-wrap') for c in range(img.shape[2])], -1)


def _fxaa(img, L, subpix):
    """FXAA (T. Lottes, NVIDIA, FXAA 3.11 «quality», the same thresholds and
    search steps), in numpy, on a tiling map. L — the brightness the edges
    are looked for in; subpix — 0…1, how much single-pixel steps are
    softened (0.75 in the original). On the card or the processor."""
    xp = X.xp(L)
    H, W = L.shape
    r = lambda a, dy, dx: xp.roll(xp.roll(a, -dy, 0), -dx, 1)   # сосед (y+dy, x+dx)
    M, N, S, E, Wl = L, r(L, -1, 0), r(L, 1, 0), r(L, 0, 1), r(L, 0, -1)
    NW, NE, SW, SE = r(L, -1, -1), r(L, -1, 1), r(L, 1, -1), r(L, 1, 1)
    mx = xp.maximum(xp.maximum(xp.maximum(M, N), xp.maximum(S, E)), Wl)
    mn = xp.minimum(xp.minimum(xp.minimum(M, N), xp.minimum(S, E)), Wl)
    rng_ = mx - mn
    edge = rng_ >= xp.maximum(0.0312, mx * 0.125)
    yy, xx = xp.nonzero(edge)
    out = img.copy()
    if yy.size == 0:
        return out, 0.0
    g = lambda a: a[yy, xx]
    m, n, s, e, w = g(M), g(N), g(S), g(E), g(Wl)
    nw, ne, sw, se = g(NW), g(NE), g(SW), g(SE)
    rg = g(rng_)
    # мягкость отдельных пикселей
    la = (2.0 * (n + s + e + w) + nw + ne + sw + se) / 12.0
    sb = xp.clip(xp.abs(la - m) / rg, 0, 1)
    sc = (-2.0 * sb + 3.0) * sb * sb
    sub = sc * sc * float(subpix)
    # куда идёт край
    ev = xp.abs(nw + sw - 2 * w) + 2 * xp.abs(n + s - 2 * m) + xp.abs(ne + se - 2 * e)
    eh = xp.abs(nw + ne - 2 * n) + 2 * xp.abs(w + e - 2 * m) + xp.abs(sw + se - 2 * s)
    horz = eh >= ev                       # край горизонтальный: поперёк — по y
    l1 = xp.where(horz, n, w)             # сосед «до» (сверху / слева)
    l2 = xp.where(horz, s, e)
    g1, g2 = xp.abs(l1 - m), xp.abs(l2 - m)
    side1 = g1 >= g2
    step = xp.where(side1, -1.0, 1.0)     # поперёк края, в сторону большего перепада
    lavg = 0.5 * (xp.where(side1, l1, l2) + m)
    gs = 0.25 * xp.maximum(g1, g2)
    # старт — на полпикселя поперёк, на самом крае
    py = yy + xp.where(horz, step * 0.5, 0.0)
    px = xx + xp.where(horz, 0.0, step * 0.5)
    dy, dx = xp.where(horz, 0.0, 1.0), xp.where(horz, 1.0, 0.0)  # вдоль края
    steps = (1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 2.0, 2.0, 2.0, 2.0, 4.0, 8.0)
    oN, oP = xp.zeros_like(m) + steps[0], xp.zeros_like(m) + steps[0]
    dN, dP = xp.zeros(m.shape, bool), xp.zeros(m.shape, bool)
    eN = eP = None
    for k, st in enumerate(steps):
        if k:
            oN = xp.where(dN, oN, oN + st)
            oP = xp.where(dP, oP, oP + st)
        vN = _sample_wrap(L, py - dy * oN, px - dx * oN) - lavg
        vP = _sample_wrap(L, py + dy * oP, px + dx * oP) - lavg
        eN = vN if eN is None else xp.where(dN, eN, vN)
        eP = vP if eP is None else xp.where(dP, eP, vP)
        dN |= xp.abs(eN) >= gs
        dP |= xp.abs(eP) >= gs
        if bool(dN.all()) and bool(dP.all()):
            break
    span = oN + oP
    dmin = xp.minimum(oN, oP)
    end = xp.where(oN < oP, eN, eP)
    good = (end < 0) != ((m - lavg) < 0)
    off = xp.where(good, 0.5 - dmin / xp.maximum(span, 1e-6), 0.0)
    off = xp.maximum(off, sub)
    sy = yy + xp.where(horz, step * off, 0.0)
    sx = xx + xp.where(horz, 0.0, step * off)
    out[yy, xx] = _sample_wrap(img, sy, sx)
    return out, float(edge.mean())


def _contour_aa(img, L, amount, radius):
    """Smoothing along contours: the direction of a contour is taken from
    the structure tensor of the brightness (gradients blurred over
    ``radius`` px); where the contour is clear (coherence) and the step is
    noticeable, the map is blurred only along it — over ±2·radius px — so
    the staircase goes and the edge itself stays sharp. On the card or the
    processor."""
    import matgen as MG
    xp = X.xp(L)
    gx = (xp.roll(L, -1, 1) - xp.roll(L, 1, 1)) * 0.5
    gy = (xp.roll(L, -1, 0) - xp.roll(L, 1, 0)) * 0.5
    r = max(0.6, float(radius))
    jxx = MG.fast_blur(gx * gx, r, mode='wrap')
    jxy = MG.fast_blur(gx * gy, r, mode='wrap')
    jyy = MG.fast_blur(gy * gy, r, mode='wrap')
    tr = jxx + jyy
    disc = xp.sqrt((jxx - jyy) ** 2 + 4 * jxy * jxy)
    coh = xp.where(tr > 1e-12, disc / xp.maximum(tr, 1e-12), 0.0) ** 2
    th = 0.5 * xp.arctan2(2 * jxy, jxx - jyy)          # направление градиента
    ty, tx = xp.cos(th), -xp.sin(th)                    # вдоль контура
    mag = xp.sqrt(xp.maximum(0.5 * (tr + disc), 0))
    ref = float(xp.percentile(mag, 99)) or 1.0
    w = xp.clip(float(amount) * coh * xp.clip(mag / (0.15 * ref), 0, 1), 0, 1)
    sel = w > 0.01
    out = img.copy()
    yy, xx = xp.nonzero(sel)
    if yy.size == 0:
        return out, 0.0
    tyy, txx = ty[sel], tx[sel]
    acc = xp.zeros((yy.size,) + img.shape[2:], xp.float32)
    tot = 0.0
    n = max(1, int(round(2 * r)))
    for k in range(-n, n + 1):
        wk = math.exp(-0.5 * (k / r) ** 2)
        acc = acc + wk * _sample_wrap(img, yy + tyy * k, xx + txx * k)
        tot += wk
    sm = acc / tot
    ww = w[sel] if img.ndim == 2 else w[sel][:, None]
    out[yy, xx] = img[yy, xx] * (1 - ww) + sm * ww
    return out, float(sel.mean())


def _mask_aa(img, amount, radius):
    """For masks and other two-level maps (alpha, threshold, posterize, two
    flat colours): the map is blurred over ``radius`` px and cut again so
    that the edge is one pixel wide: the step goes round smoothly instead
    of by stairs. A gaussian of σ px makes a step σ·√(2π) px wide, so the
    contrast round the local middle (between the darkest and the lightest
    nearby, per channel) is raised by that much, within those two levels.
    Details thinner than about radius px fade. On the card or the processor."""
    import matgen as MG
    xp, nd = X.xp(img), X.ndi(img)
    r = max(0.5, float(radius))
    k = r * math.sqrt(2 * math.pi)
    size = 2 * int(math.ceil(2 * r)) + 1

    def one(c):
        # «два уровня» берутся по месту — тёмный и светлый рядом, так
        # работает и для двух цветов, а не только для 0 и 1
        lo = nd.minimum_filter(c, size=size, mode='wrap')
        hi = nd.maximum_filter(c, size=size, mode='wrap')
        mid = 0.5 * (lo + hi)
        bl = MG.fast_blur(xp.ascontiguousarray(c), r, mode='wrap')
        return xp.clip(mid + (bl - mid) * k, lo, hi)
    out = one(img) if img.ndim == 2 else xp.stack([one(img[..., c])
                                                   for c in range(img.shape[2])], -1)
    return (img + (out - img) * float(amount)).astype(xp.float32), 1.0


def antialias(x, algo='fxaa', amount=1.0, radius=1.0):
    """Remove the staircase from edges of a map (numpy, tiling).

    algo: 'fxaa' — FXAA (edges by brightness, the pixel is resampled across
    the edge by the found offset; amount — the softening of single-pixel
    steps, 0…1); 'contour' — blur along the contour only (_contour_aa;
    amount — how much, radius — the scale of the stairs); 'mask' — blur and
    cut again, for two-level maps (_mask_aa). x: H×W or H×W×C 0…1, or a
    signed normal (−1…1: it is worked in 0…1 and normalised again).
    Returns (map, share of pixels changed by more than 1/255). On the card or
    the processor (the module of x)."""
    xp = X.xp(x)
    x = xp.asarray(x, xp.float32)
    signed = x.ndim == 3 and float(x[..., :3].min()) < -0.01
    img = x * 0.5 + 0.5 if signed else x
    if img.ndim == 2:
        L = img
    else:
        L = img[..., :3] @ xp.asarray(_LUMW) if img.shape[2] >= 3 else img[..., 0]
    if algo == 'contour':
        out, _ = _contour_aa(img, L, amount, radius)
    elif algo == 'mask':
        out, _ = _mask_aa(img, amount, radius)
    else:
        out, _ = _fxaa(img, L, amount)
    # доля пикселей, изменившихся заметно (больше 1/255 хоть в одном канале)
    dif = xp.abs(out - img)
    share = float(((dif if dif.ndim == 2 else dif.max(-1)) > 1.0 / 255).mean())
    if signed:
        out = out * 2.0 - 1.0
        nrm = out[..., :3] / xp.maximum(xp.linalg.norm(out[..., :3], axis=-1, keepdims=True), 1e-6)
        out = xp.concatenate([nrm, out[..., 3:]], -1)
    return out.astype(xp.float32), share
