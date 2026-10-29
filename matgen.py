# -*- coding: utf-8 -*-
"""
Движок для непрозрачных текстур: высота, нормаль, AO, specular/roughness,
снятие солнечных теней, бесшовность.

Отдельно от листвы: там форма берётся из альфа-канала, здесь альфы нет
и опираться приходится на саму картинку.

Engine for opaque textures: height, normal, AO, specular/roughness, removal of
sun shadows (de-lighting), seamless tiling.

Kept separate from foliage: there the shape comes from the alpha channel; here
there is no alpha, so everything has to be inferred from the picture itself
(a finished diffuse that carries no geometry).

Conventions used throughout this module:

* Colour input is sRGB float in 0…1, shape H×W×3. Anything that is summed,
  averaged or divided is done in linear light (``srgb_to_lin`` /
  ``lin_to_srgb``, a plain 2.2 power curve).
* Everything is computed on a torus. Textures tile, so every convolution,
  gradient and integration wraps around the edges: ``np.roll``, ``mode='wrap'``
  and FFT solves with periodic boundaries. A single ``mode='reflect'`` on a path
  that ends up in a map produces a visible seam when tiled. Seamlessness is
  checked as exact shift-equivariance (shifting the input shifts the map by the
  same amount, to ~1e-6).
* Normals are H×W×3 unit vectors (x right, y, z out of the surface). The sign of
  y follows the ``directx`` flag: ``directx=True`` means the green channel points
  down the image (y grows with the row index), ``False`` flips it (OpenGL).
  Slopes ``p, q`` are in image coordinates: ``p = dh/dx`` (to the right),
  ``q = dh/dy`` (down), in height units per pixel.
* Height maps handed between stages are float32 H×W in 0…1 unless a function
  says otherwise (several intermediate results are zero-mean instead; each
  docstring states which).
* Parameters called radius/scale are in pixels of the image being processed.
  Previews run on a downscaled copy; ``at_scale`` converts full-size parameters.

Main references: Frankot & Chellappa 1988 (integrating a slope field with
enforced integrability); Horn (shape from shading) and Pentland (linear shape
from shading, illuminant direction); Land's retinex and Finlayson et al. (shadow
removal via chromaticity); He, Sun & Tang 2010 (guided filter); Bavoil, Sainz &
Dimitrov 2008 (HBAO).
"""

import math

import numpy as np

import xpu as X

BUILD = 25          # версия набора файлов, проверяется при запуске

# ------------------------------------------------------------------ базовое
def srgb_to_lin(c):
    """sRGB 0…1 to linear light, as a plain 2.2 power (input clipped to 0…1)."""
    xp = X.xp(c)
    return xp.power(xp.clip(c, 0, 1), 2.2)


def lin_to_srgb(c):
    """Linear light to sRGB 0…1, inverse of ``srgb_to_lin`` (power 1/2.2)."""
    xp = X.xp(c)
    return xp.power(xp.clip(c, 0, 1), 1 / 2.2)


def luminance(lin):
    """Rec. 709 luminance of a LINEAR ...×3 array; returns the array without the
    channel axis (H×W for an image)."""
    return 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]


def fast_blur(x, sigma, mode='wrap'):
    """Gaussian blur that stays fast at large radii.

    Gaussian for small radii, three box passes for large ones. A box filter's
    run time does not depend on the radius, and three passes are already
    indistinguishable from a Gaussian — at radius 150 that is a speed-up of
    tens of times.

    x     — 2-D (or N-D) array; the dtype is preserved by ``ndimage``.
    sigma — Gaussian sigma in pixels. Below 6 an exact Gaussian is used; from 6
            up, three box passes of odd width w ≈ sqrt(4·sigma² + 1), which gives
            the same variance (three boxes of width w have variance 3·(w²−1)/12).
    mode  — edge mode, 'wrap' by default: blurring on the torus keeps a tiling
            texture seamless. Do not pass 'reflect' on any path that ends up in
            a map.

    Returns an array of the same shape as x."""
    xp = X.xp(x, sigma, mode)
    nd = X.ndi(x, sigma, mode)
    if sigma < 6:
        return nd.gaussian_filter(x, sigma, mode=mode)
    # math, не xp: у CuPy корень из числа — 0-мерный массив без round()
    w = max(3, int(round(math.sqrt(12.0 * sigma * sigma / 3.0 + 1.0))))
    if w % 2 == 0:
        w += 1
    y = x
    for _ in range(3):
        y = nd.uniform_filter(y, size=w, mode=mode)
    return y


def box(x, r):
    """Box mean over a (2r+1)×(2r+1) window in pixels (r ≥ 1), wrapping edges."""
    nd = X.ndi(x, r)
    return nd.uniform_filter(x, size=int(max(1, r)) * 2 + 1, mode='wrap')


def guided_filter(guide, src, r, eps):
    """Edge-preserving smoothing (the guided filter of He, Sun & Tang, 2010).

    Smoothing that does not smear sharp boundaries. It is needed to separate a
    smooth light gradient from the sharp edge of a shadow: an ordinary blur
    drags the shadow edge into the illumination estimate and then plants a halo
    around it.

    guide — H×W guidance image (here usually the same as src, normalised 0…1).
    src   — H×W image to smooth.
    r     — window radius in pixels (window 2r+1).
    eps   — regulariser in guide units squared: edges with local variance well
            above eps are kept, texture with variance below it is smoothed away.

    Local linear model src ≈ a·guide + b in every window, with the coefficients
    averaged over overlapping windows. All box means wrap around the edges, so
    the result stays seamless. Returns H×W, same dtype as the inputs."""
    mg, ms = box(guide, r), box(src, r)
    cov = box(guide * src, r) - mg * ms
    var = box(guide * guide, r) - mg * mg
    a = cov / (var + eps)
    b = ms - a * mg
    return box(a, r) * guide + box(b, r)


# ------------------------------------------------- снятие света и солнца ---
def saturation(rgb):
    """HSV-style saturation (max − min) / max per pixel; ...×3 in, ... out, 0…1."""
    xp = X.xp(rgb)
    mx, mn = rgb.max(-1), rgb.min(-1)
    return (mx - mn) / xp.maximum(mx, 1e-4)


def keep_saturation(orig_srgb, new_srgb, amount=1.0):
    """Restore the colourfulness the image had before de-lighting.

    Dividing by the illumination multiplies all three channels by the same
    number, so the RATIO of the channels does not change — formally the
    saturation is the same. But the eye sees not the ratio, it sees the spread
    between channels: a dark greenish corner lifted to mid brightness becomes
    plainly green, and the brightened places look more colourful than the rest
    of the texture.

    Here, for every pixel, we compute how many times brighter it became, and
    the saturation is pulled down by exactly that factor. With amount = 1 the
    absolute colour spread stays what it was; with 0 nothing is done.

    orig_srgb — H×W×3 sRGB 0…1, the image before the change.
    new_srgb  — H×W×3 sRGB 0…1, the image after it (same shape).
    amount    — 0…1 (larger values over-correct); target saturation is
                s_orig · k^(−amount), k = new/old luminance clipped to 0.05…20.

    The colour is then rescaled around its own luminance (grey axis) so that
    the new saturation hits the target; the scale factor is clipped to 0…3.
    Luminance is taken on the sRGB values directly here, which is only a
    perceptual weighting, not a physical quantity. Returns H×W×3 sRGB 0…1.
    """
    xp = X.xp(orig_srgb, new_srgb, amount)
    if amount <= 0:
        return new_srgb
    o = xp.clip(orig_srgb, 0, 1)
    n = xp.clip(new_srgb, 0, 1)
    k = xp.maximum(luminance(n), 1e-4) / xp.maximum(luminance(o), 1e-4)
    s0 = saturation(o) * xp.power(xp.clip(k, 0.05, 20.0), -amount)
    s0 = xp.clip(s0, 0.0, 1.0)
    mx, mn = n.max(-1), n.min(-1)
    g = luminance(n)
    d = mx - mn
    e = mx - g
    den = d - s0 * e
    ok = xp.abs(den) > 1e-4
    kk = xp.where(ok, s0 * xp.maximum(g, 1e-4) / xp.where(ok, den, 1.0), 1.0)
    kk = xp.clip(kk, 0.0, 3.0)
    out = g[..., None] + (n - g[..., None]) * kk[..., None]
    return xp.clip(out, 0, 1)


def delight(rgb_srgb, soft=0.7, hard=0.0, radius=48, keep=1.0,
            restore=0.0, restore_radius=3.0, sat_keep=1.0):
    """Remove baked-in lighting from a diffuse (de-lighting).

    Works in linear light on log luminance: the image is modelled as
    albedo × illumination, the illumination is estimated at a large scale and
    divided out (a retinex-style separation, after Land).

    rgb_srgb — H×W×3 sRGB 0…1.
    soft     — 0…1: remove the smooth light gradient (the overall falloff across
               the texture). The estimate is a wrap-blur of log luminance with
               sigma = radius; it is divided out as a power `soft`, clipped to
               0.3…3.
    hard     — 0…1: remove sharp sun shadows. The illumination estimate is built
               with an edge-preserving filter (guided_filter, radius radius/3,
               at least 4 px), so the shadow boundary does not turn into a halo.
               Divided out as a power `hard`, clipped to 0.25…4.
    radius   — scale of "illumination" in pixels, sensible 16…150.
    restore  — 0…1: bring the fine grain back from the original. By
               multiplication in linear space, not by overlaying on top: that
               way only the small relief detail returns and the colour does not
               shift. The grain is the original log luminance minus its blur
               with sigma = restore_radius (pixels).
    sat_keep — 0…1: hold the original saturation (see keep_saturation).
               Without it the brightened areas come out more colourful than the
               rest of the texture.
    keep     — bring back the overall brightness after the division.
               Note: the result is afterwards rescaled to the original mean
               luminance, which cancels this multiplier exactly, so in the
               current code `keep` has no visible effect.

    The mean luminance of the output equals that of the input, so the texture
    does not get globally brighter or darker. Returns H×W×3 sRGB 0…1."""
    xp = X.xp(rgb_srgb, soft, hard, radius, keep, restore, restore_radius, sat_keep)
    nd = X.ndi(rgb_srgb, soft, hard, radius, keep, restore, restore_radius, sat_keep)
    lin = srgb_to_lin(rgb_srgb)
    L = xp.maximum(luminance(lin), 1e-4)
    logL = xp.log(L)
    out = lin

    if soft > 0:
        big = fast_blur(logL, radius)
        shade = xp.exp(big - big.mean())
        out = out / xp.clip(shade, 0.3, 3.0)[..., None] ** soft

    if hard > 0:
        L2 = xp.maximum(luminance(out), 1e-4)
        g = xp.log(L2)
        gn = (g - g.min()) / max(xp.ptp(g), 1e-6)
        # eps побольше: с маленьким фильтр сохраняет и само зерно, оно попадает
        # в оценку освещения и после деления с картинки исчезает
        sm = guided_filter(gn, gn, max(4, int(radius / 3)), 0.12)
        sm = sm * max(xp.ptp(g), 1e-6) + g.min()
        shade = xp.exp(sm - sm.mean())
        out = out / xp.clip(shade, 0.25, 4.0)[..., None] ** hard

    if restore > 0:
        hp = logL - nd.gaussian_filter(logL, restore_radius, mode='wrap')
        out = out * xp.exp(restore * hp)[..., None]

    if keep != 1.0:
        out = out * keep
    m0 = float(luminance(lin).mean())
    m1 = float(max(luminance(out).mean(), 1e-5))
    out = out * (m0 / m1)                      # вернуть исходную среднюю яркость
    res = lin_to_srgb(out)
    if sat_keep > 0:
        res = keep_saturation(rgb_srgb, res, sat_keep)
    return res


# --------------------------------------------------------------- высота ----
def height_map(rgb_srgb, detail=0.55, contrast=1.0, invert=False, denoise=1.0):
    """Single-layer height from an image that has already been de-lit.

    We take not the luminance itself (then dark paint would become a pit) but a
    band of its frequencies: large albedo blotches are removed, fine relief is
    kept. This is the older one-layer path; MaterialJob uses height_variants
    and mix_heights instead.

    rgb_srgb — H×W×3 sRGB 0…1.
    detail   — 0…1: higher keeps finer relief. It sets both the denoise blur
               (sigma 0.6…3 px × denoise) and the high-pass cut
               (sigma 6…96 px).
    contrast — multiplier after normalising by the 99th percentile.
    invert   — flip high and low.
    denoise  — multiplier on the small blur; 0 disables it.

    Returns H×W in 0…1, 0.5 = mean level. Blurs wrap, so the result tiles."""
    xp = X.xp(rgb_srgb, detail, contrast, invert, denoise)
    nd = X.ndi(rgb_srgb, detail, contrast, invert, denoise)
    L = luminance(srgb_to_lin(rgb_srgb))
    # мелкое зерно — это не рельеф, а шум съёмки и краска: гасим его тем
    # сильнее, чем крупнее рельеф нас интересует
    small = denoise * (0.6 + 2.4 * (1.0 - detail))
    L = nd.gaussian_filter(L, small, mode='wrap') if small > 0 else L
    big = 6.0 + 90.0 * (1.0 - detail) ** 2         # что считать «крупным пятном»
    h = L - fast_blur(L, big)
    s = max(float(xp.percentile(xp.abs(h), 99)), 1e-5)
    h = xp.clip(h / s * contrast, -1, 1) * 0.5 + 0.5
    return 1.0 - h if invert else h


def grad_wrap(h):
    """Central differences with wrap-around at the edges.

    np.gradient computes a one-sided difference at the border, and a seamless
    texture gets a one-pixel stripe along the edge: when tiled, it lines up into
    a visible seam. Here the neighbours are taken with np.roll, i.e. on the
    torus, so the derivative of a tiling map tiles too.

    Also note: these central differences commute (∂y∂x = ∂x∂y exactly on the
    grid), which is why a slope field built with grad_wrap from any height has
    zero curl in slope_curl, not just approximately zero.

    h — H×W array. Returns (gy, gx): derivatives along rows (down the image) and
    along columns (to the right), in units of h per pixel, same shape/dtype."""
    xp = X.xp(h)
    gx = (xp.roll(h, -1, 1) - xp.roll(h, 1, 1)) * 0.5
    gy = (xp.roll(h, -1, 0) - xp.roll(h, 1, 0)) * 0.5
    return gy, gx


def normal_from_height(h, strength=1.0, directx=True,
                       w_fine=1.0, w_mid=0.0, w_big=0.0, scale=8.0):
    """Normal map from a height map, assembled from three scales.

    Each next scale is the gradient of the smoothed height, multiplied by its
    own radius: without that the large layer becomes many times weaker after
    smoothing and is not visible. Thanks to that factor the mid and large
    layers do not depend on the image size; only the fine layer does (see
    at_scale).

    h        — H×W height, normally 0…1.
    strength — overall multiplier on the slopes (internally ×40); ~0.3…3.
    directx  — True: y points down the image (DirectX layout); False: OpenGL.
    w_fine   — weight of the per-pixel gradient.
    w_mid    — weight of the gradient of h blurred with sigma = max(2, scale).
    w_big    — weight of the gradient of h blurred with sigma = max(4, 4·scale).
    scale    — radius of the mid layer in pixels (8 by default).

    All gradients and blurs wrap (grad_wrap, fast_blur), so the normal tiles.
    Returns H×W×3 unit vectors (components −1…1, z > 0), dtype of h."""
    xp = X.xp(h, strength, directx, w_fine, w_mid, w_big, scale)
    gy, gx = grad_wrap(h)
    ax, ay = w_fine * gx, w_fine * gy
    for w, r in ((w_mid, max(2.0, scale)), (w_big, max(4.0, scale * 4.0))):
        if w > 0:
            b = fast_blur(h, r)
            by, bx = grad_wrap(b)
            ax = ax + w * bx * r * 0.5
            ay = ay + w * by * r * 0.5
    nx = -ax * strength * 40.0
    ny = -ay * strength * 40.0
    if not directx:
        ny = -ny
    nz = xp.ones_like(nx)
    L = xp.sqrt(nx * nx + ny * ny + 1.0)
    return xp.stack([nx / L, ny / L, nz / L], -1)


# ------------------------------------------------------------------- AO ----
def band_radius(side, amount):
    """Convert the "scale boundary" slider into a blur radius in pixels.

    At the left end almost everything counts as large; at the right end only
    things noticeably bigger than a brick.

    side   — image side in pixels (callers pass the shorter side).
    amount — slider value, ≥ 0 (typically 0…1): radius = side / (4 + 60·amount),
             i.e. side/4 at 0 and side/64 at 1.

    Returns the radius (sigma) in pixels as a float; used by split_bands."""
    return float(side) / (4.0 + 60.0 * max(float(amount), 0.0))


def split_bands(x, radius):
    """Split into large and fine: a blurred copy and the remainder.

    The blur wraps, so each band on its own is seamless, and so is any mix of
    them.

    x      — H×W array.
    radius — blur sigma in pixels (at least 1), see band_radius.

    Returns (base, detail), both H×W float32, base + detail == x; detail is
    zero-mean."""
    xp = X.xp(x, radius)
    base = fast_blur(xp.asarray(x, xp.float32), max(1.0, float(radius)), mode='wrap')
    return base.astype(xp.float32), (x - base).astype(xp.float32)


def weigh_bands(x, radius, low=1.0, high=1.0):
    """Re-weight the large and fine bands of x: low·base + high·detail.

    radius is the split sigma in pixels (split_bands); low/high are free
    multipliers (1, 1 returns x unchanged, as float32, without blurring).
    Returns H×W float32; seamless because the split is."""
    xp = X.xp(x, radius, low, high)
    if abs(low - 1.0) < 1e-3 and abs(high - 1.0) < 1e-3:
        return xp.asarray(x, xp.float32)
    b, d = split_bands(x, radius)
    return (float(low) * b + float(high) * d).astype(xp.float32)


def chroma_albedo(rgb_srgb, radius=24.0, eps=1e-3):
    """The part of the luminance variation that is explained by COLOUR — i.e. paint.

    Here is what it rests on. A shadow multiplies all three channels by the
    same number: in shadow a leaf is just as green, only darker. So shading
    changes luminance and does NOT change chromaticity. Paint, on the other
    hand, changes exactly the chromaticity: a dark patch of moss is greener
    than the stone, rust is redder. Chromaticity is therefore the only
    physical separator of light and paint that a single image offers; the idea
    goes back to Land's retinex, and shadow removal through chromaticity /
    illuminant-invariant images is the line of work of Finlayson et al.
    (e.g. Finlayson, Hordley, Drew, "Removing shadows from images", ECCV 2002).

    So we take the logarithms of channel ratios (log R/G and log B/G — pure
    paint, the light cancels out of them) and in the neighbourhood of each
    point look at what share of the luminance variation they predict. This is
    an ordinary linear regression in a window, computed through box means as in
    the edge-preserving (guided) filter, only with two guides instead of one,
    so per pixel a 2×2 system has to be solved (closed form via the
    determinant, ridge-regularised by eps).

    What is returned is the predicted "paint" in luminance; it is subtracted,
    and what remains is shading.

    Honest limitation: on a single-coloured texture the channel ratios do not
    change at all, there is nothing to predict with, and the function returns
    almost zero. This is not a failure — from a grey picture paint cannot be
    told from shadow in principle. The model also assumes the light itself is
    one colour; a coloured fill in the shadows (e.g. bluish skylight) shifts
    the chromaticity and part of such a shadow will be taken for paint.

    rgb_srgb — H×W×3 sRGB 0…1.
    radius   — regression window radius in pixels (window 2r+1, at least 2);
               sensible 12…64. Also the scale above which variation is ignored
               (both luminance and chromaticity are high-passed with the same
               box first).
    eps      — ridge term added to the chromaticity variances (log units²).

    Returns H×W float32, in natural-log luminance units, locally zero-mean.
    All box filters wrap, so the result tiles."""
    xp = X.xp(rgb_srgb, radius, eps)
    lin = xp.clip(srgb_to_lin(rgb_srgb), 1e-4, None)
    lg = xp.log(lin)
    y = xp.log(xp.maximum(luminance(lin), 1e-4))
    r = max(2, int(round(radius)))
    # цветность: логарифмы отношений каналов. Свет из них сокращается
    c = xp.stack([lg[..., 0] - lg[..., 1], lg[..., 2] - lg[..., 1]], -1)
    y = y - box(y, r)
    c = c - xp.stack([box(c[..., 0], r), box(c[..., 1], r)], -1)
    c0, c1 = c[..., 0], c[..., 1]
    s00 = box(c0 * c0, r) + eps
    s11 = box(c1 * c1, r) + eps
    s01 = box(c0 * c1, r)
    b0 = box(c0 * y, r)
    b1 = box(c1 * y, r)
    det = s00 * s11 - s01 * s01
    det = xp.where(xp.abs(det) < 1e-12, 1e-12, det)
    a0 = (b0 * s11 - b1 * s01) / det
    a1 = (b1 * s00 - b0 * s01) / det
    return (box(a0, r) * c0 + box(a1, r) * c1).astype(xp.float32)


def g_full_log(rgb_srgb):
    """Log of linear luminance (floored at 1e-4): H×W×3 sRGB in, H×W float32."""
    xp = X.xp(rgb_srgb)
    return xp.log(xp.maximum(luminance(srgb_to_lin(rgb_srgb)), 1e-4)).astype(xp.float32)


def hard_shadow(rgb_srgb, radius=48.0):
    """Field of hard shadows: the same estimate that removes the sun on the light tab.

    The edge-preserving filter smooths the texture's grain but does not cross a
    sharp boundary, so what stays in the estimate is precisely the shadows —
    large regions that differ in brightness and are separated by a crisp edge.
    The logarithm of this field is returned.

    rgb_srgb — H×W×3 sRGB 0…1.
    radius   — pixels; the guided filter uses radius/3 (at least 4) and
               eps 0.12 on log luminance normalised to 0…1, exactly as in
               delight(hard=…).

    Returns H×W float32, natural-log luminance units, zero-mean."""
    xp = X.xp(rgb_srgb, radius)
    L = xp.maximum(luminance(srgb_to_lin(rgb_srgb)), 1e-4)
    g = xp.log(L)
    rng = max(float(xp.ptp(g)), 1e-6)
    gn = (g - g.min()) / rng
    sm = guided_filter(gn, gn, max(4, int(radius / 3)), 0.12) * rng + g.min()
    return (sm - sm.mean()).astype(xp.float32)


def shading_signal(rgb_srgb, radius=40.0, chroma=1.0, shadow=1.0,
                   chroma_radius=24.0, shadow_radius=48.0, lit=None):
    """Shading signal for height-from-light — already without paint and without
    cast shadows, as far as they can be recognised.

    The pure logic "brightness = slope" holds only for a single-coloured
    surface and only when nothing casts a shadow onto it. Real textures are not
    like that, and both corrections here are exactly about this:

      chroma — subtract what is explained by colour (chroma_albedo): that is
               paint, not relief. Weight 0…1 (MaterialJob defaults to 0.6).
      shadow — subtract the field of hard shadows: the step at a shadow's edge
               is not a slope, and integrating it means getting a cliff on flat
               ground. Weight 0…1.

    lit — an already de-lit image (H×W×3 sRGB). If it is given, the shadow field
    is taken as the difference between the original and it: this is exactly
    what the "remove light from a photo" tab shows as "diffuse without light",
    i.e. what gets subtracted is precisely what the user tuned there, not a
    second independent opinion about the same shadows. If it is not given, an
    own estimate is built (hard_shadow with shadow_radius).

    It matters what is NOT in that difference. De-lighting separates the
    illumination by scale and by edge sharpness: the overall light gradient and
    cast shadows go into it, while the relief's own shading stays in the albedo
    — otherwise the diffuse would lose all its texture. On synthetic data with a
    known answer the difference contains 0.0…0.2 of the relief shading, and the
    remainder 0.96. That is why the difference is SUBTRACTED here, not
    integrated: what has to be integrated is exactly what stayed in the albedo.

    rgb_srgb      — H×W×3 sRGB 0…1 (the COLOUR image: chroma needs colour).
    radius        — high-pass sigma in pixels; everything larger is dropped
                    from the signal and from both corrections. Sensible 16…80.
    chroma_radius — window radius of chroma_albedo, pixels.
    shadow_radius — radius for hard_shadow, pixels (ignored when lit is given).

    Returns H×W float32: high-passed natural-log luminance, approximately
    zero-mean, the input of light_azimuth / height_from_shading."""
    xp = X.xp(rgb_srgb, radius, chroma, shadow, chroma_radius, shadow_radius, lit)
    L = xp.maximum(luminance(srgb_to_lin(rgb_srgb)), 1e-4)
    g = xp.log(L)
    g = g - fast_blur(g, max(2.0, float(radius)))
    if chroma > 0:
        a = chroma_albedo(rgb_srgb, chroma_radius)
        g = g - float(chroma) * (a - fast_blur(a, max(2.0, float(radius))))
    if shadow > 0:
        if lit is None:
            sh = hard_shadow(rgb_srgb, shadow_radius)
        else:
            sh = (g_full_log(rgb_srgb) - g_full_log(lit)).astype(xp.float32)
        g = g - float(shadow) * (sh - fast_blur(sh, max(2.0, float(radius))))
    return g.astype(xp.float32)


def shading_detail(L, radius=40.0):
    """Shading without albedo, as far as that is possible at all.

    Luminance is the product of paint and light, and in the logarithm a product
    becomes a sum, so the large part (paint, the overall light gradient) is
    removed by plain subtraction of a blurred copy. What remains is the fine
    part, where the relief shading sits. Paint that is painted finely remains
    too — there is no getting away from that, and it is the main reason the
    method works better on a photograph than on a hand-painted texture.

    L      — H×W LINEAR luminance (clipped at 1e-3 before the log).
    radius — high-pass sigma in pixels (at least 2).

    Returns H×W float32, natural-log units, zero-mean (wrap blur)."""
    xp = X.xp(L, radius)
    lg = xp.log(xp.clip(L, 1e-3, None))
    return (lg - fast_blur(lg, max(2.0, float(radius)))).astype(xp.float32)


def light_azimuth(L, radius=40.0):
    """Where the light comes from — from the angular skew of the spectrum.

    If shading is the slope along the light (see height_from_shading), then in
    the spectrum the power carries a factor cos² of the angle between the
    frequency and the light direction: across the light the relief is not
    visible at all. It is enough to compute the second angular harmonic of the
    power spectrum — its phase is exactly twice the azimuth. Concretely:
    m = Σ P(k)·e^{2iθ_k} / Σ P(k) over frequencies between 2 cycles per image
    and 0.25 cycles per pixel; azimuth = −arg(m)/2 (minus: image y grows down).
    Estimating the illuminant from image statistics in this spirit goes back to
    Pentland, "Finding the illuminant direction", JOSA 1982.

    L      — H×W LINEAR luminance.
    radius — high-pass sigma in pixels passed to shading_detail.

    Returns (angle in radians in [0, 2π), how confident, 0…1). The angle is in
    screen terms: 0 — light from the right, 90° — from above. Confidence is
    2·|m| clipped to 1; (π/2, 0.0) if the image is too small to have a usable
    frequency band.

    Honest limitation: the assumption is that the relief is isotropic. The
    spectral skew comes not only from the light but also from the elongation
    of the relief itself. In brickwork the courses are horizontal, and the
    method will report light from above regardless of where it actually comes
    from. This is a known limitation, not a bug. So it is a hint for the first
    click, not the truth: the angle is set by hand.

    The direction comes out only up to a 180° turn: under such a turn bumps and
    pits swap places, and they cannot be told apart from one picture in
    principle — this is the same crater illusion. The "light from above"
    variant is chosen, as is customary in painting."""
    xp = X.xp(L, radius)
    g = shading_detail(L, radius)
    F = xp.fft.fft2(g)
    P = (F * xp.conj(F)).real
    H, W = g.shape
    u = xp.fft.fftfreq(W)[None, :]
    v = xp.fft.fftfreq(H)[:, None]
    kk = u * u + v * v
    ok = (kk > (1.0 / max(W, H) * 2) ** 2) & (kk < 0.25 ** 2)
    if not ok.any():
        return math.pi * 0.5, 0.0
    w = P[ok]
    ang = xp.arctan2(xp.broadcast_to(v, kk.shape)[ok], xp.broadcast_to(u, kk.shape)[ok])
    m = xp.sum(w * xp.exp(2j * ang)) / max(float(w.sum()), 1e-9)
    az = -float(xp.angle(m)) * 0.5        # в экранные углы: вверх это +90°
    conf = float(min(1.0, 2.0 * abs(m)))
    if math.sin(az) < 0:                  # из двух вариантов берём «сверху»
        az += math.pi
    return az % (2.0 * math.pi), conf


def height_from_shading(L, azimuth, smooth=0.25, radius=40.0, g=None):
    """Height from the lighting captured in the texture itself (shape from shading).

    This is how it differs from the other layers. They read luminance as
    height: darker means lower. But on a lit surface luminance is not height,
    it is SLOPE: under diffuse light with direction l the luminance equals N·l,
    so for gentle relief it equals, up to a constant, the derivative of height
    along the light. Hence height must not be taken from luminance but
    INTEGRATED along the light direction.

    The difference is obvious on any groove: in the "whole luminance" layer the
    dark side of a groove becomes a pit and the bright side a hump, and the
    groove comes out as a step. Integrated, the bright and dark sides add up to
    a single pit, which is what it really is.

    Solved in the spectrum: integration along a direction is division by
    (l·k), and that same factor vanishes at frequencies across the light. There
    is genuinely no relief in the picture there (along a groove turned edge-on
    to the light the luminance does not change), and that gap is filled by
    smoothing: λ|k|² is added to the denominator. With shading g ≈ −l·∇h,
    i.e. G = −i·a·H with a = l·k, the regularised solution is

        H = i·a·G / (a² + λ|k|²).

    The larger λ, the calmer the map across the light and the fewer streaks it
    has. Without λ (λ → 0) the division blows up on the line a = 0: every bit of
    noise at frequencies perpendicular to the light is amplified without bound
    and the map fills with streaks running along the light. λ is floored at
    1e-3 for that reason. This is the linear (small-slope) form of the
    classical shape-from-shading problem (Horn; Pentland's linear shape from
    shading).

    L       — H×W LINEAR luminance (used only when g is None).
    azimuth — light direction in radians, screen convention: 0 — light from the
              right, π/2 — from above (see light_azimuth). The image y axis grows
              downwards, so the vertical component is flipped internally.
    smooth  — λ, regulariser across the light, ≥ 1e-3; sensible 0.05…1.
    radius  — high-pass sigma in pixels for shading_detail (when g is None).
    g       — precomputed shading signal H×W (e.g. from shading_signal, already
              without paint and cast shadows); overrides L and radius.

    Everything is done in Fourier space on the torus, so the map is periodic by
    construction and seamlessness does not suffer. Returns H×W float32,
    zero-mean (the DC term is set to 0), in arbitrary units (log luminance ×
    pixels); callers normalise it (sfs_layer)."""
    xp = X.xp(L, azimuth, smooth, radius, g)
    if g is None:
        g = shading_detail(L, radius)
    H, W = g.shape
    kx = 2.0 * xp.pi * xp.fft.fftfreq(W)[None, :]
    ky = 2.0 * xp.pi * xp.fft.fftfreq(H)[:, None]
    # угол задаётся по-экранному: 0° — свет справа, 90° — сверху. В самой
    # картинке y растёт вниз, поэтому вертикаль переворачивается
    lx, ly = math.cos(float(azimuth)), -math.sin(float(azimuth))
    a = lx * kx + ly * ky
    lam = max(float(smooth), 1e-3)
    den = a * a + lam * (kx * kx + ky * ky)
    den[0, 0] = 1.0
    Hf = 1j * a * xp.fft.fft2(g) / den
    Hf[0, 0] = 0.0
    return xp.real(xp.fft.ifft2(Hf)).astype(xp.float32)


def height_from_cavity(c, cutoff=0.25, invert=False):
    """Height from an occlusion map (AO, cavity, "crevice map").

    A relative of the previous one. Occlusion in a crevice is the stronger the
    more concave the surface is there, so an AO map is, up to sign and a
    factor, the LAPLACIAN of height, its second derivative. Light gave the first
    derivative and was integrated once; here it has to be integrated twice,
    and that is an ordinary Poisson equation, the same one that assembles a
    seamless map from slopes. Solved in the spectrum as H = −G / (|k|² + ε),
    where g = mean(c) − c is positive where the map is darker, so dark lines
    become pits.

    Why this is needed at all: a cavity map sometimes looks like depth, and it
    is tempting to use it as height. But the dark line of a joint taken as
    height gives a groove exactly as wide as the line is drawn — vertical
    walls, no bottom. Integrated, it gives a groove with a real profile: gently
    sloping walls and a bottom, while the bricks between the joints come out as
    flat plateaus at one height.

    c      — H×W occlusion/cavity map, 0…1 (dark = crevice).
    cutoff — the scale beyond which the map is not trusted. Division by |k|²
             inflates the very lowest frequencies: any gentle unevenness that
             happens to be in the map becomes, after double integration, a
             mountain across the whole texture. The radius is given as a
             fraction of the shorter side (0.25 by default, sensible 0.05…0.5);
             it sets ε = (2π / (cutoff · min(H, W)))², so wavelengths longer
             than that are damped instead of amplified.
    invert — flip the sign (for maps where crevices are light).

    FFT on the torus, so the result tiles. Returns H×W float32, zero-mean, in
    arbitrary units; callers normalise it (height_variants uses _n01)."""
    xp = X.xp(c, cutoff, invert)
    x = xp.asarray(c, xp.float32)
    g = float(x.mean()) - x                 # положительное там, где темнее
    if invert:
        g = -g
    H, W = g.shape
    kx = 2.0 * xp.pi * xp.fft.fftfreq(W)[None, :]
    ky = 2.0 * xp.pi * xp.fft.fftfreq(H)[:, None]
    k2 = kx * kx + ky * ky
    eps = (2.0 * xp.pi / max(cutoff * min(H, W), 2.0)) ** 2
    F = -xp.fft.fft2(g) / (k2 + eps)
    F[0, 0] = 0.0
    return xp.real(xp.fft.ifft2(F)).astype(xp.float32)


def sfs_layer(L, azimuth, smooth=0.25, radius=40.0, g=None,
              scale=0.3, low=1.0, high=1.0):
    """The "from lighting" layer exactly as it enters the height mix.

    Integral of the shading along the light (height_from_shading), then the
    weighting of the large and fine bands (weigh_bands), then normalisation to
    0…1 (_n01). Factored out so that the direct normal (see
    normal_from_shading) takes EXACTLY THE SAME thing the user sees in the
    layer preview, not a similar but different quantity. Integration inflates
    the low frequencies, so the bands need their own weights here, just as the
    neural-network depth map does.

    L, azimuth, smooth, radius, g — as in height_from_shading.
    scale — "scale boundary" slider for band_radius (split sigma computed from
            the shorter side of L), typically 0…1.
    low   — multiplier on the large band.
    high  — multiplier on the fine band.

    Returns H×W float32 in 0…1 (0.5th…99.5th percentile stretch); seamless."""
    s = height_from_shading(L, azimuth, smooth, radius, g)
    # интегрирование раздувает низкие частоты, поэтому диапазонам тут
    # нужен свой вес так же, как карте от нейросети
    return _n01(weigh_bands(s, band_radius(min(L.shape), scale), low, high))


def normal_from_shading(L, azimuth=None, smooth=0.25, strength=1.0, g=None,
                        directx=True, swap=False, scale=0.3, low=1.0, high=1.0,
                        radius=40.0, bmode='unit', ao=None, invert=False):
    """Normal straight from the lighting, bypassing height: channels R and G are
    the "from lighting" layer computed for light from the left and for light
    from below.

    The method was devised by the author of the program and first assembled by
    hand in Photoshop: two screenshots of the lighting-layer preview (light at
    180° and 270°) were put into R and G, and blue was filled in by the normal
    repair. Exactly that is reproduced here; on the author's source image the
    match with his file is 0.92 and 0.94 per channel (the rest is compression
    and resampling of the screenshots).

    Why this works at all. The lighting layer is not a slope but an integral of
    the shading, and it looks like relief already lit from the side. Put into a
    slope channel, it gives a normal that behaves plausibly when the light is
    rotated, and its hollows are consistently darker under any light — as if AO
    were already baked in.

    The price. Two fields obtained independently do not add up to any height:
    the slope-field curl of such a normal is about 0.6 versus about 0.1 for an
    honest one (see slope_curl). Parallax and AO computed from the height will
    describe a different surface unless the fit (make_integrable) is enabled.

    L        — H×W LINEAR luminance.
    smooth, scale, low, high, radius, g — the same as for the lighting layer in
               the height mix (see sfs_layer), so the result matches the layer
               preview.
    strength — multiplier on R and G; at 1 the grey 0…1 preview goes into the
               channel as is, as in the author's file.
    directx  — True: DirectX layout (the author's file is DirectX, as OpenMW
               expects); False: the green channel is flipped for OpenGL.
    swap     — swap the R and G fields.
    bmode    — where blue comes from: 'unit' — from the vector length
               (z = sqrt(1 − x² − y²), as the normal repair does); 'ao' — from
               the occlusion map ao (0…1, clipped to 0.05…1), after which the
               vector is normalised: in crevices the normal tilts over more and
               darkens under any light.
    ao       — H×W occlusion map for bmode='ao'.
    invert   — turn the relief inside out (convex becomes concave; from a single
               picture these two variants cannot be told apart).
    azimuth  — not used; kept for call compatibility.

    Returns H×W×3 float32 unit vectors. Built from FFT integrals on the torus,
    so it tiles."""
    xp = X.xp(L, azimuth, smooth, strength, g, directx, swap, scale, low, high, radius, bmode, ao, invert)
    r = sfs_layer(L, math.pi, smooth, radius, g, scale, low, high)
    v = sfs_layer(L, math.pi * 1.5, smooth, radius, g, scale, low, high)
    k = float(strength)
    x = k * (2.0 * r - 1.0)
    y = k * (2.0 * v - 1.0)
    if invert:
        x, y = -x, -y
    if swap:
        x, y = y, x
    # раскладка файла автора — это DirectX (как у OpenMW); для OpenGL
    # зелёный канал разворачивается
    if not directx:
        y = -y
    if bmode == 'ao' and ao is not None:
        z = xp.clip(xp.asarray(ao, xp.float32), 0.05, 1.0)
    else:
        z = xp.sqrt(xp.clip(1.0 - x * x - y * y, 0.0, 1.0))
    n = xp.stack([x, y, z], -1)
    n = n / xp.maximum(xp.sqrt((n * n).sum(-1, keepdims=True)), 1e-6)
    return n.astype(xp.float32)


def height_from_slopes(p, q):
    """Height whose gradient is as close as possible to the given field (p, q).

    An ordinary least-squares problem: min ||∇h − (p, q)||². Its solution is a
    Poisson equation with the divergence of that field on the right-hand side,
    and in the spectrum this is again one line:
    H = −(i·kx·P + i·ky·Q) / |k|². This is the Frankot–Chellappa projection
    (Frankot & Chellappa, "A method for enforcing integrability in shape from
    shading algorithms", IEEE PAMI 1988): the curl-free part of the field is
    kept, the rest is discarded. On the torus, so the result is periodic.

    p — H×W slope along x (to the right), height units per pixel.
    q — H×W slope along y (down the image), same units.

    Returns H×W float32, zero-mean (DC set to 0), in height units of p, q
    (i.e. pixels if the slopes are dimensionless)."""
    xp = X.xp(p, q)
    H, W = p.shape
    kx = 2.0 * xp.pi * xp.fft.fftfreq(W)[None, :]
    ky = 2.0 * xp.pi * xp.fft.fftfreq(H)[:, None]
    k2 = kx * kx + ky * ky
    k2[0, 0] = 1.0
    F = -(1j * kx * xp.fft.fft2(p) + 1j * ky * xp.fft.fft2(q)) / k2
    F[0, 0] = 0.0
    return xp.real(xp.fft.ifft2(F)).astype(xp.float32)


def make_integrable(n, amount=1.0, directx=True):
    """Fit a normal map to some height.

    A slope field assembled other than from a height corresponds to no surface.
    There is exactly one way to repair that: project it onto the nearest field
    that does have a height — i.e. find the height by least squares
    (height_from_slopes, Frankot & Chellappa 1988) and take its gradient back
    (grad_wrap). After that the curl is zero exactly, not approximately (see
    slope_curl).

    This does not come for free. The projection keeps only the part of the
    field that CAN belong to a surface at all and throws the rest away. On a map
    made of two shading integrals, about a quarter of the slopes along one axis
    survives and three quarters along the other: the very sharpness for which
    the whole thing was done sits precisely in what is discarded. That is why
    this is a slider and not a checkbox — it trades consistency for looks.

    n       — H×W×3 unit normals.
    amount  — 0…1: blend of the slopes between the original field (0) and the
              projected one (1); the curl reaches zero only at 1.
    directx — layout of n and of the result (True: y down the image).

    Returns (normal, height): normal H×W×3 float32 unit vectors; height H×W
    normalised to 0…1 (_n01) — always the full projection, independent of
    amount — suitable for parallax and AO because it describes exactly the same
    surface."""
    xp = X.xp(n, amount, directx)
    nz = xp.maximum(n[..., 2], 1e-3)
    ny = n[..., 1] if directx else -n[..., 1]
    p, q = -n[..., 0] / nz, -ny / nz            # уклоны в системе картинки
    h = height_from_slopes(p, q)
    gy, gx = grad_wrap(h)
    k = min(max(float(amount), 0), 1)
    px, qy = (1 - k) * p + k * gx, (1 - k) * q + k * gy
    oy = qy if directx else -qy
    m = xp.stack([-px, -oy, xp.ones_like(px)], -1)
    m = m / xp.maximum(xp.sqrt((m * m).sum(-1, keepdims=True)), 1e-6)
    return m.astype(xp.float32), _n01(h)


def height_from_normal(n, directx=True, cutoff=0.0):
    """Height from a finished normal map — the inverse of normal_from_height.

    Slopes are taken from the normal (p = −nx/nz, q = −ny/nz, nz floored at
    1e-3; the slope length is capped at 4 × its 99th percentile, at least 3,
    so a few nearly horizontal normals cannot become spikes that take the
    whole height range) and assembled into a height by least squares (height_from_slopes,
    i.e. the Frankot–Chellappa method). If the normal was originally computed
    from a height, the height is recovered up to a constant; if not, the part
    of it that can belong to a surface at all is taken and the rest (the curl,
    see slope_curl) is lost.

    n       — H×W×3 unit normals.
    directx — layout of n (True: y down the image; False: OpenGL, green
              flipped before integrating).
    cutoff  — the scale beyond which not to trust, as a fraction of the shorter
              side; 0 — trust everything. The integral accumulates error at low
              frequencies: a small overall tilt of the normal turns into a hill
              across the whole texture. The cut subtracts a blurred (wrap) copy
              with radius cutoff × side.

    Returns height in pixels (the step to the neighbouring pixel equals the
    slope), H×W float32, zero-mean. Its range (ptp) is the relief depth in
    texture pixels. FFT on the torus, so it tiles."""
    xp = X.xp(n, directx, cutoff)
    n = xp.asarray(n, xp.float32)
    nz = xp.maximum(n[..., 2], 1e-3)
    ny = n[..., 1] if directx else -n[..., 1]
    p, q = -n[..., 0] / nz, -ny / nz
    # почти лежачие нормали (nz → 0) дают наклон в сотни раз круче остальных:
    # несколько таких пикселей превращаются в «диполи», забирающие весь
    # диапазон высоты. Наклон ограничивается четырьмя 99-ми процентилями
    s = xp.sqrt(p * p + q * q)
    smax = max(4.0 * float(xp.percentile(s, 99.0)), 3.0)
    k = xp.minimum(1.0, smax / xp.maximum(s, 1e-6))
    p, q = p * k, q * k
    h = height_from_slopes(p, q)
    if cutoff and cutoff > 0:
        h = h - fast_blur(h, max(2.0, float(cutoff) * min(h.shape)), mode='wrap')
    return (h - float(h.mean())).astype(xp.float32)


def normal_from_heightmap(h_px, directx=True):
    """Normal from a height given in pixels (step to the neighbouring pixel =
    slope).

    The exact counterpart of height_from_normal: there and back on a consistent
    normal gives the same normal. Unlike normal_from_height there is no
    strength or multi-scale weighting — the slope is taken literally.

    h_px    — H×W height in texture pixels.
    directx — True: y down the image (DirectX); False: OpenGL.

    Gradients wrap (grad_wrap), so the result tiles. Returns H×W×3 float32 unit
    vectors."""
    xp = X.xp(h_px, directx)
    gy, gx = grad_wrap(xp.asarray(h_px, xp.float32))
    ny = -gy if directx else gy
    m = xp.stack([-gx, ny, xp.ones_like(gx)], -1)
    return (m / xp.sqrt((m * m).sum(-1, keepdims=True))).astype(xp.float32)


def angle_between(a, b):
    """Mean angle between two ...×3 unit-normal maps, in degrees (float)."""
    xp = X.xp(a, b)
    d = xp.clip((xp.asarray(a) * xp.asarray(b)).sum(-1), -1.0, 1.0)
    return float(xp.degrees(xp.arccos(d)).mean())


def balance_crossover(side, balance):
    """Wavelength in pixels at which height and normal are trusted equally.

    balance 0 → 2 px (the height decides everything), 1 → the whole side (the
    normal decides everything but the mean level); geometric in between, so
    the slider moves evenly through the octaves.
    """
    b = float(np.clip(balance, 0.0, 1.0))
    return 2.0 * (max(float(side), 4.0) / 2.0) ** b


def balance_height_normal(n, h, balance=0.5, keep=0.0, directx=True):
    """Make a height map and a normal map agree, trusting each where it is good.

    The normal is a map of slopes; the height is a map of levels. Integrating
    the normal gives a height, differentiating the height gives a normal — and
    for maps made by different methods the two do not coincide. Instead of
    throwing one of them away, this finds the ONE height H that is as close as
    possible to both at once:

        min over H of  ||∇H − g||²  +  μ·||H − s·h||²

    where g are the slopes of the normal and s·h is the given height scaled
    into the normal's units. In the spectrum this is one line per frequency:

        H = (conj(D)·G + μ·s·Ĥ) / (|D|² + μ)

    with D the symbol of the same central difference that grad_wrap uses, so
    the normal taken back from H (normal_from_heightmap) matches H exactly.
    Because |D|² grows with frequency, the normal wins on fine detail and the
    height on large forms; μ sets the wavelength where they are equal
    (balance_crossover). That is also exactly the recipe for fusing a neural
    depth (good large form, no scale) with a neural normal (good detail, drifts
    on large form).

    The scale s is not a slider: it is fitted by least squares on the slopes
    of both maps in a band of about one octave around the crossover, where
    both are trusted. A negative s means the height is inside out relative to
    the normal (for example a depth map where near is dark); it is used as is,
    and reported.

    Args:
        n: H×W×3 unit normals, layout by ``directx``.
        h: H×W height, any units (usually 0…1).
        balance: 0…1, see balance_crossover.
        keep: 0…1, how much of the normal's own slopes to put back on top of
            the agreed ones, in slope space. 0 — the output normal is exactly
            the normal of the output height; 1 — the input normal unchanged.
            In between: the part of the normal that no height can explain
            (its curl) comes back partly.
        directx: layout of ``n`` and of the returned normal.

    Returns:
        (height_px, normal, info): height in pixels (zero mean, float32, the
        step to the neighbour equals the slope), H×W×3 unit normals, and a dict
        with 'scale' (s, pixels per unit of h), 'corr' (correlation of the two
        maps' slopes in the fitting band, −1…1), 'crossover' (px), 'depth_px'
        (ptp of the height), 'curl_in', 'curl_out' (slope_curl of the input and
        output normal, DirectX-equivalent), 'angle_n' (mean angle between the
        output and the input normal, degrees), 'angle_h' (between the output
        and the normal of s·h, degrees).
    """
    xp = X.xp(n, h, balance, keep, directx)
    n = xp.asarray(n, xp.float32)
    h = xp.asarray(h, xp.float32)
    Hh, Ww = h.shape
    nz = xp.maximum(n[..., 2], 0.05)
    ny = n[..., 1] if directx else -n[..., 1]
    p, q = -n[..., 0] / nz, -ny / nz             # наклоны в раскладке DirectX
    kx = 2.0 * xp.pi * xp.fft.fftfreq(Ww)[None, :]
    ky = 2.0 * xp.pi * xp.fft.fftfreq(Hh)[:, None]
    Dx = 1j * xp.sin(kx)                         # символ центральной разности
    Dy = 1j * xp.sin(ky)
    D2 = (xp.abs(Dx) ** 2 + xp.abs(Dy) ** 2)
    P, Q = xp.fft.fft2(p), xp.fft.fft2(q)
    H0 = xp.fft.fft2(h - float(h.mean()))
    lam = balance_crossover(min(Hh, Ww), balance)
    kc = 2.0 * xp.pi / lam
    # подгонка масштаба: в полосе около точки равновесия, где верим обоим
    kk = xp.sqrt(kx * kx + ky * ky)
    kk[0, 0] = 1e-9
    w = xp.exp(-0.5 * (xp.log2(kk / min(kc, xp.pi * 0.7))) ** 2) + 0.02
    w[0, 0] = 0.0
    dh_x, dh_y = Dx * H0, Dy * H0
    num = float(xp.sum(w * xp.real(xp.conj(dh_x) * P + xp.conj(dh_y) * Q)))
    den = float(xp.sum(w * (xp.abs(dh_x) ** 2 + xp.abs(dh_y) ** 2)))
    s = num / max(den, 1e-12)
    ng = float(xp.sum(w * (xp.abs(P) ** 2 + xp.abs(Q) ** 2)))
    corr = num / max(xp.sqrt(den * ng), 1e-12)
    if balance <= 0.0:
        F = s * H0
    else:
        mu = (2.0 * xp.sin(min(kc, xp.pi) * 0.5)) ** 2 if balance < 1.0 else 0.0
        den_k = D2 + mu
        den_k[0, 0] = 1.0
        F = (xp.conj(Dx) * P + xp.conj(Dy) * Q + mu * s * H0) / den_k
        if mu == 0.0:
            # на частотах, которых центральная разность не видит (Найквист),
            # нормаль ничего не говорит — там оставляем высоту
            blind = D2 < 1e-9
            F = xp.where(blind, s * H0, F)
    F[0, 0] = 0.0
    Hout = xp.real(xp.fft.ifft2(F)).astype(xp.float32)
    nh = normal_from_heightmap(Hout, True)
    k = min(max(float(keep), 0.0), 1.0)
    if k > 0:
        gy, gx = grad_wrap(Hout)
        px_ = gx + k * (p - gx)
        qy_ = gy + k * (q - gy)
        m = xp.stack([-px_, -qy_, xp.ones_like(px_)], -1)
        nout = (m / xp.sqrt((m * m).sum(-1, keepdims=True))).astype(xp.float32)
    else:
        nout = nh
    nin_dx = xp.stack([n[..., 0], ny, n[..., 2]], -1)
    n_hs = normal_from_heightmap(s * (h - float(h.mean())), True)
    info = {'scale': s, 'corr': corr, 'crossover': lam,
            'depth_px': float(xp.ptp(Hout)),
            'curl_in': slope_curl(nin_dx), 'curl_out': slope_curl(nout),
            'angle_n': angle_between(nout, nin_dx),
            'angle_h': angle_between(nout, n_hs)}
    if not directx:
        nout = nout.copy()
        nout[..., 1] = -nout[..., 1]
    return Hout, nout, info


def slope_curl(n):
    """How far the slope field fails to add up to any height.

    For a normal computed from a height the mixed derivatives are equal and the
    curl of the slope field is zero. The curl ∂p/∂y − ∂q/∂x is a measure of
    non-integrability; make_integrable projects it to exactly zero.

    n — H×W×3 unit normals. Slopes are p = −nx/nz, q = −ny/nz with ny taken as
        is, i.e. the DirectX layout (y down the image) is assumed; for an
        OpenGL normal flip green first, otherwise the value is meaningless.

    The number is returned as a fraction of the field itself: mean |curl|
    divided by the mean magnitude of all four first derivatives of (p, q).
    About 0.1 (typically ≈0.13) — an honest normal; about 0.6 — slopes that
    have no common height. Derivatives wrap (grad_wrap). Returns a float."""
    xp = X.xp(n)
    nz = xp.maximum(n[..., 2], 1e-3)
    p, q = -n[..., 0] / nz, -n[..., 1] / nz
    gy_p, gx_p = grad_wrap(p)
    gy_q, gx_q = grad_wrap(q)
    d = xp.sqrt(gy_p ** 2 + gx_p ** 2 + gy_q ** 2 + gx_q ** 2)
    return float(xp.abs(gy_p - gx_q).mean() / max(float(d.mean()), 1e-9))


def guess_green(n):
    """Tell the green layout of a normal map of unknown origin from its curl.

    A normal that came from any surface has (almost) no curl in its true
    layout. Flipping green turns ∂p/∂y − ∂q/∂x into ∂p/∂y + ∂q/∂x, which for a
    real surface is twice a genuine derivative — large. So the layout with the
    smaller curl is the right one. On DeepBump's output the two readings are
    0.17 and 0.54; on normals computed from a height, 0.00 and about 1.

    n — H×W×3 unit normals as stored.
    Returns (is_directx, curl_as_directx, curl_as_opengl).
    """
    xp = X.xp(n)
    n = xp.asarray(n, xp.float32)
    c_dx = slope_curl(n)
    c_gl = slope_curl(n * xp.array([1.0, -1.0, 1.0], xp.float32))
    return c_dx <= c_gl, c_dx, c_gl


def normal_ao(n):
    """Occlusion from the normal alone.

    Under a uniform sky a point sees the larger share of it the closer its
    normal is to the outward one: the visible fraction of the hemisphere is
    (1 + n.z) / 2. The physics is simple and honest, but it knows only about
    the tilt and nothing about the neighbours.

    n — H×W×3 unit normals. Returns H×W in 0…1 (1 = fully open), dtype of n."""
    xp = X.xp(n)
    return xp.clip(0.5 * (1.0 + n[..., 2]), 0.0, 1.0)


def _masked_grad(h, mask):
    """Slope of h along x and y over the object only: the central difference
    where both neighbours are in the mask, a one-sided one where only one is,
    0 where neither (wraps round, like the blurs)."""
    xp = X.xp(h)
    m = mask > 0.5
    out = []
    for ax in (1, 0):
        hp, hm = xp.roll(h, -1, ax), xp.roll(h, 1, ax)
        mp, mm = xp.roll(m, -1, ax), xp.roll(m, 1, ax)
        g = xp.where(mp & mm, (hp - hm) * 0.5,
                     xp.where(mp, hp - h, xp.where(mm, h - hm, 0.0)))
        out.append(g.astype(h.dtype))
    return out


def cavity_ao(h, radius=28.0, strength=0.9, scales=3, mask=None):
    """AO from hollows, over several scales.

    A point darkens by as much as it lies below its surroundings. The method is
    cruder than an honest horizon search (hbao), but more robust on
    photographic textures: there fine grain produces false "cliffs", and the
    horizon starts darkening bumps instead of pits.

    h        — H×W height, 0…1.
    radius   — blur sigma of the largest scale in pixels; sensible 8…64.
    strength — 0…1 overall darkening.
    scales   — number of scales; each next one halves the radius (stops below
               1 px).

    Per scale: c = blur(h, r) − h, normalised by its 99th percentile of |c|,
    clipped to 0…1, and ao *= 1 − strength·1.6/scales·c. Blurs wrap
    (fast_blur), so the result tiles. Returns H×W in 0…1 (1 = open), dtype of
    h.

    mask     — None, or H×W 0/1: which pixels are the object (opaque). Then
               the surroundings are averaged over the object only
               (blur(h·mask) / blur(mask)), so the transparent background
               no longer counts as a pit: no grey halo around a leaf and no
               dark rim along its edge. Near the edge that mean is one-sided
               (from inside only), which on a slope down to the edge reads
               as a pit too; the part due to the slope (the slope times the
               shift of the mean's centre) is taken off. Outside the mask the
               result is 1."""
    xp = X.xp(h, radius, strength, scales)
    ao = xp.ones_like(h)
    for i in range(int(scales)):
        r = radius * (0.5 ** i)
        if r < 1.0:
            break
        if mask is None:
            c = fast_blur(h, r) - h
            s = max(float(xp.percentile(xp.abs(c), 99)), 1e-5)
        else:
            # среднее окружение только по непрозрачному: фон не «яма»
            bm = xp.maximum(fast_blur(mask, r), 1e-4)
            mean = fast_blur(h * mask, r) / bm
            # у края среднее берётся с одной, внутренней стороны, и на склоне к
            # краю (лист, ветка) оно выше самой точки — края темнели контуром,
            # хотя за краем ничего нет и заслонять нечему. Центр окружения там
            # сдвинут внутрь на r²·∇(blur mask)/blur mask; вычитаем то, что на
            # этом сдвиге даёт сам склон. Склон — разностью соседей по предмету
            # (у края — односторонней). Внутри предмета сдвиг 0 — ямы и стыки
            # листьев темнеют как раньше
            ox = r * r * (xp.roll(bm, -1, 1) - xp.roll(bm, 1, 1)) * 0.5 / bm
            oy = r * r * (xp.roll(bm, -1, 0) - xp.roll(bm, 1, 0)) * 0.5 / bm
            gx, gy = _masked_grad(h, mask)
            c = (mean - h - (gx * ox + gy * oy)) * mask
            inside = xp.abs(c)[mask > 0.5]
            # нижний предел — от размаха высоты: на ровном предмете c — это
            # только погрешность размытия, её нельзя растягивать до 1
            s = max(float(xp.percentile(inside, 99)) if inside.size else 0.0,
                    1e-3 * float(xp.abs(h).max()), 1e-5)
        ao *= xp.clip(1.0 - strength * 1.6 / scales * xp.clip(c / s, 0, 1), 0, 1)
    if mask is not None:
        ao = ao * mask + (1.0 - mask)
    return xp.clip(ao, 0, 1)



def hbao(h, radius=24.0, amp=12.0, strength=0.8, dirs=16, steps=8):
    """Horizon-based occlusion (after HBAO: Bavoil, Sainz & Dimitrov 2008).

    From every point we look around in a circle and find how high the relief
    rises. The higher the horizon around, the less light gets through. Shifts
    are done with wrap-around (np.roll), so for a seamless texture the result
    is seamless too.

    h        — H×W height, 0…1. AO is computed on a smoothed copy (sigma
               radius/10, at least 1 px): per-pixel grain is not a wall, and a
               horizon over it is meaningless.
    radius   — search distance in pixels; sensible 8…64.
    amp      — height scale: h·amp is the height in pixels, so it sets how
               steep the relief is relative to the search distance.
    strength — 0…1 overall darkening.
    dirs     — number of directions around the circle.
    steps    — samples per direction, evenly spaced up to radius.

    Per direction the maximum elevation slope is taken, converted to the sine
    of the horizon angle, and averaged over directions. Returns H×W float32 in
    0…1 (1 = open)."""
    xp = X.xp(h, radius, amp, strength, dirs, steps)
    # AO считаем по сглаженному рельефу: попиксельное зерно — не стенка,
    # и горизонт по нему получается бессмысленный
    H = fast_blur(h.astype(xp.float32), max(1.0, radius / 10.0)) * amp
    occ = xp.zeros_like(H)
    for k in range(dirs):
        # направления — простые числа, а не массивы: иначе на CuPy round() падает
        a = 2 * math.pi * k / dirs
        dx, dy = math.cos(a), math.sin(a)
        best = xp.zeros_like(H)
        for s in range(1, steps + 1):
            d = radius * s / steps
            ox, oy = int(round(dx * d)), int(round(dy * d))
            if ox == 0 and oy == 0:
                continue
            sh = xp.roll(xp.roll(H, oy, 0), ox, 1)
            best = xp.maximum(best, (sh - H) / math.hypot(ox, oy))
        occ += best / xp.sqrt(1.0 + best * best)          # sin(угла горизонта)
    occ /= dirs
    return xp.clip(1.0 - strength * xp.clip(occ, 0, 1), 0, 1)


def bent_from_height(h, radius=24.0, amp=12.0, dirs=10, steps=6):
    """Bent normal: the direction from which diffuse light actually reaches a
    point.

    An ordinary normal points where the surface faces and does not know that
    half of the sky above it is blocked by a neighbouring ledge. Here, with the
    same horizon search as for AO (hbao), the middle of the REMAINING open part
    of the sky is taken. In a crevice between bricks such a normal turns along
    the crevice, and in the engine the crevice stops being flatly lit — and
    nothing has to be done for that, everything already sits in the normal map.

    h, radius, amp — as in hbao (height 0…1, search distance in pixels, height
                     scale); the height is smoothed the same way.
    dirs, steps    — number of directions and samples per direction.

    Per direction the midpoint of the open arc (between the horizon and the
    zenith) is weighted by the open fraction 1 − sin(horizon) and summed.
    Shifts wrap around, so on a seamless texture the result is seamless.
    Returns H×W×3 float32 unit vectors in image axes (x right, y down the
    image, i.e. DirectX layout; bend_normal flips y for OpenGL).
    """
    xp = X.xp(h, radius, amp, dirs, steps)
    H = fast_blur(h.astype(xp.float32), max(1.0, radius / 10.0)) * amp
    bx = xp.zeros_like(H)
    by = xp.zeros_like(H)
    bz = xp.zeros_like(H)
    for k in range(int(dirs)):
        # направления — простые числа, а не массивы: иначе на CuPy round() падает
        a = 2 * math.pi * k / dirs
        dx, dy = math.cos(a), math.sin(a)
        best = xp.zeros_like(H)
        for s in range(1, int(steps) + 1):
            d = radius * s / steps
            ox, oy = int(round(dx * d)), int(round(dy * d))
            if ox == 0 and oy == 0:
                continue
            sh = xp.roll(xp.roll(H, oy, 0), ox, 1)
            best = xp.maximum(best, (sh - H) / math.hypot(ox, oy))
        sin_h = best / xp.sqrt(1.0 + best * best)
        half = 0.5 * (xp.arcsin(xp.clip(sin_h, 0, 1)) + xp.pi * 0.5)
        w = 1.0 - sin_h
        # знак обратный: xp.roll(H, ox) кладёт в точку p значение из p − o,
        # то есть горизонт щупается в направлении МИНУС (dx, dy). Для самой
        # окклюзии это незаметно, а bent normal без этого разворачивается
        # ровно в сторону стенки, а не от неё
        c = xp.cos(half) * w
        bx -= c * dx
        by -= c * dy
        bz += xp.sin(half) * w
    b = xp.stack([bx, by, bz], -1)
    L = xp.sqrt((b * b).sum(-1, keepdims=True))
    return (b / xp.maximum(L, 1e-6)).astype(xp.float32)


def bend_normal(n, bent, amount=1.0, directx=True):
    """Blend a bent normal into the ordinary one.

    After this the map is no longer purely geometric: for parallax and for
    anything that derives relief from the normal, keep a separate ordinary copy.

    n       — H×W×3 unit normals in the layout given by directx.
    bent    — H×W×3 from bent_from_height (DirectX layout; y is flipped here
              for OpenGL), or None.
    amount  — 0…1: linear blend weight, then renormalised.
    directx — layout of n and of the result.

    Returns n unchanged if amount ≤ 0 or bent is None, otherwise H×W×3 float32
    unit vectors."""
    xp = X.xp(n, bent, amount, directx)
    if amount <= 0 or bent is None:
        return n
    b = bent.copy()
    if not directx:
        b[..., 1] = -b[..., 1]
    a = min(max(float(amount), 0), 1)
    out = n * (1.0 - a) + b * a
    L = xp.sqrt((out * out).sum(-1, keepdims=True))
    return (out / xp.maximum(L, 1e-6)).astype(xp.float32)


# --------------------------------------------- specular / roughness --------
def material_maps(rgb_srgb, h, ao, f0=0.04, gloss=0.3, gloss_var=0.35,
                  metal=0.0, tint=0.0, cavity=0.5):
    """Specular colour, shininess, roughness and metalness from the diffuse.

    Returns (spec_rgb, shininess, roughness, metalness) — all in 0..1.

    rgb_srgb  — H×W×3 sRGB 0…1 (normally the de-lit diffuse).
    h         — height; currently unused, kept for the call signature.
    ao        — H×W occlusion 0…1 (1 = open).
    f0        — reflectance at normal incidence. Dielectrics 0.02…0.08, metals
                0.5…1.0. Above this ceiling the reflection becomes brighter
                than the incoming light and the picture goes off — in old games
                this is exactly what produced the "glowing" highlight. Clipped
                to 0.95 for dielectrics.
    gloss     — 0…1 base glossiness: roughness = 1 − gloss + gloss_var·(0.5 −
                normalised luminance) — dark and loose is rougher, light and
                smooth is smoother.
    gloss_var — how much luminance modulates roughness.
    metal     — 0…1 metalness, constant over the texture: blends the specular
                colour from grey f0 (dielectric) to albedo·f0 (metal).
    tint      — 0…1: tint the specular colour with the albedo hue at the same
                strength.
    cavity    — 0…1: scales roughness by (1 − 0.4·cavity·(1 − ao)).

    Outputs: spec_rgb H×W×3 sRGB, computed in linear light, multiplied by
    0.35 + 0.65·ao (no highlight in crevices) and clipped to 0.9 linear;
    shininess = (1 − roughness)²; roughness clipped to 0.045…1; metalness
    H×W filled with metal.
    """
    xp = X.xp(rgb_srgb, h, ao, f0, gloss, gloss_var, metal, tint, cavity)
    lin = srgb_to_lin(rgb_srgb)
    L = luminance(lin)
    Ln = (L - L.min()) / max(xp.ptp(L), 1e-6)

    # шероховатость: тёмное и рыхлое — шершавее, светлое и гладкое — глаже
    rough = xp.clip(1.0 - gloss + gloss_var * (0.5 - Ln), 0.045, 1.0)
    rough = xp.clip(rough * (1.0 - cavity * 0.4 * (1.0 - ao)), 0.045, 1.0)

    metalness = xp.full_like(L, float(metal))

    # цвет отражения: у металла — из альбедо, у диэлектрика — серый
    base = min(max(float(f0), 0.0), 0.95)
    dielectric = xp.stack([xp.full_like(L, base)] * 3, -1)
    tinted = lin * base / xp.maximum(L[..., None], 1e-4)
    spec_lin = dielectric * (1 - metal) + (lin * min(max(float(f0), 0.0), 1.0)) * metal
    if tint > 0:
        spec_lin = spec_lin * (1 - tint) + tinted * tint
    spec_lin = spec_lin * (0.35 + 0.65 * ao)[..., None]     # в щелях блика нет
    spec_lin = xp.clip(spec_lin, 0.0, 0.9)
    shininess = xp.clip((1.0 - rough) ** 2, 0.0, 1.0)
    return lin_to_srgb(spec_lin), shininess, rough, metalness


# ------------------------------------------------------------ бесшовность --
RELIEF_SOURCES = ('height', 'ao', 'ext')
RELIEF_MODES = ('overlay', 'soft', 'multiply', 'burn')


def relief_into_diffuse(d, s, mode='overlay', amount=0.0, radius=None,
                        low=1.0, high=1.0, sat_keep=1.0, occl=False):
    """Blend relief into the diffuse — what is done in Photoshop by laying a
    height or AO map over the texture.

    d      — diffuse, H×W×3 sRGB 0…1.
    s      — H×W source map 0…1: height (high is lighter), AO (1 = open) or an
             external depth map.
    mode   — 'overlay', 'soft' (soft light), 'multiply', 'burn' (linear burn);
             see RELIEF_MODES.
    amount — strength, 0 — nothing changes; sensible 0…3.
    radius, low, high — weight of the large and fine bands in the source (see
             weigh_bands; radius is the split sigma in pixels, None disables
             it); it is precisely the high-frequency part of the height that
             gives the effect Photoshop users get through "high pass".
    sat_keep — hold saturation (see keep_saturation): without it the darkened
             parts go grey and the lightened ones go acid-coloured.
    occl   — the source is an occlusion (1 = open, like AO). Then multiply takes
             it as is, physically; otherwise the source is first converted to
             that form.

    Strength scale. The source is measured in its own standard deviations
    (SD), not in raw units: height has a spread of 0.2, AO of 0.1, and the same
    "strength" would give different results on them. At strength 1 a point one
    SD above the mean gets an overlay layer of 0.6 instead of the neutral 0.5 —
    noticeable but not crude (on a 2048 wall: on average 9 steps out of 255);
    3 is already like "Curves" with a steep slope.

    The colour spaces differ on purpose. Multiply is physical — it is the
    fraction of light that got through — so it is done in linear space, as the
    former "bake AO" was (d · ao^amount). Overlay, soft light and linear burn
    are defined in display space, and computing them in linear means getting
    something other than what Photoshop gives; so they are done in sRGB.

    Overlay and soft light are neutral at 0.5, so the source is centred:
    B = 0.5 + amount · 0.1 · (s − mean) / SD. Otherwise the whole picture gets
    lighter or darker. Multiply and burn only darken and are neutral at 1, not
    0.5: AO already works that way; a height is converted so that the highest
    point (99.5th percentile) is untouched and each SD below it darkens by 0.1.
    Burn is d + B − 1 with B = 1 − 0.35·amount·(1 − s); without the 0.35 it hit
    four times harder than overlay at the same strength. Soft light uses the
    W3C formulation (as in Photoshop).

    Purely per-pixel apart from the wrap-around band split, so seamlessness is
    preserved. Returns H×W×3 float32 sRGB 0…1."""
    xp = X.xp(d, s, mode, amount, radius, low, high, sat_keep, occl)
    d = xp.clip(xp.asarray(d, xp.float32), 0, 1)
    if amount <= 0:
        return d
    x = xp.clip(xp.asarray(s, xp.float32), 0, 1)
    if radius is not None and (abs(low - 1.0) > 1e-3 or abs(high - 1.0) > 1e-3):
        m = float(x.mean())
        x = m + weigh_bands(x - m, radius, low, high)
    a = float(amount)
    sd = max(float(x.std()), 1e-4)
    if mode in ('multiply', 'burn'):
        # эти два режима только затемняют и нейтральны при 1, а не при 0.5.
        # AO так и устроено (1 — открыто); высоту приводим к тому же виду:
        # самое высокое не трогается, каждое СКО ниже него темнит на 0.1
        if not occl:
            top = float(xp.percentile(x, 99.5))
            x = xp.clip(1.0 - 0.10 * (top - x) / sd, 0.0, 1.0)
    if mode == 'multiply':
        out = lin_to_srgb(srgb_to_lin(d) * xp.power(xp.clip(x, 1e-4, 1.0), a)[..., None])
    elif mode == 'burn':
        # линейное затемнение: d + B − 1, где B = 1 − 0.35·a·(1 − s). Без
        # множителя оно при той же силе било вчетверо сильнее перекрытия
        out = d - (0.35 * a * (1.0 - x))[..., None]
    else:
        B = xp.clip(0.5 + a * 0.10 * (x - float(x.mean())) / sd, 0.0, 1.0)[..., None]
        if mode == 'soft':
            # мягкий свет в формулировке W3C (как в Фотошопе)
            D = xp.where(d <= 0.25, ((16 * d - 12) * d + 4) * d, xp.sqrt(d))
            out = xp.where(B <= 0.5, d - (1 - 2 * B) * d * (1 - d),
                           d + (2 * B - 1) * (D - d))
        else:
            out = xp.where(d <= 0.5, 2 * d * B, 1 - 2 * (1 - d) * (1 - B))
    out = xp.clip(out, 0, 1).astype(xp.float32)
    if sat_keep > 0:
        out = keep_saturation(d, out, sat_keep)
    return out.astype(xp.float32)


def seam_weights(shape, band):
    """Edge replacement weight for (H, W): 1 at the very edge, 0 deeper than band px.

    Linear ramp in the distance to the nearest edge. Returns H×W float32 0…1."""
    xp = X.xp(shape, band)
    H, W = shape
    ys = xp.minimum(xp.arange(H), xp.arange(H)[::-1]).astype(xp.float32)
    xs = xp.minimum(xp.arange(W), xp.arange(W)[::-1]).astype(xp.float32)
    d = xp.minimum(ys[:, None], xs[None, :])
    return xp.clip(1.0 - d / max(band, 1.0), 0.0, 1.0)


def make_seamless(img, band):
    """Make a texture tile by replacing a band along the edges.

    The band along the edges is replaced with content from the middle of the
    texture, shifted by half (np.roll by H/2, W/2). The middle is continuous,
    so the seam disappears. The same weight is applied to all maps, so they do
    not drift apart.

    img  — H×W or H×W×3 array (any map).
    band — band width in pixels; ≤ 0 returns img unchanged.

    Returns an array of the same shape (a normal map has to be renormalised
    afterwards, as MaterialJob.all_maps does)."""
    xp = X.xp(img, band)
    if band <= 0:
        return img
    a = xp.asarray(seam_weights(img.shape[:2], band))
    if img.ndim == 3:
        a = a[..., None]
    sh = xp.roll(xp.roll(img, img.shape[0] // 2, 0), img.shape[1] // 2, 1)
    return img * (1 - a) + sh * a


# ------------------------------------------------------------- пресеты -----
# height: (детальность, контраст, инверсия)  ao: (радиус, сила)
# spec:   (f0, глянец, разброс глянца, металличность, подкраска)
# F0 — отражение в упор. У всех диэлектриков оно около 0.04 и почти не
# зависит от того, камень это, дерево или ткань: разница между ними в
# шероховатости, а не в силе отражения. Разводить их по F0 — старая привычка
# из до-PBR времён, и именно она даёт блики ярче света.
# Металлы: железо ≈0.56, серебро и алюминий ≈0.9, у них цвет отражения берётся
# из альбедо. Ржавчина — уже не металл, а окисел, то есть диэлектрик.
PRESETS = {
    'stone':      dict(height=(0.55, 1.0, False), nstr=1.0, ao=(24, 0.9),
                       spec=(0.040, 0.30, 0.30, 0.0, 0.0), soft=0.5, hard=0.0),
    'stone_worn': dict(height=(0.45, 1.1, False), nstr=1.1, ao=(30, 1.0),
                       spec=(0.040, 0.12, 0.45, 0.0, 0.0), soft=0.6, hard=0.2),
    'brick':      dict(height=(0.30, 1.3, False), nstr=1.2, ao=(34, 1.0),
                       spec=(0.040, 0.16, 0.40, 0.0, 0.0), soft=0.5, hard=0.1),
    'plaster':    dict(height=(0.50, 0.8, False), nstr=0.8, ao=(22, 0.7),
                       spec=(0.040, 0.14, 0.30, 0.0, 0.0), soft=0.6, hard=0.1),
    'metal':      dict(height=(0.70, 0.9, False), nstr=0.8, ao=(16, 0.6),
                       spec=(0.70, 0.78, 0.15, 1.0, 1.0), soft=0.4, hard=0.0),
    'metal_rust': dict(height=(0.55, 1.2, False), nstr=1.1, ao=(22, 0.9),
                       spec=(0.35, 0.20, 0.55, 0.35, 0.8), soft=0.5, hard=0.1),
    'wood':       dict(height=(0.50, 1.0, False), nstr=1.0, ao=(20, 0.8),
                       spec=(0.040, 0.30, 0.35, 0.0, 0.1), soft=0.5, hard=0.1),
    'bark':       dict(height=(0.35, 1.25, False), nstr=1.3, ao=(28, 1.0),
                       spec=(0.040, 0.10, 0.40, 0.0, 0.0), soft=0.6, hard=0.2),
    'cloth':      dict(height=(0.65, 0.9, False), nstr=0.9, ao=(18, 0.7),
                       spec=(0.040, 0.10, 0.25, 0.0, 0.0), soft=0.4, hard=0.0),
    'leather':    dict(height=(0.60, 1.0, False), nstr=1.0, ao=(20, 0.8),
                       spec=(0.040, 0.22, 0.30, 0.0, 0.0), soft=0.5, hard=0.1),
    'skin':       dict(height=(0.70, 0.7, False), nstr=0.6, ao=(20, 0.6),
                       spec=(0.040, 0.38, 0.20, 0.0, 0.1), soft=0.7, hard=0.0),
    'ground':     dict(height=(0.60, 1.0, False), nstr=1.0, ao=(26, 0.9),
                       spec=(0.040, 0.08, 0.30, 0.0, 0.0), soft=0.6, hard=0.3),
    'moss':       dict(height=(0.45, 1.0, False), nstr=1.1, ao=(30, 1.0),
                       spec=(0.040, 0.06, 0.25, 0.0, 0.0), soft=0.6, hard=0.2),
    'glass':      dict(height=(0.75, 0.8, False), nstr=0.7, ao=(14, 0.5),
                       spec=(0.045, 0.92, 0.08, 0.0, 0.2), soft=0.3, hard=0.0),
    'snow':       dict(height=(0.55, 0.8, False), nstr=0.7, ao=(24, 0.7),
                       spec=(0.040, 0.35, 0.20, 0.0, 0.0), soft=0.7, hard=0.3),
}
PRESET_IDS = list(PRESETS)
PRESET_NAMES = PRESET_IDS          # совместимость со старым кодом


# ------------------------------------------------------- зоны материалов ---
def blend_maps(base, over, mask):
    """Two different material settings on one texture.

    The maps are computed, each with its own parameters, and stitched together
    by a mask. The results are blended, not the parameters: for height and AO
    intermediate settings would give not an intermediate result but a third
    one, resembling nothing. The normal is renormalised after blending.

    base — dict name → map (H×W or H×W×3), e.g. from MaterialJob.all_maps.
    over — dict of the same form; a key missing here falls back to base.
    mask — H×W, 0…1 (1 = over), clipped.

    Returns a new dict with the keys of base."""
    xp = X.xp(base, over, mask)
    m = xp.clip(mask, 0, 1).astype(xp.float32)
    out = {}
    for k, a in base.items():
        b = over.get(k, a)
        w = m[..., None] if a.ndim == 3 else m
        out[k] = a * (1 - w) + b * w
    n = out.get('normal')
    if n is not None:
        L = xp.sqrt((n ** 2).sum(-1, keepdims=True))
        out['normal'] = n / xp.maximum(L, 1e-6)
    return out


# ------------------------------------------------- параллакс OpenMW -------
# Как OpenMW читает высоту из альфы карты нормалей (_nh). Из исходников
# движка, files/shaders/lib/material/parallax.glsl:
#     offset = eyeDir.xy * (height * PARALLAX_SCALE + PARALLAX_BIAS)
# Отсюда: смещения нет при height = 0.5 (альфа 128), полный размах альфы —
# 4 % длины одного повтора текстуры. Отдельной настройки глубины в движке нет.
PARALLAX_SCALE = 0.04
PARALLAX_BIAS = -0.02
# Размер игрового юнита: движок считает 22.1 юнита на фут (MWSE, Game Units)
CM_PER_UNIT = 30.48 / 22.1
PARALLAX_REFS = ('mean', 'median', 'mode', 'manual')

_BAYER8 = (np.array([[0, 32, 8, 40, 2, 34, 10, 42], [48, 16, 56, 24, 50, 18, 58, 26],
                     [12, 44, 4, 36, 14, 46, 6, 38], [60, 28, 52, 20, 62, 30, 54, 22],
                     [3, 35, 11, 43, 1, 33, 9, 41], [51, 19, 59, 27, 49, 17, 57, 25],
                     [15, 47, 7, 39, 13, 45, 5, 37], [63, 31, 55, 23, 61, 29, 53, 21]],
                    np.float32) + 0.5) / 64.0


def parallax_range_cm(repeat_units):
    """Full relief range OpenMW can show, in centimetres: 4 % of the repeat length.

    OpenMW's parallax offset is eyeDir.xy · (h · 0.04 − 0.02), so the full alpha
    range 0…255 spans PARALLAX_SCALE = 0.04 of one texture repeat, with no
    offset at alpha 128. A game unit is 1/22.1 ft (CM_PER_UNIT). With a repeat
    of 128 units this is 7.1 cm.

    repeat_units — game units covered by one texture repeat. Returns a float."""
    return PARALLAX_SCALE * float(repeat_units) * CM_PER_UNIT


def parallax_alpha(h, repeat_units=128.0, depth_cm=None, ref='median',
                   ref_level=0.5, dither=True):
    """Alpha for _nh: the height laid into a window around 128 so that the relief
    comes out at the requested depth in OpenMW.

    OpenMW offsets the texture lookup by eyeDir.xy · (h · 0.04 − 0.02), so alpha
    128 (h = 0.5) is neutral — it coincides with the geometry — and the full
    alpha range equals 4 % of one repeat (parallax_range_cm). The engine has no
    separate depth setting, so depth is controlled only through how much of the
    alpha range the height occupies.

    h            — height, any scale (usually 0…1). Its range is taken between
                   the 0.5th and 99.5th percentiles.
    repeat_units — how many game units one texture repeat covers on the model
                   (22.1 units per foot). Everything depends on it: the engine
                   coefficient is set as a fraction of the repeat, not in
                   centimetres.
    depth_cm     — full relief range (from the lowest to the highest place) in
                   centimetres. None — all that is available. More than the
                   engine can show is not possible: the depth is cut down to
                   the limit rather than stretching alpha past the edge.
    ref          — which height goes to 128, i.e. coincides with the geometry:
                   'mean', 'median', 'mode' (the most frequent level, i.e. the
                   largest flat area — on masonry this is the face of the stones
                   or the joint filling, whichever takes more space; 64-bin
                   histogram) or 'manual' (ref_level — fraction from the bottom
                   to the top of the relief, 0…1).
    dither       — ordered noise of 1/255 before writing to 8 bits. On a narrow
                   window, steps are visible without it. The matrix is 8×8
                   (Bayer), so on a texture whose side is a multiple of 8 the
                   noise is seamless.

    A flat plate (all height weights at zero) has no relief and nothing to
    stretch: the alpha is then 0.5 everywhere.

    Returns (alpha, info). alpha is H×W float32 in 0…1 (128/255 neutral). info
    is a dict: repeat_cm and avail_cm — repeat length and available range in
    cm; step_mm — size of one alpha level in mm; depth_cm / want_cm — the depth
    used and the one requested; used — how many alpha levels are occupied;
    clipped — fraction of pixels that did not fit and were clipped; up_cm /
    down_cm — how far the relief rises above the surface and goes into it
    (cm); ref — the height value mapped to 128."""
    xp = X.xp(h, repeat_units, depth_cm, ref, ref_level, dither)
    h = xp.asarray(h, xp.float32)
    avail = parallax_range_cm(repeat_units)
    lo, hi = (float(v) for v in xp.percentile(h, [0.5, 99.5]))
    span = max(hi - lo, 1e-6)
    want = avail if depth_cm is None else max(float(depth_cm), 0.0)
    # больше, чем движок может показать, не бывает: урезаем до предела, а не
    # растягиваем альфу за край (это обрезало бы пятую часть картинки)
    depth = min(want, avail)
    if ref == 'mean':
        r = float(h.mean())
    elif ref == 'mode':
        hist, edges = xp.histogram(xp.clip(h, lo, hi), bins=64, range=(lo, hi))
        i = int(xp.argmax(hist))
        r = float(0.5 * (edges[i] + edges[i + 1]))
    elif ref == 'manual':
        r = lo + min(max(float(ref_level), 0), 1) * span
    else:
        r = float(xp.median(h))
    # ровная плита (все веса высоты в нуле) — рельефа нет, растягивать нечего
    k = 0.0 if hi - lo < 1e-4 else depth / max(avail, 1e-6) / span
    a = 0.5 + (h - r) * k
    clipped = float(((a < 0.0) | (a > 1.0)).mean())
    if dither:
        H, W = a.shape
        t = xp.tile(xp.asarray(_BAYER8), (H // 8 + 1, W // 8 + 1))[:H, :W]
        a = a + (t - 0.5) / 255.0
    a = xp.clip(a, 0.0, 1.0).astype(xp.float32)
    used = int(len(xp.unique((a * 255 + 0.5).astype(xp.uint8))))
    info = dict(repeat_cm=float(repeat_units) * CM_PER_UNIT, avail_cm=avail,
                step_mm=avail * 10.0 / 255.0, depth_cm=depth,
                want_cm=want, clipped=clipped, used=used,
                up_cm=(hi - r) * k * avail, down_cm=(r - lo) * k * avail, ref=r)
    return a, info


def parallax_height_px(h, info, repeat_units, width_px):
    """The same height that goes into the alpha, but in texture pixels (the step
    to the neighbouring pixel is the slope).

    Needed so that the normal (normal_from_heightmap) describes exactly the
    surface the parallax will show.

    h            — the height passed to parallax_alpha.
    info         — the info dict returned by parallax_alpha (uses up_cm,
                   down_cm, ref).
    repeat_units — game units per texture repeat, as for parallax_alpha.
    width_px     — width of the FULL-size texture in pixels (not of a preview
                   copy).

    Returns H×W float32 height in pixels, zero at the reference level."""
    xp = X.xp(h, info, repeat_units, width_px)
    span_cm = info['up_cm'] + info['down_cm']
    lo_hi = max(float(xp.percentile(h, 99.5) - xp.percentile(h, 0.5)), 1e-6)
    h_cm = (xp.asarray(h, xp.float32) - info['ref']) * (span_cm / lo_hi)
    px_cm = float(repeat_units) * CM_PER_UNIT / max(float(width_px), 1.0)
    return h_cm / px_cm


# Параметры, заданные в пикселях полноразмерной текстуры. Превью считается на
# уменьшенной копии, и если оставить их как есть, каждый радиус на копии
# захватит вдвое большую часть картинки: превью покажет не то, что потом
# сохранится (на стене 2048 наклон нормали выходил 46° в превью против 41° в
# файле, AO совпадало лишь на 0.7). См. at_scale.
PX_PARAMS = ('light_radius', 'guided_radius', 'ao_radius', 'n_scale', 'h_amp',
             'sfs_radius', 'sfs_chroma_radius')
PX_DEFAULTS = {'sfs_radius': 40.0, 'sfs_chroma_radius': 48.0}


def at_scale(p, px):
    """Full-size parameters, recomputed for a copy downscaled by a factor 1/px.

    Radii and the height amplitude are multiplied by px (PX_PARAMS; missing
    ones are filled from PX_DEFAULTS first). Otherwise every radius on the
    copy would grab twice as large a part of the picture, and the preview
    would show something other than what gets saved.

    Of the normal, only the fine layer (n_fine) is rescaled: it takes the height
    step to the neighbouring pixel, and on a downscaled copy the same relief
    gives twice the step. The mid and large layers are multiplied by their own
    radius (see normal_from_height) and do not depend on the size by
    themselves. Verified on a 2048 wall: difference between the preview and the
    downscaled saved result 0.033, versus 0.108 if the whole normal strength is
    divided instead.

    p  — parameter dict for the full-size image.
    px — scale of the copy (0.5 for a half-size preview).

    Returns p itself when px == 1, otherwise a new dict with the rescaled
    values and 'px' set."""
    if abs(float(px) - 1.0) < 1e-6:
        return p
    q = dict(PX_DEFAULTS)
    q.update(p)
    for k in PX_PARAMS:
        if k in q:
            q[k] = float(q[k]) * px
    if 'n_fine' in q:
        q['n_fine'] = float(q['n_fine']) * px
    q['px'] = float(px)
    return q


class MaterialJob:
    """Computes the whole set of maps for an opaque texture.

    rgb_srgb — H×W×3 sRGB 0…1 source diffuse (stored as float32).
    px       — how many times smaller the picture is than the full-size one
               (0.5 for a fast preview). Parameters are passed already rescaled
               (at_scale); px is needed for the fixed radii inside the
               computations themselves.

    Each stage (de-lighting, height variants, height mix, shading signal,
    direct normal, AO, bent normal) is cached under a key of the parameters it
    depends on, so moving one slider recomputes only the stages downstream of
    it. The order in all_maps matters: the direct normal and its fit
    (make_integrable) are applied BEFORE AO, otherwise height, AO and parallax
    drift apart and describe different surfaces."""

    def __init__(self, rgb_srgb, px=1.0):
        """Store the source (float32) and the preview scale, start with empty caches."""
        self.src = rgb_srgb.astype(np.float32)
        self.px = float(px)
        self.reset()

    def reset(self):
        """Drop all cached results and inputs (guide, external depth, ESRGAN maps)."""
        self._restore = 0.0
        self.extern = None
        self.guide = None
        self._vkey = None
        self._dl = self._h = None
        self._bent = None
        self._bent_key = None
        self._sfs_g = None
        self._g_key = None
        self._nd = self._nd_key = None
        self.esr_n = self.esr_f = None
        self._dl_key = self._h_key = self._ao_key = None

    def delit(self, soft, hard, radius, keep, restore=0.0, sat_keep=1.0):
        """De-lit diffuse (see delight for the parameters), cached.

        The grain-restore radius is 3 px of the full-size image (3·px here).
        restore is remembered and reused by later calls that re-derive the
        de-lit image (sfs_signal, _base). A new result invalidates the height
        mix cache. Returns H×W×3 sRGB 0…1."""
        self._restore = restore
        key = (round(soft, 3), round(hard, 3), int(radius), round(keep, 3),
               round(restore, 3), round(sat_keep, 3))
        if self._dl_key != key:
            self._dl = delight(self.src, soft, hard, radius, keep, self._restore,
                               restore_radius=3.0 * self.px, sat_keep=sat_keep)
            self._dl_key = key
            self._h_key = None
        return self._dl

    def set_guide(self, arr):
        """Height hint: H×W in −1…1, or None to clear.

        Not painting values over the map, but telling the algorithm where to
        raise and where to lower what it found by itself. The shape inside the
        area is still taken from the picture (see mix_heights: guide_amt,
        guide_flat)."""
        self.guide = None if arr is None else np.clip(arr.astype(np.float32), -1, 1)
        self._h_key = None

    def set_esr(self, normal=None, franken=None):
        """Outputs of the ESRGAN networks (see esrgan.py), already resized to
        this picture.

        normal  — NormalMapGenerator output, H×W×3 RGB 0…1, or None.
        franken — FrankenMapGenerator output, H×W×3 (R grey, G roughness,
                  B height), or None.

        Invalidates the height caches."""
        self.esr_n = None if normal is None else np.asarray(normal, np.float32)
        self.esr_f = None if franken is None else np.asarray(franken, np.float32)
        self._vkey = None
        self._h_key = None

    def set_extern(self, arr):
        """External depth map, already resized to the texture: H×W 0…1, or None."""
        self.extern = None if arr is None else np.clip(arr.astype(np.float32), 0, 1)
        self._vkey = None
        self._h_key = None

    def _base(self, p, from_delit):
        """Source for height/shading: the de-lit image if from_delit, else the original."""
        return (self.delit(p['soft'], p['hard'], p['light_radius'], p['keep'],
                           self._restore, p.get('sat_keep', 1.0))
                if from_delit else self.src)

    def sfs_signal(self, p, from_delit=False):
        """Shading signal for the "from lighting" layer and for the normal from
        lighting — one and the same, with the same corrections, regardless of
        whether the layer itself is included in the height mix. Cached by its
        own parameters.

        The signal is taken from the COLOUR picture: from luminance alone paint
        cannot be told from shadow. If hard shadows have already been removed on
        the light tab (p['hard'] > 0 and from_delit is False), the shadow field
        is taken from there (shading_signal's lit argument).

        p          — parameter dict (sfs_*, hard, soft, keep, light_radius,
                     sat_keep).
        from_delit — use the de-lit image as the source.

        Returns H×W float32 (see shading_signal)."""
        key = (bool(from_delit), round(p.get('sfs_radius', 40.0), 1),
               round(p.get('sfs_chroma', 0.6), 3), round(p.get('sfs_shadow', 1.0), 3),
               round(p.get('sfs_chroma_radius', 48.0), 1), int(p.get('light_radius', 48)),
               round(p.get('hard', 0.0), 3), round(p.get('soft', 0.0), 3),
               round(p.get('keep', 1.0), 3), round(self._restore, 3),
               round(p.get('sat_keep', 1.0), 3))
        if self._g_key != key:
            base = self._base(p, from_delit)
            # сигнал берётся с ЦВЕТНОЙ картинки: по одной яркости краску от
            # тени не отличить. Если резкие тени уже сняты на вкладке света,
            # поле теней берём оттуда: человек его там и настраивал, а два
            # независимых мнения об одних и тех же тенях ни к чему
            lit = None
            if not from_delit and p.get('hard', 0.0) > 0:
                lit = self.delit(p['soft'], p['hard'], p['light_radius'],
                                 p['keep'], self._restore, p.get('sat_keep', 1.0))
            self._sfs_g = shading_signal(base, p.get('sfs_radius', 40.0),
                                         p.get('sfs_chroma', 0.6),
                                         p.get('sfs_shadow', 1.0),
                                         p.get('sfs_chroma_radius', 48.0),
                                         p.get('light_radius', 48), lit)
            self._g_key = key
        return self._sfs_g

    def height(self, p, from_delit=False):
        """Mixed height map for parameters p, cached in two levels.

        By default height is taken from the ORIGINAL (from_delit=False): on a
        photographic texture the shadow is the only cue to the relief, and
        removing the light first means throwing away what is being looked for.

        Level 1 (height_variants) depends on the detail, radii, shading and
        external-map settings; level 2 (mix_heights) on the layer weights w_*,
        contrast, gamma, h_amp, max_tilt, invert and the guide. The shading
        layer is computed only when w_sfs > 0, the ESRGAN and external layers
        only when those maps are set. A new mix invalidates the AO cache.

        Returns H×W in 0…1."""
        # По умолчанию высоту берём с ОРИГИНАЛА. На фотографической текстуре
        # тень и есть единственный признак рельефа: сначала снять свет, а потом
        # искать по ней форму — значит выбросить то, что ищем.
        base = (self.delit(p['soft'], p['hard'], p['light_radius'], p['keep'],
                           self._restore, p.get('sat_keep', 1.0))
                if from_delit else self.src)
        ex = getattr(self, 'extern', None)
        if ex is not None and p.get('ext_invert'):
            ex = 1.0 - ex
        az = p.get('sfs_az', None) if p.get('w_sfs', 0.0) > 0 else None
        vkey = (round(p['h_detail'], 3), round(p['guided_radius'], 2),
                round(p['macro_falloff'], 3), bool(from_delit),
                id(ex) if ex is not None else 0, bool(p.get('ext_invert')),
                None if az is None else round(az, 4),
                round(p.get('sfs_smooth', 0.25), 3),
                round(p.get('sfs_radius', 40.0), 1),
                round(p.get('sfs_chroma', 0.6), 3), round(p.get('sfs_shadow', 1.0), 3),
                round(p.get('hard', 0.0), 3), round(p.get('soft', 0.0), 3),
                round(p.get('keep', 1.0), 3), int(p.get('light_radius', 48)),
                round(self._restore, 3), round(p.get('sat_keep', 1.0), 3),
                round(p.get('sfs_scale', 0.3), 3), round(p.get('sfs_low', 1.0), 3),
                round(p.get('sfs_high', 1.0), 3),
                bool(p.get('ext_cavity')), round(p.get('ext_cut', 0.25), 3),
                id(self.esr_f) if self.esr_f is not None else 0)
        if getattr(self, '_vkey', None) != vkey:
            L = luminance(srgb_to_lin(base))
            g = self.sfs_signal(p, from_delit) if az is not None else None
            self._parts = height_variants(L, p['h_detail'], p['guided_radius'],
                                          p['macro_falloff'], ex, az,
                                          p.get('sfs_smooth', 0.25),
                                          p.get('sfs_radius', 40.0),
                                          bool(p.get('ext_cavity')),
                                          p.get('ext_cut', 0.25), g,
                                          p.get('sfs_scale', 0.3),
                                          p.get('sfs_low', 1.0),
                                          p.get('sfs_high', 1.0), self.px)
            if self.esr_f is not None:
                # высота из франкен-карты (канал B) — ещё один слой смеси
                self._parts['esr'] = _n01(self.esr_f[..., 2])
            self._vkey = vkey
            self._h_key = None
        w = {k: p['w_' + k] for k in ('band', 'multi', 'lum', 'guided')}
        if self.esr_f is not None:
            w['esr'] = p.get('w_esr', 0.0)
        if az is not None:
            w['sfs'] = p.get('w_sfs', 0.0)
        if getattr(self, 'extern', None) is not None:
            w['ext'] = p.get('w_ext', 0.0)
        g = getattr(self, 'guide', None)
        key = (vkey, tuple(sorted((k, round(v, 3)) for k, v in w.items())),
               round(p['h_contrast'], 3), round(p.get('h_gamma', 1.0), 3),
               round(p['h_amp'], 2), round(p.get('guide_amt', 0.0), 3),
               round(p.get('guide_flat', 0.0), 3), (g is None) or float(np.abs(g).sum()),
               round(p['max_tilt'], 1), bool(p['h_invert']))
        if self._h_key != key:
            self._h = mix_heights(self._parts, w, p['h_contrast'], p['h_amp'],
                                  p['max_tilt'], p['h_invert'], p.get('h_gamma', 1.0),
                                  g, p.get('guide_amt', 0.0), p.get('guide_flat', 0.0))
            self._h_key = key
            self._ao_key = None
        return self._h

    def ao(self, h, n, p, extra=None):
        """Ambient occlusion for the final height h and normal n, cached.

        A weighted mix of relief AO (weight ao_w_cav: cavity_ao, or hbao when
        ao_horizon is set) and normal AO (weight ao_w_norm: normal_ao, scaled by
        ao_strength); both weights at zero give all ones.

        h     — H×W height 0…1.
        n     — H×W×3 unit normals.
        p     — parameter dict.
        extra — hashable key of everything that changes h or n besides the
                height itself (direct normal, its fit, ESRGAN normal).

        Returns H×W in 0…1 (1 = open)."""
        # extra — всё, что меняет h или n помимо самой высоты (нормаль по
        # освещению и её подгонка). Без этого затенение оставалось бы от
        # прошлого положения ползунков
        key = (extra, round(p['ao_radius'], 2), round(p['h_amp'], 2),
               round(p['ao_strength'], 3),
               round(p.get('ao_w_cav', 1.0), 3), round(p.get('ao_w_norm', 0.0), 3),
               bool(p.get('ao_horizon', False)), self._h_key,
               round(p.get('n_fine', 1.0), 3), round(p.get('n_mid', 0.0), 3),
               round(p.get('n_big', 0.0), 3), round(p.get('n_scale', 8.0), 2),
               # AO «по наклону нормали» зависит от всего, что меняет нормаль
               round(p.get('n_strength', 1.0), 3), bool(p.get('directx', True)),
               bool(p.get('plx_normal')), p.get('plx_repeat'), p.get('plx_depth'),
               p.get('plx_ref'), p.get('plx_level'))
        if self._ao_key != key:
            wc = max(p.get('ao_w_cav', 1.0), 0.0)
            wn = max(p.get('ao_w_norm', 0.0), 0.0)
            tot = wc + wn
            if tot <= 0:
                self._aov = np.ones_like(h)
            else:
                cav = (hbao(h, p['ao_radius'], p['h_amp'], p['ao_strength'])
                       if p.get('ao_horizon') else
                       cavity_ao(h, p['ao_radius'], p['ao_strength']))
                nao = 1.0 - p['ao_strength'] * (1.0 - normal_ao(n))
                self._aov = np.clip((wc * cav + wn * nao) / tot, 0, 1)
            self._ao_key = key
        return self._aov

    def all_maps(self, p, band=0):
        """p — parameter dict. Returns all maps as one set.

        band — width in pixels of the edge band replaced by make_seamless; 0
               leaves the maps as computed (they are already seamless by
               construction when the source is).

        Order: de-lighting → height → normal from height (or, with plx_normal,
        the normal consistent with the parallax depth) → ESRGAN normal mix →
        direct normal from shading and its fit (make_integrable, which may
        replace the height) → AO → bent normal → specular/roughness → ESRGAN
        roughness → tone curves → relief into diffuse → make_seamless →
        parallax alpha from the seamless height. The direct normal and its fit
        come BEFORE AO; otherwise height, AO and parallax drift apart.

        Returns a dict: diffuse (H×W×3 sRGB 0…1, with relief blended in),
        diffuse_base (the de-lit diffuse without it), height (H×W 0…1), ao
        (H×W 0…1), normal (H×W×3 unit vectors, DirectX or OpenGL per
        p['directx']), spec (H×W×3 sRGB), shininess, rough, metal (H×W 0…1);
        plus nh_alpha (H×W 0…1) and plx_info when p['plx_repeat'] is set. Also
        sets self.direct_curl (slope_curl of the direct normal, or None)."""
        d = self.delit(p['soft'], p['hard'], p['light_radius'], p['keep'],
                       p.get('restore', 0.0), p.get('sat_keep', 1.0))
        h = self.height(p, p.get('h_from_delit', False))
        n = normal_from_height(h, p['n_strength'], p['directx'],
                               p.get('n_fine', 1.0), p.get('n_mid', 0.0),
                               p.get('n_big', 0.0), p.get('n_scale', 8.0))
        plx = p.get('plx_repeat') is not None
        if plx and p.get('plx_normal'):
            # нормаль, согласованная с параллаксом: та же высота, та же
            # глубина в сантиметрах, пересчитанная в уклон на пиксель. Сила
            # нормали тогда не задаётся — она следует из глубины. Ширина
            # берётся полноразмерная (px): на копии превью пиксель крупнее
            _, inf = parallax_alpha(h, p['plx_repeat'], p.get('plx_depth'),
                                    p.get('plx_ref', 'median'), p.get('plx_level', 0.5),
                                    False)
            hp = parallax_height_px(h, inf, p['plx_repeat'], h.shape[1] / self.px)
            n = normal_from_heightmap(hp * self.px, p['directx'])
        k_esr = float(np.clip(p.get('esr_nmix', 0.0), 0, 1))
        if k_esr > 0 and self.esr_n is not None:
            # нормаль от NormalMapGenerator. Сила — множитель на наклон (x и y
            # до нормировки); зелёный разворачивается, если соглашение модели
            # не совпало с выбранным
            ne = self.esr_n * 2.0 - 1.0
            ny = ne[..., 1] if p['directx'] else -ne[..., 1]
            if p.get('esr_nflip'):
                ny = -ny
            kk = float(p.get('esr_nk', 1.0))
            ne = np.stack([ne[..., 0] * kk, ny * kk, np.maximum(ne[..., 2], 1e-3)], -1)
            ne = ne / np.sqrt((ne * ne).sum(-1, keepdims=True))
            n = ne if k_esr >= 0.999 else n * (1 - k_esr) + ne * k_esr
            n = n / np.maximum(np.sqrt((n * n).sum(-1, keepdims=True)), 1e-6)
        esr_key = (round(k_esr, 3), round(float(p.get('esr_nk', 1.0)), 3),
                   bool(p.get('esr_nflip')), id(self.esr_n)) if k_esr > 0 else None
        nd_key = None
        mix = float(np.clip(p.get('n_direct_mix', 0.0), 0, 1))
        if mix > 0:
            # нормаль по освещению (R и G): см. normal_from_shading. Слой
            # берётся с теми же настройками, что слой освещения в смеси высот,
            # поэтому выглядит так же, как его превью
            from_delit = p.get('h_from_delit', False)
            bmode = p.get('n_direct_b', 'unit')
            g = self.sfs_signal(p, from_delit)
            nd_key = (self._g_key, round(p.get('sfs_smooth', 0.25), 3),
                      round(p.get('sfs_scale', 0.3), 3), round(p.get('sfs_low', 1.0), 3),
                      round(p.get('sfs_high', 1.0), 3), round(p.get('sfs_radius', 40.0), 1),
                      round(p.get('n_direct', 1.0), 3), bool(p.get('n_direct_swap')),
                      bool(p.get('n_direct_inv')), bool(p['directx']), bmode,
                      self._h_key if bmode == 'ao' else None,
                      round(p['ao_radius'], 2) if bmode == 'ao' else None,
                      round(p['ao_strength'], 3) if bmode == 'ao' else None,
                      round(float(p.get('n_direct_fix', 0.0)), 3), round(mix, 3))
            if self._nd_key != nd_key:
                L = luminance(srgb_to_lin(self._base(p, from_delit)))
                ao_b = (cavity_ao(h, p['ao_radius'], p['ao_strength'])
                        if bmode == 'ao' else None)
                nd = normal_from_shading(
                    L, None, p.get('sfs_smooth', 0.25), p.get('n_direct', 1.0), g,
                    p['directx'], bool(p.get('n_direct_swap')),
                    p.get('sfs_scale', 0.3), p.get('sfs_low', 1.0),
                    p.get('sfs_high', 1.0), p.get('sfs_radius', 40.0),
                    bmode, ao_b, bool(p.get('n_direct_inv')))
                hf = None
                fix = float(p.get('n_direct_fix', 0.0))
                if fix > 0:
                    # подогнать под высоту: см. make_integrable. При полной
                    # подгонке высотой становится та, что отвечает этой
                    # нормали, и тогда параллакс с AO описывают ту же
                    # поверхность
                    nd, hf = make_integrable(nd, fix, p['directx'])
                # slope_curl считает в раскладке DirectX; для OpenGL разворачиваем
                self._nd = (nd, hf, slope_curl(
                    nd if p['directx'] else nd * np.array([1.0, -1.0, 1.0], np.float32)))
                self._nd_key = nd_key
            nd, hf, self.direct_curl = self._nd
            if hf is not None:
                fix = float(p.get('n_direct_fix', 0.0))
                h = h * (1 - fix) + hf * fix
            n = nd if mix >= 0.999 else (n * (1 - mix) + nd * mix)
            n = n / np.maximum(np.sqrt((n * n).sum(-1, keepdims=True)), 1e-6)
        else:
            self.direct_curl = None
        ao = self.ao(h, n, p, (nd_key, esr_key))
        if p.get('n_bent', 0.0) > 0:
            key = (self._h_key, nd_key, round(p['ao_radius'], 2), round(p['h_amp'], 2))
            if self._bent_key != key:
                self._bent = bent_from_height(h, p['ao_radius'], p['h_amp'])
                self._bent_key = key
            n = bend_normal(n, self._bent, p['n_bent'], p['directx'])
        spec, shin, rough, metal = material_maps(
            d, h, ao, p['f0'], p['gloss'], p['gloss_var'], p['metal'], p['tint'])
        k_r = float(np.clip(p.get('esr_rough', 0.0), 0, 1))
        if k_r > 0 and self.esr_f is not None:
            # шероховатость из франкен-карты (канал G) вместо посчитанной
            rough = rough * (1 - k_r) + self.esr_f[..., 1] * k_r
            shin = np.clip((1.0 - rough) ** 2, 0.0, 1.0)   # как в material_maps
        ao = tone(ao, p.get('ao_gamma', 1.0), p.get('ao_contrast', 1.0))
        rough = tone(rough, p.get('rough_gamma', 1.0), p.get('rough_contrast', 1.0))
        d0 = d
        amt = float(p.get('rd_amt', 0.0))
        if amt > 0:
            # подмешать рельеф в диффуз: см. relief_into_diffuse
            srcn = p.get('rd_src', 'height')
            if srcn == 'ao':
                sm = ao
            elif srcn == 'ext' and getattr(self, 'extern', None) is not None:
                sm = self.extern
            else:
                sm = h
            d = relief_into_diffuse(d, sm, p.get('rd_mode', 'overlay'), amt,
                                    band_radius(min(h.shape), p.get('rd_scale', 0.3)),
                                    p.get('rd_low', 1.0), p.get('rd_high', 1.0),
                                    p.get('rd_sat', 1.0), occl=(sm is ao))
        out = dict(diffuse=d, diffuse_base=d0, height=h, ao=ao, normal=n,
                   spec=spec, shininess=shin, rough=rough, metal=metal)
        if band > 0:
            out = {k: make_seamless(v, band) for k, v in out.items()}
            n = out['normal']
            L = np.sqrt((n ** 2).sum(-1, keepdims=True))
            out['normal'] = n / np.maximum(L, 1e-6)
        if plx:
            # альфа для _nh считается по уже бесшовной высоте
            out['nh_alpha'], out['plx_info'] = parallax_alpha(
                out['height'], p['plx_repeat'], p.get('plx_depth'),
                p.get('plx_ref', 'median'), p.get('plx_level', 0.5),
                bool(p.get('plx_dither', True)))
        return out


# ------------------------------------------------- варианты высоты --------
def height_variants(L, detail=0.45, guided_radius=8.0, macro_falloff=0.7, extern=None,
                    sfs_az=None, sfs_smooth=0.25, sfs_radius=40.0,
                    ext_cavity=False, ext_cut=0.25, sfs_g=None,
                    sfs_scale=0.3, sfs_low=1.0, sfs_high=1.0, px=1.0):
    """Four ways to get height out of luminance. Each has its own disease, so
    they are not chosen between but mixed (mix_heights). Two optional layers
    come on top: an external depth map and the shading-based layer.

      band   — band-pass: only fine relief; a wide crevice stays grey inside,
               only its edge turns dark.
      multi  — multi-scale: the same bands at several sizes with decaying
               weight; the large shape returns without drowning the detail.
      lum    — the whole luminance: maximal large shape, but all the paint goes
               into the relief too (a dark stone becomes a pit).
      guided — edge-preserving: the same luminance with the grain removed and
               region borders kept sharp; a wide crevice stays deep across its
               whole width.
      ext    — external depth map (MiDaS, a scan, hand painting), or its
               double integral when ext_cavity is set (height_from_cavity).
      sfs    — from the lighting captured in the texture (sfs_layer).

    L             — H×W LINEAR luminance.
    detail        — 0…1, as in height_map (denoise blur and band-pass cut).
    guided_radius — guided-filter radius in pixels (at least 2).
    macro_falloff — 0…1 weight decay per scale of the multi layer (scale step
                    ×2.5, from 4 px up to 1/6 of the shorter side).
    extern        — H×W 0…1 external depth map, or None.
    sfs_az        — light azimuth in radians for the sfs layer, or None to skip.
    sfs_smooth, sfs_radius, sfs_g, sfs_scale, sfs_low, sfs_high — passed to
                    sfs_layer.
    ext_cavity    — treat extern as an occlusion map and integrate it.
    ext_cut       — cutoff for height_from_cavity (fraction of the shorter side).
    px            — scale of the copy (see at_scale): the radii hard-coded here
                    rely on it.

    All filters wrap, so every layer tiles. Returns a dict of H×W arrays in
    0…1 (percentile stretch, _n01) with keys band, multi, lum, guided, ext, sfs;
    ext and sfs are zeros when not requested."""
    xp = X.xp(L, detail, guided_radius, macro_falloff, extern, sfs_az, sfs_smooth, sfs_radius, ext_cavity, ext_cut, sfs_g, sfs_scale, sfs_low, sfs_high, px)
    nd = X.ndi(L, detail, guided_radius, macro_falloff, extern, sfs_az, sfs_smooth, sfs_radius, ext_cavity, ext_cut, sfs_g, sfs_scale, sfs_low, sfs_high, px)
    small = (0.6 + 2.4 * (1.0 - detail)) * px
    x = nd.gaussian_filter(L, small, mode='wrap')

    # 1. полосовой: только мелкий рельеф. Крупной формы нет вообще —
    #    широкая щель внутри остаётся серой, тёмным получается только её край
    big = (6.0 + 90.0 * (1.0 - detail) ** 2) * px
    band = x - fast_blur(x, big)

    # 2. многомасштабный: те же полосы на нескольких размерах со спадающим
    #    весом. Крупная форма возвращается, но не забивает мелочь
    acc = xp.zeros_like(x)
    w, r, tot = 1.0, 4.0 * px, 0.0
    rmax = min(L.shape) / 6.0
    prev = fast_blur(x, r)
    while r < rmax and w > 0.02:
        r2 = min(r * 2.5, rmax)
        nxt = fast_blur(x, r2)
        b = prev - nxt
        acc += w * b / max(float(xp.percentile(xp.abs(b), 99)), 1e-6)
        tot += w
        w *= macro_falloff
        prev, r = nxt, r2
    multi = acc / max(tot, 1e-6)

    # 3. яркость целиком: крупная форма максимальная, но и краска целиком
    #    уходит в рельеф — тёмный камень станет ямой
    lum = x

    # 4. краесохраняющий: та же яркость, но зерно снято, а границы областей
    #    оставлены резкими. Широкая щель остаётся глубокой по всей ширине
    guided = guided_filter(L, L, max(2, int(guided_radius)), 0.02)

    # 5. готовая карта глубины со стороны: MiDaS, скан, ручная рисовка.
    #    Она не выводится из картинки вообще и потому знает то, чего яркость
    #    знать не может — крупную форму без примеси краски
    if extern is None:
        ext = xp.zeros_like(x)
    elif ext_cavity:
        # карту приняли за затенение: тёмное — это щель, а не низкое место.
        # Тогда её надо не класть высотой, а проинтегрировать (см.
        # height_from_cavity), и борозда получит настоящий профиль
        ext = _n01(height_from_cavity(extern.astype(xp.float32), ext_cut))
    else:
        ext = _n01(extern.astype(xp.float32))

    # 6. по освещению, снятому на самой текстуре: яркость это не высота, а
    #    уклон вдоль света, поэтому её интегрируют, а не берут как есть
    if sfs_az is None:
        sfs = xp.zeros_like(x)
    else:
        sfs = sfs_layer(L, sfs_az, sfs_smooth, sfs_radius, sfs_g,
                        sfs_scale, sfs_low, sfs_high)

    return {'band': _n01(band), 'multi': _n01(multi),
            'lum': _n01(lum), 'guided': _n01(guided), 'ext': ext, 'sfs': sfs}


def soft01(t, knee=0.04):
    """Bring a stretched map t (its main range already 0…1) into 0…1 without
    cutting the tails flat.

    A hard clip turns everything beyond the stretch into a plateau: the tops
    of the highest bumps and the bottoms of the deepest pits become flat, and
    a normal made from such a height shows flat blobs. Here t is kept as is
    on knee…1 − knee, and beyond that bent hyperbolically, 1 − k²/(t − 1 + 2k)
    at the top (and the mirror at the bottom): continuous with slope 1 at
    the knee, still rising however far out, never reaching 0 or 1, so the
    shape of a peak survives (a point 20 % beyond the range still differs
    from one 10 % beyond by about a step of 255)."""
    xp = X.xp(t)
    k = float(knee)
    hi = 1.0 - k * k / xp.maximum(t - 1.0 + 2.0 * k, k)
    lo = k * k / xp.maximum(k - t + k, k)
    return xp.where(t > 1.0 - k, hi, xp.where(t < k, lo, t))


def _n01(x):
    """Stretch the 0.5th…99.5th percentiles of x to 0…1, the tails bent in
    softly (soft01) instead of clipped; same shape."""
    xp = X.xp(x)
    lo, hi = xp.percentile(x, [0.5, 99.5])
    return soft01((x - lo) / max(float(hi - lo), 1e-6))


def tone(x, gamma=1.0, contrast=1.0, pivot=0.5):
    """Tone for a greyscale map: first contrast around the middle, then gamma.

    The value is raised to the power 1/gamma, as with the middle slider of
    Levels in image editors: gamma above 1 lifts the darks (brighter), below 1
    sinks them (darker).

    x        — array in 0…1 (clipped).
    gamma    — exponent parameter, floored at 0.05; 1 = off.
    contrast — multiplier around pivot; 1 = off.
    pivot    — contrast centre, 0.5 by default.

    Returns an array of the same shape in 0…1."""
    xp = X.xp(x, gamma, contrast, pivot)
    y = xp.clip((xp.clip(x, 0, 1) - pivot) * contrast + pivot, 0.0, 1.0)
    if abs(gamma - 1.0) > 1e-3:
        y = xp.power(y, 1.0 / max(gamma, 0.05))
    return y


def mix_heights(parts, weights, contrast=1.0, amp=20.0, max_tilt_deg=75.0, invert=False,
                gamma=1.0, guide=None, guide_amt=0.0, guide_flat=0.0):
    """Add the height variants up as weighted layers without letting the contrast
    be overcranked.

    If after all the multipliers the relief turns out steeper than the limit,
    the whole map is pressed down proportionally. Otherwise the normal goes
    past 90° and pixels facing away from the camera appear in the engine.

    With all weights at zero a flat plate at 0.5 is returned (then passed
    through tone, which keeps it flat). All zeros mean "no relief", not "take
    some layer at your discretion": earlier the band-pass layer was silently
    substituted here, and it looked as if the sliders did not work.

    parts        — dict of H×W layers 0…1 from height_variants (must contain
                   'band').
    weights      — dict name → weight; weights ≤ 0 are ignored. Each layer is
                   mean-subtracted, the weighted sum divided by the total weight
                   and stretched to 0…1 (_n01).
    contrast, gamma — tone curve (see tone).
    amp          — height scale: h·amp is the height in pixels used for the
                   tilt check (h_amp in MaterialJob, after at_scale).
    max_tilt_deg — slope limit in degrees at the 99.5th percentile of the tilt.
    invert       — flip high and low (applied last).
    guide        — H×W hint in −1…1 (MaterialJob.set_guide), or None.
    guide_amt    — how much of the guide is added to the height (can be
                   negative).
    guide_flat   — 0…1: flatten the marked area — inside it the height is pulled
                   towards its own (|guide|-weighted) mean, so the area becomes a
                   plateau rather than relief.

    Gradients wrap (grad_wrap). Returns H×W in 0…1."""
    xp = X.xp(parts, weights, contrast, amp, max_tilt_deg, invert, gamma, guide, guide_amt, guide_flat)
    tot = sum(max(w, 0.0) for w in weights.values())
    if tot <= 0:
        # все веса в нуле значит «рельефа нет», а не «возьми какой-нибудь
        # слой на своё усмотрение». Раньше сюда подставлялся полосовой, и
        # выглядело это так, будто ползунки не работают
        h = xp.full_like(parts['band'], 0.5)
    else:
        h = xp.zeros_like(parts['band'])
        for k, w in weights.items():
            if w > 0:
                h = h + w * (parts[k] - float(parts[k].mean()))
        h = _n01(h / tot)
    h = tone(h, gamma, contrast)

    if guide is not None and (guide_amt != 0.0 or guide_flat > 0):
        g = xp.clip(guide, -1.0, 1.0)
        if guide_flat > 0:
            # выровнять размеченное: внутри области высота тянется к своему
            # среднему, то есть область становится площадкой, а не рельефом
            w = xp.abs(g)
            tot = float(w.sum())
            if tot > 1e-3:
                lvl = float((h * w).sum() / tot)
                h = h * (1 - guide_flat * w) + lvl * (guide_flat * w)
        h = xp.clip(h + guide_amt * g, 0.0, 1.0)

    gy, gx = grad_wrap(h * max(amp, 1e-3))
    tilt = xp.degrees(xp.arctan(xp.sqrt(gx * gx + gy * gy)))
    p = float(xp.percentile(tilt, 99.5))
    if p > max_tilt_deg:
        k = math.tan(math.radians(max_tilt_deg)) / max(math.tan(math.radians(p)), 1e-6)
        h = xp.clip((h - 0.5) * k + 0.5, 0, 1)
    return 1.0 - h if invert else h
