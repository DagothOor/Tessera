# -*- coding: utf-8 -*-
"""
Repair module: exposure recovery, masking and region selection.
Модуль восстановления: пересветы, провалы, маски и выделение областей.

Three families of tools live here.

1. Exposure masks. A photographed or hand-painted texture often has areas
   that carry no usable signal any more: blown highlights clipped to white
   and crushed shadows clipped to black. They are found by brightness plus
   local contrast — a bright area that still has detail is not blown, it is
   simply light.

2. Repair. Three approaches, from cheapest to most capable:
     * tone      — rebalance the level of the region to its surroundings.
                   Keeps whatever detail is left, invents nothing.
     * smooth    — biharmonic inpainting. Fills the hole with the smoothest
                   surface that matches the border. Correct for gradients,
                   useless for texture.
     * patch     — exemplar transfer. Finds the place in the same texture
                   that best continues the border of the hole and copies it
                   in, with a Poisson-style boundary match so the seam does
                   not show. This is what modern content-aware fill does,
                   in its simplest single-offset form.

3. Region selection. Turns a rough brush stroke into a region the algorithm
   finds by itself: colour similarity to the stroke, either contiguous
   (stops at contours) or across the whole texture. Also builds masks from a
   single channel with a threshold range.

Conventions. Images are float RGB (H, W, 3) in 0…1 as loaded (sRGB-encoded
values; nothing here converts to linear light), masks are (H, W) in 0…1
where > 0.5 means "damaged" / "selected". Textures tile, so every blur uses
mode='wrap', exposure masks are computed on a wrap-padded copy, and repair()
rolls the texture so that a patch crossing the tile edge is repaired as one
piece.
"""

import numpy as np
from scipy import ndimage as ndi

import xpu as X

BUILD = 25


def _gpu(a):
    """A host array moved to the card when the card is in use (else as is):
    the heavy parts below (exposure masks, the donor search) run there, the
    rest of the repair stays on the processor."""
    return X.to_dev(np.ascontiguousarray(a)) if X.active() else a


def luminance(rgb):
    """Rec. 709 luma of an (..., 3) array, applied as-is (no linearisation); (...)."""
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def _blur(x, r, mode='wrap'):
    """Gaussian blur, sigma r px (at least 0.5), wrapping round the edges;
    float32, in the module of x."""
    # заворот, а не отражение: текстуры замощаются, и всё, что считается
    # по краю, должно смыкаться с противоположным краем
    return X.ndi(x).gaussian_filter(x.astype(X.xp(x).float32), max(r, 0.5), mode=mode)


def local_contrast(x, radius=6.0):
    """Spread of brightness in the neighbourhood (local standard deviation).

    A blown highlight is not simply light, it is light WITHOUT texture.
    x: float (H, W), 0…1; radius: Gaussian sigma of the window in px
    (default 6). Returns float32 (H, W), ≥ 0 (at most 0.5 for 0…1 input).
    Seamless: the window wraps round the tile edges.
    """
    m = _blur(x, radius)
    v = _blur(x * x, radius) - m * m
    return np.sqrt(np.maximum(v, 0.0))


def _band_contrast(x, need, radius=6.0, step=0.03, band=0.06):
    """Texture around each pixel among neighbours of about its own brightness.

    For a dark spot this is the texture left INSIDE the spot, not the jump to
    its lighter surroundings, so a thin black crack counts as featureless
    just like a big black hole. x: float (H, W), 0…1; need: bool (H, W),
    the pixels whose value is wanted (others get 1.0); radius: Gaussian
    sigma of the window, px. The brightness range is cut into levels every
    ``step``; for each level the local standard deviation is taken over the
    neighbours within ±``band`` of it (in a square window of side
    2·radius + 1), and a pixel gets the value of its nearest level.
    Returns float32 (H, W), ≥ 0, in the module of x.
    """
    xp, nd = X.xp(x), X.ndi(x)
    out = xp.ones(x.shape, xp.float32)
    if not bool(need.any()):
        return out
    lev = xp.clip(xp.rint(x / step), 0, int(round(1.0 / step))).astype(xp.int32)
    size = 2 * int(round(radius)) + 1

    def box(a):
        return nd.uniform_filter(a, size, mode='wrap')
    for k in X.to_host(xp.unique(lev[need])).tolist():
        c = k * step
        w = (xp.abs(x - c) <= band).astype(xp.float32)
        # квадратное окно вместо гауссова: уровней бывает до двух десятков,
        # а скользящая сумма в несколько раз быстрее
        n = xp.maximum(box(w), 1e-3)
        m = box(x * w) / n
        v = box(x * x * w) / n - m * m
        sel = need & (lev == k)
        out[sel] = xp.sqrt(xp.maximum(v[sel], 0.0))
    return out


# ------------------------------------------------------------------ маски --
def exposure_masks(rgb, hi=0.90, lo=0.08, flat=0.030, radius=6.0, soft=1.5,
                   slack=0.12, grow=2.0, reach=6.0):
    """Find blown highlights and crushed shadows; seamless wrapper over
    _exposure_masks.

    The image is wrapped around its edges with a margin larger than the
    filters reach (max(64, reach + |grow| + 4·radius) px, but at most half
    the image side), processed and cropped back. Otherwise a spot at the edge
    of the texture would not see its continuation on the other side, and the
    mask would break off at the seam. On textures smaller than about
    2·margin the cap at half the side can make the margin shorter than the
    filters reach.

    rgb: float (H, W, 3), 0…1. Parameters are those of _exposure_masks
    (thresholds on luminance 0…1, distances in px): hi 0.90, lo 0.08,
    flat 0.03, radius 6, soft 1.5, slack 0.12, grow 2 (may be negative),
    reach 6.

    Returns (blown, crushed): float32 (H, W) soft masks, 0…1, in the module
    of rgb; the work itself runs on the card when it is in use.
    """
    H, W = rgb.shape[:2]
    P = int(min(max(64, reach + abs(grow) + 4 * radius), H // 2, W // 2))
    dev = X.is_dev(rgb)
    big = X.xp(rgb).pad(rgb, ((P, P), (P, P), (0, 0)), mode='wrap')
    if not dev:
        big = _gpu(big)
    b, c = _exposure_masks(big, hi, lo, flat, radius, soft, slack, grow, reach)
    b, c = b[P:P + H, P:P + W], c[P:P + H, P:P + W]
    return (b, c) if dev else (X.to_host(b), X.to_host(c))


def _exposure_masks(rgb, hi=0.90, lo=0.08, flat=0.030, radius=6.0, soft=1.5,
                    slack=0.12, grow=2.0, reach=6.0):
    """Returns (blown, crushed) — soft masks 0…1, float32 (H, W).

    Not seamless by itself; call exposure_masks, which wrap-pads the image.

    rgb    — float (H, W, 3), clipped to 0…1 before use; luminance is Rec. 709.
    hi/lo  — brightness level (0…1) from which an area counts as damaged
             (blown at L ≥ hi, crushed at L ≤ lo), like a threshold drawn
             with Curves: moving it selects more or less, never jumps;
    flat   — how much texture a pixel may still have around it and count
             as damaged: the local standard deviation among neighbours of
             about the same brightness (_band_contrast) must stay below
             flat/2. It does not depend on hi/lo, so every slider acts
             monotonically: raising lo or flat only adds to the mask.
             Larger = also textured dark/bright areas are taken; from about
             0.08 up only the brightness decides;
    radius — Gaussian sigma of the texture window, px;
    soft   — feather of the final masks: Gaussian sigma in px, then ×1.3 and
             clipped (0 = hard edges);
    slack  — the soft rim: from the selected spots the mask grows over
             pixels up to slack past the level (hi − slack / lo + slack) ...
    reach  — ... but no further than reach px, counted along those pixels
             (not in a straight line, so the rim follows the spot's own
             shape). Without the rim the slope of the spot stays outside
             and the repair takes its sample from it;
    grow   — margin in pixels (whole iterations). Positive expands the mask.
             NEGATIVE removes from it everything narrower than about
             2·|grow| px and leaves the rest as it was: the dark joints
             between stones are thin lines, a broken-out spot is wide, so
             −4…−8 keeps the joints and repairs the spots.
    In the module of rgb (card or processor).
    """
    xp, nd = X.xp(rgb), X.ndi(rgb)
    L = luminance(xp.clip(rgb, 0, 1))
    tex = _band_contrast(L, (L <= lo) | (L >= hi), radius)

    def pick(cand, loose):
        """Pixels of cand with little texture of their own, plus the rim."""
        m = cand & (tex < flat / 2.0)
        it = int(round(reach))
        if it > 0 and bool(m.any()):
            m = nd.binary_dilation(m, iterations=it, mask=loose | m, brute_force=True)
        return m

    blown = pick(L >= hi, L >= max(hi - slack, 0.02))
    crushed = pick(L <= lo, L <= min(lo + slack, 0.98))
    it = int(round(abs(grow)))
    if it:
        if grow > 0:
            blown = nd.binary_dilation(blown, iterations=it, brute_force=True)
            crushed = nd.binary_dilation(crushed, iterations=it, brute_force=True)
        else:
            blown = nd.binary_opening(blown, iterations=it, brute_force=True)
            crushed = nd.binary_opening(crushed, iterations=it, brute_force=True)
    b = blown.astype(xp.float32)
    c2 = crushed.astype(xp.float32)
    if soft > 0:
        b = xp.clip(_blur(b, soft) * 1.3, 0, 1)
        c2 = xp.clip(_blur(c2, soft) * 1.3, 0, 1)
    return b.astype(xp.float32), c2.astype(xp.float32)


def clean_mask(m, min_area=24, grow=0.0, thr=0.5):
    """Remove isolated pixels and, if needed, expand the mask.

    m: float (H, W) mask; thr: binarisation threshold (default 0.5);
    min_area: connected pieces smaller than this many px are dropped
    (≤ 1 = keep all); grow: expansion radius in px, done as a Gaussian blur
    with sigma grow re-thresholded at 0.35 (0 = none). Returns bool (H, W).
    Blur wraps round the edges, but connectivity for min_area does not, so a
    piece crossing the tile edge is measured as two pieces.
    """
    b = m > thr
    if min_area > 1:
        lab, n = ndi.label(b)
        if n:
            area = np.bincount(lab.ravel())
            keep = area >= min_area
            keep[0] = False
            b = keep[lab]
    if grow > 0:
        b = _blur(b.astype(np.float32), grow) > 0.35
    return b


def channel_mask(rgb, channel='luma', lo=0.0, hi=1.0, soft=0.02, alpha=None):
    """Mask from a single channel of the existing picture: a range of values
    with soft edges. This is the "mask from what is already there".

    rgb: float (H, W, 3), clipped to 0…1. channel: 'luma' (Rec. 709),
    'R' / 'G' / 'B', 'A' (uses alpha, float (H, W); all ones if None),
    'sat' (HSV saturation (max − min) / max), 'hue' (0…1: 0 red, 1/3 green,
    2/3 blue; grey pixels get 0); anything else falls back to 'luma'.
    lo / hi: the kept range of the channel value, 0…1; soft: width of the
    linear ramp at each end (min 1e-4). The hue range does not wrap past
    red, so lo > hi gives an empty mask. Returns float32 (H, W), 0…1.
    """
    rgb = np.clip(rgb, 0, 1)
    if channel == 'luma':
        x = luminance(rgb)
    elif channel in ('R', 'G', 'B'):
        x = rgb[..., 'RGB'.index(channel)]
    elif channel == 'A':
        x = np.ones(rgb.shape[:2], np.float32) if alpha is None else alpha
    elif channel == 'sat':
        mx, mn = rgb.max(-1), rgb.min(-1)
        x = (mx - mn) / np.maximum(mx, 1e-4)
    elif channel == 'hue':
        mx, mn = rgb.max(-1), rgb.min(-1)
        d = np.maximum(mx - mn, 1e-6)
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        h = np.where(mx == r, ((g - b) / d) % 6,
                     np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) / 6.0
        x = np.where(mx - mn < 1e-4, 0.0, h)
    else:
        x = luminance(rgb)
    s = max(soft, 1e-4)
    up = np.clip((x - lo) / s, 0, 1)
    dn = np.clip((hi - x) / s, 0, 1)
    return np.clip(up * dn, 0, 1).astype(np.float32)


def _edge_preserving(rgb, r=4, eps=0.02):
    """Smooth out grain but keep region boundaries sharp.

    Without this, a colour selection on a photographic texture falls apart
    into speckle: neighbouring pixels of one and the same brick differ from
    each other more than the brick differs from the joint.

    This is the guided filter (He, Sun & Tang, "Guided Image Filtering",
    2010) with the luminance as the guide: per window each channel is fitted
    as a·g + b, and the averaged a, b are applied. rgb: float (H, W, C);
    r: window radius in px (box of 2r + 1); eps: regulariser in luminance²
    units — larger smooths across weaker edges. Box filters wrap round the
    edges (seamless). Returns float (H, W, C).
    """
    g = luminance(rgb)
    mg = ndi.uniform_filter(g, size=2 * r + 1, mode='wrap')
    vg = ndi.uniform_filter(g * g, size=2 * r + 1, mode='wrap') - mg * mg
    out = np.empty_like(rgb)
    for c in range(rgb.shape[2]):
        p = rgb[..., c]
        mp = ndi.uniform_filter(p, size=2 * r + 1, mode='wrap')
        cov = ndi.uniform_filter(g * p, size=2 * r + 1, mode='wrap') - mg * mp
        a = cov / (vg + eps)
        b = mp - a * mg
        out[..., c] = (ndi.uniform_filter(a, size=2 * r + 1, mode='wrap') * g
                       + ndi.uniform_filter(b, size=2 * r + 1, mode='wrap'))
    return out


def wand(rgb, seeds, tol=0.10, contiguous=True, feather=1.0, smooth=4, clean=24):
    """Turn a brush stroke into a region: take the mean colour of the stroke
    and add everything that looks like it.

    rgb: float (H, W, 3), 0…1. seeds: bool (H, W), the stroke (None or empty
    returns an all-zero mask). tol: RMS colour distance per channel (0…1
    units) that still counts as similar, typical 0.05…0.25. feather: Gaussian
    sigma in px of the result edge (0 = hard). smooth: radius in px of the
    edge-preserving pre-filter (0 = off). clean: pieces smaller than this
    many px are dropped after a morphological closing and opening
    (0 = no clean-up). Returns float32 (H, W), 0…1; the stroke itself is
    always included.

    contiguous — only what is connected to the stroke, so the selection
    stops at contours by itself. Without it all similar places across the
    whole texture are found at once — for example all the joints of the
    masonry with a single stroke. Connectivity does not wrap round the tile
    edge.

    Before the colour comparison the picture is smoothed with an
    edge-preserving filter (_edge_preserving), and the result is cleaned up
    with morphology: otherwise on a photograph you get not a region but a
    speckle of separate pixels.
    """
    if seeds is None or not seeds.any():
        return np.zeros(rgb.shape[:2], np.float32)
    img = _edge_preserving(rgb, int(max(smooth, 1))) if smooth else rgb
    ref = img[seeds].reshape(-1, img.shape[2]).mean(0)
    d = np.sqrt(((img - ref) ** 2).sum(-1) / img.shape[2])
    near = d <= max(tol, 1e-3)
    near |= seeds
    if clean > 0:
        near = ndi.binary_closing(near, iterations=2)
        near = ndi.binary_opening(near, iterations=1) | seeds
        lab, n = ndi.label(near)
        if n:
            area = np.bincount(lab.ravel())
            keep = area >= clean
            keep[0] = False
            near = keep[lab] | seeds
    if contiguous:
        lab, n = ndi.label(near)
        keep = np.zeros(n + 1, bool)
        keep[np.unique(lab[seeds])] = True
        keep[0] = False
        near = keep[lab]
    out = near.astype(np.float32)
    return _blur(out, feather) if feather > 0 else out


# --------------------------------------------------------- восстановление --
def _masked_blur(x, m, r):
    """Normalised convolution: blur(x·m) / blur(m), sigma r px, wrapping; (H, W)."""
    num = _blur(x * m, r)
    den = np.maximum(_blur(m, r), 1e-4)
    return num / den


def tone_fix(rgb, mask, radius=40.0, strength=1.0, keep_grain=1.0):
    """The gentlest method: the area is not invented anew but brought to the
    level of its neighbours. Whatever is left of the texture stays in place.

    The gain is (Gaussian mean of the luminance of good pixels, sigma radius)
    / (Gaussian mean of all luminance, sigma radius/2), clipped to 0.2…5 and
    raised to strength, and multiplies the colour inside the mask.

    rgb: float (H, W, 3), 0…1 (used as given, not linearised). mask: float
    (H, W), 0…1, also the blend weight. radius: px, the neighbourhood the
    level is taken from (typical 20…80). strength: 0…1+, exponent of the gain
    (0 = no change). keep_grain: 0…1; below 1 the repaired area is blended
    towards a 2 px blur of itself, removing grain/noise left in it.
    Returns float (H, W, 3), 0…1.
    """
    m = np.clip(mask, 0, 1)[..., None]
    lin = np.clip(rgb, 1e-4, 1.0)
    L = luminance(lin)
    good = (1.0 - np.clip(mask, 0, 1)).astype(np.float32)
    around = _masked_blur(L, good, radius)
    inside = np.maximum(_blur(L, radius * 0.5), 1e-4)
    k = np.clip(around / inside, 0.2, 5.0) ** strength
    out = lin * k[..., None]
    if keep_grain < 1.0:
        out = out * (1 - (1 - keep_grain) * m) + _blur(out, 2.0) * ((1 - keep_grain) * m)
    return np.clip(rgb * (1 - m) + out * m, 0, 1)


def smooth_fill(rgb, mask, max_side=512):
    """Smooth fill: biharmonic inpainting (skimage inpaint_biharmonic) — the
    smoothest surface that meets the edges of the hole. Ideal for gradients
    and flat surfaces, does not restore texture.

    rgb: float (H, W, 3), 0…1; mask: float (H, W), > 0.5 is filled.
    max_side: for speed, images larger than this (px) are solved on every
    k-th pixel (k = max(H, W) // max_side) and scaled back up bilinearly
    (wrapping round the tile), so the fill stays smooth. The
    result is blended in with a 1.2 px feather. Falls back to tone_fix if
    scikit-image is not available. The solve does not wrap round the edges;
    repair() handles patches that cross the tile edge. Returns float
    (H, W, 3), 0…1 (rgb itself if the mask is empty).
    """
    try:
        from skimage.restoration import inpaint_biharmonic
    except Exception:
        return tone_fix(rgb, mask)
    b = mask > 0.5
    if not b.any():
        return rgb
    h, w = rgb.shape[:2]
    k = max(1, int(max(h, w) / max_side))
    if k > 1:
        small = rgb[::k, ::k]
        sm = b[::k, ::k]
        r = inpaint_biharmonic(small, sm, channel_axis=-1)
        r = np.asarray(r, np.float32)
        # обратно — плавно, а не повтором пикселей: иначе заплатка на большой
        # текстуре собиралась из квадратиков k×k
        r = ndi.zoom(r, (h / r.shape[0], w / r.shape[1], 1), order=1,
                     mode='grid-wrap')[:h, :w]
        if r.shape[:2] != (h, w):
            pad = ((0, h - r.shape[0]), (0, w - r.shape[1]), (0, 0))
            r = np.pad(r, pad, mode='edge')
        out = r
    else:
        out = np.asarray(inpaint_biharmonic(rgb, b, channel_axis=-1), np.float32)
    a = np.maximum(b, np.clip(_blur(b.astype(np.float32), 1.2), 0, 1))[..., None]
    return np.clip(rgb * (1 - a) + out * a, 0, 1)


def _components(mask, limit=24, min_area=48):
    """Split the mask into holes to repair one by one, each with its own donor.

    Small stuff is not worth it: the donor search for a three-pixel dot
    takes as long as for a spot half the texture in size, and the difference
    is not visible. mask: float (H, W), > 0.5 counts; limit: at most this
    many holes, largest first; min_area: holes smaller than this many px are
    skipped (patch_fill leaves them unrepaired). Returns a list of bool
    (H, W) masks.
    """
    lab, n = ndi.label(mask > 0.5)
    if n == 0:
        return []
    area = np.bincount(lab.ravel())
    area[0] = 0
    order = [i for i in np.argsort(area)[::-1][:limit] if area[i] >= min_area]
    return [(lab == i) for i in order]


def _spectra(L, valid):
    """The FFTs of _best_offset that do not depend on the hole: those of L,
    L² and the valid mask — computed once for all holes, on the card when
    it is in use."""
    Ld, Vd = _gpu(L.astype(np.float32)), _gpu(valid.astype(np.float32))
    xp = X.xp(Ld)
    return {'L': Ld, 'FL': xp.fft.rfft2(Ld), 'FL2': xp.fft.rfft2(Ld * Ld),
            'FV': xp.fft.rfft2(Vd)}


def _best_offset(L, valid, band, cover, forbid, wrap=True, spec=None):
    """Exact brute-force search over ALL offsets at once for the donor place.

    The quality of an offset o is the sum of squared differences between the
    ring around the hole and what lies on the same ring shifted by o:
    err(o) = Σ R·(L(p) − L(p+o))² = Σ R·L² − 2·Σ R·L·L(p+o) + Σ R·L(p+o)².
    Written out, the sum falls apart into three cross-correlations, and a
    correlation is computed in full through the fast Fourier transform.
    Trying one offset at a time would cost seconds, here it is milliseconds,
    and all offsets are tried, not a sample.

    Even so, this is the slowest part of the program: every hole costs six
    forward and three inverse FFTs of the whole texture, and patch_fill calls
    it once per hole (up to 24).

    L: float32 (H, W) luminance (matching uses luminance only, not colour).
    valid: bool (H, W), undamaged pixels. band: bool, the ring around the
    hole. cover: bool, ring AND hole — the donor has to be intact over all of
    it (≥ 98 % valid pixels). forbid: bool (H, W) over offsets, offsets
    excluded outright. The correlations are circular, so offsets wrap round
    the tile; the wrap argument is currently unused.

    spec — _spectra(L, valid), shared by all holes of one repair (then the
    search runs where the spectra are: on the card when it is in use).

    Returns (dy, dx) in −H/2…H/2, −W/2…W/2, or None if the ring has fewer
    than 12 px or no offset is admissible.
    """
    if int(band.sum()) < 12:
        return None
    if spec is None:
        spec = _spectra(L, valid)
    xp = X.xp(spec['L'])
    Ld = spec['L']
    R = xp.asarray(band).astype(xp.float32)
    K = xp.asarray(cover).astype(xp.float32)   # кольцо И сама дырка: донор должен быть цел весь
    FR = xp.conj(xp.fft.rfft2(R))
    FK = xp.conj(xp.fft.rfft2(K))
    FRL = xp.conj(xp.fft.rfft2(R * Ld))
    sh = Ld.shape
    c1 = xp.fft.irfft2(spec['FL'] * FRL, sh)          # sum R*L*L(+o)
    c2 = xp.fft.irfft2(spec['FL2'] * FR, sh)          # sum R*L(+o)^2
    cv = xp.fft.irfft2(spec['FV'] * FK, sh)           # донор должен быть целым и под дыркой тоже
    const = float((R * Ld * Ld).sum())
    err = const - 2.0 * c1 + c2
    err = xp.where(cv > 0.98 * float(K.sum()), err, np.inf)
    err[xp.asarray(forbid)] = np.inf
    if not bool(xp.isfinite(err).any()):
        return None
    i = int(xp.argmin(err))
    oy, ox = divmod(i, sh[1])
    if oy > sh[0] // 2:
        oy -= sh[0]
    if ox > sh[1] // 2:
        ox -= sh[1]
    return oy, ox


def _boundary_fit(a, b, ys, xs, by, bx):
    """How much darker or lighter the donor is than the place it is put into.

    Computed over the ring and continued inwards as a tilted plane: a
    constant difference plus two slopes, least-squares fitted to a − b on the
    ring pixels. Diffusion (a true Poisson blend) would give the same, but on
    a wide hole it goes astray, whereas the plane is always stable.

    a: (H, W) target channel; b: (H, W) the same channel shifted so that
    b[p] is the donor value for p; ys, xs: hole pixel coordinates; by, bx:
    ring pixel coordinates. Returns float (len(ys),), the correction to add
    to the donor at each hole pixel.
    """
    n = len(by)
    A = np.stack([np.ones(n, np.float32), bx.astype(np.float32), by.astype(np.float32)], 1)
    d = a[by, bx] - b[by, bx]
    coef, *_ = np.linalg.lstsq(A, d, rcond=None)
    B = np.stack([np.ones(len(ys), np.float32), xs.astype(np.float32), ys.astype(np.float32)], 1)
    return B @ coef


def patch_fill(rgb, mask, ring=8, feather=1.5, wrap=True):
    """Patch transfer. For every hole the whole texture is searched for the
    place whose surroundings best continue the surroundings of the hole, and
    the content is carried over from there. The brightness difference at the
    seam is removed by fitting at the edge (_boundary_fit), so the seam does
    not show.

    One offset per hole instead of full PatchMatch (Barnes et al., 2009). On
    tiles and on uniform texture — and game textures almost always are such —
    the result is the same, and it computes an order of magnitude faster.
    Still the slowest repair: _best_offset runs once per hole.

    rgb: float (H, W, 3), 0…1; mask: float (H, W), > 0.5 is repaired.
    ring: width in px of the ring around the hole used for matching (min 2).
    feather: Gaussian sigma in px of the final blend (×1.4, clipped).
    wrap: currently unused — the search and the donor lookup always wrap
    round the tile. Offsets smaller than the hole's bounding box in both
    axes are forbidden, so the donor never overlaps the hole itself. The 24
    largest holes of at least 48 px get a donor patch (see _components); every
    other masked pixel is filled smoothly (smooth_fill on a window around each
    small hole, at full resolution). Returns float (H, W, 3), 0…1.
    """
    H, W = rgb.shape[:2]
    out = rgb.astype(np.float32).copy()
    L = luminance(rgb).astype(np.float32)
    valid = mask <= 0.5
    done = np.zeros(mask.shape, bool)
    spec = None
    for comp in _components(mask):
        band = ndi.binary_dilation(comp, iterations=int(max(ring, 2))) & ~comp & valid
        ys, xs = np.nonzero(comp)
        ch = ys.max() - ys.min() + 1
        cw = xs.max() - xs.min() + 1
        # запрещаем сдвиги, при которых донор накрывает саму дырку
        oy = np.minimum(np.arange(H), H - np.arange(H))[:, None]
        ox = np.minimum(np.arange(W), W - np.arange(W))[None, :]
        forbid = (oy < ch) & (ox < cw)
        if spec is None:
            spec = _spectra(L, valid)        # общее для всех дырок — один раз
        off = _best_offset(L, valid, band, band | comp, forbid, wrap, spec)
        if off is None:
            continue
        dy, dx = off
        sy, sx = (ys + dy) % H, (xs + dx) % W
        by, bx = np.nonzero(band)
        donor = rgb[sy, sx].astype(np.float32).copy()
        for c in range(rgb.shape[2]):
            donor[:, c] += _boundary_fit(rgb[..., c], np.roll(np.roll(rgb[..., c], -dy, 0), -dx, 1),
                                         ys, xs, by, bx)
        out[ys, xs] = donor
        done |= comp
    # мелкие дырки (и всё сверх 24 крупных) заплаткой не ищутся — раньше они
    # так и оставались битыми. Их заливаем гладко, каждую в своём окне на
    # полном разрешении
    rest = (mask > 0.5) & ~done
    if rest.any():
        lab, _ = ndi.label(rest)
        for sl in ndi.find_objects(lab):
            y0, y1 = max(sl[0].start - 16, 0), min(sl[0].stop + 16, H)
            x0, x1 = max(sl[1].start - 16, 0), min(sl[1].stop + 16, W)
            out[y0:y1, x0:x1] = smooth_fill(out[y0:y1, x0:x1],
                                            rest[y0:y1, x0:x1].astype(np.float32))
    # сама дырка заменяется целиком, растушёвка идёт только наружу: у мелкого
    # пятна размытая маска не дотягивала до 1, и оно оставалось полусветлым
    hole = (mask > 0.5).astype(np.float32)
    a = np.maximum(hole, np.clip(_blur(hole, feather) * 1.4, 0, 1))[..., None]
    return np.clip(rgb * (1 - a) + out * a, 0, 1)


# ---------------------------------------------- заливка без чёрного -----
def fill_source(rgb, mask, dark=0.08):
    """Pixels a fill may take its colours from: intact AND not black.

    rgb: float (H, W, 3), 0…1; mask: float (H, W), > 0.5 is damaged; dark:
    luminance at or below which a pixel counts as black and is not used
    (the repair node passes its shadow threshold). Falls back to all intact
    pixels, then to all pixels, if nothing is left. Returns bool (H, W).
    """
    ok = mask <= 0.5
    keep = ok & (luminance(np.clip(rgb, 0, 1)) > dark)
    if keep.sum() >= 16:
        return keep
    return ok if ok.sum() >= 16 else np.ones(mask.shape, bool)


def _blend_hole(rgb, mask, out, feather=1.5):
    """Put ``out`` into the hole: replaced fully inside, feathered outward."""
    hole = (mask > 0.5).astype(np.float32)
    a = np.maximum(hole, np.clip(_blur(hole, feather) * 1.4, 0, 1))[..., None]
    return np.clip(rgb * (1 - a) + out * a, 0, 1).astype(np.float32)


def mean_fill(rgb, mask, keep):
    """Fill the hole with one colour: the mean of the source pixels.

    rgb: float (H, W, 3), 0…1; mask: float (H, W), > 0.5 is filled; keep:
    bool (H, W) from fill_source. Returns float32 (H, W, 3), 0…1.
    """
    c = rgb[keep].reshape(-1, rgb.shape[2]).mean(0)
    out = np.broadcast_to(c.astype(np.float32), rgb.shape)
    return _blend_hole(rgb, mask, out)


def _push_pull(cw, w):
    """Fill the gaps of a weighted image by a pyramid (push-pull).

    cw: float (H, W, C), colour already multiplied by the weight; w: float
    (H, W), weight 0…1 (0 = unknown). The image is halved again and again
    (2×2 averages of cw and w) until nothing is unknown, then brought back
    level by level: where a level has its own data it keeps it, where it has
    none it takes the smooth enlargement of the coarser level. The result is
    a smooth blend of the known colours across the gaps, without the rings a
    fixed blur radius leaves. Wraps round the tile. Returns float32 (H, W, C).
    """
    H, W = w.shape
    if float(w.min()) > 0 or min(H, W) <= 2:
        return (cw / np.maximum(w, 1e-6)[..., None]).astype(np.float32)
    ph, pw = H % 2, W % 2
    cw2 = np.pad(cw, ((0, ph), (0, pw), (0, 0)), mode='wrap')
    w2 = np.pad(w, ((0, ph), (0, pw)), mode='wrap')
    cw2 = 0.25 * (cw2[0::2, 0::2] + cw2[1::2, 0::2] + cw2[0::2, 1::2] + cw2[1::2, 1::2])
    w2 = 0.25 * (w2[0::2, 0::2] + w2[1::2, 0::2] + w2[0::2, 1::2] + w2[1::2, 1::2])
    c2 = _push_pull(cw2, w2)
    up = ndi.zoom(c2, ((H + ph) / c2.shape[0], (W + pw) / c2.shape[1], 1), order=1,
                  mode='grid-wrap')[:H, :W]
    own = cw / np.maximum(w, 1e-6)[..., None]
    k = np.clip(w * 4.0, 0.0, 1.0)[..., None]
    return (own * k + up * (1.0 - k)).astype(np.float32)


def gradient_fill(rgb, mask, keep):
    """Fill the hole with a smooth gradient of the surrounding colours.

    The colours of the source pixels (keep) are spread into the hole by a
    pyramid (_push_pull): a small hole gets the colour of its edge, a big one
    a soft blend of the colours around it. Black and damaged pixels do not
    take part, so the fill does not darken. Wraps round the tile. Returns
    float32 (H, W, 3), 0…1.
    """
    w = keep.astype(np.float32)
    field = _push_pull(rgb.astype(np.float32) * w[..., None], w)
    # вне дырки — сама картинка: растушёвка края смешивает заливку с ней
    out = rgb.astype(np.float32).copy()
    need = mask > 0.5
    out[need] = field[need]
    return _blend_hole(rgb, mask, out)


def _palette(px, k=4, iters=12, seed=0):
    """Main colours of the pixels px (N, 3): plain k-means.

    Returns (colours (k', 3), share of pixels of each (k',)), sorted from
    dark to light; k' ≤ k (empty clusters are dropped).
    """
    rng = np.random.default_rng(seed)
    if len(px) > 20000:
        px = px[rng.choice(len(px), 20000, replace=False)]
    k = max(1, min(k, len(px)))
    # начальные центры — по квантилям яркости: без случайных «пустых» центров
    lum = luminance(px)
    order = np.argsort(lum)
    c = px[order[((np.arange(k) + 0.5) / k * len(px)).astype(int)]].copy()
    for _ in range(iters):
        d = ((px[:, None, :] - c[None, :, :]) ** 2).sum(-1)
        lab = d.argmin(1)
        for j in range(k):
            if (lab == j).any():
                c[j] = px[lab == j].mean(0)
    d = ((px[:, None, :] - c[None, :, :]) ** 2).sum(-1)
    lab = d.argmin(1)
    share = np.bincount(lab, minlength=k).astype(np.float64)
    used = share > 0
    c, share = c[used], share[used] / share[used].sum()
    o = np.argsort(luminance(c))
    return c[o].astype(np.float32), share[o]


def spots_fill(rgb, mask, keep, k=4, scale=None, seed=0):
    """Fill the hole with spots of the texture's main colours (not black).

    The main colours of the source pixels are found (_palette); a smooth
    random field cut at its quantiles splits the hole into spots, each colour
    taking the same share of the hole as it has in the texture, neighbouring
    spots in order of brightness so the edges stay soft. A fine grain as
    strong as the texture's own is added on top. scale: spot size in px
    (Gaussian sigma of the field), default min(H, W) / 40. Wraps round the
    tile; the result is the same for the same seed.

    The spot field is drawn on a fixed grid (256 cells along the shorter
    side) and brought to the picture's size, so the spots do not depend on
    the resolution: the preview copy, its refined pass and the saved full
    size get the same spots (drawn per pixel, every resolution had its own,
    and the picture changed at each recompute). Returns float32 (H, W, 3).
    """
    from texio import resize_wrap
    H, W = mask.shape
    need = mask > 0.5
    pal, share = _palette(rgb[keep].reshape(-1, rgb.shape[2]), k, seed=seed)
    rng = np.random.default_rng(seed)
    s = float(scale) if scale else max(3.0, min(H, W) / 40.0)
    m = float(min(H, W))
    gh, gw = max(8, int(round(256 * H / m))), max(8, int(round(256 * W / m)))
    sg = s * 256.0 / m                       # размер пятна в клетках сетки
    fg = _blur(rng.standard_normal((gh, gw)).astype(np.float32), sg) \
        + 0.5 * _blur(rng.standard_normal((gh, gw)).astype(np.float32), sg / 2.5)
    f = resize_wrap(fg, H, W).astype(np.float32)
    v = f[need]
    cuts = np.quantile(v, np.cumsum(share)[:-1]) if len(pal) > 1 else []
    idx = np.searchsorted(cuts, f) if len(pal) > 1 else np.zeros((H, W), int)
    out = pal[np.clip(idx, 0, len(pal) - 1)]
    out = np.stack([_blur(out[..., ch], 1.5) for ch in range(out.shape[2])], -1)
    # зерно: столько же мелкой ряби, сколько у самой текстуры
    L = luminance(np.clip(rgb, 0, 1))
    g = float(np.std((L - _blur(L, 2.0))[keep]))
    grain = _blur(rng.standard_normal((H, W)).astype(np.float32), 0.7)
    grain *= g / max(float(np.std(grain)), 1e-6)
    out = out + grain[..., None]
    return _blend_hole(rgb, mask, np.clip(out, 0, 1))


FILLS = ('mean', 'gradient', 'spots')
METHODS = ('tone', 'smooth', 'patch') + FILLS


def _clean_cut(mask):
    """The row and column with no damaged pixels (or the fewest).

    mask: float (H, W); pixels > 0.05 count as damaged. Returns (row, col).
    """
    m = mask > 0.05
    return int(np.argmin(m.sum(1))), int(np.argmin(m.sum(0)))


def repair(rgb, mask, method='patch', radius=40.0, strength=1.0, keep=None, seed=0):
    """Repair the damaged areas.

    The texture tiles, so a spot that crosses the edge is one spot, not two.
    The picture is rolled round so that the seam falls on a clean row and
    column, repaired and rolled back: otherwise the two halves of such a spot
    would be filled independently, and a seam would show at the tile joint.

    rgb: float (H, W, 3), 0…1. mask: float (H, W), 0…1, > 0.5 is repaired
    (e.g. from exposure_masks). method: one of METHODS — 'tone' (tone_fix),
    'smooth' (smooth_fill), 'mean' (mean_fill), 'gradient' (gradient_fill),
    'spots' (spots_fill) or 'patch' (patch_fill, also the fallback for an
    unknown name). radius (px) and strength apply to 'tone' only. keep:
    bool (H, W), the pixels the three fills may take colours from (default
    fill_source with its default black level). seed: the draw of the
    'spots' fill. Returns float (H, W, 3),
    0…1; rgb itself if nothing is masked.
    """
    if mask is None or not np.any(mask > 0.5):
        return rgb
    if method in FILLS:
        # эти заливки сами замыкаются через край, сдвиг шва им не нужен
        keep = fill_source(rgb, mask) if keep is None else keep
        if method == 'spots':
            return spots_fill(rgb, mask, keep, seed=int(seed))
        fn = {'mean': mean_fill, 'gradient': gradient_fill}[method]
        return fn(rgb, mask, keep)
    y, x = _clean_cut(mask)
    if y or x:
        out = _repair(np.roll(rgb, (-y, -x), (0, 1)), np.roll(mask, (-y, -x), (0, 1)),
                      method, radius, strength)
        return np.roll(out, (y, x), (0, 1))
    return _repair(rgb, mask, method, radius, strength)


def _repair(rgb, mask, method='patch', radius=40.0, strength=1.0):
    """Dispatch to tone_fix / smooth_fill / patch_fill without the seam roll."""
    if method == 'tone':
        return tone_fix(rgb, mask, radius, strength)
    if method == 'smooth':
        return smooth_fill(rgb, mask)
    return patch_fill(rgb, mask)
