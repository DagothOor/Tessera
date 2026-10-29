# -*- coding: utf-8 -*-
"""
Texelith — 3D preview with parallax.
Maps shown on a tilted plane with parallax.

Why this is needed. A height map and a normal map lie in different ways, and
lie unnoticeably as long as you look at the texture head-on: a normal seen
straight on looks convincing almost always. Parallax works exactly the other
way round — head-on it gives nothing, the whole effect appears at a grazing
view, and that is also where all its diseases come out: splitting into
stair-steps, swimming edges, too much depth that makes a brick slide away
from its own joint.

So here a real plane is drawn in perspective, which can be tilted, and the
same thing the engine computes is honestly computed on it:

  * single-step offset (offset)       — as in the oldest implementations: the
    texture is shifted by the depth under the point. Cheap, but swims at a
    slant;
  * with occlusion (occlusion, POM)    — the ray goes into the surface in steps
    and looks for where it pierces the relief. This is what modern engines
    do, including OpenMW;
  * plus self-shadowing: from the found point a ray goes towards the light,
    and if the relief along the way is higher, the point is in shadow.

A fourth mode, 'openmw', reproduces OpenMW's own parallax exactly
(files/shaders/lib/material/parallax.glsl): a single step
offset = eyeDir.xy · (a · 0.04 − 0.02), without division by z, where a is the
alpha of the _nh texture with the neutral at 128 (a ≈ 0.5 gives no shift).
The depth slider does not apply there: the depth follows from the alpha and
the engine's coefficients.

Texture sampling imitates a GPU: a mip pyramid, the level chosen per pixel
from the screen-space footprint with anisotropy capped at 16, and bilinear
filtering inside the level. The ray march reads the height from the same mip
level as the colour, so far away it walks over the smoothed relief.

Nothing depends on OpenGL: everything is computed with numpy, so it works
wherever the program itself works.

Texelith — 3D preview with parallax.
Просмотр карт на наклонной плоскости с параллаксом.

Зачем это нужно. Карта высоты и карта нормалей врут по-разному, и врут
незаметно, пока смотришь на текстуру в лоб: нормаль в упор выглядит
убедительно почти всегда. Параллакс же работает ровно наоборот — в лоб он
не даёт ничего, а весь эффект появляется на скользящем взгляде, и там же
вылезают все его болезни: расслоение на ступеньки, поплывшие края, слишком
большая глубина, из-за которой кирпич уезжает от собственного шва.

Поэтому здесь рисуется настоящая плоскость в перспективе, которую можно
наклонить, и по ней честно считается то же самое, что считает движок:

  * смещение по одному шагу (offset)  — как в самых старых реализациях:
    текстура сдвигается на глубину под точкой. Дёшево, но на скосе плывёт;
  * с перекрытием (occlusion, POM)    — луч идёт внутрь поверхности шагами
    и ищет, где он протыкает рельеф. Это то, что делают современные движки,
    включая OpenMW;
  * плюс собственная тень: из найденной точки луч идёт к источнику света,
    и если по дороге рельеф выше — точка в тени.

Ничего не зависит от OpenGL: всё считается numpy, поэтому работает везде,
где работает сама программа.
"""

import numpy as np

import xpu as X

BUILD = 25

MODES = ('flat', 'offset', 'occlusion', 'openmw')

# Параллакс OpenMW, files/shaders/lib/material/parallax.glsl:
#     offset = eyeDir.xy * (height * 0.04 - 0.02)
# один шаг, без деления на z, высота — альфа _nh, нейтраль 128
OPENMW_SCALE, OPENMW_BIAS = 0.04, -0.02


def _rot(yaw, pitch):
    """Axes of the plane: right, down and out (towards the camera).

    The plane is rotated by ``pitch`` about the camera's x axis, then by ``yaw``
    about its y axis (R = Ry · Rx), both in radians. The camera looks along +z;
    with zero angles the plane faces it.

    Returns:
        (T, B, N): float32 unit 3-vectors in camera space — T along +u (right),
        B along +v (down the texture), N the outward normal (towards the camera).
    """
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], np.float32)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], np.float32)
    R = Ry @ Rx
    T = R @ np.array([1, 0, 0], np.float32)
    B = R @ np.array([0, 1, 0], np.float32)
    N = R @ np.array([0, 0, -1], np.float32)
    return T, B, N


def _samp(arr, u, v, flat=None):
    """Point sampling with wrap-around.

    Nearest neighbour: when marching into the relief, bilinear filtering costs
    four times as much and gives no difference in the found intersection point.

    Args:
        arr: H×W or H×W×C array.
        u, v: 1-D arrays of texture coordinates, 1.0 = one repeat (any value;
            wrapped modulo the size).
        flat: optional precomputed ``arr.reshape(H * W, -1)``.

    Returns:
        N×C array (C = 1 for a 2-D ``arr``).

    Not used by render() at the moment: the ray march samples the height
    bilinearly from the point's mip level instead (see render.hs).
    """
    xp = X.xp(arr, u, v, flat)
    H, W = arr.shape[:2]
    xi = xp.mod((u * W).astype(xp.int64), W)
    yi = xp.mod((v * H).astype(xp.int64), H)
    idx = yi * W + xi
    f = flat if flat is not None else arr.reshape(H * W, -1)
    return f[idx]


# ------------------------------------------------- мипы и фильтрация -------
# Видеокарта берёт текстуру не с полного размера, а с того уровня мипов, где
# один тексель примерно равен пикселю экрана, и сглаживает между соседями.
# Без этого вдали картинка рябит, а вблизи видны квадраты. Здесь то же:
# пирамида уменьшенных копий, уровень по размеру пикселя на поверхности,
# билинейная выборка внутри уровня. Анизотропия (скошенный взгляд)
# учитывается как в видеокартах: уровень по меньшей стороне следа пикселя,
# но не больше чем в 16 раз меньше большей.
_PYR = {}


def _pyramid(arr):
    """Build (once) and return the mip pyramid of ``arr``.

    Level 0 is ``arr`` as contiguous float32; each next level halves both sides
    with a 2×2 box average (an odd last row / column is dropped), until the
    shorter side reaches 1. Cached by id(arr); the entry keeps a reference to
    ``arr`` and is checked with ``is``, so a recycled id cannot hit a stale
    pyramid. The cache is cleared wholesale when it holds more than 24 arrays.
    Modifying ``arr`` in place after the first call is not noticed.

    Returns:
        List of float32 arrays, finest first.
    """
    xp = X.xp(arr)
    k = id(arr)
    hit = _PYR.get(k)
    if hit is not None and hit[0] is arr:
        return hit[1]
    lv = [xp.ascontiguousarray(arr, xp.float32)]
    while min(lv[-1].shape[:2]) > 1:
        a = lv[-1]
        h2, w2 = max(1, a.shape[0] // 2), max(1, a.shape[1] // 2)
        a = a[:h2 * 2, :w2 * 2]
        lv.append(a.reshape((h2, 2, w2, 2) + a.shape[2:]).mean(axis=(1, 3)))
    if len(_PYR) > 24:
        _PYR.clear()
    _PYR[k] = (arr, lv)
    return lv


def _bilinear(a, u, v):
    """Bilinear sample with wrap-around from one pyramid level.

    Args:
        a: H×W or H×W×C level.
        u, v: 1-D texture coordinates, 1.0 = full width / height; texel centres
            at (i + 0.5) / W, as on a GPU. Neighbours are taken modulo the size,
            so filtering across the tile edge is seamless.

    Returns:
        N or N×C array (float32 for a float32 level).
    """
    xp = X.xp(a, u, v)
    H, W = a.shape[:2]
    x = u * W - 0.5
    y = v * H - 0.5
    x0 = xp.floor(x)
    y0 = xp.floor(y)
    fx = (x - x0).astype(xp.float32)
    fy = (y - y0).astype(xp.float32)
    x0 = xp.mod(x0.astype(xp.int64), W)
    y0 = xp.mod(y0.astype(xp.int64), H)
    x1 = xp.mod(x0 + 1, W)
    y1 = xp.mod(y0 + 1, H)
    if a.ndim == 3:
        fx, fy = fx[:, None], fy[:, None]
    return ((a[y0, x0] * (1 - fx) + a[y0, x1] * fx) * (1 - fy)
            + (a[y1, x0] * (1 - fx) + a[y1, x1] * fx) * fy)


def _sample(arr, u, v, lvl):
    """Sample from mip level ``lvl`` (an integer per point), bilinearly.

    Points are grouped by level; levels above the coarsest are clamped to it.
    There is no blending between levels (no trilinear filtering).

    Args:
        arr: H×W or H×W×C map (its pyramid is taken from _pyramid).
        u, v: 1-D texture coordinates, 1.0 = one repeat.
        lvl: 1-D int array, same length as u.

    Returns:
        float32 array of shape (N,) + arr.shape[2:].
    """
    xp = X.xp(arr, u, v, lvl)
    lv = _pyramid(arr)
    out = xp.empty((len(u),) + arr.shape[2:], xp.float32)
    top = len(lv) - 1
    for L in xp.unique(lvl):
        m = lvl == L
        out[m] = _bilinear(lv[min(int(L), top)], u[m], v[m])
    return out


def _levels(u, v, idx, w, h, tex_w, tex_h):
    """Mip level for each point: by how many texels fit into a screen pixel.

    Horizontally and vertically. The texture coordinates of the visible points
    are scattered back into the w×h screen, and their screen-space derivatives
    are taken by finite differences to the neighbouring pixel (see d). The pixel
    footprint along x and y is measured in texels of level 0; as in GPUs with
    anisotropic filtering, the level follows the smaller side of the footprint,
    but that side is taken as no less than 1/16 of the larger one (anisotropy
    capped at 16): level = floor(log2(footprint)), at least 0. Pixels without
    a usable neighbour get footprint 1 (level 0).

    Args:
        u, v: texture coordinates of the visible points (1.0 = one repeat).
        idx: their flat indices in the w×h screen.
        w, h: screen size in pixels.
        tex_w, tex_h: texture size at level 0 in pixels.

    Returns:
        int32 array of levels, aligned with u.
    """
    xp = X.xp(u, v, idx, w, h, tex_w, tex_h)
    U = xp.full(w * h, xp.nan, xp.float32)
    V = xp.full(w * h, xp.nan, xp.float32)
    U[idx], V[idx] = u, v
    U, V = U.reshape(h, w), V.reshape(h, w)

    def d(A, ax):
        """Absolute finite difference of A along axis ``ax``, same shape as A.

        Forward difference (0 at the last row / column), replaced by the backward one
        where the forward one is NaN, i.e. where the neighbour lies off the plane.
        """
        g = xp.abs(xp.diff(A, axis=ax, append=xp.take(A, [-1], axis=ax)))
        b = xp.abs(xp.diff(A, axis=ax, prepend=xp.take(A, [0], axis=ax)))
        return xp.where(xp.isnan(g), b, g)
    fx = xp.hypot(d(U, 1) * tex_w, d(V, 1) * tex_h)
    fy = xp.hypot(d(U, 0) * tex_w, d(V, 0) * tex_h)
    fx, fy = xp.nan_to_num(fx, nan=1.0), xp.nan_to_num(fy, nan=1.0)
    big, small = xp.maximum(fx, fy), xp.minimum(fx, fy)
    foot = xp.maximum(small, big / 16.0)
    lvl = xp.floor(xp.log2(xp.maximum(foot, 1.0))).astype(xp.int32)
    return lvl.reshape(-1)[idx]


def _shade(maps, u, v, Lt, Vt, amb=0.22, key=1.10, hemi=False, lvl=None, sss_pow=2.0):
    """Same lighting computation as in the flat preview, in the tilted plane's frame.

    The vectors are expressed in the coordinate system of the tilted plane
    (T, B, N from _rot), so the normal map can be used as is.

    Ambient light. hemi off — as in OpenMW: one value for the whole surface
    (amb · 0.9), independent of the normal, no AO map (occlusion is visible only
    if it is baked into the diffuse). hemi on — as in engines where AO and bent
    normals are applied: ambient over a hemisphere by the normal,
    amb · 1.6 · (0.45 + 0.55 · n_z), multiplied by AO; direct light is not touched
    by AO — that is how AO is meant to work. With a constant ambient a bent normal
    has no visible effect at all and looks broken; the hemispherical ambient is
    what makes it show.

    Direct light: Lambert, key · max(n·L, 0); specular: Blinn-Phong,
    (n·h)^(2 + 400 · shininess) times the specular colour.

    Args:
        maps: dict with 'normal' (H×W×3 vectors in −1…1, renormalised after
            sampling), 'diffuse' and 'spec' (H×W×3, 0…1), 'shininess' (H×W,
            0…1) and, for hemi, 'ao' (H×W, 0…1, 1 = open).
        u, v: 1-D texture coordinates (1.0 = one repeat).
        Lt, Vt: N×3 unit light and view directions in the plane's frame.
        amb: ambient level; key: direct light level.
        hemi: hemispherical ambient with AO, see above.
        lvl: mip level per point (default: all 0).

    Returns:
        N×3 float32 colours in 0…1.
    """
    xp = X.xp(maps, u, v, Lt, Vt, amb, key, hemi, lvl)
    if lvl is None:
        lvl = xp.zeros(len(u), xp.int32)
    n = _sample(maps['normal'], u, v, lvl).reshape(-1, 3)
    n = n / xp.maximum(xp.sqrt((n * n).sum(-1, keepdims=True)), 1e-6)
    dif = _sample(maps['diffuse'], u, v, lvl).reshape(-1, 3)
    sp = _sample(maps['spec'], u, v, lvl).reshape(-1, 3)
    sh = _sample(maps['shininess'], u, v, lvl).reshape(-1)
    dot = xp.clip((n * Lt).sum(-1), 0, 1)
    hv = Lt + Vt
    hv = hv / xp.maximum(xp.sqrt((hv * hv).sum(-1, keepdims=True)), 1e-6)
    ndh = xp.clip((n * hv).sum(-1), 0, 1)
    spec = (ndh ** xp.maximum(2.0 + 400.0 * sh, 1.0))[..., None] * sp
    if hemi:
        ao = _sample(maps['ao'], u, v, lvl).reshape(-1)
        sky = amb * 1.6 * (0.45 + 0.55 * xp.clip(n[..., 2], 0, 1)) * ao
    else:
        sky = xp.full(len(n), amb * 0.9, xp.float32)
    col = dif * (sky + key * dot)[..., None] + spec
    if maps.get('sss') is not None and maps.get('sss_a') is not None:
        # просвет: прошедший сквозь лист свет идёт вперёд, по ходу луча, —
        # виден, когда источник за листом, (V·−L)^узость (как foliage.backlit)
        back = xp.clip(-(Vt * Lt).sum(-1), 0.0, 1.0) ** max(float(sss_pow), 0.1)
        tr = _sample(maps['sss'], u, v, lvl).reshape(-1, 3)
        ta = _sample(maps['sss_a'], u, v, lvl).reshape(-1)
        col = col + tr * (ta * back)[..., None]
    return xp.clip(col, 0, 1)


def render(maps, w, h, yaw=0.35, pitch=0.9, dist=2.6, tiles=3.0,
           depth=0.06, steps=24, mode='occlusion', shadow=True,
           light=(-0.47, 0.67, 0.57), fov=42.0, bg=(0.055, 0.06, 0.065),
           grid=False, hemi=False, sss_pow=2.0):
    """One frame of the 3D view.

    Args:
        maps: dict of maps, as for _shade, plus 'height' (H×W, 0…1, 1 = top of
            the relief) and optionally 'nh_alpha' (H×W 0…1, the alpha that goes
            into _nh; mode 'openmw' falls back to 'height' without it) and
            'alpha' (H×W 0…1, the texture's transparency: below 0.5 the plane
            is cut away, as the alpha test of foliage in the game).
        w, h: frame size in pixels.
        yaw, pitch: plane rotation in radians (see _rot).
        dist: camera distance to the plane centre; the plane spans −1…1 along
            T and B, i.e. it is 2 units wide.
        tiles: how many texture repeats fit across the plane. A non-square
            texture keeps square texels: along the other axis a repeat is
            th/tw as long, so more of them fit.
        depth: depth of the relief as a fraction of one tile. 0.05 is 5 % of
            the tile width: for a brick wall that is about right.
        steps: how many steps the ray takes into the relief (at least 4 are
            used). Few steps — the relief splits into stair-steps; many — more
            expensive to compute. The number is fixed per call, it does not
            depend on the viewing angle. Self-shadowing uses max(4, steps // 2).
        mode: one of MODES. 'flat' — no parallax; 'offset' — a single step,
            (u, v) − V.xy / V.z · depth · (1 − height); 'occlusion' — POM: linear
            steps into the relief, then linear interpolation between the last
            two, so the relief is not seen in layers; 'openmw' — exactly what the
            engine computes: offset = eyeDir.xy · (a · 0.04 − 0.02), one step, no
            division by z, neutral alpha 128 (OpenMW
            files/shaders/lib/material/parallax.glsl); ``depth`` is not used
            there.
        shadow: self-shadowing (only in 'occlusion' mode with depth > 0): a march
            from the found point towards the light; the lit factor is clipped to
            0.25…1.
        light: light direction RELATIVE TO THE SURFACE (x along T, y along B,
            z out of it), as in the flat preview, normalised here — so the light
            does not drift when the plane is tilted.
        fov: vertical field of view in degrees.
        bg: background RGB 0…1.
        grid: draw the borders of the tiles.
        hemi: hemispherical ambient with AO (see _shade).
        sss_pow: narrowness of the translucency beam, if maps has 'sss'
            (H×W×3 colour) and 'sss_a' (H×W amount): light behind the leaf
            shows through it (see _shade).

    The height used by the ray march is sampled from the point's own mip level.
    The far part of the plane is faded towards the background so the perspective
    reads.

    Returns:
        h×w×3 float32 in 0…1.
    """
    xp = X.xp(maps, w, h, yaw, pitch, dist, tiles, depth, steps, mode, shadow, light, fov, bg, grid, hemi)
    T, B, N = (xp.asarray(a) for a in _rot(yaw, pitch))     # оси — туда же, где карты
    f = (0.5 * h) / xp.tan(xp.radians(fov) * 0.5)
    ys, xs = xp.mgrid[0:h, 0:w].astype(xp.float32)
    d = xp.stack([xs - w * 0.5, ys - h * 0.5, xp.full((h, w), f, xp.float32)], -1)
    d /= xp.maximum(xp.sqrt((d * d).sum(-1, keepdims=True)), 1e-6)
    d = d.reshape(-1, 3)

    P0 = xp.array([0.0, 0.0, float(dist)], xp.float32)
    den = d @ N
    vis = den < -1e-4                       # смотрим на лицевую сторону
    out = xp.tile(xp.array(bg, xp.float32), (h * w, 1))
    if not vis.any():
        return out.reshape(h, w, 3)

    dv = d[vis]
    tt = (P0 @ N) / (dv @ N)
    hit = dv * tt[:, None]
    loc = hit - P0
    u = (loc @ T) * 0.5 + 0.5               # 0…1 по плоскости
    v = (loc @ B) * 0.5 + 0.5
    inside = (tt > 0) & (u > 0) & (u < 1) & (v > 0) & (v < 1)
    if not inside.any():
        return out.reshape(h, w, 3)

    th, tw = maps['diffuse'].shape[:2]
    # неквадратная текстура: тексели квадратные, поэтому повтор по высоте
    # короче повтора по ширине в th/tw раз — иначе картинка растянута
    asp = tw / float(th)
    u, v = u[inside] * tiles, v[inside] * (tiles * asp)
    lvl = _levels(u, v, xp.nonzero(vis)[0][inside], w, h, tw, th)
    V = -dv[inside]
    Vt = xp.stack([V @ T, V @ B, V @ N], -1)
    Vt[:, 2] = xp.maximum(Vt[:, 2], 1e-3)
    # свет задаётся ОТНОСИТЕЛЬНО поверхности, как и в плоском превью: то же
    # кольцо, тот же смысл. Иначе при наклоне плоскости освещение уезжает
    # само собой и сравнивать два просмотра становится не с чем
    L = xp.asarray(light, xp.float32)
    L = L / max(float(xp.sqrt((L * L).sum())), 1e-6)
    Lt = xp.tile(L, (len(u), 1))

    hm = xp.ascontiguousarray(maps['height'], xp.float32)
    hflat = hm.reshape(-1, 1)

    def hs(uu, vv, sel=None):
        """Height at (uu, vv), sampled from each point's own mip level.

        The levels are ``lvl``, or ``lvl[sel]`` for a subset of the points.

        ``sel`` is the boolean mask of points being sampled when only some of them are
        still marching. Using the colour's mip level keeps the far relief smooth, as on
        a GPU, instead of breaking up into needles.
        """
        # высота с уровня мипов точки: вдали луч идёт по сглаженному
        # рельефу, как у видеокарты, и не рассыпается на иглы
        return _sample(hm, uu, vv, lvl if sel is None else lvl[sel])

    if mode == 'openmw':
        # ровно то, что посчитает движок, по той самой альфе, что уйдёт в
        # _nh. Глубина здесь не задаётся ползунком: она следует из альфы и
        # коэффициента движка. Направление — как у смещения выше: высокое
        # уезжает к зрителю
        am = xp.ascontiguousarray(maps.get('nh_alpha', maps['height']), xp.float32)
        a = _sample(am, u, v, lvl)
        s = a * OPENMW_SCALE + OPENMW_BIAS
        u, v = u + Vt[:, 0] * s, v + Vt[:, 1] * s
    elif mode != 'flat' and depth > 0:
        # шаг смещения по текстуре на всю глубину рельефа
        px = Vt[:, 0] / Vt[:, 2] * depth
        py = Vt[:, 1] / Vt[:, 2] * depth * asp     # та же длина в мире, в долях высоты
        if mode == 'offset':
            k = 1.0 - hs(u, v)
            u, v = u - px * k, v - py * k
        else:
            n_st = max(int(steps), 4)
            # шагов тем больше, чем косее взгляд: в лоб они не нужны вовсе
            lay = 1.0 / n_st
            cu, cv = u.copy(), v.copy()
            cd = xp.zeros(len(u), xp.float32)
            ch = 1.0 - hs(cu, cv)
            done = xp.zeros(len(u), bool)
            pu, pv, pd, ph = cu.copy(), cv.copy(), cd.copy(), ch.copy()
            for _ in range(n_st):
                go = ~done
                if not go.any():
                    break
                pu[go], pv[go], pd[go], ph[go] = cu[go], cv[go], cd[go], ch[go]
                cu[go] -= px[go] * lay
                cv[go] -= py[go] * lay
                cd[go] += lay
                ch[go] = 1.0 - hs(cu[go], cv[go], go)
                done |= cd >= ch
            # точка пересечения между последними двумя шагами: без этого
            # рельеф виден слоями, как горбушка на срезе
            a = ch - cd
            b = pd - ph
            wgt = xp.clip(a / xp.maximum(a + b, 1e-6), 0.0, 1.0)
            u = cu * (1.0 - wgt) + pu * wgt
            v = cv * (1.0 - wgt) + pv * wgt

    col = _shade(maps, u, v, Lt, Vt, hemi=hemi, lvl=lvl, sss_pow=sss_pow)

    if shadow and mode == 'occlusion' and depth > 0 and float(L[2]) > 0.02:
        # собственная тень: из найденной точки идём к свету и смотрим,
        # не перекрывает ли её рельеф выше по дороге
        lz = xp.maximum(Lt[:, 2], 1e-3)
        sx = Lt[:, 0] / lz * depth
        sy = Lt[:, 1] / lz * depth * asp
        h0 = hs(u, v)
        n_sh = max(4, int(steps) // 2)
        occ = xp.zeros(len(u), xp.float32)
        for i in range(1, n_sh + 1):
            k = i / n_sh
            hk = hs(u + sx * k, v + sy * k)
            occ = xp.maximum(occ, (hk - (h0 + k * (1.0 - h0))) * (1.0 - k))
        lit = xp.clip(1.0 - 6.0 * xp.clip(occ, 0, 1), 0.25, 1.0)
        col = col * lit[..., None]

    # плоскость уходит вдаль — дальний край чуть притушен, иначе перспектива
    # не читается и картинка выглядит наклеенной
    fade = xp.clip(1.35 - 0.32 * tt[inside] / max(float(dist), 1e-3), 0.35, 1.0)
    col = col * fade[..., None] + xp.array(bg, xp.float32) * (1.0 - fade)[..., None]

    if grid:
        eu, ev = xp.abs(xp.mod(u, 1.0) - 0.5), xp.abs(xp.mod(v, 1.0) - 0.5)
        edge = ((eu > 0.497) | (ev > 0.497)).astype(xp.float32)[..., None]
        col = col * (1 - edge * 0.5) + xp.array([0.35, 0.55, 0.8], xp.float32) * edge * 0.5

    idx = xp.nonzero(vis)[0][inside]
    if maps.get('alpha') is not None:
        # проверка альфы, как у листвы в игре: где альфа ниже половины —
        # пикселя нет, виден фон
        am = xp.ascontiguousarray(maps['alpha'], xp.float32)
        keep = _sample(am, u, v, lvl) >= 0.5
        idx, col = idx[keep], col[keep]
    out[idx] = col
    return xp.clip(out.reshape(h, w, 3), 0, 1)
