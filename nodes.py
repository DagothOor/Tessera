# -*- coding: utf-8 -*-
"""
Texelith — node graph.

Why it is needed when the tabs already compute everything. The tabs are one
hard-wired order: repair, remove the light, assemble the height from layers,
the normal from it, AO from that. The order is right, and in nine cases out of
ten there is no reason to change it. The tenth case is when you need, for
example, to remove the light TWICE with different radii and add the results,
or compute the height from the lighting but take the normal from that same
height after mixing in the network's map, or send one and the same map into
two different tone settings and compare. The tabs cannot express that.

The graph computes nothing new: every node is a call of the same function
that stands behind the corresponding slider. The only new thing here is that
they can be connected any way you like.

The design is as simple as possible:

  * a node knows its type, its position on the canvas and its parameter
    values;
  * a link is "the output of such-and-such node is plugged into the input of
    such-and-such node";
  * evaluation goes from the end: the output node is asked for its result, it
    asks its inputs, and so on down to the source. Every node is computed once
    per pass and remembers its answer together with the key it was obtained
    from — so editing one parameter does not recompute the whole tree.

Cycles are forbidden: an attempt to connect so that a ring would appear does
not create the link. This is checked by walking along the links (a new link
a → b would close a ring exactly when a is already reachable from b).

Conventions for the node functions registered with @node. Each is called as
fn(i, p, ctx): ``i`` — dict input port → array (an unconnected optional input
is simply absent), ``p`` — dict parameter name → value, ``ctx`` — the
evaluation context (Graph.eval): 'src' the source texture, 'key' a string
identifying it, optionally 'sub' a progress callable(i, n) for long nodes. It
returns a dict output port → array. Value kinds on the wires: RGB — H×W×3 sRGB
0…1; GRAY — H×W float32, usually 0…1 (most nodes stretch their result with
_n01); NRM — H×W×3 unit vectors in −1…1 in the layout chosen by the node's
``directx`` flag; ANY — anything (used by the output node and the packed _nh).
The ESRGAN nodes need their models downloaded beforehand (see esrgan.py).

Texelith — узловой граф.

Зачем он нужен, если вкладки и так всё считают. Вкладки — это один жёстко
заданный порядок: починить, снять свет, собрать высоту из слоёв, из неё
нормаль, из неё AO. Порядок правильный и в девяти случаях из десяти менять
его незачем. Десятый случай — это когда нужно, например, снять свет ДВАЖДЫ
с разными радиусами и сложить, или посчитать высоту по освещению, а нормаль
взять от неё же, но после подмешивания карты от сети, или отправить одну и
ту же карту в два разных тона и сравнить. Во вкладках такого не выразить.

Граф ничего нового не считает: каждый узел — это вызов той же функции, что
стоит за соответствующим ползунком. Новое тут только одно — их можно
соединять как угодно.

Устройство самое простое:

  * узел знает свой тип, положение на холсте и значения параметров;
  * связь — это «выход такого-то узла воткнут во вход такого-то»;
  * счёт идёт с конца: спрашиваем у выходного узла результат, он спрашивает
    у своих входов, и так до источника. Каждый узел считается один раз за
    проход и запоминает ответ вместе с ключом, из которого он получен, —
    поэтому правка одного параметра не пересчитывает всё дерево.

Циклы запрещены: при попытке соединить так, чтобы получилось кольцо, связь
не создаётся. Проверяется это обходом вверх по входам.
"""

import hashlib
import json
import math
import os

import numpy as np

import xpu as X

BUILD = 25          # версия набора файлов, проверяется при запуске

# Типы значений на проводах. Нужны только для двух вещей: подсветить, что
# куда втыкается, и не дать соединить карту высоты с картинкой.
RGB, GRAY, NRM, ANY = 'rgb', 'gray', 'normal', 'any'


def kinds_fit(out_kind, in_kind):
    """Can an output of out_kind be wired into an input of in_kind: the same
    kind, ANY on either side, or grey ↔ colour (converted on the way: grey
    becomes the same value in R, G and B; colour becomes its brightness)."""
    return (out_kind == in_kind or ANY in (out_kind, in_kind)
            or {out_kind, in_kind} == {GRAY, RGB})


def fit_value(v, in_kind):
    """A value brought to what an input of in_kind expects: a grey map
    into a colour input → three equal channels; a colour picture into a grey
    input → its brightness (Rec. 709 on the stored values). Other values
    pass as they are."""
    if v is None or not hasattr(v, 'ndim'):
        return v
    xp = X.xp(v)
    if in_kind == RGB and v.ndim == 2:
        return xp.stack([v, v, v], -1)
    if in_kind == GRAY and v.ndim == 3 and v.shape[2] >= 3:
        return (0.2126 * v[..., 0] + 0.7152 * v[..., 1] + 0.0722 * v[..., 2]).astype(v.dtype)
    return v

KIND_COLOR = {RGB: '#c8a45c', GRAY: '#8a9298', NRM: '#6ea8dc', ANY: '#b0b6bb'}


class Spec:
    """Description of one node type.

    Attributes:
        key: type name, e.g. 'delight'; also the key in REG.
        ins: input ports, list of (name, kind), e.g. [('in', RGB)].
        outs: output ports, list of (name, kind), e.g. [('out', RGB)].
        params: list of (name, kind, default, min, max). kind 'f' — float and
            'i' — int, shown as a slider over min…max; 'b' — bool (min/max 0, 1);
            'c' — choice: min is the tuple of allowed values, max is None;
            'd' — data edited in the canvas, not in the inspector (the patch
            node's list of patches), min and max None.
        fn: the node function fn(inputs dict, params dict, ctx) → dict of
            outputs by port name (see the module docstring).
        size: (width, height) hint for the node box on the canvas, px; stored
            with the spec, the current canvas code does not read it.
        cat: category for the add menu (a key of CATEGORIES).
        opt: names of input ports that may stay unconnected; the node function
            then simply does not find them in its input dict.
        presets: dict preset name → {parameter: value}. If the node has a
            choice parameter 'preset', picking a value in the inspector writes
            these values into the other parameters (the sliders move); the
            node function itself ignores 'preset'.
    """

    presets = {}
    preset_fn = None    # f(params, name, value) → новые значения ползунков или None
    hidden = False      # True — не показывать в меню (оставлен для старых проектов)
    upgrade = None      # f(старые параметры) → параметры: перевод проектов прошлых сборок
    groups = ()         # ((ключ i18n вкладки, (параметры…)), …): параметры в инспекторе по вкладкам
    show_if = None      # f(параметры, имя) → показывать ли ползунок в инспекторе (None — все)

    def __init__(self, key, ins, outs, params, fn, size=(150, 0), cat='other', opt=()):
        """Store the fields described in the class docstring."""
        self.key = key          # 'delight'
        self.ins = ins          # [('in', RGB)]
        self.outs = outs        # [('out', RGB)]
        self.params = params    # [(имя, вид, умолчание, мин, макс)]
        self.fn = fn            # функция(входы_словарь, параметры_словарь) -> словарь
        self.size = size
        self.cat = cat          # раздел меню добавления
        self.opt = tuple(opt)   # входы, которые можно не подключать


REG = {}


def label(key, name):
    """Caption of port or parameter ``name`` of node type ``key``: the
    node's own caption 'np.<key>.<name>' if i18n has one (the same name means
    different things in different nodes — 'amount' is opacity in one and
    effect % in another), else the common 'np.<name>'."""
    import i18n
    k = 'np.%s.%s' % (key, name)
    return i18n.t(k) if k in i18n.S else i18n.t('np.' + name)


def hint(key, name):
    """Explanation of parameter ``name`` of node ``key`` ('' if none):
    'np.<key>.<name>.info', else 'np.<name>.info'."""
    import i18n
    for k in ('np.%s.%s.info' % (key, name), 'np.%s.info' % name):
        if k in i18n.S:
            return i18n.t(k)
    return ''


# Узлы, которым входы отдаются на процессоре: нейросети (у них свой
# onnxruntime или PyTorch, который сам работает на видеокарте; «Две сети»
# после сетей сама переносит фильтры на видеокарту), «Листва» (нарезка на
# листья — процессор, прожилки и AO кроны она сама считает на видеокарте),
# починка (repair.py: тяжёлые части — маски пересвета и поиск донора — сама
# переносит на видеокарту). Остальные — на видеокарте, если она включена.
HOST_NODES = {'repair', 'depth_net', 'deepbump', 'esrgan_normal',
              'esrgan_franken', 'supermat', 'foliage', 'nh_fuse'}

# Разделы меню добавления узла, в этом порядке.
CATEGORIES = ('input', 'prepare', 'adjust', 'filter', 'height', 'normal', 'neural', 'maps',
              'foliage', 'openmw', 'output')


def node(key, ins, outs, params, size=(150, 0), cat='other', opt=(), hidden=False):
    """Decorator factory that registers a node type in REG.

    Used as ``@node(key, ins, outs, params)`` above a function fn(i, p, ctx);
    the arguments are those of Spec. The function itself is returned unchanged,
    so it can still be called directly.
    """
    def deco(fn):
        """Register ``fn`` as REG[key] and return it unchanged."""
        REG[key] = Spec(key, ins, outs, params, fn, size, cat, opt)
        REG[key].hidden = hidden
        return fn
    return deco


# Сколько юнитов игры покрывает один повтор текстуры: задано числом или
# пропорционально ширине текстуры (юнитов на пиксель)
REPEAT_MODES = ('units', 'per_px')
CM_PER_UNIT = 30.48 / 22.1          # 22,1 юнита в футе (MWSE), как в matgen
# общие настройки рельефа у «Параллакса для OpenMW», «Рельефа для OpenMW» и AO
# в сантиметрах (уровень поверхности — только у первых двух, у AO его нет):
# имя настройки в узле → общее имя (галочка «relief_sync»)
RELIEF_SYNC = {'parallax_nh': {'depth': 'depth_cm', 'repeat': 'repeat',
                               'repeat_mode': 'repeat_mode', 'units_px': 'units_px',
                               'ref': 'ref', 'level': 'level'},
               'ao': {'depth_cm': 'depth_cm', 'repeat': 'repeat',
                      'repeat_mode': 'repeat_mode', 'units_px': 'units_px'},
               'relief_omw': {'depth_cm': 'depth_cm', 'repeat': 'repeat',
                              'repeat_mode': 'repeat_mode', 'units_px': 'units_px',
                              'ref': 'ref', 'level': 'level'}}


def repeat_units(p, width_px):
    """Game units covered by one repeat of the texture (along its width):
    the 'repeat' setting, or with repeat_mode 'per_px' units_px × the
    full-size width in pixels — then a texture twice as wide covers twice
    as much."""
    if p.get('repeat_mode') == 'per_px':
        return max(1.0, float(p.get('units_px', 0.5)) * float(width_px))
    return float(p.get('repeat', 128))


def _px(ctx):
    """Scale of the picture being evaluated relative to the full-size one.

    1.0 when saving; 0.5 when the editor previews a large texture on a
    half-size copy. Nodes multiply their radii in pixels by it, so the preview
    shows what will be saved (the same rule as matgen.at_scale)."""
    return float(ctx.get('px', 1.0))


# ------------------------------------------------------------- сами узлы --
@node('source', [], [('rgb', RGB), ('a', GRAY)], [], cat='input')
def _n_source(i, p, ctx):
    """Source node: ``ctx['src']`` (H×W×3 sRGB 0…1) on 'rgb' and its alpha
    (``ctx['alpha']``, H×W 0…1; all ones when the texture has none) on 'a'."""
    src = X.to_dev(ctx['src'])
    xp = X.xp(src)
    a = X.to_dev(ctx.get('alpha'))
    if a is None:
        a = xp.ones(src.shape[:2], xp.float32)
    return {'rgb': src, 'a': a}


@node('repair', [('rgb', RGB)], [('rgb', RGB), ('mask', GRAY), ('overlay', RGB)],
      [('hi', 'f', 0.90, 0.6, 1.0), ('lo', 'f', 0.08, 0.0, 0.4),
       ('flat', 'f', 0.030, 0.005, 0.12), ('grow', 'i', 2, -16, 16),
       ('rim', 'i', 6, 0, 24), ('mix', 'f', 1.0, 0.0, 1.0),
       ('method', 'c', 'patch', ('patch', 'tone', 'smooth', 'mean', 'gradient', 'spots'),
        None), ('seed', 'i', 1, 1, 99)], cat='prepare')
def _n_repair(i, p, ctx):
    """Repair blown-out and crushed spots (repair.py).

    The masks come from repair.exposure_masks (blown-out + crushed, summed and
    clipped): hi — luminance above which a spot counts as blown out (0.6…1);
    lo — below which as crushed (0…0.4); flat — how little texture must remain to
    confirm it (0.005…0.12); grow — margin in pixels, positive expands the mask,
    negative removes from it what is narrower than about 2·|grow| (keeps the dark
    joints between stones); rim — how far, px, the mask creeps from a found
    spot over its slope (pixels up to 0.12 past the level), so the repair does
    not take its sample from the slope; on dark textures the slope is the
    neighbouring dark stone itself, and 0 keeps the repair to the spots
    (repair.exposure_masks reach). method — 'patch', 'tone', 'smooth', or one of the
    fills that take colours only from intact pixels brighter than lo: 'mean' (one
    average colour), 'gradient' (smooth blend of the surrounding colours),
    'spots' (spots of the texture's main colours; seed — which draw of spots,
    the same on the preview and in the file) — see repair.repair; mix —
    0…1 blend of the repaired result over the original.

    If the mask covers practically nothing (< 1e-6 of the pixels above 0.5), the
    input is returned unchanged. Outputs: 'rgb' — H×W×3 float32 sRGB 0…1;
    'mask' — what was selected for repair, H×W float32 0…1 (white = repaired);
    'overlay' — the input with the selection tinted red, to see what is
    selected and where (both picked in the 2D view's port list).
    """
    xp = X.xp(i)
    import repair as RP
    src = i['rgb']
    b, c = RP.exposure_masks(src, hi=p['hi'], lo=p['lo'], flat=p['flat'],
                             grow=float(p['grow']) * _px(ctx),
                             reach=float(p.get('rim', 6)) * _px(ctx))
    m = xp.clip(b + c, 0, 1).astype(xp.float32)
    red = xp.asarray([1.0, 0.1, 0.1], xp.float32)
    over = (src * (1 - 0.6 * m[..., None]) + red * (0.6 * m[..., None])).astype(xp.float32)
    if (m > 0.5).mean() < 1e-6:
        return {'rgb': src, 'mask': m, 'overlay': over}
    # заливки берут цвета только с целого и не чёрного: темнее порога провала
    keep = RP.fill_source(src, m, dark=float(p['lo']))
    out = RP.repair(src, m, p['method'], keep=keep, seed=int(p.get('seed', 1)))
    k = float(p['mix'])
    return {'rgb': xp.clip(src * (1 - k) + out * k, 0, 1).astype(xp.float32), 'mask': m,
            'overlay': over}


@node('patch', [('h', ANY)], [('h', ANY)],
      [('auto', 'b', True, 0, 1), ('seamless', 'b', True, 0, 1),
       ('feather', 'f', 4.0, 0.5, 40.0), ('patches', 'd', [], None, None)], cat='prepare')
def _n_patch(i, p, ctx):
    """Patch tool, by hand, as in Photoshop: each patch is an outline drawn
    in the 2D view over the place to fix and dragged onto the place to take
    from (imgfilters.apply_patch), applied in order.

    auto — the editor picks the source itself as soon as an outline is
    drawn (imgfilters.find_source: the place whose surroundings match best);
    the node function only reads the stored shift.
    'patches' — list of {'poly': [[x, y], …], 'dx': …, 'dy': …}, all in
    fractions of the texture size, so the same patches fit the preview copy
    and the full size. seamless — gradient-domain blend (no seam, colour
    brought to the surroundings); off — plain copy with an edge feathered by
    ``feather`` px. Works on any map: colour, grey or a normal. On the card
    or the processor (the membrane of a seamless patch is solved on the
    processor). Readout: the number of patches.
    """
    import imgfilters as F
    xp = X.xp(i)
    h = xp.asarray(i['h'], xp.float32)
    feather = float(p['feather']) * _px(ctx)
    for pt in p.get('patches') or []:
        h = F.apply_patch(h, pt['poly'], pt['dx'], pt['dy'], bool(p['seamless']), feather)
    return {'h': h, '_info': _info(patches=len(p.get('patches') or []))}


@node('paint_mask', [('rgb', ANY)], [('over', RGB), ('mask', GRAY)],
      [('base', 'c', 'black', ('black', 'white'), None), ('feather', 'f', 0.0, 0.0, 100.0),
       ('invert', 'b', False, 0, 1), ('strokes', 'd', [], None, None)],
      cat='input', opt=('rgb',))
def _n_paint_mask(i, p, ctx):
    """A mask painted by hand with a brush in the 2D view.

    While the node is open in the inspector, the left button paints white
    and the right one black (the brush size, hardness and opacity are set in
    the inspector and stored with each stroke). 'strokes' — the list of
    strokes (imgfilters.paint_strokes), in fractions of the texture size, so
    they fit the preview copy and the full size. base — the mask before any
    stroke, black (nothing selected) or white; feather — blur of the whole
    mask, px of the full-size texture; invert — 1 − mask.

    Outputs: 'over' — the picture from the 'rgb' input (the source texture
    if nothing is connected) with the black part of the mask tinted red, as
    the quick mask in Photoshop, to see where to paint; 'mask' — the mask
    itself, white = 1. Readout: the number of strokes.
    """
    import imgfilters as F
    import matgen as MG
    ref = i.get('rgb')
    if ref is None:
        ref = ctx['src']
    ref = np.asarray(X.to_host(ref), np.float32)
    h, w = ref.shape[:2]
    strokes = p.get('strokes') or []
    m = F.paint_strokes(strokes, h, w, 1.0 if p['base'] == 'white' else 0.0)
    fe = float(p['feather']) * _px(ctx)
    if fe > 0.3:
        m = np.clip(np.asarray(X.to_host(MG.fast_blur(X.to_dev(m), fe, mode='wrap'))), 0, 1)
    if p['invert']:
        m = 1.0 - m
    m = np.asarray(m, np.float32)
    rgb = ref[..., :3] if ref.ndim == 3 else np.stack([ref] * 3, -1)
    if rgb.shape[-1] < 3:
        rgb = np.repeat(rgb[..., :1], 3, -1)
    k = (0.5 * (1.0 - m))[..., None]
    over = np.clip(rgb, 0, 1) * (1 - k) + np.array([0.85, 0.1, 0.1], np.float32) * k
    return {'over': X.to_dev(over.astype(np.float32)), 'mask': X.to_dev(m),
            '_info': _info(strokes=len(strokes))}


@node('delight', [('rgb', RGB)], [('rgb', RGB), ('light', GRAY)],
      [('soft', 'f', 0.5, 0.0, 1.0), ('hard', 'f', 0.0, 0.0, 1.0),
       ('radius', 'i', 48, 4, 300), ('sat_keep', 'f', 0.7, 0.0, 1.0)], cat='prepare')
def _n_delight(i, p, ctx):
    """Remove baked-in lighting (matgen.delight).

    soft — 0…1, remove the smooth light gradient; hard — 0…1, remove sharp
    shadows; radius — scale of "illumination" in pixels (4…300); sat_keep — 0…1,
    hold the original saturation.

    Outputs: 'rgb' — the de-lit texture, H×W×3 sRGB 0…1; 'light' — the lighting
    estimate itself, i.e. the difference of the log linear luminances of the
    original and the result, stretched to 0…1 by _n01.
    """
    xp = X.xp(i)
    import matgen as MG
    src = i['rgb']
    out = MG.delight(src, p['soft'], p['hard'],
                     max(2, int(round(p['radius'] * _px(ctx)))), sat_keep=p['sat_keep'])
    # заодно отдаём саму оценку освещения: разницу оригинала и результата
    a = xp.maximum(MG.luminance(MG.srgb_to_lin(src)), 1e-4)
    b = xp.maximum(MG.luminance(MG.srgb_to_lin(out)), 1e-4)
    return {'rgb': out, 'light': _n01(xp.log(a) - xp.log(b))}


@node('height_lum', [('rgb', RGB)], [('h', GRAY)],
      [('detail', 'f', 0.45, 0.05, 1.0),
       ('mode', 'c', 'multi', ('band', 'multi', 'lum', 'guided'), None),
       ('radius', 'i', 8, 2, 40)], cat='height')
def _n_height_lum(i, p, ctx):
    """Height from luminance (matgen.height_variants on linear luminance).

    mode — which of the variants to output: 'band', 'multi', 'lum' or 'guided';
    detail — 0.05…1, higher keeps finer relief (sets the denoise blur and the
    band-pass cut); radius — guided-filter radius in pixels (2…40). The macro
    falloff is fixed at 0.7.

    Output 'h': H×W float32 0…1.
    """
    import matgen as MG
    L = MG.luminance(MG.srgb_to_lin(i['rgb']))
    parts = MG.height_variants(L, p['detail'], float(p['radius']) * _px(ctx), 0.7,
                               px=_px(ctx))
    return {'h': parts[p['mode']]}


@node('height_light', [('rgb', RGB)], [('h', GRAY), ('signal', GRAY)],
      [('az', 'i', 135, 0, 359), ('auto_az', 'b', False, 0, 1),
       ('smooth', 'f', 0.25, 0.02, 1.5),
       ('chroma', 'f', 0.6, 0.0, 1.5), ('shadow', 'f', 1.0, 0.0, 1.5),
       ('radius', 'i', 40, 6, 200)], cat='height')
def _n_height_light(i, p, ctx):
    """Height from the lighting captured in the texture (shape from shading).

    The shading signal (matgen.shading_signal) has paint (chroma, 0…1.5) and cast
    shadows (shadow, 0…1.5) subtracted; it is then integrated along the light
    direction by matgen.height_from_shading. az — light azimuth in degrees
    (0…359); smooth — regulariser λ (0.02…1.5) that fills the directions across
    the light where the picture holds no relief; radius — scale in pixels
    (6…200).

    The method only works when the picture holds DIRECTIONAL light (painted
    or photographed with a side light). A scan lit evenly has no slope
    information and gives blobs. Readouts: the azimuth estimated from the
    picture (matgen.light_azimuth) and how directional the light looks
    (0…1): below about 0.3 there is nothing to integrate; a high value on
    bricks or wood can come from the courses or the grain, not from light.
    auto_az — use the estimated azimuth instead of ``az``.

    Outputs: 'h' — H×W float32 0…1; 'signal' — the shading signal itself, 0…1
    (both stretched by _n01).
    """
    import matgen as MG
    rgb = i['rgb']
    r = float(p['radius']) * _px(ctx)
    g = MG.shading_signal(rgb, r, p['chroma'], p['shadow'],
                          24.0 * _px(ctx), 48.0 * _px(ctx))
    L = MG.luminance(MG.srgb_to_lin(rgb))
    est, conf = MG.light_azimuth(L, r)
    az = est if p.get('auto_az') else math.radians(p['az'])
    h = MG.height_from_shading(L, az, p['smooth'], r, g)
    return {'h': _n01(h), 'signal': _n01(g),
            '_info': _info(angle_est=math.degrees(est), light_dir=float(conf))}


@node('depth_net', [('rgb', RGB)], [('h', GRAY)],
      [('model', 'c', 'da2s', ('da2s', 'midas'), None),
       ('fine', 'b', False, 0, 1), ('tile', 'f', 0.5, 0.15, 0.75)], cat='neural')
def _n_depth_net(i, p, ctx):
    """Height from a neural depth network (depthnet.compute).

    model — 'da2s' (Depth Anything V2 Small) or 'midas' (MiDaS v2.1 Small); fine —
    use the windowed "fine" mode; tile — window side as a fraction of the shorter
    image side (0.15…0.75). The model must be downloaded and onnxruntime
    installed (see depthnet.py).

    Output 'h': H×W float32 0…1 (depthnet.normalize, 1st…99th percentile).
    """
    import depthnet as DN
    rgb = i['rgb']
    tile = int(min(rgb.shape[:2]) * p['tile'])
    def run():
        """Compute when not cached; a missing model is said plainly."""
        _need_model(DN.have_model(p['model']), DN.MODELS[p['model']]['title'])
        return DN.compute(rgb, p['model'], fine=bool(p['fine']), tile=tile)
    # «v2» в метке: после смены способа (сборка 23) старые записи кэша не берутся
    d = disk_cached('depth_v2_' + p['model'], rgb[..., :3],
                    {'fine': bool(p['fine']), 'tile': tile}, run)
    return {'h': DN.normalize(d), '_info': _info(device=_device(p['model']))}


@node('from_ao', [('h', GRAY)], [('h', GRAY)],
      [('cutoff', 'f', 0.25, 0.05, 0.8), ('invert', 'b', False, 0, 1)], cat='height')
def _n_from_ao(i, p, ctx):
    """Height from an occlusion / cavity map by double integration.

    See matgen.height_from_cavity.

    cutoff — scale beyond which the map is not trusted, as a fraction of the
    shorter side (0.05…0.8); invert — for maps where crevices are light.

    Output 'h': H×W float32 0…1 (_n01).
    """
    import matgen as MG
    return {'h': _n01(MG.height_from_cavity(i['h'], p['cutoff'], bool(p['invert'])))}


@node('bands', [('h', GRAY)], [('h', GRAY)],
      [('scale', 'f', 0.3, 0.05, 1.0), ('low', 'f', 1.0, 0.0, 1.5),
       ('high', 'f', 1.0, 0.0, 2.0)], cat='height')
def _n_bands(i, p, ctx):
    """Re-weight the large and fine bands of a height map.

    scale — "scale boundary" slider (0.05…1), turned into a split radius by
    matgen.band_radius from the shorter side; low — multiplier on the large band
    (0…1.5); high — on the fine band (0…2). Output 'h': H×W float32 0…1 (_n01).
    """
    import matgen as MG
    h = i['h']
    r = MG.band_radius(min(h.shape[:2]), p['scale'])
    return {'h': _n01(MG.weigh_bands(h, r, p['low'], p['high']))}


@node('tone', [('h', GRAY)], [('h', GRAY)],
      [('gamma', 'f', 1.0, 0.2, 3.0), ('contrast', 'f', 1.0, 0.2, 3.0),
       ('invert', 'b', False, 0, 1)], cat='height', hidden=True)
def _n_tone(i, p, ctx):
    """Tone curve for a grey map (matgen.tone): contrast around 0.5, then gamma.

    Hidden from the add menu — Levels, Brightness/Contrast and Invert do the
    same; kept so that old projects still open.

    The input is clipped to 0…1 first. gamma, contrast — 0.2…3; invert — output
    1 − h. Output 'h': H×W 0…1.
    """
    xp = X.xp(i)
    import matgen as MG
    h = MG.tone(xp.clip(i['h'], 0, 1), p['gamma'], p['contrast'])
    return {'h': 1.0 - h if p['invert'] else h}


@node('mix', [('a', GRAY), ('b', GRAY)], [('h', GRAY)],
      [('wa', 'f', 1.0, 0.0, 2.0), ('wb', 'f', 1.0, 0.0, 2.0),
       ('stretch', 'b', True, 0, 1), ('divisor', 'f', 2.0, 0.5, 8.0)],
      cat='height', opt=('a', 'b'))
def _n_mix(i, p, ctx):
    """Weighted sum of two height maps with their means removed.

    stretch on: h = wa·(a − mean a) + wb·(b − mean b), stretched linearly so
    that its minimum is 0 and its maximum 1 — full contrast, however dull the
    inputs are, with nothing pushed beyond 0…1 (the earlier stretch of the
    1st…99th percentile pushed 2 % of the pixels out and flattened them).

    stretch off: no stretching, the inputs' own contrast is kept:
    h = level + (wa·(a − mean a) + wb·(b − mean b)) / divisor, level being the
    weighted mean of the two means. With both weights 1, divisor 2 is the plain
    average (it cannot leave 0…1), divisor 1 the plain sum of the reliefs. What
    still goes beyond 0…1 is bent in softly (matgen.soft01), not clipped; the
    readout 'beyond' gives its share.

    Either input may be left unconnected (Graph.eval allows that for this
    node): then the other one is passed through unchanged, not re-normalised;
    with neither connected the output is None. With both weights at zero the
    output is a flat 0.5 plate, as in matgen.mix_heights.
    """
    xp = X.xp(i)
    a, b = i.get('a'), i.get('b')
    if a is None:
        return {'h': b}
    if b is not None and p['wa'] + p['wb'] <= 0:
        # оба веса в нуле — рельефа нет: ровная плита, как у mix_heights
        return {'h': xp.full(a.shape[:2], 0.5, xp.float32)}
    if b is None:
        return {'h': a}
    ma, mb = float(a.mean()), float(b.mean())
    d = p['wa'] * (a - ma) + p['wb'] * (b - mb)
    if p['stretch']:
        # растяжка по настоящим минимуму и максимуму: ни один пиксель не
        # выходит за 0…1 и ничего не сжимается у краёв. Прежняя растяжка по
        # 1-му…99-му процентилю выталкивала 2 % пикселей за край, и смесь
        # чистых входов получала клип
        lo, hi = float(d.min()), float(d.max())
        if hi - lo < 1e-6:
            return {'h': xp.full(d.shape[:2], 0.5, xp.float32)}
        return {'h': ((d - lo) / (hi - lo)).astype(xp.float32)}
    # без растяжки: делим на делитель и ставим на средний уровень входов
    import matgen as MG
    level = (p['wa'] * ma + p['wb'] * mb) / (p['wa'] + p['wb'])
    h = level + d / max(p['divisor'], 1e-6)
    out = float(((h < 0) | (h > 1)).mean()) * 100.0
    return {'h': MG.soft01(h).astype(xp.float32),
            '_info': _info(beyond='%.1f %%' % out)}


@node('nh_fuse', [('rgb', RGB), ('n', NRM), ('h', GRAY)], [('h', GRAY), ('n', NRM)],
      [('use_n', 'b', True, 0, 1), ('use_h', 'b', True, 0, 1),
       ('overlap', 'f', 0.5, 0.25, 0.75), ('model', 'c', 'da2s', ('da2s', 'midas'), None),
       ('fine', 'b', False, 0, 1), ('tile', 'f', 0.5, 0.15, 0.75),
       ('n_directx', 'b', True, 0, 1), ('hp_n', 'f', 40.0, 0.0, 500.0),
       ('hp_h', 'f', 40.0, 0.0, 500.0), ('wa', 'f', 1.0, 0.0, 2.0), ('wb', 'f', 1.0, 0.0, 2.0),
       ('stretch', 'b', True, 0, 1), ('divisor', 'f', 2.0, 0.5, 8.0),
       ('strength', 'f', 1.0, 0.05, 3.0), ('amp', 'f', 1.0, 0.05, 5.0),
       ('directx', 'b', True, 0, 1)], cat='neural', opt=('rgb', 'n', 'h'))
def _n_nh_fuse(i, p, ctx):
    """Height and normal from a picture by two networks — the usual chain as
    one node: DeepBump normal → height (normal_to_height) → High Pass hp_n
    px; network depth (depth_net: model, fine, tile) → High Pass hp_h px;
    the two mixed («Mix»: weights wa, wb, stretch / divisor); the normal made
    from the result («Normal from height»: strength, amp, layout directx).
    The same functions as the separate nodes, so the result is the same as
    the hand-made chain. The networks get their inputs on the processor
    (they have their own runtime); everything after them runs on the video
    card when it is on.

    use_n / use_h switch a branch off: then only the other one works (its
    height through its High Pass, and the normal from it). A ready normal or
    height may be fed instead of a network ('n' — layout n_directx, 'h'):
    then that network is not run. A High Pass radius of 0 switches that
    filter off. Raises ValueError if neither branch has anything to work on.
    """
    rgb = i.get('rgb')
    n, d = i.get('n'), i.get('h')
    if n is None and p['use_n'] and rgb is not None:
        n = _n_deepbump({'rgb': rgb}, {'overlap': p['overlap'], 'directx': True}, ctx)['n']
        ndx = True
    else:
        ndx = bool(p['n_directx'])
        if not p['use_n']:
            n = None
    if d is None and p['use_h'] and rgb is not None:
        d = _n_depth_net({'rgb': rgb}, {'model': p['model'], 'fine': p['fine'],
                                        'tile': p['tile']}, ctx)['h']
    elif not p['use_h']:
        d = None
    if n is None and d is None:
        raise ValueError(t_('nh_fuse.none'))
    # сети отработали на процессоре (у них своя видеокарта через onnxruntime /
    # PyTorch); дальше обычные фильтры — на видеокарте, если она включена
    n = None if n is None else X.to_dev(n)
    d = None if d is None else X.to_dev(d)

    def hp(h, r):
        if r <= 0:
            return h
        return _n_highpass({'h': h}, {'radius': r, 'remove': 1.0, 'stretch': True,
                                      'keep_edges': False, 'edge_level': 25}, ctx)['h']
    a = b = None
    if n is not None:
        a = _n_normal_to_height({'n': n}, {'cutoff': 0.0, 'directx': ndx}, ctx)['h']
        a = hp(a, float(p['hp_n']))
    if d is not None:
        b = hp(d, float(p['hp_h']))
    h = _n_mix({'a': a, 'b': b}, {'wa': p['wa'], 'wb': p['wb'], 'stretch': p['stretch'],
                                  'divisor': p['divisor']}, ctx)['h']
    nn = _n_normal({'h': h}, {'strength': p['strength'], 'amp': p['amp'], 'fine': 1.0,
                              'mid': 0.0, 'big': 0.0, 'directx': p['directx']}, ctx)['n']
    used = ('both' if a is not None and b is not None else 'n' if a is not None else 'h')
    return {'h': h, 'n': nn, '_info': _info(inputs=t_('nh_fuse.' + used))}


REG['nh_fuse'].groups = (
    ('ed.tab.nets', ('use_n', 'use_h', 'overlap', 'model', 'fine', 'tile', 'n_directx')),
    ('ed.tab.mix', ('hp_n', 'hp_h', 'wa', 'wb', 'stretch', 'divisor')),
    ('ed.tab.normal', ('strength', 'amp', 'directx')))


def t_(key):
    """i18n text by key, imported late (nodes.py does not import i18n at the top)."""
    import i18n
    return i18n.t(key)


# ------------------------------------------------------------ фильтры ----
# Фильтры и коррекции — как в Photoshop и GIMP, с теми же настройками.
# Размеры — в пикселях текстуры (как в Photoshop); на уменьшенной копии для
# превью они уменьшаются вместе с ней, так что превью совпадает с файлом.
# Пороги и уровни — в шагах 0…255, как в диалогах Photoshop.
# Фильтры принимают и серую карту, и цветную картинку (каждый канал отдельно,
# см. _per_channel); коррекции цвета — только цветную.

def _r_px(p, ctx, name='radius'):
    """Radius parameter in pixels of the array being processed (scaled for
    the preview copy), never below 0.3 px."""
    return max(0.3, float(p[name]) * _px(ctx))


def _keep_range(h, out, stretch):
    """Back into 0…1: stretched by percentiles (_n01), or centred on the
    input's mean and clipped, so the level stays where it was."""
    xp = X.xp(out)
    if stretch:
        return _n01(out)
    return xp.clip(out - float(out.mean()) + float(h.mean()), 0, 1).astype(xp.float32)


def _edge_low(h, r, level):
    """Large-scale part of h that does not leak across sharp edges: one pass
    of the self-guided filter (imgfilters.surface_blur) with the box radius
    matched to a Gaussian of sigma r (≈ r·√3)."""
    import imgfilters as F
    return F.surface_blur(h, max(1, int(round(r * 1.732))), level)


@node('highpass', [('h', GRAY)], [('h', GRAY), ('low', GRAY)],
      [('radius', 'f', 40.0, 0.5, 500.0), ('remove', 'f', 1.0, 0.0, 1.0),
       ('stretch', 'b', True, 0, 1), ('keep_edges', 'b', False, 0, 1),
       ('edge_level', 'i', 25, 1, 128)], cat='filter')
def _n_highpass(i, p, ctx):
    """High Pass (Photoshop: Other → High Pass): take away what is larger
    than ``radius`` px, keep the fine relief.

    The large part is a Gaussian blur of that radius on the torus; remove —
    0…1, how much of it is subtracted (1 = all, as in Photoshop). stretch —
    stretch the result to 0…1 by percentiles; off — keep the input's mean
    level and clip.

    keep_edges — estimate the large part with an edge-preserving filter
    instead of the blur. A blur spreads a sharp bright spot into a wide hump,
    and subtracting the hump leaves a dark ring (halo) round the spot; the
    edge-preserving estimate keeps steps higher than edge_level (levels
    1…128 of 255) inside the large part, so no ring is left.

    Outputs: 'h' the filtered map; 'low' the part that was taken away (0…1).
    Readout: the share of the map's spread (variance) that was in the large
    part.
    """
    import matgen as MG
    h = i['h']
    r = _r_px(p, ctx)
    if p.get('keep_edges'):
        low = _edge_low(h, r, float(p['edge_level']) / 255.0)
    else:
        low = MG.fast_blur(h, r, mode='wrap')
    out = h - float(p['remove']) * (low - float(low.mean()))
    share = float(low.var()) / max(float(h.var()), 1e-12)
    return {'h': _keep_range(h, out, p['stretch']), 'low': _n01(low),
            '_info': _info(low_share=100.0 * share)}


@node('lowpass', [('h', GRAY)], [('h', GRAY)],
      [('radius', 'f', 4.0, 0.3, 250.0)], cat='filter')
def _n_lowpass(i, p, ctx):
    """Gaussian Blur: sigma = radius px, on the torus."""
    import matgen as MG
    h = i['h']
    return {'h': MG.fast_blur(h, _r_px(p, ctx), mode='wrap').astype(X.xp(h).float32)}


@node('box_blur', [('h', GRAY)], [('h', GRAY)],
      [('radius', 'f', 4.0, 1.0, 250.0)], cat='filter')
def _n_box_blur(i, p, ctx):
    """Box Blur: the mean over a square of side 2·radius + 1 px."""
    import imgfilters as F
    return {'h': F.box_blur(i['h'], _r_px(p, ctx))}


@node('motion_blur', [('h', GRAY)], [('h', GRAY)],
      [('angle', 'i', 0, -180, 180), ('distance', 'f', 10.0, 1.0, 500.0)], cat='filter')
def _n_motion_blur(i, p, ctx):
    """Motion Blur: the mean along a line ``distance`` px long at ``angle``
    degrees (0 — horizontal)."""
    import imgfilters as F
    return {'h': F.motion_blur(i['h'], p['angle'], float(p['distance']) * _px(ctx))}


@node('edge_smooth', [('h', GRAY)], [('h', GRAY)],
      [('radius', 'f', 5.0, 1.0, 100.0), ('threshold', 'i', 15, 2, 255)], cat='filter')
def _n_edge_smooth(i, p, ctx):
    """Surface Blur: smooths within ``radius`` px, but steps higher than
    ``threshold`` (levels of 255) stay sharp (self-guided filter)."""
    import imgfilters as F
    return {'h': F.surface_blur(i['h'], _r_px(p, ctx), float(p['threshold']) / 255.0)}


@node('antialias', [('h', ANY)], [('h', ANY)],
      [('algo', 'c', 'fxaa', ('fxaa', 'contour', 'mask'), None),
       ('amount', 'f', 1.0, 0.0, 1.0), ('radius', 'f', 1.0, 0.5, 4.0)], cat='filter')
def _n_antialias(i, p, ctx):
    """Anti-aliasing — the staircase taken off edges (imgfilters.antialias):
    'fxaa' — FXAA; 'contour' — blur along contours only; 'mask' — blur and
    cut again, for two-level maps (masks, alpha, threshold, two flat
    colours). amount — how much (for FXAA the softening of one-pixel
    steps); radius — px of the full-size texture ('contour', 'mask').
    Works on grey, colour and normals. Readout: share of pixels touched.
    """
    import imgfilters as F
    out, share = F.antialias(i['h'], p['algo'], float(p['amount']),
                             max(0.5, float(p['radius']) * _px(ctx)))
    return {'h': out, '_info': _info(touched=share * 100.0)}


@node('sharpen', [('h', GRAY)], [('h', GRAY)],
      [('amount', 'i', 100, 1, 500), ('radius', 'f', 1.0, 0.1, 250.0),
       ('threshold', 'i', 0, 0, 255)], cat='filter')
def _n_sharpen(i, p, ctx):
    """Unsharp Mask: amount % of the difference from a Gaussian blur of
    ``radius`` px is added where it exceeds ``threshold`` levels."""
    import imgfilters as F
    return {'h': F.unsharp_mask(i['h'], float(p['amount']) / 100.0, _r_px(p, ctx),
                                float(p['threshold']) / 255.0)}


@node('add_noise', [('h', GRAY)], [('h', GRAY)],
      [('amount', 'f', 5.0, 0.1, 100.0), ('gaussian', 'b', True, 0, 1),
       ('mono', 'b', True, 0, 1), ('seed', 'i', 1, 0, 999)], cat='filter')
def _n_add_noise(i, p, ctx):
    """Add Noise: amount % (uniform ±amount, or Gaussian with σ = amount/2);
    mono — the same noise in every channel of a colour picture. The same
    seed gives the same noise. Not scaled for the preview: noise is per
    pixel."""
    import imgfilters as F
    h = i['h']
    xp = X.xp(h)
    a = float(p['amount']) / 100.0
    if h.ndim == 2:
        return {'h': F.add_noise(h, a, p['gaussian'], p['seed'])}
    if p['mono']:
        n = F.add_noise(xp.zeros(h.shape[:2], xp.float32) + 0.5, a, p['gaussian'], p['seed'])
        out = xp.clip(h[..., :3] + (n - 0.5)[..., None], 0, 1)
    else:
        out = xp.stack([F.add_noise(h[..., c], a, p['gaussian'], p['seed'], c)
                        for c in range(3)], axis=-1)
    if h.shape[-1] > 3:
        out = xp.concatenate([out, h[..., 3:]], axis=-1)
    return {'h': out.astype(xp.float32)}


@node('median', [('h', GRAY)], [('h', GRAY)],
      [('radius', 'i', 1, 1, 25)], cat='filter')
def _n_median(i, p, ctx):
    """Median over a round window of ``radius`` px: removes specks, keeps
    edges."""
    import imgfilters as F
    return {'h': F.median(i['h'], max(1, int(round(int(p['radius']) * _px(ctx)))))}


@node('dust', [('h', GRAY)], [('h', GRAY)],
      [('radius', 'i', 2, 1, 25), ('threshold', 'i', 20, 0, 255)], cat='filter')
def _n_dust(i, p, ctx):
    """Dust & Scratches: a pixel that differs from the median of ``radius``
    px by more than ``threshold`` levels is replaced by that median."""
    import imgfilters as F
    return {'h': F.dust_scratches(i['h'], max(1, int(round(int(p['radius']) * _px(ctx)))),
                                  float(p['threshold']) / 255.0)}


@node('denoise', [('h', GRAY)], [('h', GRAY)],
      [('strength', 'f', 1.0, 0.1, 5.0), ('detail', 'f', 0.0, 0.0, 1.0)], cat='filter')
def _n_denoise(i, p, ctx):
    """Reduce Noise: non-local means; strength — against the noise level
    estimated from the picture; detail — share of the removed difference put
    back. On the video card when it is on (imgfilters.reduce_noise)."""
    import imgfilters as F
    return {'h': F.reduce_noise(i['h'], p['strength'], p['detail'])}


@node('maximum', [('h', GRAY)], [('h', GRAY)],
      [('radius', 'i', 1, 1, 50)], cat='filter')
def _n_maximum(i, p, ctx):
    """Maximum (GIMP: Dilate): the largest value within ``radius`` px —
    light areas grow."""
    import imgfilters as F
    return {'h': F.maximum(i['h'], max(1, int(round(int(p['radius']) * _px(ctx)))))}


@node('minimum', [('h', GRAY)], [('h', GRAY)],
      [('radius', 'i', 1, 1, 50)], cat='filter')
def _n_minimum(i, p, ctx):
    """Minimum (GIMP: Erode): the smallest value within ``radius`` px — dark
    areas grow."""
    import imgfilters as F
    return {'h': F.minimum(i['h'], max(1, int(round(int(p['radius']) * _px(ctx)))))}


@node('offset', [('h', GRAY)], [('h', GRAY)],
      [('dx', 'f', 50.0, -50.0, 50.0), ('dy', 'f', 50.0, -50.0, 50.0)], cat='filter')
def _n_offset(i, p, ctx):
    """Offset with wrap-around, in % of the width and height: 50/50 brings
    the tile edges to the middle — a seam, if there is one, is in plain
    view."""
    import imgfilters as F
    return {'h': F.offset(i['h'], float(p['dx']) / 100.0, float(p['dy']) / 100.0)}


@node('emboss', [('h', GRAY)], [('h', GRAY)],
      [('angle', 'i', 135, -180, 180), ('height', 'f', 3.0, 0.5, 50.0),
       ('amount', 'i', 100, 1, 500)], cat='filter')
def _n_emboss(i, p, ctx):
    """Emboss: grey 0.5 plus the difference between the values ``height`` px
    ahead and behind along ``angle`` (where the light comes from), times
    amount %."""
    import imgfilters as F
    return {'h': F.emboss(i['h'], p['angle'], float(p['height']) * _px(ctx),
                          float(p['amount']) / 100.0)}


@node('edges', [('h', GRAY)], [('h', GRAY)],
      [('method', 'c', 'sobel', ('sobel', 'prewitt', 'scharr', 'laplace'), None),
       ('invert', 'b', False, 0, 1)], cat='filter')
def _n_edges(i, p, ctx):
    """Find Edges (GIMP: Edge Detect): strength of the change — Sobel,
    Prewitt, Scharr or Laplace — stretched to 0…1; invert — dark lines on
    white, as Photoshop shows them."""
    import imgfilters as F
    return {'h': F.find_edges(i['h'], p['method'], bool(p['invert']))}


# Узлы, которые берут и серое, и цветное: каждый канал отдельно.
FILTER_NODES = ('highpass', 'lowpass', 'box_blur', 'motion_blur', 'edge_smooth', 'sharpen',
                'median', 'dust', 'denoise', 'maximum', 'minimum', 'offset', 'emboss',
                'edges')


def _per_channel(fn):
    """Wrap a grey-map filter so it also takes an H×W×3 (or ×4) picture.

    A 2-D input goes straight to ``fn``. A colour one is filtered channel by
    channel (alpha, if any, is passed through untouched) and the channels
    are stacked back. 'stretch' is not applied to colour: stretching each
    channel on its own would shift the hues, so each channel keeps its own
    mean level instead (the "off" behaviour). Readouts come from the first
    channel.
    """
    def run(i, p, ctx):
        h = i['h']
        if h is None or h.ndim == 2:
            # карты дальше ждут float32, даже если пришла float64
            return {k: (v.astype(X.xp(v).float32) if hasattr(v, 'ndim') and
                        not k.startswith('_') else v)
                    for k, v in fn(i, p, ctx).items()}
        xp = X.xp(h)
        q = dict(p)
        if 'stretch' in q:
            q['stretch'] = False
        n = min(h.shape[2], 3)
        outs = [fn(dict(i, h=xp.ascontiguousarray(h[..., c])), q, ctx) for c in range(n)]
        res = {}
        for k, v in outs[0].items():
            if k.startswith('_') or v is None:
                res[k] = v
                continue
            planes = [X.xp(o[k]).asarray(o[k]) for o in outs]
            planes = [X.to_dev(pl) if X.is_dev(h) else X.to_host(pl) for pl in planes]
            if h.shape[2] > 3 and k == 'h':
                planes.append(h[..., 3])          # альфа — как была
            res[k] = xp.stack(planes, axis=-1).astype(xp.float32)
        return res
    run.__doc__ = fn.__doc__
    return run


for _k in FILTER_NODES:
    _sp = REG[_k]
    _sp.fn = _per_channel(_sp.fn)
    _sp.ins = [(n, ANY if k == GRAY else k) for n, k in _sp.ins]
    _sp.outs = [(n, ANY if k == GRAY else k) for n, k in _sp.outs]
# «Добавить шум» сам различает серое и цветное (общий шум на все каналы)
REG['add_noise'].ins = [('h', ANY)]
REG['add_noise'].outs = [('h', ANY)]


@node('bandpass', [('h', GRAY)], [('h', GRAY)],
      [('hi_cut', 'f', 0.004, 0.001, 0.2), ('lo_cut', 'f', 0.08, 0.005, 0.5),
       ('stretch', 'b', True, 0, 1)], cat='filter', hidden=True)
def _n_bandpass(i, p, ctx):
    """Band-pass, kept only so that old projects still open: it is the same
    as Gaussian Blur followed by High Pass, so it is not in the add menu.
    Difference of two blurs, sizes as fractions of the shorter side."""
    import matgen as MG
    h = i['h']
    a, b = sorted((float(p['hi_cut']), float(p['lo_cut'])))
    r1 = max(0.5, a * min(h.shape[:2]))
    r2 = max(0.5, b * min(h.shape[:2]))
    out = MG.fast_blur(h, r1, mode='wrap') - MG.fast_blur(h, r2, mode='wrap')
    return {'h': _keep_range(h, out, p['stretch'])}


# ------------------------------------------------------------ коррекция --
def _keep_alpha(src, out):
    """Put the alpha of a 4-channel source back onto a 3-channel result."""
    xp = X.xp(out)
    if src.ndim == 3 and src.shape[-1] > 3 and out.ndim == 3:
        return xp.concatenate([out, src[..., 3:]], axis=-1)
    return out


def _all_channels(h, f):
    """Apply f to a grey map, or to the colour channels of a picture
    (alpha kept)."""
    if h.ndim == 2:
        return f(h)
    return _keep_alpha(h, f(h[..., :3]))


@node('levels', [('h', ANY)], [('h', ANY)],
      [('in_black', 'i', 0, 0, 254), ('in_white', 'i', 255, 1, 255),
       ('gamma', 'f', 1.0, 0.1, 9.99), ('out_black', 'i', 0, 0, 255),
       ('out_white', 'i', 255, 0, 255)], cat='adjust')
def _n_levels(i, p, ctx):
    """Levels: input black and white points, midtone gamma (above 1 —
    lighter), output black and white points; levels of 255."""
    import imgfilters as F
    a = [float(p[k]) / 255.0 for k in ('in_black', 'in_white')]
    o = [float(p[k]) / 255.0 for k in ('out_black', 'out_white')]
    return {'h': _all_channels(i['h'], lambda v: F.levels(v, a[0], a[1], p['gamma'],
                                                           o[0], o[1]))}


CURVE_LINE = [[0.0, 0.0], [1.0, 1.0]]


@node('curves', [('h', ANY)], [('h', ANY)],
      [('pts', 'd', CURVE_LINE, None, None)], cat='adjust')
def _n_curves(i, p, ctx):
    """Curves, as the Curves dialogue: 'pts' — the points [input, output]
    0…1, any number from two up, edited on the curve in the inspector; a
    smooth cubic spline between them (imgfilters.curve_lut). The default is
    the straight line. Points out of order or closer than 1/1000 along the
    input are tidied before use."""
    import imgfilters as F
    pts = []
    for x, y in sorted((float(a), float(b)) for a, b in (p.get('pts') or CURVE_LINE)):
        if not pts or x - pts[-1][0] > 1e-3:
            pts.append((min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)))
    if len(pts) < 2:
        pts = [(0.0, 0.0), (1.0, 1.0)]
    lut = F.curve_lut(pts)
    return {'h': _all_channels(i['h'], lambda v: F.apply_lut(v, lut))}


def _curves_upgrade(old):
    """Projects of earlier build 24 files stored five sliders c0…c100 (output
    level of 255 at input 0, 25, 50, 75, 100 %): turned into points."""
    if 'pts' not in old and 'c0' in old:
        xs = (('c0', 0.0, 0), ('c25', 0.25, 64), ('c50', 0.5, 128), ('c75', 0.75, 191),
              ('c100', 1.0, 255))
        old['pts'] = [[x, round(float(old.get(k, d)) / 255.0, 4)] for k, x, d in xs]
    return old


REG['curves'].upgrade = _curves_upgrade



@node('brightness', [('h', ANY)], [('h', ANY)],
      [('brightness', 'i', 0, -150, 150), ('contrast', 'i', 0, -100, 100)], cat='adjust')
def _n_brightness(i, p, ctx):
    """Brightness/Contrast: brightness in levels is added; contrast % scales
    the spread round the mean (100 — twice, −100 — flat)."""
    import imgfilters as F
    return {'h': _all_channels(i['h'], lambda v: F.brightness_contrast(
        v, float(p['brightness']) / 255.0, float(p['contrast']) / 100.0))}


@node('invert', [('h', ANY)], [('h', ANY)], [], cat='adjust')
def _n_invert(i, p, ctx):
    """Invert: 1 − value (alpha kept)."""
    xp = X.xp(i)
    return {'h': _all_channels(i['h'], lambda v: (1.0 - xp.clip(v, 0, 1)).astype(xp.float32))}


@node('threshold', [('h', ANY)], [('h', GRAY)],
      [('level', 'i', 128, 1, 255)], cat='adjust')
def _n_threshold(i, p, ctx):
    """Threshold: white where the brightness is at least ``level`` (of 255),
    black elsewhere. A colour picture is judged by its brightness."""
    import imgfilters as F
    xp = X.xp(i)
    h = i['h']
    v = h if h.ndim == 2 else F.luma(h[..., :3])
    return {'h': (v >= float(p['level']) / 255.0).astype(xp.float32)}


@node('posterize', [('h', ANY)], [('h', ANY)],
      [('levels', 'i', 4, 2, 255)], cat='adjust')
def _n_posterize(i, p, ctx):
    """Posterize: so many levels per channel."""
    import imgfilters as F
    return {'h': _all_channels(i['h'], lambda v: F.posterize(v, p['levels']))}


@node('equalize', [('h', ANY)], [('h', ANY)], [], cat='adjust')
def _n_equalize(i, p, ctx):
    """Equalize: every channel's histogram made flat (as in GIMP)."""
    import imgfilters as F
    h = i['h']
    if h.ndim == 2:
        return {'h': F.equalize(h)}
    xp = X.xp(h)
    out = xp.stack([F.equalize(xp.ascontiguousarray(h[..., c])) for c in range(3)], axis=-1)
    return {'h': _keep_alpha(h, out)}


@node('auto_contrast', [('h', ANY)], [('h', ANY)],
      [('clip_lo', 'f', 0.1, 0.0, 10.0), ('clip_hi', 'f', 0.1, 0.0, 10.0)], cat='adjust')
def _n_auto_contrast(i, p, ctx):
    """Auto Contrast (GIMP: Stretch Contrast): clip_lo % of the pixels go
    black, clip_hi % white, the rest is stretched; all channels together, so
    hues stay."""
    import imgfilters as F
    return {'h': _all_channels(i['h'], lambda v: F.auto_contrast(
        v, float(p['clip_lo']) / 100.0, float(p['clip_hi']) / 100.0))}


@node('exposure', [('rgb', RGB)], [('rgb', RGB)],
      [('stops', 'f', 0.0, -5.0, 5.0), ('offset', 'f', 0.0, -0.5, 0.5),
       ('gamma', 'f', 1.0, 0.1, 9.99)], cat='adjust')
def _n_exposure(i, p, ctx):
    """Exposure: in linear light ×2^stops, plus offset, then gamma — the same
    three settings as in Photoshop."""
    import imgfilters as F
    h = i['rgb']
    return {'rgb': _keep_alpha(h, F.exposure(h[..., :3], p['stops'], p['offset'], p['gamma']))}


@node('hue_sat', [('rgb', RGB)], [('rgb', RGB)],
      [('hue', 'i', 0, -180, 180), ('saturation', 'i', 0, -100, 100),
       ('lightness', 'i', 0, -100, 100)], cat='adjust')
def _n_hue_sat(i, p, ctx):
    """Hue/Saturation: hue shift in degrees; saturation and lightness −100…100
    as in Photoshop."""
    import imgfilters as F
    h = i['rgb']
    return {'rgb': _keep_alpha(h, F.hue_saturation(h, p['hue'], p['saturation'] / 100.0,
                                                   p['lightness'] / 100.0))}


@node('color_balance', [('rgb', RGB)], [('rgb', RGB)],
      [('tone', 'c', 'midtones', ('shadows', 'midtones', 'highlights'), None),
       ('cyan_red', 'i', 0, -100, 100), ('magenta_green', 'i', 0, -100, 100),
       ('yellow_blue', 'i', 0, -100, 100), ('keep_light', 'b', True, 0, 1)], cat='adjust')
def _n_color_balance(i, p, ctx):
    """Color Balance for one tonal range: three sliders −100…100; keep_light —
    the brightness is kept as it was. For several ranges chain several
    nodes."""
    import imgfilters as F
    h = i['rgb']
    return {'rgb': _keep_alpha(h, F.color_balance(
        h, p['tone'], p['cyan_red'] / 100.0, p['magenta_green'] / 100.0,
        p['yellow_blue'] / 100.0, bool(p['keep_light'])))}


@node('desaturate', [('rgb', RGB)], [('h', GRAY)],
      [('mode', 'c', 'luma', ('luma', 'average', 'lightness', 'max', 'r', 'g', 'b'), None)],
      cat='adjust')
def _n_desaturate(i, p, ctx):
    """Desaturate to a grey map: by brightness (Rec. 709), the average, the
    lightness (max + min)/2, the maximum, or one channel."""
    import imgfilters as F
    return {'h': F.desaturate(i['rgb'][..., :3], p['mode'])}


BLEND_MODES = ('mix', 'multiply', 'screen', 'overlay', 'soft', 'darken', 'lighten')


def _blend(a, b, mode):
    """Photoshop blend of layer b over base a, both sRGB 0…1, as Photoshop
    does it (in display space). Soft light — the W3C/PDF formula."""
    xp = X.xp(a)
    if mode == 'multiply':
        return a * b
    if mode == 'screen':
        return 1.0 - (1.0 - a) * (1.0 - b)
    if mode == 'overlay':
        return xp.where(a < 0.5, 2.0 * a * b, 1.0 - 2.0 * (1.0 - a) * (1.0 - b))
    if mode == 'soft':
        d = xp.where(a <= 0.25, ((16.0 * a - 12.0) * a + 4.0) * a, xp.sqrt(a))
        return xp.where(b <= 0.5, a - (1.0 - 2.0 * b) * a * (1.0 - a),
                        a + (2.0 * b - 1.0) * (d - a))
    if mode == 'darken':
        return xp.minimum(a, b)
    if mode == 'lighten':
        return xp.maximum(a, b)
    return b


@node('mix_rgb', [('a', RGB), ('b', RGB), ('mask', GRAY)], [('rgb', RGB)],
      [('mode', 'c', 'mix', BLEND_MODES, None), ('amount', 'f', 1.0, 0.0, 1.0)],
      cat='prepare', opt=('b', 'mask'))
def _n_mix_rgb(i, p, ctx):
    """Blend two colour pictures (diffuse), as layers in Photoshop.

    a — the bottom layer, b — the top one; mode — how b is laid over a
    (mix = plain cover, multiply, screen, overlay, soft light, darken,
    lighten); amount — the top layer's opacity 0…1; mask (optional, grey
    0…1) — where the top layer applies (1 = fully). Without b the output is
    a. If b has another size it is resized to a's on the torus. Alpha of a
    is kept. Readout: mean change in 0…255 steps.
    """
    xp = X.xp(i)
    a = i['a']
    b = i.get('b')
    if b is None:
        return {'rgb': a}
    a3 = xp.clip(a[..., :3], 0, 1)
    b3 = xp.clip(b[..., :3], 0, 1)
    if b3.shape[:2] != a3.shape[:2]:
        from texio import resize_wrap
        b3 = X.to_dev(resize_wrap(X.to_host(b3), a3.shape[0], a3.shape[1]).astype('float32'))
    w = float(p['amount'])
    m = i.get('mask')
    if m is not None:
        if m.ndim == 3:
            m = m[..., :3].mean(axis=-1)
        if m.shape[:2] != a3.shape[:2]:
            raise ValueError('mask size %dx%d differs from the picture %dx%d'
                             % (m.shape[1], m.shape[0], a3.shape[1], a3.shape[0]))
        w = xp.clip(m, 0, 1)[..., None] * w
    out = a3 + (_blend(a3, b3, p['mode']) - a3) * w
    out = xp.clip(out, 0, 1).astype(xp.float32)
    if a.shape[-1] > 3:
        out = xp.concatenate([out, a[..., 3:]], axis=-1)
    return {'rgb': out, '_info': _info(steps=float(xp.abs(out[..., :3] - a3).mean() * 255))}


@node('normal', [('h', GRAY)], [('n', NRM)],
      [('strength', 'f', 1.0, 0.05, 3.0), ('amp', 'f', 1.0, 0.05, 5.0),
       ('fine', 'f', 1.0, 0.0, 2.0), ('mid', 'f', 0.0, 0.0, 2.0),
       ('big', 'f', 0.0, 0.0, 2.0), ('directx', 'b', True, 0, 1)], cat='normal')
def _n_normal(i, p, ctx):
    """Normal map from height (matgen.normal_from_height).

    The input is clipped to 0…1 and multiplied by amp (0.05…5; 1 gives the same
    normal as the classic Material tab at the same strength).
    strength — overall slope multiplier (0.05…3); fine, mid, big — weights of the
    per-pixel, mid-scale and large-scale gradients (0…2); directx — layout of the
    result (True: y down the image).

    Output 'n': H×W×3 unit vectors.
    """
    xp = X.xp(i)
    import matgen as MG
    h = xp.clip(i['h'], 0, 1) * p['amp']
    k = _px(ctx)
    return {'n': MG.normal_from_height(h, p['strength'], bool(p['directx']),
                                       p['fine'] * k, p['mid'], p['big'], 8.0 * k)}


@node('normal_light', [('rgb', RGB)], [('n', NRM)],
      [('smooth', 'f', 0.25, 0.02, 1.5), ('chroma', 'f', 0.6, 0.0, 1.5),
       ('shadow', 'f', 1.0, 0.0, 1.5), ('scale', 'f', 0.3, 0.05, 1.0),
       ('low', 'f', 1.0, 0.0, 1.5), ('high', 'f', 1.0, 0.0, 2.0),
       ('strength', 'f', 1.0, 0.2, 3.0), ('blue', 'c', 'unit', ('unit', 'ao'), None),
       ('invert', 'b', False, 0, 1), ('fix', 'f', 0.0, 0.0, 1.0),
       ('swap', 'b', False, 0, 1), ('directx', 'b', True, 0, 1)], cat='normal')
def _n_normal_light(i, p, ctx):
    """Normal straight from the lighting, bypassing height (matgen.normal_from_shading).

    R and G are the "from lighting" layer computed for two light directions; the
    shading signal uses a fixed radius of 40 px. smooth, scale, low, high — as for
    the lighting layer (matgen.sfs_layer); chroma, shadow — paint and shadow
    correction of the signal; strength — multiplier on R and G (0.2…3); blue —
    'unit' (z from the vector length) or 'ao' (z from an occlusion map computed
    here from the lighting layer with cavity_ao, radius 28, strength 0.9);
    invert — turn the relief inside out; swap — swap the R and G fields; fix —
    0…1, fit the result to a height (matgen.make_integrable), trading sharpness
    for consistency with height, AO and parallax; directx — layout.

    Output 'n': H×W×3 unit vectors.
    """
    import matgen as MG
    rgb = i['rgb']
    k = _px(ctx)
    g = MG.shading_signal(rgb, 40.0 * k, p['chroma'], p['shadow'], 24.0 * k, 48.0 * k)
    L = MG.luminance(MG.srgb_to_lin(rgb))
    ao = None
    if p['blue'] == 'ao':
        h = MG.sfs_layer(L, math.pi * 0.5, p['smooth'], 40.0 * k, g,
                         p['scale'], p['low'], p['high'])
        ao = MG.cavity_ao(h * 20.0 * k, 28.0 * k, 0.9)
    n = MG.normal_from_shading(L, None, p['smooth'], p['strength'], g,
                               bool(p['directx']), bool(p['swap']),
                               p['scale'], p['low'], p['high'], 40.0 * k,
                               p['blue'], ao, bool(p['invert']))
    if p['fix'] > 0:
        n, _ = MG.make_integrable(n, p['fix'], bool(p['directx']))
    return {'n': n}


GREEN = ('auto', 'directx', 'opengl')


def _green(n, mode, want_dx):
    """Bring a stored normal to the wanted layout.

    mode — what the stored map is: 'directx', 'opengl' or 'auto' (decided by
    matgen.guess_green from the curl). Returns (normal, stored_was_directx,
    curl_dx, curl_gl)."""
    xp = X.xp(n, mode, want_dx)
    import matgen as MG
    is_dx, c_dx, c_gl = MG.guess_green(n)
    if mode != 'auto':
        is_dx = (mode == 'directx')
    if is_dx != bool(want_dx):
        n = n * xp.array([1.0, -1.0, 1.0], xp.float32)
    return n, is_dx, c_dx, c_gl


@node('as_normal', [('rgb', RGB)], [('n', NRM)],
      [('green', 'c', 'directx', GREEN, None), ('directx', 'b', True, 0, 1)],
      cat='input')
def _n_as_normal(i, p, ctx):
    """Treat the picture as a ready normal map: 0…1 into vectors −1…1.

    The vectors are renormalised (length floored at 1e-6); alpha is dropped.
    green — the layout the picture has ('auto' decides by the curl, see
    matgen.guess_green); directx — the layout to output. Output 'n': H×W×3
    float32 unit vectors; readout: which layout was found and both curls.
    """
    xp = X.xp(i)
    n = xp.asarray(i['rgb'], xp.float32)[..., :3] * 2.0 - 1.0
    n = n / xp.maximum(xp.sqrt((n * n).sum(-1, keepdims=True)), 1e-6)
    n, is_dx, c_dx, c_gl = _green(n, p['green'], p['directx'])
    return {'n': n.astype(xp.float32),
            '_info': _info(green='DirectX' if is_dx else 'OpenGL',
                           curl_dx=c_dx, curl_gl=c_gl)}


@node('normal_to_height', [('n', NRM)], [('h', GRAY)],
      [('cutoff', 'f', 0.0, 0.0, 0.8), ('directx', 'b', True, 0, 1)], cat='height')
def _n_normal_to_height(i, p, ctx):
    """Height from a normal map (matgen.height_from_normal, Frankot–Chellappa).

    cutoff — scale beyond which not to trust, as a fraction of the shorter side,
    0 = trust everything (0…0.8); directx — layout of the input normal.

    Output 'h': H×W float32 0…1, stretched by the 0.2 and 99.8 percentiles and
    clipped: single extreme pixels do not squeeze the rest into a flat grey.
    """
    xp = X.xp(i)
    import matgen as MG
    h = MG.height_from_normal(i['n'], bool(p['directx']), p['cutoff'])
    q = X.to_host(xp.percentile(h, xp.asarray([0.2, 99.8], xp.float32)))
    lo, hi = float(q[0]), float(q[1])
    return {'h': xp.clip((h - lo) / max(hi - lo, 1e-6), 0, 1).astype(xp.float32)}


@node('esrgan_normal', [('rgb', RGB)], [('n', NRM)],
      [('strength', 'f', 1.0, 0.2, 3.0), ('flip', 'b', False, 0, 1),
       ('directx', 'b', True, 0, 1)], cat='neural')
def _n_esrgan_normal(i, p, ctx):
    """Normal from the NormalMapGenerator network (esrgan.py).

    The model must already be downloaded — that is done with the button in the
    ESRGAN section of the material tab, so that the person sees the licence
    before downloading.

    strength — multiplier on x and y before renormalisation (0.2…3); directx —
    as in esrgan.normal_from_output: True keeps the model's green, False negates
    it; flip — additionally negate green. z is floored at 1e-3. Progress of the
    window passes goes to ``ctx['sub']``.

    Output 'n': H×W×3 unit vectors.

    Raises:
        ValueError: the model is not downloaded (text: i18n 'nd.nomodel').
    """
    xp = X.xp(i)
    import esrgan as ES
    if not ES.have_model('normal'):
        _need_model(False, ES.MODELS['normal']['title'])
    out = disk_cached('esr_normal', i['rgb'][..., :3], {},
                      lambda: ES.run('normal', i['rgb'], progress=ctx.get('sub')))
    n = out * 2.0 - 1.0
    ny = n[..., 1] if p['directx'] else -n[..., 1]
    if p['flip']:
        ny = -ny
    k = float(p['strength'])
    n = xp.stack([n[..., 0] * k, ny * k, xp.maximum(n[..., 2], 1e-3)], -1)
    return {'n': n / xp.sqrt((n * n).sum(-1, keepdims=True)),
            '_info': _info(device=_device('esr_normal'))}


@node('esrgan_franken', [('rgb', RGB)], [('h', GRAY), ('rough', GRAY), ('gray', GRAY)], [], cat='neural')
def _n_esrgan_franken(i, p, ctx):
    """Franken map from the FrankenMapGenerator network.

    Height (channel B), roughness (G) and the texture in greyscale (R).

    The model must already be downloaded (see _n_esrgan_normal). Outputs 'h',
    'rough', 'gray': H×W float32 0…1 as the network gives them, without
    re-normalisation.

    Raises:
        ValueError: the model is not downloaded (text: i18n 'nd.nomodel').
    """
    import esrgan as ES
    if not ES.have_model('franken'):
        _need_model(False, ES.MODELS['franken']['title'])
    g, r, h = ES.franken_split(disk_cached(
        'esr_franken', i['rgb'][..., :3], {},
        lambda: ES.run('franken', i['rgb'], progress=ctx.get('sub'))))
    return {'h': h, 'rough': r, 'gray': g, '_info': _info(device=_device('esr_franken'))}


@node('relief_omw', [('h', GRAY), ('n', NRM)], [('h', GRAY), ('n', NRM)],
      [('repeat_mode', 'c', 'units', REPEAT_MODES, None),
       ('repeat', 'i', 128, 16, 1024), ('units_px', 'f', 0.5, 0.01, 8.0),
       ('depth_cm', 'f', 7.0, 0.1, 30.0),
       ('ref', 'c', 'median', ('mean', 'median', 'mode', 'manual'), None),
       ('level', 'f', 0.5, 0.0, 1.0), ('relief_sync', 'b', False, 0, 1),
       ('directx', 'b', True, 0, 1)], cat='openmw', opt=('n',))
def _n_relief_omw(i, p, ctx):
    """Relief for OpenMW: the height brought into the range the game shows,
    and the normal of exactly that relief.

    Output 'h' — the height laid into the parallax window as the alpha of
    _nh carries it (matgen.parallax_alpha, without dithering): 0.5 is the
    surface itself (the 'ref' level), the whole 0…1 is OpenMW's parallax
    range, 4 % of the repeat; the relief takes depth_cm of it, no more than
    the engine can show. Output 'n' — the normal from the real slopes of
    that relief: height in centimetres on a texture covering 'repeat' game
    units along its width (or units_px × its width, repeat_units). So the
    two agree with each other and with the parallax in the game.
    «Parallax for OpenMW» fed with this 'h' gives the same alpha (the
    mapping is linear, applying it again changes nothing). relief_sync —
    depth and repeat kept the same as in the parallax and AO nodes that
    have it ticked. directx — layout of the normal (of the input one too).

    Optional input 'n' — a ready normal (from a network, a bake, the
    game's own _n). Then the output normal is that one, with its detail
    kept, but its slopes scaled by one factor so that it agrees with the
    relief in centimetres: the factor is the least-squares fit of the
    input's slopes to the slopes of the height. If the fit comes out
    negative along Y, the input has the other green layout and is flipped.
    The readouts give the factor and how well the input agrees with the
    height (correlation of slopes, %; low means the two maps describe
    different reliefs, and the factor is unreliable).

    Readouts: depth used and the engine limit, cm; one alpha step, mm;
    share of pixels clipped; repeat, units; mean tilt, degrees; with an
    input normal also the factor, the agreement and the Y flip.
    """
    xp = X.xp(i)
    import matgen as MG
    h = xp.asarray(i['h'], xp.float32)
    k = _px(ctx)
    wcur = float(h.shape[1])
    rep = repeat_units(p, wcur / max(k, 1e-6))
    a, inf = MG.parallax_alpha(h, rep, p['depth_cm'], p['ref'], p['level'], False)
    # альфа 0…1 — это весь диапазон параллакса; высота в сантиметрах и в пикселях
    hpx = (a - 0.5) * inf['avail_cm'] * (wcur / (rep * CM_PER_UNIT))
    n = MG.normal_from_heightmap(hpx, bool(p['directx']))
    extra = {}
    nin = i.get('n')
    if nin is not None:
        nin = xp.asarray(nin, xp.float32)[..., :3]
        if tuple(nin.shape[:2]) != tuple(h.shape):
            # нормаль другого размера — под размер высоты (с переносом через край)
            from texio import resize_wrap
            r = resize_wrap(np.asarray(X.to_host(nin), np.float32), h.shape[0], h.shape[1])
            nin = X.to_dev(r) if X.is_dev(h) else r
        # уклоны: у нормали (nx, ny, nz) уклон −nx/nz, −ny/nz; у почти
        # отвесных мест (nz → 0) уклон уходит в бесконечность — их не берём
        zi = xp.maximum(nin[..., 2], 0.05)
        sx_i, sy_i = nin[..., 0] / zi, nin[..., 1] / zi
        sx_h, sy_h = n[..., 0] / n[..., 2], n[..., 1] / n[..., 2]
        # знак по Y: другой формат зелёного даёт отрицательное совпадение
        flip = float((sy_i * sy_h).sum()) < 0 < float((sx_i * sx_h).sum())
        if flip:
            sy_i = -sy_i
        # один множитель на обе оси — наименьшие квадраты
        num = float((sx_i * sx_h).sum() + (sy_i * sy_h).sum())
        den = float((sx_i * sx_i).sum() + (sy_i * sy_i).sum())
        hh = float((sx_h * sx_h).sum() + (sy_h * sy_h).sum())
        gain = max(num / den, 0.0) if den > 1e-12 else 0.0
        corr = num / math.sqrt(den * hh) if den > 1e-12 and hh > 1e-12 else 0.0
        m = xp.stack([sx_i * gain, sy_i * gain, xp.ones_like(sx_i)], -1)
        n = (m / xp.sqrt((m * m).sum(-1, keepdims=True))).astype(xp.float32)
        extra = dict(n_gain=gain, n_corr=corr * 100.0, n_flip=flip)
    tilt = float(xp.degrees(xp.arccos(xp.clip(n[..., 2], -1, 1))).mean())
    out = {'h': a, 'n': n,
           '_info': _info(depth_cm_used=inf['depth_cm'], avail_cm=inf['avail_cm'],
                          step_mm=inf['step_mm'], clipped_pc=inf['clipped'] * 100.0,
                          repeat_u=rep, tilt_mean=tilt, **extra)}
    # нормаль и высота про разный рельеф — множитель ненадёжен, сказать об этом
    if extra and extra['n_corr'] < 50.0:
        out['_warn'] = 'relief_omw.n_mismatch'
    return out


@node('parallax_nh', [('n', NRM), ('h', GRAY)], [('nh', ANY)],
      [('repeat_mode', 'c', 'units', REPEAT_MODES, None),
       ('repeat', 'i', 128, 16, 1024), ('units_px', 'f', 0.5, 0.01, 8.0),
       ('depth', 'f', 7.0, 0.1, 30.0), ('relief_sync', 'b', False, 0, 1),
       ('ref', 'c', 'median', ('mean', 'median', 'mode', 'manual'), None),
       ('level', 'f', 0.5, 0.0, 1.0), ('dither', 'b', True, 0, 1)], cat='openmw')
def _n_parallax_nh(i, p, ctx):
    """Packing for OpenMW: normal in RGB, height in alpha with the neutral at 128.

    See matgen.parallax_alpha. The output is RGBA 0…1; the output node writes it
    as _nh. OpenMW's parallax shader reads the height from the alpha of the _nh
    texture and shifts the texture coordinates by eyeDir.xy·(a·0.04 − 0.02), so
    alpha 128 means "exactly at the geometry" and the usable depth follows from
    the texture's size on the model.

    repeat — game units covered by one repeat of the texture (16…1024); depth —
    full relief depth in centimetres (0.1…30); ref — which height goes to 128:
    'mean', 'median', 'mode' or 'manual' (then level, 0…1 from bottom to top of
    the relief); dither — 8×8 ordered noise of 1/255 before 8-bit quantisation.
    repeat_mode 'per_px' — the repeat is units_px × the full-size width
    instead (repeat_units). relief_sync — depth and repeat kept the same as
    in the AO nodes that have it ticked (the editor copies them).

    Output 'nh': H×W×4 float32 0…1 (RGB = n·0.5 + 0.5, clipped). Readout:
    the repeat in units.
    """
    xp = X.xp(i)
    import matgen as MG
    rep = repeat_units(p, i['h'].shape[1] / max(_px(ctx), 1e-6))
    a, _ = MG.parallax_alpha(i['h'], rep, p['depth'], p['ref'], p['level'],
                             bool(p['dither']))
    n = i['n']
    return {'nh': xp.concatenate([xp.clip(n * 0.5 + 0.5, 0, 1), a[..., None]], -1),
            '_info': _info(repeat_u=rep)}


@node('ao', [('h', GRAY), ('alpha', GRAY), ('n', NRM)], [('ao', GRAY)],
      [('method', 'c', 'cavity', ('cavity', 'lsao', 'gtao'), None),
       ('units', 'c', 'pct', ('pct', 'px', 'cm'), None),
       ('radius', 'i', 28, 4, 120), ('radius_pct', 'f', 5.5, 0.2, 25.0),
       ('radius_cm', 'f', 10.0, 0.1, 200.0),
       ('strength', 'f', 0.9, 0.0, 2.0),
       ('depth', 'f', 80.0, 1.0, 200.0), ('depth_pct', 'f', 15.0, 0.2, 40.0),
       ('depth_cm', 'f', 7.0, 0.1, 30.0),
       ('repeat_mode', 'c', 'units', REPEAT_MODES, None),
       ('repeat', 'i', 128, 16, 1024), ('units_px', 'f', 0.5, 0.01, 8.0),
       ('relief_sync', 'b', False, 0, 1),
       ('dirs', 'i', 16, 4, 32), ('falloff', 'b', False, 0, 1),
       ('n_depth', 'b', True, 0, 1),
       ('gamma', 'f', 1.0, 0.2, 3.0), ('contrast', 'f', 1.0, 0.2, 3.0),
       ('alpha_cut', 'f', 0.5, 0.01, 0.99), ('dir_amount', 'f', 0.0, 0.0, 1.0),
       ('azimuth', 'i', 90, 0, 359), ('spread', 'i', 60, 10, 180)],
      cat='maps', opt=('alpha', 'n'))
def _n_ao(i, p, ctx):
    """Ambient occlusion from height (matgen.cavity_ao), then a tone curve.

    The height is clipped to 0…1 and scaled ×20 first. radius — blur sigma of the
    largest scale in pixels (4…120); strength — darkening (0…2); gamma, contrast —
    0.2…3 (matgen.tone). Output 'ao': H×W 0…1, 1 = fully open.

    method 'lsao' — line-sweep ambient obscurance instead
    (imgfilters.line_sweep_ao): a true horizon search in ``dirs``
    directions over a height of ``depth`` px (height 0…1 × depth, px of the
    full-size texture); radius is then the falloff distance, strength
    scales the occlusion. Unlike 'cavity' the result depends on the real
    relief depth, not on the spread of the height map.

    units 'pct' (default) — radius and depth are radius_pct / depth_pct
    per cent of the texture's larger side, so the same project gives the
    same look at any texture size; 'px' — radius / depth in pixels of the
    full-size texture (projects made before the switch keep 'px'); 'cm' —
    radius_cm / depth_cm in centimetres of the game world, the texture
    covering 'repeat' game units along its width (or units_px × its width
    with repeat_mode 'per_px', see repeat_units); relief_sync keeps depth
    and repeat the same as in «Parallax for OpenMW» nodes with it ticked.

    falloff (LSAO, GTAO) — off by default, as in physics: every occluder
    counts however far it is (the sweep runs a whole turn round the tile);
    on — occluders weaken with distance on the scale of 'radius' (ambient
    obscurance, an artistic choice, not physics).

    'n' (optional, LSAO and GTAO) — the normal map of the same relief.
    With n_depth on, the depth is not taken from the slider but found from
    it (imgfilters.height_scale: the height scale at which the slopes of
    the height match the normal's, either layout of green); a fit under
    10 % means the normal is of another relief, and the slider is used. For
    GTAO the surface tilt is then taken from the normal too (its fine
    detail), with the layout the fit found. Readouts: the depth used (% of
    the larger side and px) and how well the normal fits.

    method 'gtao' — ground-truth AO (imgfilters.gtao) with the same horizon
    sweep: the cosine-weighted share of sky light reaching the surface,
    taking its own tilt into account; depth, radius, dirs as for 'lsao'.

    'alpha' (optional) — the texture's alpha, e.g. the source's 'a'. Pixels
    with alpha below alpha_cut are background: they are left out of the
    surroundings and get AO 1, so leaves and grass get no halo.

    dir_amount — how much the sky light comes from one side: the occlusion
    of light from ``azimuth`` (degrees, 0 — right, 90 — top), spread over
    ±``spread`` degrees, is multiplied in (imgfilters.directional_occlusion,
    a horizon search over the same height — scaled ×20 for 'cavity', by
    ``depth`` for 'lsao').
    """
    xp = X.xp(i)
    import matgen as MG
    k = _px(ctx)
    mask = None
    if i.get('alpha') is not None:
        mask = (i['alpha'] >= float(p['alpha_cut'])).astype(xp.float32)
    if p.get('units') == 'pct':
        # проценты от большей стороны — сразу в пикселях того, что считаем
        side = float(max(i['h'].shape[:2]))
        rpx = float(p['radius_pct']) / 100.0 * side
        dpx = float(p['depth_pct']) / 100.0 * side
    elif p.get('units') == 'cm':
        # сантиметры мира: повтор текстуры по ширине — repeat юнитов
        wcur = float(i['h'].shape[1])
        rep = repeat_units(p, wcur / max(k, 1e-6))
        px_per_cm = wcur / (rep * CM_PER_UNIT)
        rpx = float(p['radius_cm']) * px_per_cm
        dpx = float(p['depth_cm']) * px_per_cm
    else:
        rpx, dpx = float(p['radius']) * k, float(p['depth']) * k
    info = {}
    nin = i.get('n')
    nrm = None
    if nin is not None and p.get('method') in ('lsao', 'gtao'):
        import imgfilters as F
        if tuple(nin.shape[:2]) != tuple(i['h'].shape[:2]):
            raise ValueError(t_('ao.nsize'))
        n3 = xp.asarray(nin[..., :3], xp.float32)
        hc = xp.clip(i['h'], 0, 1)
        hs, sx, sy, fit = F.height_scale(n3, hc)
        info['n_fit'] = fit * 100.0
        if fit >= 0.1:
            # масштаб высоты, при котором её склоны совпадают с наклонами
            # нормали: hs = hc·k, отсюда k
            kn = float((hs * hc).sum()) / max(float((hc * hc).sum()), 1e-12)
            if p.get('n_depth', True):
                dpx = kn
            # наклон точки для GTAO — из самой нормали, в раскладке, что нашла подгонка
            nz = xp.maximum(n3[..., 2], 0.05)
            tx, ty = sx * n3[..., 0] / nz, sy * n3[..., 1] / nz
            inv = 1.0 / xp.sqrt(tx * tx + ty * ty + 1.0)
            nrm = xp.stack([tx * inv, ty * inv, inv], -1)
        else:
            info['n_bad'] = t_('ao.nbad')
    if p.get('method') in ('lsao', 'gtao'):
        side = float(max(i['h'].shape[:2]))
        info['depth_used'] = dpx / side * 100.0
        info['depth_px'] = dpx / max(k, 1e-6)
        if p.get('units') == 'cm':
            info['depth_cm_used'] = dpx / px_per_cm
            info['repeat_u'] = rep
    if p.get('method') in ('lsao', 'gtao'):
        import imgfilters as F
        # без спада — как в физике: затенитель считается на любом расстоянии
        rr = rpx if p.get('falloff') else 1e6
        if p['method'] == 'lsao':
            lo = F.line_sweep_ao(xp.clip(i['h'], 0, 1) * dpx, rr, int(p['dirs']), mask=mask)
        else:
            lo = F.gtao(xp.clip(i['h'], 0, 1) * dpx, rr, int(p['dirs']), mask=mask,
                        normal=nrm)
        ao = xp.clip(1.0 - float(p['strength']) * (1.0 - lo), 0.0, 1.0)
    else:
        ao = MG.cavity_ao(xp.clip(i['h'], 0, 1) * 20.0 * k, rpx, p['strength'], mask=mask)
    ao = MG.tone(ao, p['gamma'], p['contrast'])
    amt = float(p.get('dir_amount', 0.0))
    if amt > 0:
        # свет неба в основном с одной стороны: тень от горизонта в ту сторону
        import imgfilters as F
        # глубина та же, что у основного AO: у LSAO — своя настройка
        dep = dpx if p.get('method') in ('lsao', 'gtao') else 20.0 * k
        d = F.directional_occlusion(xp.clip(i['h'], 0, 1) * dep, p['azimuth'],
                                    rpx * 1.5, spread=p['spread'])
        ao = ao * (1.0 - amt + amt * d)
    if mask is not None:
        ao = ao * mask + (1.0 - mask)      # кривая не должна серить фон
    return {'ao': ao.astype(xp.float32), '_info': _info(**info)}


@node('bent', [('n', NRM), ('h', GRAY)], [('n', NRM), ('bent', NRM)],
      [('amount', 'f', 0.6, 0.0, 1.0), ('radius', 'i', 28, 4, 120),
       ('directx', 'b', True, 0, 1)], cat='normal', opt=('n',))
def _n_bent(i, p, ctx):
    """Bent normal from height (matgen.bent_from_height): the middle of the
    open part of the sky above each point.

    radius — search distance, 4…120 px; the height scale is fixed at 12.
    Outputs: 'bent' — the bent normal itself, for engines that read a
    separate bent-normal map (Unreal, Unity HDRP), in the layout given by
    directx; 'n' — the incoming normal with the bent one blended in by
    amount (for engines without such a map; no longer purely geometric, so
    keep an unbent copy for parallax). 'n' may stay unconnected — then only
    'bent' is produced.
    """
    xp = X.xp(i)
    import matgen as MG
    k = _px(ctx)
    b = MG.bent_from_height(xp.clip(i['h'], 0, 1), float(p['radius']) * k, 12.0 * k)
    dx = bool(p['directx'])
    pure = b if dx else b * xp.asarray([1.0, -1.0, 1.0], xp.float32)
    n = i.get('n')
    return {'n': MG.bend_normal(n, b, p['amount'], dx) if n is not None else None,
            'bent': pure.astype(xp.float32)}


# Материал текстуры для узлов блика и PBR: один или несколько материалов из
# materials.MATERIALS по шкале блеска (параметр 'mats', правится палитрой в
# инспекторе), или, если не выбран ни один, свои числа ползунками; и имя
# текстуры, если включено «по имени».
import materials as MAT

MAT_PARAMS = [
    # общий материал у узлов «PBR», «PBR → OpenMW», «Отражение» с этой галкой:
    # правка в одном копируется в другие (editor.Editor.sync_mat)
    ('mat_sync', 'b', False, 0, 1),
    ('mats', 'd', [], None, None),          # [[ключ, доля], …] — materials.layers
    # полосы на шкале в своём порядке, а не по блеску самих материалов
    # (переставляются перетаскиванием на шкале; в инспекторе отдельно не видна)
    ('mats_manual', 'b', False, 0, 1),
    ('auto_name', 'b', False, 0, 1),
    ('clip_dull', 'f', 2.0, 0.0, 40.0), ('clip_shiny', 'f', 2.0, 0.0, 40.0),
    ('f0', 'f', 0.04, 0.0, 0.2), ('f0_shiny', 'f', 0.04, 0.0, 0.2),
    ('rough_lo', 'f', 0.0, 0.0, 1.0), ('rough_hi', 'f', 1.0, 0.0, 1.0),
    ('metal_lo', 'f', 0.0, 0.0, 1.0), ('metal_hi', 'f', 1.0, 0.0, 1.0),
    ('fit', 'f', 0.0, 0.0, 1.0),
    ('rough_default', 'f', 0.7, 0.05, 1.0), ('metal_default', 'f', 0.0, 0.0, 1.0),
]


def _material_maps(i, p, ctx, xp):
    """Base colour (linear), laid roughness, metalness, per-pixel f0, AO and
    readouts for the PBR nodes (see materials.lay, materials.effective)."""
    import matgen as MG
    base = MG.srgb_to_lin(xp.clip(i['base'][..., :3], 0, 1))
    shp = base.shape[:2]
    q, found = MAT.effective(p, ctx.get('name', ''))
    rou, met, f0, info = MAT.lay(i.get('rough'), i.get('metal'), q, shp, xp)
    ao = xp.clip(i['ao'], 0, 1) if i.get('ao') is not None else xp.ones(shp, xp.float32)
    if found:
        info['materials'] = MAT.names(found)
    elif p.get('auto_name'):
        info['materials'] = t_('mat.none')     # галка стоит, а в имени ничего не нашлось
    return base, rou, met, f0, ao, info


@node('spec', [('rgb', RGB), ('ao', GRAY)],
      [('spec', RGB), ('rough', GRAY), ('metal', GRAY)],
      [pr for pr in MAT_PARAMS if pr[0] not in ('rough_default', 'metal_default')]
      + [('gloss', 'f', 0.3, 0.0, 1.0), ('gloss_var', 'f', 0.35, 0.0, 1.0),
         ('metal', 'f', 0.0, 0.0, 1.0), ('tint', 'f', 0.0, 0.0, 1.0)],
      cat='maps', opt=('ao',))
def _n_spec(i, p, ctx):
    """Specular colour, roughness and metalness estimated from the diffuse
    (matgen.material_maps), then laid into the material range.

    Inputs: 'rgb' (the de-lit diffuse) is required; 'ao' is optional and
    defaults to all ones. gloss, gloss_var, metal, tint — the estimate (see
    material_maps). The estimated roughness is then laid, by ``fit``, into
    the range of the texture's materials, from the dullest to the shiniest
    (materials.lay — as in «PBR → OpenMW»), and the specular colour is made
    from the per-pixel f0 of that range: mix(f0, base colour, metalness), in
    linear, tinted by ``tint``, × (0.35 + 0.65·AO), clipped to 0.9. With fit
    0 and no material from the name the estimate is returned as is.

    Outputs: 'spec' H×W×3, 'rough' and 'metal' H×W, all 0…1.
    """
    xp = X.xp(i)
    import matgen as MG
    rgb = i['rgb'][..., :3]
    ao = i.get('ao')
    if ao is None:
        ao = xp.ones(rgb.shape[:2], xp.float32)
    ao = xp.clip(ao, 0, 1)
    s, sh, r, m = MG.material_maps(rgb, None, ao, p['f0'], p['gloss'],
                                   p['gloss_var'], p['metal'], p['tint'])
    q, found = MAT.effective(p, ctx.get('name', ''))
    if float(q.get('fit', 0.0)) <= 0:
        out = {'spec': s, 'rough': r, 'metal': m}
        if p.get('auto_name'):
            out['_info'] = _info(materials=t_('mat.none'))
        return out
    q = dict(q, metal_default=p['metal'])
    r, m, f0, info = MAT.lay(r, None, q, rgb.shape[:2], xp)
    hist = info.pop('_hist', None)
    lin = MG.srgb_to_lin(xp.clip(rgb, 0, 1))
    m3 = m[..., None]
    spec = f0[..., None] * (1 - m3) + lin * m3
    if p['tint'] > 0:
        L = MG.luminance(lin)
        tinted = lin * f0[..., None] / xp.maximum(L[..., None], 1e-4)
        spec = spec * (1 - p['tint']) + tinted * p['tint']
    spec = xp.clip(spec * (0.35 + 0.65 * ao)[..., None], 0.0, 0.9)
    if found:
        info['materials'] = MAT.names(found)
    elif p.get('auto_name'):
        info['materials'] = t_('mat.none')
    return {'spec': MG.lin_to_srgb(spec).astype(xp.float32), 'rough': r, 'metal': m,
            '_mat_hist': hist,
            '_info': _info(rough=float(r.mean()), metal=float(m.mean()), **info)}


@node('seamless', [('h', GRAY)], [('h', GRAY)], [('band', 'i', 24, 4, 128)], cat='prepare')
def _n_seamless(i, p, ctx):
    """Make a map tile: replace an edge band with half-shifted content.

    See matgen.make_seamless.

    band — band width in pixels (4…128). Output 'h': H×W.
    """
    import matgen as MG
    return {'h': MG.make_seamless(i['h'], max(2, int(round(p['band'] * _px(ctx)))))}


# ------------------------------------------------ кэш нейросетей на диске --
CACHE_LIMIT = 2 << 30          # 2 ГБ; сверх этого удаляются самые старые


def cache_dir():
    """``~/.texelith/cache``, created on demand: results of the neural nodes."""
    import i18n
    d = os.path.join(i18n.APP_DIR, 'cache')
    os.makedirs(d, exist_ok=True)
    return d


def disk_cached(tag, arr, params, fn):
    """Return fn(), remembered on disk by the exact input picture and settings.

    A network pass takes seconds to minutes; reopening a project or running a
    batch a second time over the same folder should not repeat it. The key is
    a SHA-1 of the tag, the parameters and the bytes of the input array, so a
    changed pixel or setting means a new pass. The result may be one array or
    a dict of arrays; it is stored as .npz. When the folder grows past
    CACHE_LIMIT, the least recently used files are removed. Any error of the
    cache itself (disk full, no permission) is ignored: then fn() is simply
    computed.
    """
    a = np.ascontiguousarray(arr, np.float32)
    hsh = hashlib.sha1()
    hsh.update(repr((tag, sorted(params.items()), a.shape)).encode('utf-8'))
    hsh.update(a.tobytes())
    path = os.path.join(cache_dir(), tag + '_' + hsh.hexdigest() + '.npz')
    try:
        with np.load(path) as z:
            res = {k: z[k] for k in z.files}
        os.utime(path)
        return res['_'] if list(res) == ['_'] else res
    except Exception:
        pass
    res = fn()
    try:
        tmp = path + '.part.npz'
        if isinstance(res, dict):
            np.savez(tmp, **res)
        else:
            np.savez(tmp, _=res)
        os.replace(tmp, path)
        _prune_cache()
    except Exception:
        pass
    return res


def _prune_cache():
    """Delete the least recently used cache files above CACHE_LIMIT bytes."""
    d = cache_dir()
    files = []
    for f in os.listdir(d):
        p = os.path.join(d, f)
        try:
            st = os.stat(p)
            files.append((st.st_mtime, st.st_size, p))
        except OSError:
            pass
    total = sum(f[1] for f in files)
    for _, size, p in sorted(files):
        if total <= CACHE_LIMIT:
            break
        try:
            os.remove(p)
            total -= size
        except OSError:
            pass


def _need_model(ok, title):
    """Raise a readable error when a network's model file is missing."""
    if not ok:
        from i18n import t
        raise ValueError(t('nd.nomodel', title))


def _device(key):
    """Where a network ran: 'CUDA', 'WebGPU', 'DirectML' or 'CPU'; '—' if it
    has not run in this session (the result came from the disk cache)."""
    import depthnet as DN
    p = DN.provider_name(key)
    return (DN.provider_short(p) or 'CPU') if p else '—'


def _info(**kw):
    """Format node readouts: dict of label key → value, turned into text by the UI.

    Nodes return it under the output key '_info' as a list of (key, value)
    pairs; the i18n key 'ni.<key>' gives the label. Kept as data rather than a
    ready string so the language can change after the evaluation.
    """
    return [(k, v) for k, v in kw.items()]


# ---------------------------------------------------------- новые узлы ----
@node('deepbump', [('rgb', RGB)], [('n', NRM), ('h', GRAY)],
      [('overlap', 'f', 0.5, 0.25, 0.75), ('directx', 'b', True, 0, 1)],
      cat='neural')
def _n_deepbump(i, p, ctx):
    """Normal from the DeepBump network (deepbump.py), plus its integral.

    overlap — share of the 256-px window shared with its neighbours (more —
    smoother seams between windows, more passes); directx — layout of the
    result. The network is run on the torus, so the map tiles.

    Outputs: 'n' — H×W×3 unit vectors; 'h' — the height integrated from it
    (matgen.height_from_normal, Frankot–Chellappa), 0…1. The height of a
    network normal is good in detail and unreliable in the large form — feed
    it to 'nh_balance' together with a depth map rather than use it alone.

    Raises:
        ValueError: the model is not downloaded.
    """
    xp = X.xp(i)
    import deepbump as DB
    import matgen as MG
    if not DB.have_model():
        _need_model(False, DB.MODEL['title'])
    rgb = i['rgb']
    n = disk_cached('deepbump', rgb[..., :3], {'o': round(float(p['overlap']), 3)},
                    lambda: DB.normal(rgb, p['overlap'], True, ctx.get('sub')))
    h = MG.height_from_normal(n, True)
    info = _info(depth_px=float(xp.ptp(h)), curl=MG.slope_curl(n),
                 device=_device('deepbump'))
    if not p['directx']:
        n = n * xp.array([1, -1, 1], xp.float32)
    return {'n': n, 'h': _n01(h), '_info': info}


@node('nh_balance', [('n', NRM), ('h', GRAY)], [('h', GRAY), ('n', NRM)],
      [('balance', 'f', 0.5, 0.0, 1.0), ('keep', 'f', 0.0, 0.0, 1.0),
       ('gain', 'f', 1.0, 0.0, 4.0), ('directx', 'b', True, 0, 1)],
      cat='height')
def _n_nh_balance(i, p, ctx):
    """Make a normal and a height agree without throwing either away.

    See matgen.balance_height_normal. balance — 0: the height decides, 1: the
    normal decides; in between, the normal is trusted for detail finer than the
    crossover wavelength and the height for larger forms. keep — how much of
    the normal's own slopes to put back over the agreed ones (0: output normal
    exactly matches the output height). gain — multiplier on the fitted scale
    of the height (1: as fitted). directx — layout of the input and output
    normal.

    Outputs: 'h' — agreed height, 0…1 (min–max; its depth in pixels is in the
    readout), 'n' — agreed normal. Readouts: crossover, fitted scale and the
    correlation it rests on, depth in px, curl before/after, and how far the
    output normal is from each input.
    """
    xp = X.xp(i)
    import matgen as MG
    n, h = i['n'], xp.asarray(i['h'], xp.float32)
    lam = MG.balance_crossover(min(h.shape[:2]), p['balance'])
    Hh, nn, inf = MG.balance_height_normal(n, h, p['balance'], p['keep'],
                                           bool(p['directx']))
    if abs(float(p['gain']) - 1.0) > 1e-6:
        # подгонка даёт масштаб, gain его поправляет: пересчёт с тем же
        # равновесием, но с высотой, заранее умноженной на gain
        Hh, nn, inf = MG.balance_height_normal(n, h * float(p['gain']), p['balance'],
                                               p['keep'], bool(p['directx']))
    rng = max(float(xp.ptp(Hh)), 1e-6)
    info = _info(crossover=lam / _px(ctx), scale=inf['scale'], corr=inf['corr'],
                 depth_px=inf['depth_px'] / _px(ctx), curl_in=inf['curl_in'],
                 curl_out=inf['curl_out'], angle_n=inf['angle_n'],
                 angle_h=inf['angle_h'])
    return {'h': (Hh - float(Hh.min())) / rng, 'n': nn, '_info': info}


NMIX_MODES = ('mix', 'add', 'rnm')


@node('normal_mix', [('a', NRM), ('b', NRM)], [('n', NRM)],
      [('amount', 'f', 0.5, 0.0, 1.0), ('mode', 'c', 'mix', NMIX_MODES, None)],
      cat='normal', opt=('b',))
def _n_normal_mix(i, p, ctx):
    """Combine two normal maps of the same layout.

    mode:
      'mix' — mixing (default): the normal of the surface that lies between
          the two, (1 − amount)·A + amount·B. Done on slopes, not on vectors:
          averaging the vectors themselves flattens the result (two opposite
          tilts give a flat plate, and even equal ones come out lower). Use it
          to combine two estimates of the SAME relief — e.g. DeepBump and the
          normal from a height.
      'add' — adding: the relief of B laid on top of A with weight amount,
          A + amount·B, also on slopes. For a detail map over a base: the
          tilts add up, the result is steeper than either.
      'rnm' — the same "detail over base" by reoriented normal mapping
          (Barré-Brisebois & Hill, 2012), as in game engines; amount fades
          B toward flat first. Close to 'add'.
    Without B, A passes unchanged. Readouts: mean tilt of A, B and the result.
    Older graphs with 'linear' / 'detail' are read as 'mix' / 'rnm'.
    """
    xp = X.xp(i)
    a, b = i['a'], i.get('b')
    if b is None:
        return {'n': a}
    k = float(p['amount'])
    mode = {'linear': 'mix', 'detail': 'rnm'}.get(p['mode'], p['mode'])
    if mode in ('mix', 'add'):
        # в наклонах, как у поверхностей: z не меньше 0.05, чтобы почти
        # отвесные пиксели не давали бесконечный наклон
        za = xp.maximum(a[..., 2:3], 0.05)
        zb = xp.maximum(b[..., 2:3], 0.05)
        sa, sb = a[..., :2] / za, b[..., :2] / zb
        s = sa * (1.0 - k) + sb * k if mode == 'mix' else sa + sb * k
        m = xp.concatenate([s, xp.ones_like(za)], -1)
    else:
        bb = b * xp.array([k, k, 1.0], xp.float32)
        bb = bb / xp.sqrt((bb * bb).sum(-1, keepdims=True))
        t = a + xp.array([0, 0, 1.0], xp.float32)
        u = bb * xp.array([-1.0, -1.0, 1.0], xp.float32)
        m = t * (t * u).sum(-1, keepdims=True) / xp.maximum(t[..., 2:3], 1e-6) - u
    m = m.astype(xp.float32)
    m = m / xp.maximum(xp.sqrt((m * m).sum(-1, keepdims=True)), 1e-6)

    def tilt(v):
        return float(xp.degrees(xp.arccos(xp.clip(v[..., 2], -1, 1))).mean())
    return {'n': m, '_info': _info(tilt_a=tilt(a), tilt_b=tilt(b), tilt_out=tilt(m))}


@node('normal_fix', [('n', NRM)], [('n', NRM)],
      [('strength', 'f', 1.0, 0.0, 3.0), ('min_z', 'f', 0.0, 0.0, 0.6),
       ('flip', 'b', False, 0, 1)], cat='normal')
def _n_normal_fix(i, p, ctx):
    """Adjust a normal map: tilt strength, green flip, limit on the tilt.

    strength — multiplier on x and y before renormalising (1: unchanged);
    flip — negate green (switch OpenGL ↔ DirectX); min_z — squeeze xy so z is
    at least this (normalfix.fix); 0 turns it off. Readouts: mean tilt before
    and after, degrees.
    """
    xp = X.xp(i)
    import normalfix as NF
    n = xp.asarray(i['n'], xp.float32)
    m = n * xp.array([p['strength'], -1.0 if p['flip'] else 1.0, 1.0], xp.float32) \
        * xp.array([1.0, p['strength'], 1.0], xp.float32)
    m[..., 2] = xp.maximum(m[..., 2], 1e-4)
    m = m / xp.sqrt((m * m).sum(-1, keepdims=True))
    if p['min_z'] > 0:
        m = NF.decode(NF.fix(NF.encode(m), False, True, float(p['min_z']), False))
    tilt = lambda v: float(xp.degrees(xp.arccos(xp.clip(v[..., 2], -1, 1))).mean())
    return {'n': m.astype(xp.float32), '_info': _info(tilt_in=tilt(n), tilt_out=tilt(m))}


@node('normal_flatfix', [('n', ANY)], [('n', ANY), ('mask', GRAY)],
      [('spread', 'f', 0.5, 0.05, 5.0), ('level', 'f', 0.3, 0.05, 5.0),
       ('min_area', 'i', 30, 4, 5000), ('grow', 'i', 3, 0, 16),
       ('detail', 'b', True, 0, 1)], cat='normal')
def _n_normal_flatfix(i, p, ctx):
    """Repair flat-cut patches of a normal map (imgfilters.fix_flat_normals)
    or plateaus of a height map (imgfilters.fix_flat_height) — whichever is
    connected; for a height the flat test is ``level`` (levels of 255).

    A blown-out or crushed spot in the source leaves a network nothing to
    see, and it answers with one constant normal — a flat blob. Such
    patches are found (the direction varies by less than ``spread`` degrees
    over 5×5 px, at least ``min_area`` px, grown by ``grow`` px) and filled
    with relief taken from the place whose surroundings match best (detail
    on: a seamless patch, as the auto patch does) or with a smooth
    continuation of the tilt round them (detail off). Outputs: 'n', and
    'mask' — what was filled. Readout: the filled share, %. On the card or
    the processor (the membranes are solved on the processor).
    """
    import imgfilters as F
    xp = X.xp(i)
    k = _px(ctx)
    n = xp.asarray(i['n'], xp.float32)
    area, grow = max(4, int(round(p['min_area'] * k * k))), int(round(p['grow'] * k))
    if n.ndim == 2:
        # высота: плато — там, где она не меняется (порог в уровнях из 255)
        out, m = F.fix_flat_height(n, float(p['level']) / 255.0, area, grow, bool(p['detail']))
    else:
        out, m = F.fix_flat_normals(n, p['spread'], area, grow, bool(p['detail']))
    return {'n': out, 'mask': m.astype(xp.float32),
            '_info': _info(filled=100.0 * float(m.mean()))}


@node('shadows', [('h', GRAY)], [('light', GRAY), ('mask', GRAY)],
      [('azimuth', 'i', 135, 0, 359), ('elevation', 'i', 35, 1, 89),
       ('depth', 'f', 20.0, 1.0, 200.0), ('softness', 'f', 4.0, 0.5, 30.0),
       ('radius', 'i', 64, 4, 400)], cat='maps')
def _n_shadows(i, p, ctx):
    """Cast shadows of the relief in screen space (imgfilters.light_shadow):
    a far light from ``azimuth`` (degrees, 0 — right, 90 — top) at
    ``elevation`` above the surface; the height 0…1 is ``depth`` px tall;
    the search goes ``radius`` px towards the light; softness — the light's
    angular size, degrees. Outputs: 'light' — 1 lit, 0 in shadow (for
    «Remove shading»); 'mask' — the opposite, 1 in shadow. Readout: share in
    shadow, %.
    """
    import imgfilters as F
    xp = X.xp(i)
    k = _px(ctx)
    lit = F.light_shadow(xp.clip(i['h'], 0, 1) * float(p['depth']) * k, p['azimuth'],
                         p['elevation'], float(p['radius']) * k, p['softness'])
    return {'light': lit, 'mask': (1.0 - lit).astype(xp.float32),
            '_info': _info(in_shadow=100.0 * float((lit < 0.5).mean()))}


@node('unshade', [('rgb', RGB), ('shade', GRAY)], [('rgb', RGB)],
      [('strength', 'f', 1.0, 0.0, 2.0), ('floor', 'f', 0.25, 0.05, 1.0),
       ('auto', 'b', True, 0, 1), ('no_halo', 'b', True, 0, 1)], cat='prepare')
def _n_unshade(i, p, ctx):
    """Remove baked shading from the picture (imgfilters.remove_shading):
    connect an AO map or the 'light' output of «Shadows» to 'shade' (1 = lit).
    In linear light the colour is divided by shade^strength; floor — the
    least divisor, so pitch-black spots do not blow up into noise.

    auto: the strength is found from the data (imgfilters.shade_strength —
    how much the picture really darkens where the shade map says so), and
    'strength' multiplies it (1 — as found). no_halo: no pixel is made
    brighter than the lit places of its own colour nearby
    (imgfilters.halo_guard, window 3 % of the size) — against the light
    rims where the shade map darkens what the picture does not. Readouts:
    mean change in steps of 255, the strength found, the share of pixels
    held back.
    """
    import imgfilters as F
    xp = X.xp(i)
    rgb = i['rgb']
    shade = i['shade']
    if tuple(shade.shape[:2]) != tuple(rgb.shape[:2]):
        raise ValueError(t_('unshade.size'))
    s = float(p['strength'])
    info = {}
    if p['auto']:
        s0, used = F.shade_strength(rgb, shade, p['floor'], 4.0 * _px(ctx))
        info['auto_used'] = used * 100.0
        if s0 is None:
            info['auto_s'] = t_('unshade.noauto')
        else:
            info['auto_s'] = s0
            s = s0 * s
    out = F.remove_shading(rgb, shade, s, p['floor'])
    if p['no_halo']:
        out, held = F.halo_guard(rgb, out, max(2.0, 0.03 * max(rgb.shape[:2])))
        info['halo_px'] = held * 100.0
    if rgb.shape[-1] > 3:
        out = xp.concatenate([out, rgb[..., 3:]], -1)
    return {'rgb': out, '_info': _info(steps=float(xp.abs(out[..., :3] - rgb[..., :3]).mean()) * 255,
                                       **info)}


def _unshade_upgrade(old):
    """Projects made before 'auto' and 'no_halo' keep working as they did."""
    old.setdefault('auto', False)
    old.setdefault('no_halo', False)
    return old


REG['unshade'].upgrade = _unshade_upgrade


@node('light_back', [('rgb', RGB), ('albedo', RGB)], [('light', GRAY), ('light_rgb', RGB)],
      [('floor', 'f', 0.03, 0.005, 0.3), ('smooth', 'f', 1.5, 0.0, 30.0),
       ('norm', 'c', 'lit1', ('lit1', 'avg05', 'raw'), None)], cat='prepare')
def _n_light_back(i, p, ctx):
    """The light that was taken off the picture: original ÷ albedo, in
    linear light (e.g. the source and SuperMat's albedo).

    'light' — by brightness, 'light_rgb' — per channel (the tint of the
    light). Both are a ratio of blurs, blur(original) / blur(albedo), with
    the blur ``smooth`` px of the full-size texture, so single noisy pixels
    of the albedo do not turn into spikes; floor — the least albedo
    brightness divided by (linear), so black spots do not blow up. norm —
    how the ratio is brought to 0…1: 'lit1' — the most lit 1 % become 1 (like
    an AO map: 1 = lit, ready for «Remove shading»), 'avg05' — the average
    becomes 0.5, 'raw' — the ratio as it is. Beyond 1 (glare, highlights
    the network did not keep) it is bent in softly. If the albedo has
    another size it is brought to the original's. Readouts: the scale of
    the ratio before norm (how much brighter or darker the network made
    the albedo) and the share above the 0…1 range.
    """
    import matgen as MG
    xp = X.xp(i)
    o = i['rgb'][..., :3]
    a = i['albedo'][..., :3]
    if tuple(a.shape[:2]) != tuple(o.shape[:2]):
        from texio import resize_wrap
        a = X.to_dev(resize_wrap(np.asarray(X.to_host(a), np.float32),
                                 o.shape[0], o.shape[1]).astype(np.float32)) \
            if X.is_dev(o) else resize_wrap(np.asarray(a, np.float32), o.shape[0], o.shape[1])
    lo, la = MG.srgb_to_lin(o), MG.srgb_to_lin(a)
    r = float(p['smooth']) * _px(ctx)

    def bl(x):
        return MG.fast_blur(xp.ascontiguousarray(x), r, mode='wrap') if r >= 0.3 else x
    fl = float(p['floor'])
    lum = bl(MG.luminance(lo)) / xp.maximum(bl(MG.luminance(la)), fl)
    rgb = xp.stack([bl(lo[..., c]) / xp.maximum(bl(la[..., c]), fl) for c in range(3)], -1)
    if p['norm'] == 'lit1':
        k = 1.0 / max(float(xp.percentile(lum, 99)), 1e-6)
    elif p['norm'] == 'avg05':
        k = 0.5 / max(float(lum.mean()), 1e-6)
    else:
        k = 1.0
    scale = float(xp.median(lum))
    lum, rgb = lum * k, rgb * k
    over = float((lum > 1.0).mean()) * 100.0
    return {'light': MG.soft01(lum).astype(xp.float32),
            'light_rgb': MG.soft01(rgb).astype(xp.float32),
            '_info': _info(ratio=scale, beyond='%.1f %%' % over)}


@node('relief', [('rgb', RGB), ('s', GRAY)], [('rgb', RGB)],
      [('mode', 'c', 'overlay', ('overlay', 'soft', 'multiply', 'burn'), None),
       ('amount', 'f', 1.0, 0.0, 3.0), ('occl', 'b', False, 0, 1),
       ('sat_keep', 'f', 1.0, 0.0, 1.0)], cat='maps')
def _n_relief(i, p, ctx):
    """Lay relief into the diffuse, as in Photoshop (matgen.relief_into_diffuse).

    s — a height (high is lighter) or, with occl on, an AO map (1 = open).
    mode — overlay, soft light, multiply (linear light, physical) or linear
    burn. amount — strength in standard deviations of the source. Readout:
    mean change in 0…255 steps.
    """
    xp = X.xp(i)
    import matgen as MG
    d = i['rgb'][..., :3]
    out = MG.relief_into_diffuse(d, xp.clip(i['s'], 0, 1), p['mode'], p['amount'],
                                 None, 1.0, 1.0, p['sat_keep'], bool(p['occl']))
    return {'rgb': out, '_info': _info(steps=float(xp.abs(out - d).mean() * 255))}


@node('pbr_openmw', [('base', RGB), ('metal', GRAY), ('rough', GRAY), ('ao', GRAY)],
      [('diffuse', RGB), ('spec', ANY), ('gloss', RGB)],
      MAT_PARAMS + [('metal_dark', 'f', 0.7, 0.0, 1.0), ('ao_diffuse', 'f', 1.0, 0.0, 2.0),
                    ('ao_spec', 'f', 1.0, 0.0, 2.0), ('spec_gain', 'f', 1.0, 0.0, 3.0),
                    ('env_rough', 'f', 1.0, 0.0, 1.0)],
      cat='openmw', opt=('metal', 'rough', 'ao'))
def _n_pbr_openmw(i, p, ctx):
    """Turn PBR maps into what OpenMW reads: a diffuse and a _spec map.

    OpenMW has no PBR shader. Its object shader takes the specular colour from
    the RGB of _spec and the Blinn-Phong exponent from its alpha
    (``shininess = specTex.a * 255.0``, files/shaders/compatibility/
    objects.frag). So:

      * specular colour = mix(f0 grey, base colour, metalness), in linear,
        times AO^ao_spec (crevices do not shine) times spec_gain;
      * exponent = 2/α² − 2 with α = roughness² (the usual Beckmann ↔
        Blinn-Phong match), clipped to 1…255, stored as alpha / 255;
      * OpenMW's specular term is not energy-normalised, so the colour is
        multiplied by the Blinn-Phong normalisation (n + 8) / 8π, capped at
        1 — rough surfaces get a faint broad sheen, glossy ones keep the full
        colour;
      * OpenMW adds the specular to the lit diffuse in display (gamma)
        space, with no conversion. So the map stores not srgb(specular) but
        the display-space increment srgb(d + specular) − srgb(d) (both
        linear), d being the diffuse blurred over 2 % of the size: on screen
        that gives what linear light would, and the specular does not follow
        the pixel detail of the diffuse (dark gaps between threads would
        otherwise get several times more specular and glitter);
      * diffuse = base colour × (1 − metal_dark · metalness) × AO^ao_diffuse,
        in linear: a metal has almost no diffuse; metal_dark < 1 keeps some,
        since vanilla lighting has no reflections to carry the colour.

    Material range (materials.lay): the incoming rough and metal maps are
    laid, by ``fit``, into the range of the materials the texture is made
    of — 'mats', one or several, each in its own band of the shininess
    scale, f0 included (with none chosen — the sliders); clip_dull /
    clip_shiny — the extreme pixels left out when the spread is found;
    with auto_name the materials named in the texture file name are used
    instead. Unconnected maps: the middle of the main material (or
    rough_default, metal_default); no AO — no occlusion. Readouts: mean
    metalness, roughness, exponent, the incoming maps' spread and the
    materials recognised by name.

    Output 'gloss' — the reflection mask for Morrowind environment maps:
    the gloss map of a NIF (NiTexturingProperty, gloss slot), by which
    OpenMW multiplies the reflection of the environment map
    (objects.frag: envEffect *= texture2D(glossMap, …).xyz; vanilla
    Morrowind reads it only with the Morrowind Code Patch option). The
    reflected share at head-on view is the same specular colour f0 → base
    colour by metalness, dimmed by AO; env_rough — how much roughness dims
    it, (1 − rough)² at 1: an environment map cannot blur, so a rough
    surface showing a sharp mirror image would look like chrome. Like the
    specular, the reflection is added to the picture in display space, so
    the map stores the display-space increment srgb(d + r) − srgb(d) for a
    white environment (d — the smoothed diffuse, as for _spec): a metal
    (d ≈ 0) gets srgb(r), a dull plastic only a few steps. Readout: mean
    gloss in steps of 255.
    """
    xp = X.xp(i)
    import matgen as MG
    base, rou, met, f0, ao, info = _material_maps(i, p, ctx, xp)
    m3 = met[..., None]
    spec = (f0[..., None] * (1 - m3) + base * m3) * (ao ** p['ao_spec'])[..., None] * p['spec_gain']
    a = xp.maximum(rou * rou, 0.02)
    expo = xp.clip(2.0 / (a * a) - 2.0, 1.0, 255.0)
    spec = spec * xp.clip((expo + 8.0) / (8.0 * xp.pi), 0.0, 1.0)[..., None]
    dif = xp.clip(base * (1.0 - p['metal_dark'] * m3) * (ao ** p['ao_diffuse'])[..., None], 0, 1)
    # OpenMW складывает блик с диффузом в гамма-пространстве, без перевода в
    # линейное (objects.frag: gl_FragData.xyz += matSpec · getSpecular). Чтобы
    # на экране вышло то же, что физически — srgb(диффуз + блик) в линейном, —
    # в карту пишется именно эта прибавка: srgb(d + s) − srgb(d). Запись
    # srgb(s), как было раньше, завышала тусклый блик в 4…8 раз: ткань блестела.
    # Прибавку считаем от сглаженного диффуза, а не от каждого пикселя: иначе
    # тёмные щели между нитями получали блик в разы сильнее светлых нитей, и
    # ткань в игре искрилась. Физически блик от цвета нити не зависит
    dif_g = MG.lin_to_srgb(dif)
    r_s = max(2.0, 0.02 * min(dif.shape[:2]))
    dif_s = xp.stack([MG.fast_blur(xp.ascontiguousarray(dif[..., c]), r_s, mode='wrap')
                      for c in range(3)], -1)
    spec_g = xp.clip(MG.lin_to_srgb(xp.clip(dif_s + spec, 0, 1)) - MG.lin_to_srgb(dif_s), 0, 1)
    # маска отражения для карты окружения (gloss в NIF): та же доля отражения
    # при взгляде в лоб, без нормировки Блинна-Фонга — у карты окружения её нет
    sm = 1.0 - float(p.get('env_rough', 1.0)) * (1.0 - (1.0 - rou) ** 2)
    refl = (f0[..., None] * (1 - m3) + base * m3) * (sm * ao ** p['ao_spec'])[..., None]
    gloss = xp.clip(MG.lin_to_srgb(xp.clip(dif_s + refl, 0, 1)) - MG.lin_to_srgb(dif_s), 0, 1)
    out_spec = xp.concatenate([spec_g, (expo / 255.0)[..., None]], -1).astype(xp.float32)
    hist = info.pop('_hist', None)              # гистограмма блеска — для шкалы в инспекторе
    return {'diffuse': xp.clip(dif_g, 0, 1).astype(xp.float32),
            'spec': out_spec, 'gloss': gloss.astype(xp.float32), '_mat_hist': hist,
            '_info': _info(metal=float(met.mean()), rough=float(rou.mean()),
                           exponent=float(expo.mean()),
                           spec_steps=float(spec_g.mean()) * 255.0,
                           gloss_steps=float(gloss.mean()) * 255.0, **info)}


@node('pbr', [('base', RGB), ('metal', GRAY), ('rough', GRAY), ('ao', GRAY)],
      [('base', RGB), ('rough', GRAY), ('metal', GRAY), ('ao', GRAY), ('orm', RGB),
       ('smooth', GRAY), ('ms', ANY), ('specular', RGB), ('spec_diffuse', RGB)],
      MAT_PARAMS, cat='output', opt=('metal', 'rough', 'ao'))
def _n_pbr(i, p, ctx):
    """Standard PBR maps for any engine, with the same material range as
    «PBR → OpenMW» (materials.lay).

    Outputs: 'base' — base colour (sRGB); 'rough', 'metal', 'ao'; 'orm' —
    packed R = AO, G = roughness, B = metalness (glTF, Unreal); 'smooth' —
    1 − roughness (Unity); 'ms' — RGBA with metalness in R and smoothness in
    A (Unity's metallic map); 'specular' and 'spec_diffuse' — the
    specular-glossiness workflow: specular colour = mix(f0, base, metal),
    diffuse = base × (1 − metal); gloss there is 'smooth'.
    """
    xp = X.xp(i)
    import matgen as MG
    base, rou, met, f0, ao, info = _material_maps(i, p, ctx, xp)
    m3 = met[..., None]
    sm = (1.0 - rou).astype(xp.float32)
    zero = xp.zeros_like(rou)
    spec = MG.lin_to_srgb(xp.clip(f0[..., None] * (1 - m3) + base * m3, 0, 1))
    sdif = MG.lin_to_srgb(xp.clip(base * (1 - m3), 0, 1))
    hist = info.pop('_hist', None)
    return {'_mat_hist': hist, 'base': xp.clip(i['base'][..., :3], 0, 1).astype(xp.float32), 'rough': rou,
            'metal': met, 'ao': ao.astype(xp.float32),
            'orm': xp.stack([ao, rou, met], -1).astype(xp.float32), 'smooth': sm,
            'ms': xp.stack([met, zero, zero, sm], -1).astype(xp.float32),
            'specular': xp.clip(spec, 0, 1).astype(xp.float32),
            'spec_diffuse': xp.clip(sdif, 0, 1).astype(xp.float32),
            '_info': _info(metal=float(met.mean()), rough=float(rou.mean()), **info)}


for _k in ('pbr_openmw', 'pbr', 'spec'):
    REG[_k].preset_fn = MAT.preset_update


def _mat_upgrade(old):
    """Projects with the two lists «dullest» / «shiniest» (build 24 and
    earlier): the materials named there become the 'mats' list."""
    if 'mats' not in old:
        keys = []
        for k in (old.get('mat_dull'), old.get('mat_shiny')):
            if k and k != 'custom' and k in MAT.MATERIALS and k not in keys:
                keys.append(k)
        old['mats'] = [[k, 1.0] for k in keys]
    return old


def _mat_show(params, name):
    """Inspector: 'mats_manual' has no control of its own — the material
    scale sets it (dragging a band, the «by shininess» button)."""
    return name != 'mats_manual'


for _k in ('pbr_openmw', 'pbr', 'spec'):
    REG[_k].upgrade = _mat_upgrade
    REG[_k].show_if = _mat_show
    REG[_k].groups = (
        ('ed.tab.materials', ('mat_sync', 'mats', 'clip_dull', 'clip_shiny', 'auto_name', 'fit')),
        ('ed.tab.ranges', ('f0', 'f0_shiny', 'rough_lo', 'rough_hi', 'metal_lo', 'metal_hi',
                           'rough_default', 'metal_default')))


@node('sss_map', [('mask', GRAY)],
      [('sss', GRAY), ('alpha', GRAY), ('color', RGB), ('rgba', ANY)],
      [('material', 'c', 'sss_marble', ('custom',) + MAT.SSS_KEYS, None),
       ('length', 'f', MAT.SSS_LENGTH['sss_marble'], 0.0, 100.0),
       ('full_at', 'f', 6.0, 0.5, 50.0),
       ('col_r', 'f', MAT.SSS_COLOR['sss_marble'][0], 0.0, 1.0),
       ('col_g', 'f', MAT.SSS_COLOR['sss_marble'][1], 0.0, 1.0),
       ('col_b', 'f', MAT.SSS_COLOR['sss_marble'][2], 0.0, 1.0)],
      cat='maps', opt=('mask',))
def _n_sss_map(i, p, ctx):
    """Subsurface scattering strength from the physics of the material.

    'length' — how far light spreads under the surface, mm
    (materials.diffusion_length; picking a material from the list sets it
    from the measured coefficients). SSS = length / full_at, capped at 1:
    full_at is the length counted as full SSS — the shader's SSS has no
    unit of its own (OpenMW-PBR adds a glow at the light/shadow boundary,
    0.1 × SSS), so this is the one free choice; the ratios between
    materials stay physical. The optional 'mask' (0…1) says where the
    material is: SSS × mask. Outputs: 'sss' and 'alpha' = 1 − SSS — what
    the alpha of _spec for OpenMW-PBR stores; 'color' — the colour of the
    scattering (col_r, col_g, col_b; picking a material sets them from the
    same measurements, materials.SSS_COLOR: which light goes furthest under
    the surface); 'rgba' — that colour with SSS in the alpha, as the _sss
    file of the foliage tab and the 3D view take it.
    """
    m = i.get('mask')
    v = min(1.0, float(p['length']) / max(float(p['full_at']), 1e-3))
    if m is None:
        shp = ctx['src'].shape[:2]
        xp = X.xp(ctx['src'])
        s = xp.full(shp, v, xp.float32)
    else:
        xp = X.xp(m)
        s = (xp.clip(m, 0, 1) * v).astype(xp.float32)
    c = xp.asarray(np.array([p['col_r'], p['col_g'], p['col_b']], np.float32))
    col = xp.broadcast_to(xp.clip(c, 0, 1), s.shape + (3,)).astype(xp.float32)
    return {'sss': s, 'alpha': (1.0 - s).astype(xp.float32), 'color': col,
            'rgba': xp.concatenate([col, s[..., None]], -1),
            '_info': _info(sss=float(s.mean()), length=float(p['length']))}


def _sss_preset(params, name, val):
    """Inspector hook: picking a material sets its diffusion length and
    the colour of its scattering."""
    if name != 'material' or val not in MAT.SSS_LENGTH:
        return None
    r, g, b = MAT.SSS_COLOR[val]
    return {'material': val, 'length': MAT.SSS_LENGTH[val],
            'col_r': r, 'col_g': g, 'col_b': b}


REG['sss_map'].preset_fn = _sss_preset


def _ao_preset(params, name, value):
    """Picking the AO method moves «Strength» to that method's own value:
    1.0 for LSAO and GTAO — the occlusion as computed, without any boost
    (GTAO then is the physical share of sky light: cosine law, no falloff);
    0.9 for 'cavity', which has no physical scale (its default)."""
    if name in ('units', 'repeat_mode'):
        return {name: value}                # инспектор перерисуется под единицы
    if name != 'method':
        return None
    return {'method': value, 'strength': {'lsao': 1.0, 'gtao': 1.0}.get(value, 0.9)}


def _ao_show(params, name):
    """Inspector: only the sliders of the chosen units (pixels or per cent);
    the horizon settings only for LSAO and GTAO."""
    u = params.get('units')
    horizon = params.get('method') in ('lsao', 'gtao')
    if name in ('depth', 'depth_pct', 'depth_cm', 'dirs', 'falloff', 'n_depth') \
            and not horizon:
        return False
    if name in ('radius', 'depth'):
        return u == 'px'
    if name in ('radius_pct', 'depth_pct'):
        return u == 'pct'
    if name in ('radius_cm', 'depth_cm', 'repeat_mode', 'relief_sync'):
        return u == 'cm'
    if name in ('repeat', 'units_px'):
        return u == 'cm' and _repeat_show(params, name)
    return True


def _repeat_show(params, name):
    """Inspector: 'repeat' in units mode, 'units_px' in per-pixel mode."""
    per = params.get('repeat_mode') == 'per_px'
    if name == 'repeat':
        return not per
    if name == 'units_px':
        return per
    return True


def _ao_upgrade(old):
    """Projects made before the units switch were in pixels, and before the
    falloff switch LSAO/GTAO always had the falloff: they stay so."""
    if old and 'units' not in old:
        old['units'] = 'px'
    if old and 'falloff' not in old:
        old['falloff'] = True           # LSAO и GTAO тогда считались со спадом
    return old


REG['ao'].preset_fn = _ao_preset
REG['parallax_nh'].show_if = _repeat_show
REG['relief_omw'].show_if = _repeat_show
REG['ao'].show_if = _ao_show
REG['ao'].upgrade = _ao_upgrade




@node('albedo_match', [('albedo', RGB), ('ref', RGB), ('n', NRM), ('h', GRAY)],
      [('rgb', RGB), ('shadow', GRAY)],
      [('light', 'f', 1.0, 0.0, 1.0), ('cast', 'f', 1.0, 0.0, 1.0),
       ('saturation', 'f', 0.5, 0.0, 1.0), ('spread', 'f', 1.0, 0.0, 1.0),
       ('detail', 'f', 0.0, 0.0, 2.0),
       ('detail_r', 'f', 2.0, 0.5, 20.0), ('dark_cut', 'f', 0.02, 0.0, 0.2),
       ('shadows', 'b', True, 0, 1)], cat='maps',
      opt=('n', 'h'))
def _n_albedo_match(i, p, ctx):
    """Fit a generated albedo (SuperMat, de-light) to the source diffuse, so
    it does not drift too far: overall lightness, colour cast and saturation
    (imgfilters.match_albedo, in OKLab), plus the source's fine detail for
    albedos that came out soft. Only overall values are matched — local
    ones would bring the baked lighting back.

    'ref' of another size is resized to the albedo's on the torus.
    Readouts: lightness difference before matching (OKLab L × 100) and the
    source/albedo saturation ratio.

    With a normal connected ('n', the same layout as anywhere in the graph —
    the signs are found from the data) lightness and cast come not from the
    means but from the relief (imgfilters.relief_fit): the baked light is
    fitted to source / albedo, and the albedo is scaled so that flat places
    keep the source's colour. 'shadows': cast shadows of the fitted light
    are masked out of the fit — over the height 'h', or, without it, over a
    height integrated from the normal (both layouts tried, the one that
    fits the normal kept). Output 'shadow' — that mask (1 = cast shadow).
    Readouts then: the gain, light azimuth (0° — from the right, 90° — from
    the top) and elevation, shadowed and fitted share of pixels.

    'spread' — the spread of lightness on a large scale brought towards the
    source's (what is seen from afar; imgfilters.match_albedo, readout
    spread_k); 'dark_cut' — black defects of the source (dark below it) are
    left out of that measurement and of the detail.
    """
    import imgfilters as F
    alb, ref = i['albedo'], i['ref']
    if ref.shape[:2] != alb.shape[:2]:
        from texio import resize_wrap
        ref = X.to_dev(resize_wrap(X.to_host(ref[..., :3]), alb.shape[0], alb.shape[1])) \
            if X.is_dev(alb) else resize_wrap(X.to_host(ref[..., :3]), alb.shape[0], alb.shape[1])
    gain = None
    xp = X.xp(alb)
    shadow = xp.zeros(alb.shape[:2], xp.float32)
    rinfo = {}
    nrm = i.get('n')
    if nrm is not None:
        if tuple(nrm.shape[:2]) != tuple(alb.shape[:2]):
            raise ValueError(t_('albedo_match.size'))
        h = i.get('h')
        if h is not None and tuple(h.shape[:2]) != tuple(alb.shape[:2]):
            raise ValueError(t_('albedo_match.size'))
        if p['shadows'] and h is None:
            import matgen as MG
            # высота из нормали: раскладка зелёного неизвестна — берём ту, что лучше
            # объясняет наклоны нормали
            best = None
            for dx in (True, False):
                hh = MG.height_from_normal(nrm, dx, 0.0)
                fit_q = F.height_scale(xp.asarray(nrm, xp.float32), hh)[3]
                if best is None or fit_q > best[0]:
                    best = (fit_q, hh)
            h = best[1]
        gain, sh, rinfo = F.relief_fit(alb, ref, nrm, h if p['shadows'] else None)
        shadow = sh.astype(xp.float32)
    out, info = F.match_albedo(alb[..., :3], ref[..., :3], p['light'], p['cast'],
                               p['saturation'], p['detail'], float(p['detail_r']) * _px(ctx),
                               gain, float(p.get('spread', 0.0)), float(p.get('dark_cut', 0.0)))
    if gain is not None:
        info.pop('light_diff', None)
        info.update(rinfo)
    return {'rgb': out, 'shadow': shadow, '_info': _info(**info)}


def _albedo_match_upgrade(old):
    """Projects made before «spread» and «dark_cut» existed keep their old
    result: both off."""
    old.setdefault('spread', 0.0)
    old.setdefault('dark_cut', 0.0)
    return old


REG['albedo_match'].upgrade = _albedo_match_upgrade


@node('foliage', [('rgb', RGB), ('a', GRAY), ('hand', GRAY), ('depth', GRAY)],
      [('n', NRM), ('h', GRAY), ('ao', GRAY), ('bent', NRM),
       ('sss', RGB), ('sss_a', GRAY), ('seg', RGB), ('branch', RGB)],
      [('mode', 'c', 'leaf_silh', ('leaf_silh', 'leaf_dense', 'leaf_lum'), None),
       ('split', 'f', 0.35, 0.0, 1.0), ('min_area', 'i', 60, 0, 1200),
       ('fold', 'f', 0.9, -1.5, 1.5), ('profile', 'f', 2.0, 1.0, 5.0),
       ('detail', 'f', 0.22, 0.0, 0.6), ('seed', 'i', 3, 1, 50), ('pad', 'i', 8, 0, 24),
       ('tilt', 'f', 0.45, 0.0, 1.5), ('thin', 'f', 3.5, 1.5, 12.0),
       ('blade', 'f', 0.0, 0.0, 1.0), ('bend', 'f', 0.10, -1.0, 1.0),
       ('clump_mid', 'f', 0.5, -1.5, 1.5), ('clump_big', 'f', 0.25, -1.5, 1.5),
       ('clump_scale', 'i', 24, 6, 90),
       ('dome_w', 'f', 0.0, -1.2, 1.2), ('dome_h', 'f', 0.0, -1.2, 1.2),
       ('directx', 'b', True, 0, 1),
       ('h_mode', 'c', 'depth', ('depth', 'normal', 'shape'), None),
       ('depth_w', 'f', 0.5, 0.0, 1.0),
       ('ao_str', 'f', 1.0, 0.0, 1.5), ('ao_spread', 'i', 12, 0, 48),
       ('ao_bulge', 'i', 6, 0, 30), ('ao_radius', 'i', 24, 4, 80),
       ('ao_tilt', 'f', 0.4, 0.0, 1.0), ('ao_gamma', 'f', 1.0, 0.2, 3.0),
       ('ao_contrast', 'f', 1.0, 0.2, 3.0),
       ('sss_str', 'f', 1.0, 0.0, 2.0), ('sss_base', 'f', 0.15, 0.0, 1.5),
       ('sss_veins', 'f', 0.9, 0.0, 2.5), ('sss_behind', 'f', 1.0, 0.0, 2.0),
       ('sss_reach', 'i', 14, 2, 60), ('sss_layers', 'f', 0.5, 0.0, 2.5),
       ('sss_pigment', 'f', 0.7, 0.0, 2.0), ('sss_edge', 'f', 0.15, 0.0, 1.5),
       ('sss_branch', 'f', 1.0, 0.0, 1.0), ('sss_tint', 'f', 0.35, 0.0, 1.0),
       ('sss_gamma', 'f', 1.0, 0.2, 3.0), ('sss_contrast', 'f', 1.0, 0.2, 3.0)],
      cat='foliage', opt=('hand', 'depth'))
def _n_foliage(i, p, ctx):
    """Foliage card (texture with alpha): the engine of the «Foliage» tabs
    (foliage.py) as one node — the leaves are cut out of the silhouette once
    and every map below is made from that one cut.

    Inputs: 'rgb' and 'a' — the picture and its alpha (the source node's
    'a'); 'hand' (optional) — the hand mask of branches, as the brush of
    the tab: white = a branch (gets rounding only, no tilt), e.g. from the
    «Paint a mask» node; 'depth' (optional) — an outside depth (light =
    closer: a depth network, a hand-made map): each leaf then takes its
    depth from it — its mean over the leaf — instead of the random draw,
    mixed as (1 − depth_w)·random + depth_w·outside (leaf_depth's net), so
    that leaves which the outside depth sees as equal still part; that one
    crown depth feeds the height, the AO and the translucency. Outputs: 'n' — per-leaf normals (Normals.run with the same settings
    and defaults as the «Foliage: normals» tab: leaf cut, shape, orientation,
    clumps — the large relief of a lush crown — and the card's overall
    convexity), layout by directx; 'h' — foliage height by h_mode: 'depth'
    — the crown depth (leaf_depth: every leaf at its own random depth, the
    same draw as the AO and the translucency, plus its bulge; steps between
    leaves are allowed — the per-leaf tilts of the normal have curl anyway,
    so the height need not be its integral); 'normal' — the part of the
    normal that is a surface (matgen.height_from_normal: the volume of
    clumps and of the card, smooth, no steps); 'shape' — as before, how deep
    inside its piece a point is (height_map: every leaf a ridge from its
    edge). 0 on the background, 0…1 over the leaves; 'ao' —
    crown occlusion (ao_map); 'bent' — the
    crown's bent normal; 'sss', 'sss_a' — translucency colour and amount;
    'seg' — how the silhouette was cut into leaves (a random colour per
    piece) and 'branch' — what counts as a leaf (green) and as a branch or
    needle (red), the "element split" and "leaves and branches" views of the
    tab, to check the cut.

    Only what has no node of its own is here: de-lighting the colour is the
    «Remove light» node. Sizes in pixels (min_area, pad, clump_scale,
    ao_spread, ao_bulge, ao_radius) are scaled for the preview copy. The cut into leaves (watershed) is on the processor; the vein filter and the horizon walk of the AO run on the video card when it is on (foliage.py). The result is kept in the disk cache.
    Raises ValueError if the alpha is empty (foliage.Source).
    """
    import foliage as FO
    import matgen as MG
    rgb = np.clip(X.to_host(i['rgb'])[..., :3], 0, 1).astype(np.float32)
    a = np.clip(X.to_host(i['a']), 0, 1).astype(np.float32)
    k = _px(ctx)
    force = None
    if i.get('hand') is not None:
        b = np.asarray(X.to_host(i['hand']), np.float32)
        if b.ndim == 3:
            b = b[..., :3].mean(-1)
        if b.shape != a.shape:
            from texio import resize_wrap
            b = resize_wrap(b, a.shape[0], a.shape[1])
        force = b >= 0.5
    net = None
    if i.get('depth') is not None:
        net = np.asarray(X.to_host(i['depth']), np.float32)
        if net.ndim == 3:
            net = net[..., :3].mean(-1)
        if net.shape != a.shape:
            from texio import resize_wrap
            net = resize_wrap(net, a.shape[0], a.shape[1])
    nw = float(p['depth_w']) if net is not None else 0.0
    q = dict(p, px=round(k, 4),
             force=0 if force is None else hash(np.packbits(force).tobytes()),
             net=0 if net is None else hash(np.round(net[::4, ::4], 3).tobytes()))

    def make():
        """All foliage maps for this picture and these settings."""
        src = FO.Source('', np.concatenate([rgb, a[..., None]], -1))
        nrm = FO.Normals(src)
        n = nrm.run(p['mode'][5:], p['split'], int(p['seed']), p['thin'], p['tilt'],
                    p['fold'], p['detail'], p['profile'], float(p['pad']) * k,
                    p['dome_w'], p['dome_h'], force, p['clump_mid'], p['clump_big'],
                    float(p['clump_scale']) * k, int(round(p['min_area'] * k * k)),
                    p['blade'], p['bend'])
        spread, rad = float(p['ao_spread']) * k, float(p['ao_radius']) * k
        bulge = float(p['ao_bulge']) * k
        occ0, bent = FO.crown_occlusion(src, nrm, None, spread, bulge, rad, int(p['seed']),
                                        p['ao_tilt'], want_bent=True, net=net, net_w=nw)
        occ = FO.crown_occlusion(src, nrm, n, spread, bulge, rad, int(p['seed']), p['ao_tilt'],
                                 net=net, net_w=nw)
        ao = FO.ao_map(src, strength=p['ao_str'], gamma=p['ao_gamma'],
                       contrast=p['ao_contrast'], occ=occ)
        z = FO.leaf_depth(nrm, src.mask, spread, bulge, int(p['seed']), net=net, net_w=nw)
        back = FO.behind_leaves(z, src.mask, float(p['sss_reach']) * k, max(spread, 1.0))
        col, amt = FO.translucency(src, nrm, p['sss_str'], p['sss_base'], p['sss_veins'],
                                   p['sss_layers'], p['sss_pigment'], p['sss_edge'],
                                   p['sss_branch'], p['sss_tint'], p['sss_gamma'],
                                   p['sss_contrast'], FO.trans_color(src, p['sss_tint']),
                                   FO.vein_map(src), occ, p['sss_behind'], back)
        msk = src.mask
        if p['h_mode'] == 'depth':
            hh = z
        elif p['h_mode'] == 'normal':
            nn = np.where(msk[..., None], n, np.array([0.0, 0.0, 1.0], np.float32))
            hh = MG.height_from_normal(nn, True)
        else:
            hh = FO.height_map(nrm)
        if p['h_mode'] != 'shape' and msk.any():
            # по листьям — в 0…1 (1-й…99-й процентиль, хвосты мягко), фон 0
            v = hh[msk]
            lo, hi = float(np.percentile(v, 1)), float(np.percentile(v, 99))
            hh = MG.soft01((hh - lo) / max(hi - lo, 1e-6)) * msk
        return {'n': n, 'h': np.asarray(hh, np.float32), 'ao': ao,
                'bent': bent, 'sss': col, 'sss_a': amt,
                'seg': nrm.segments_picture().astype(np.float32),
                'branch': nrm.branch_picture().astype(np.float32)}

    m = disk_cached('foliage_v6', np.concatenate([rgb, a[..., None]], -1), q, make)
    out = {kk: np.asarray(v, np.float32) for kk, v in m.items()}
    if not p['directx']:
        for kk in ('n', 'bent'):
            out[kk] = out[kk] * np.array([1.0, -1.0, 1.0], np.float32)
    return out


# у листвы ползунков много — в инспекторе они по вкладкам, как разделы вкладки
REG['foliage'].groups = (
    ('ed.tab.leaves', ('mode', 'split', 'min_area', 'seed')),
    ('ed.tab.shape', ('fold', 'profile', 'detail', 'pad')),
    ('ed.tab.orient', ('tilt', 'thin', 'blade', 'bend', 'directx', 'h_mode')),
    ('ed.tab.clump', ('clump_mid', 'clump_big', 'clump_scale', 'dome_w', 'dome_h')),
    ('ed.tab.ao', ('depth_w', 'ao_str', 'ao_spread', 'ao_bulge', 'ao_radius', 'ao_tilt', 'ao_gamma',
                   'ao_contrast')),
    ('ed.tab.sss', ('sss_str', 'sss_base', 'sss_veins', 'sss_behind', 'sss_reach',
                    'sss_layers', 'sss_pigment', 'sss_edge', 'sss_branch', 'sss_tint',
                    'sss_gamma', 'sss_contrast')))


_SM_RECENT = []      # [(res, отпечаток 64×64, карты res × res)], последние 6


def _supermat_recent(x, res, maps=None):
    """Memory of the last SuperMat results, matched by a small print.

    The fast preview, the fine preview and the full-size save give SuperMat
    the same texture at different sizes; brought to res × res they differ
    only by resampling noise, and the model would be run (four passes) for
    each. A 64×64 print of the res × res input is kept with the result; an
    input whose print differs by less than 1/255 on average reuses it.
    With ``maps`` given, stores them; returns the stored maps or None."""
    from texio import resize_wrap
    pr = resize_wrap(x, 64, 64)
    if maps is not None:
        _SM_RECENT.insert(0, (res, pr, maps))
        del _SM_RECENT[6:]
        return maps
    for r, q, mm in _SM_RECENT:
        if r == res and float(np.abs(q - pr).mean()) < 1.0 / 255.0:
            return mm
    return None


@node('supermat', [('rgb', RGB)],
      [('base', RGB), ('rough', GRAY), ('metal', GRAY)],
      [('res', 'c', 512, (512, 768, 1024), None)],
      cat='neural')
def _n_supermat(i, p, ctx):
    """PBR maps from SuperMat (MIT), run in its own environment.

    See supermat_bridge.py: SuperMat is installed separately, Texelith only
    starts it. res — the square size the model works at (it was trained at
    512); the texture is resized to it and the maps back, on the torus.

    Outputs: 'base' — base colour without lighting, sRGB 0…1; 'rough',
    'metal' — 0…1. Seamless (four passes rolled by half a period, see
    supermat_worker.py). SuperMat gives no normal and no height: take them
    from DeepBump, a depth network or the height nodes.

    Raises:
        ValueError: SuperMat is not set up (message: what is missing).
    """
    import supermat_bridge as SB
    bad = SB.problem()
    if bad:
        from i18n import t
        raise ValueError('SuperMat: ' + t(bad))
    from texio import resize_wrap
    rgb = np.clip(X.to_host(i['rgb'])[..., :3], 0, 1).astype(np.float32)
    H, W = rgb.shape[:2]
    res = int(p['res'])
    # модель всё равно работает на res × res, поэтому быстрое и подробное
    # превью и сохранение — одна и та же задача: сначала ищем уже посчитанное
    # для почти такой же картинки (см. _supermat_recent), потом диск
    x = resize_wrap(rgb, res, res)
    m = _supermat_recent(x, res)
    if m is None:
        m = disk_cached('supermat_v3', x, {'res': res}, lambda: SB.run(x, res, back=False))
        _supermat_recent(x, res, m)
    m = {k: np.clip(resize_wrap(v, H, W), 0, 1).astype(np.float32) for k, v in m.items()}
    return {'base': m['basecolor'], 'rough': m['roughness'], 'metal': m['metalness'],
            '_info': _info(rough=float(m['roughness'].mean()),
                           metal=float(m['metalness'].mean()),
                           device=SB.device() or '—')}


OUT_NAMES = ('_nh', '_n', '_spec', 'diffuse', '_ao', '_h', '_rough', '_metal', '_fixed',
             '_basecolor', '_orm', '_smooth', '_ms', '_specular', '_bent', '_gloss', '_sss')
# Набор, в который пишется выход: 'main' — обычная папка вывода; 'pbr' — рядом
# с ней папка «<папка вывода>_PBR» (для шейдера OpenMW-PBR: тот читает _spec
# иначе — R металл, G шероховатость, B AO, A 1 − SSS — и диффуз без AO)
OUT_SETS = ('main', 'pbr')
# откуда брать канал узла «Выход по каналам»: яркость или один канал источника
PACK_CH = ('luma', 'r', 'g', 'b', 'a')


@node('out', [('any', ANY), ('alpha', GRAY)], [('any', ANY)],
      [('name', 'c', '_h', OUT_NAMES, None), ('set', 'c', 'main', OUT_SETS, None)],
      cat='output', opt=('alpha',))
def _n_out(i, p, ctx):
    """Output node: pass the single input through unchanged (None if unconnected).

    The 'name' parameter is the file suffix the program writes the result with:
    '_nh', '_n', '_spec', '_ao', '_h', '_rough', '_metal', '_fixed', or
    'diffuse' — the source's own name (a replacement diffuse; the source alpha
    is carried over). A 'diffuse' output is written only into an output
    folder different from the source folder, never over the source.

    A 'diffuse' output of a colour picture also carries an alpha as a fourth
    channel, so that the 2D view shows it (channel A) as it will be in the
    file: the 'alpha' input if something is connected to it, else the source
    alpha when the texture has one (leaves, grilles). An alpha input replaces
    the source's own alpha; that is reported as '_warn' (the node is marked
    in the graph, the readouts say why). On other outputs the alpha input is
    ignored, also with a warning (their alpha means something else: the
    height in _nh, the shininess in _spec).

    'set' (OUT_SETS): 'main' writes into the output folder; 'pbr' into the
    folder next to it named '<output folder>_PBR' (project.pbr_root), so
    that two versions of the same file — for vanilla OpenMW and for the
    OpenMW-PBR shader — do not overwrite each other.
    """
    v = i.get('any')
    ain = i.get('alpha')
    a_src = ctx.get('alpha') if isinstance(ctx, dict) else None
    has_src = a_src is not None and float(a_src.min()) < 0.999
    warn = None
    if p['name'] == 'diffuse' and v is not None and v.ndim == 3:
        xp = X.xp(v)
        a = None
        if ain is not None:
            a = ain[..., :3].mean(-1) if ain.ndim == 3 else ain
            if tuple(a.shape[:2]) != tuple(v.shape[:2]):
                a, warn = None, 'out.alpha_size'
            elif has_src:
                warn = 'out.alpha_replaced'
        if a is None and has_src and tuple(a_src.shape[:2]) == tuple(v.shape[:2]):
            a = a_src
        if a is not None:
            v = xp.concatenate([v[..., :3], xp.clip(xp.asarray(a, xp.float32), 0, 1)[..., None]],
                               -1)
    elif ain is not None:
        warn = 'out.alpha_ignored'
    out = {'any': v}
    if warn:
        out['_warn'] = warn
    return out


def _pack_channel(v, ch, xp):
    """One channel for «Выход по каналам» from an input value: a grey map
    as it is; from a picture its brightness (ch 'luma', Rec. 709) or one of
    its channels 'r', 'g', 'b', 'a' (no alpha → 1). A signed normal (−1…1)
    is first brought to the 0…1 of a file, as a normal map stores it."""
    v = xp.asarray(v, xp.float32)
    if v.ndim == 2:
        return xp.clip(v, 0, 1)
    if float(v[..., :3].min()) < -0.01:
        v = xp.concatenate([v[..., :3] * 0.5 + 0.5, v[..., 3:]], -1)
    if ch == 'luma':
        g = 0.2126 * v[..., 0] + 0.7152 * v[..., 1] + 0.0722 * v[..., 2]
    elif ch == 'a':
        g = v[..., 3] if v.shape[2] > 3 else xp.ones(v.shape[:2], xp.float32)
    else:
        g = v[..., 'rgb'.index(ch)]
    return xp.clip(g, 0, 1).astype(xp.float32)


@node('out_pack', [('r', ANY), ('g', ANY), ('b', ANY), ('a', ANY)], [('any', ANY)],
      [('name', 'c', '_spec', OUT_NAMES, None), ('set', 'c', 'main', OUT_SETS, None),
       ('r_ch', 'c', 'luma', PACK_CH, None), ('r_val', 'f', 0.0, 0.0, 1.0),
       ('g_ch', 'c', 'luma', PACK_CH, None), ('g_val', 'f', 0.0, 0.0, 1.0),
       ('b_ch', 'c', 'luma', PACK_CH, None), ('b_val', 'f', 0.0, 0.0, 1.0),
       ('a_ch', 'c', 'luma', PACK_CH, None), ('a_fill', 'b', False, 0, 1),
       ('a_val', 'f', 1.0, 0.0, 1.0)],
      cat='output', opt=('r', 'g', 'b', 'a'))
def _n_out_pack(i, p, ctx):
    """Output by channels: a file whose R, G, B and (optional) A are
    picked one by one.

    Each of the inputs 'r', 'g', 'b', 'a' takes any map; '<c>_ch' says what
    of it goes into that channel (_pack_channel: brightness, or the R, G, B
    or A of the source). An unconnected R, G or B is filled with '<c>_val';
    an unconnected A — with 'a_val' if 'a_fill' is on, else the file has no
    alpha of its own (three channels; for 'diffuse' the source alpha is used
    then, as by «Выход»). 'name' and
    'set' — as in «Выход». Every connected map must be of the same size,
    else the node stops with an error. Readouts: the mean of each channel.
    """
    vals = {c: i.get(c) for c in 'rgba'}
    ref = next((v for v in vals.values() if v is not None), None)
    if ref is None:
        return {'any': None}
    xp = X.xp(ref)
    shp = tuple(ref.shape[:2])
    chans = []
    for c in 'rgba':
        v = vals[c]
        if v is None:
            if c == 'a' and not p['a_fill']:
                break
            chans.append(xp.full(shp, float(p[c + '_val']), xp.float32))
            continue
        if tuple(v.shape[:2]) != shp:
            raise ValueError(t_('out_pack.size') % (c.upper(), v.shape[1], v.shape[0],
                                                    shp[1], shp[0]))
        chans.append(_pack_channel(v, p[c + '_ch'], xp))
    out = xp.stack(chans, -1).astype(xp.float32)
    means = {k: float(chans[n].mean()) for n, k in enumerate('RGBA'[:len(chans)])}
    return {'any': out, '_info': _info(**means)}


def _n01(x):
    """Stretch the 1st…99th percentile of ``x`` to 0…1; returns float32.

    The tails beyond are bent in softly (matgen.soft01), not clipped: a hard
    clip made the tops and bottoms of a height flat, and the normal made from
    it showed flat blobs. A constant map comes out as a flat low value.
    """
    import matgen as MG
    xp = X.xp(x)
    x = xp.asarray(x, xp.float32)
    lo, hi = float(xp.percentile(x, 1)), float(xp.percentile(x, 99))
    return MG.soft01((x - lo) / max(hi - lo, 1e-6)).astype(xp.float32)


# ---------------------------------------------------------------- граф ----
class Graph:
    """The node graph: nodes, links, evaluation with a cache, JSON.

    Attributes:
        nodes: dict id → {'type', 'x', 'y', 'params'}; x, y are canvas pixels.
        links: list of (src_id, src_port, dst_id, dst_port).
        _seq: running counter for new ids (ids are never reused).
        _cache: dict id → list of (key, outputs), the KEEP most recent results
            per node (see key_of, eval).

    Structural changes (remove, connect, disconnect, from_json) clear the whole
    cache; parameter edits need not, because the parameters are part of the key.
    """
    KEEP = 2            # сколько последних ответов помнить на узел

    def __init__(self):
        """Create an empty graph."""
        self.nodes = {}      # id -> {'type','x','y','params'}
        self.links = []      # (src_id, src_port, dst_id, dst_port)
        self._seq = 0
        self._cache = {}

    # --- правка
    def add(self, key, x=40, y=40):
        """Add a node of type ``key`` at canvas position (x, y) with default parameters.

        Returns:
            The new id, '<key><n>' with n from the running counter.

        Raises:
            KeyError: unknown node type.
        """
        self._seq += 1
        nid = '%s%d' % (key, self._seq)
        sp = REG[key]
        pars = {}
        for pr in sp.params:
            # список (заплатки) — свой у каждого узла, а не общий из описания
            pars[pr[0]] = list(pr[2]) if isinstance(pr[2], list) else pr[2]
        self.nodes[nid] = {'type': key, 'x': x, 'y': y, 'params': pars}
        return nid

    def remove(self, nid):
        """Remove node ``nid`` (no error if absent), its links and its cache
        entries. The rest of the cache stays: its keys are by content (node
        type, parameters, everything upstream), so the nodes this one fed get
        new keys by themselves."""
        self.nodes.pop(nid, None)
        self.links = [l for l in self.links if l[0] != nid and l[2] != nid]
        self._cache.pop(nid, None)

    def connect(self, a, ap, b, bp):
        """Connect output a.ap to input b.bp.

        An input can take only one wire (an existing wire into b.bp is replaced),
        rings are forbidden. The port kinds must fit (kinds_fit: equal, ANY, or
        grey ↔ colour, converted when evaluated).
        The ring check is by reachability: the new link a → b closes a ring exactly
        when a is already reachable from b downstream (_reaches(b, a)); this also
        rejects a == b. The cache is kept: the keys of the nodes below change
        by themselves (key_of), and a duplicate of a node finds the original's
        answer instead of computing it again.

        Returns:
            True if connected; False if refused (unknown node or port, kind mismatch,
            ring).
        """
        if a not in self.nodes or b not in self.nodes:
            return False
        ka = dict(REG[self.nodes[a]['type']].outs).get(ap)
        kb = dict(REG[self.nodes[b]['type']].ins).get(bp)
        if ka is None or kb is None:
            return False
        if not kinds_fit(ka, kb):
            return False
        if self._reaches(b, a):
            return False                       # получилось бы кольцо
        self.links = [l for l in self.links if not (l[2] == b and l[3] == bp)]
        self.links.append((a, ap, b, bp))
        return True

    def disconnect(self, b, bp):
        """Remove the wire going into input b.bp (if any); the cache stays (see
        connect)."""
        self.links = [l for l in self.links if not (l[2] == b and l[3] == bp)]

    def _reaches(self, start, target):
        """Tell whether ``target`` is reachable from ``start`` going down the links.

        Used for the ring check. Iterative depth-first walk; start == target counts
        as reachable.
        """
        seen, stack = set(), [start]
        while stack:
            cur = stack.pop()
            if cur == target:
                return True
            if cur in seen:
                continue
            seen.add(cur)
            stack += [l[2] for l in self.links if l[0] == cur]
        return False

    def inputs_of(self, nid):
        """Return the wires into ``nid``: dict input port → (source id, source port)."""
        return {l[3]: (l[0], l[1]) for l in self.links if l[2] == nid}

    # --- счёт
    def key_of(self, nid, ctx_key):
        """Cache key of node ``nid``: what its output depends on.

        The key is the repr of: the node type, ``ctx_key`` (identifies the source
        texture), its parameters sorted by name (floats rounded to 4 decimals by
        _round) and, recursively, for every connected input the port names and the
        key of the upstream node. So the key changes exactly when something that can
        change this node's output changes: its own parameters, any parameter or wire
        upstream, or the source. Without the upstream part a slider moved above would
        not invalidate the nodes below, and without ``ctx_key`` a new texture (or the
        full-size pass after the preview) would reuse maps of the old one. Parameter
        changes smaller than 1e-4 do not cause a recomputation.

        Returns:
            str. Computed recursively on every call, without memoisation, which is
            cheap for graphs of this size.
        """
        n = self.nodes[nid]
        parts = [n['type'], ctx_key,
                 tuple(sorted((k, _round(v)) for k, v in n['params'].items()))]
        for port, (sid, sport) in sorted(self.inputs_of(nid).items()):
            parts.append((port, sport, self.key_of(sid, ctx_key)))
        return repr(parts)

    def eval(self, nid, ctx, progress=None):
        """Values of all outputs of a node.

        Computed on demand and remembered: the key includes the parameters of all
        nodes upstream, so editing one slider recomputes only what lies below it.
        Every node is computed at most once per pass (a node shared by two branches is
        a cache hit the second time); each node keeps its KEEP latest results (the
        fast and the fine preview).

        Args:
            nid: node id.
            ctx: evaluation context, dict: 'src' — source texture H×W×3 sRGB 0…1
                (read by the source node); 'key' — str identifying the source, part
                of the cache key ('' if absent); 'sub' — optional progress
                callable(i, n) for long nodes (ESRGAN windows).
            progress: optional callable(node id), called right before a node is
                actually computed (not on cache hits).

        Returns:
            Dict output port → value, as returned by the node function. The same
            object is kept in the cache, so it must not be modified in place.

        Raises:
            ValueError: a required input is not connected (the message names the
                node and the port); ports listed in the spec's ``opt`` are exempt. Exceptions of
                the node functions propagate.
        """
        key = self.key_of(nid, ctx.get('key', ''))
        hits = self._cache.get(nid) or []
        for k, v in hits:
            if k == key:
                return v
        # ключ не зависит от номера узла: копия узла с теми же входами и
        # настройками (дубликат) берёт готовый ответ оригинала, а не считает заново
        for other, oh in list(self._cache.items()):
            if other != nid:
                for k, v in oh:
                    if k == key:
                        self._cache[nid] = [(key, v)] + hits[:self.KEEP - 1]
                        return v
        ins = {}
        sp = REG[self.nodes[nid]['type']]
        kin = dict(sp.ins)
        for port, (sid, sport) in self.inputs_of(nid).items():
            # серое в цветной вход — тремя одинаковыми каналами, цветное в
            # серый — яркостью (см. kinds_fit)
            ins[port] = fit_value(self.eval(sid, ctx, progress)[sport], kin.get(port))
        # где считается узел: у процессорных (сети, починка) входы — на
        # процессор, у остальных — на видеокарту, если она включена
        ins = {k: (X.to_host(v) if sp.key in HOST_NODES else X.to_dev(v))
               for k, v in ins.items()}
        need = [p for p, _ in sp.ins if p not in ins and p not in sp.opt]
        if need:
            from i18n import t
            raise ValueError(t('g.noinput', nid, t('np.' + need[0])))
        if progress:
            progress(nid)
        out = sp.fn(ins, self.nodes[nid]['params'], ctx)
        # два последних ответа на узел: превью считается то на быстрой копии,
        # то на подробной, и одна запись вытесняла бы другую
        self._cache[nid] = [(key, out)] + [h for h in hits if h[0] != key][:self.KEEP - 1]
        return out

    def order(self):
        """All node ids, every node after all nodes feeding it.

        Used to evaluate a whole graph for previews: going in this order each
        node finds its inputs already in the cache.
        """
        seen, res = set(), []

        def visit(nid):
            """Depth-first: put every input of nid before nid itself."""
            if nid in seen:
                return
            seen.add(nid)
            for sid, _ in self.inputs_of(nid).values():
                visit(sid)
            res.append(nid)
        for nid in list(self.nodes):
            visit(nid)
        return res

    def outputs(self):
        """Return the ids of all output nodes ('out', 'out_pack')."""
        return [n for n, v in self.nodes.items() if v['type'] in ('out', 'out_pack')]

    # --- файл
    def to_json(self):
        """Serialise the graph to JSON text.

        Nodes (with positions and parameters), links and the id counter are saved.

        The cache is not saved; link tuples become lists; non-ASCII is kept as is.
        """
        return json.dumps({'nodes': self.nodes, 'links': self.links,
                           'seq': self._seq}, ensure_ascii=False, indent=1)

    def from_json(self, text):
        """Load a graph saved by to_json, replacing the current content.

        Nodes of types not in REG are dropped together with their links. The graph
        may have been saved by an earlier version: missing parameters take their
        defaults, parameters that no longer exist are dropped. Links become tuples;
        the id counter is restored from 'seq' (default: the number of nodes). The
        cache is cleared. Links to ports a node type no longer has are dropped;
        kinds and rings of the loaded links are not checked again.
        """
        d = json.loads(text)
        self.nodes = {k: v for k, v in d['nodes'].items() if v['type'] in REG}
        # граф мог быть сохранён прошлой версией: недостающие параметры
        # берутся по умолчанию, исчезнувшие отбрасываются
        for v in self.nodes.values():
            old = v.get('params', {})
            if REG[v['type']].upgrade:
                old = REG[v['type']].upgrade(dict(old))
            v['params'] = {pr[0]: old.get(pr[0], list(pr[2]) if isinstance(pr[2], list)
                                          else pr[2]) for pr in REG[v['type']].params}
        # связь к гнезду, которого у узла больше нет (узел поменялся между
        # версиями), отбрасывается: иначе граф ссылался бы на пустоту
        self.links = [tuple(x) for x in d['links']
                      if x[0] in self.nodes and x[2] in self.nodes
                      and x[1] in dict(REG[self.nodes[x[0]]['type']].outs)
                      and x[3] in dict(REG[self.nodes[x[2]]['type']].ins)]
        self._seq = int(d.get('seq', len(self.nodes)))
        self._cache.clear()


def _round(v):
    """Round floats to 4 decimals for the cache key; other values are returned as is."""
    return round(v, 4) if isinstance(v, float) else v


def default_graph():
    """Same as what the tabs do: repair, de-light, height, normal, AO.

    Good as a starting point — it can be rewired from there. The graph actually
    built is source → delight → height_lum → normal → out (named '_n'), plus an
    'ao' node fed by the same height whose output is left unconnected; there is no
    repair node.

    Returns:
        A new Graph.
    """
    g = Graph()
    s = g.add('source', 30, 30)
    d = g.add('delight', 30, 130)
    h = g.add('height_lum', 30, 260)
    n = g.add('normal', 260, 260)
    a = g.add('ao', 260, 400)
    o = g.add('out', 470, 260)
    g.connect(s, 'rgb', d, 'rgb')
    g.connect(d, 'rgb', h, 'rgb')
    g.connect(h, 'h', n, 'h')
    g.connect(h, 'h', a, 'h')
    g.connect(n, 'n', o, 'any')
    g.nodes[o]['params']['name'] = '_n'
    return g
