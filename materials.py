# -*- coding: utf-8 -*-
"""Real-world materials for the specular / PBR nodes, and laying generated
maps into the range of the materials a texture is made of.

f0 — reflectance at normal incidence for dielectrics, linear, from the
refractive index: f0 = ((n − 1)/(n + 1))². Water n 1.33 → 0.020, ice and
snow 1.31 → 0.018, glass, plastics, varnish 1.5 → 0.040, quartz, granite,
chitin ~1.55 → 0.045, gems 1.75…1.8 → 0.075, skin 0.028, hair 0.046,
teeth 0.058 (the usual PBR reference tables agree with these). Metals have
no f0 of their own here: in the metalness workflow their specular colour is
the base colour, so for them metal is 0.9…1.

Roughness ranges are typical values used in PBR reference tables and
material libraries (plaster ~0.86, raw wood ~0.68, steel ~0.25, mirror 0),
not physical constants: roughness depends on how a surface was worked and
worn. That is why every material has a range, not a number.

words — what a texture file name may contain for the material to be
recognised (English game naming plus Morrowind's: dwrv, chitin, ebony,
redware, netch…). A word matches when some part of the name between
'_', '-', digits or spaces starts with it: 'tx_woodfloor_01' → wood.

Материалы для узлов блика и PBR. f0 диэлектриков — из показателя
преломления (физика); диапазоны шероховатости — типичные значения из
справочных таблиц PBR, не константы: шероховатость зависит от обработки.
"""

import re

import numpy as np

import xpu as X

BUILD = 25          # версия набора файлов, проверяется при запуске

# ключ: f0 (лин.), шероховатость (от, до), металличность (от, до), слова
MATERIALS = {
    # камень и минералы
    'stone': (0.045, (0.60, 0.95), (0.0, 0.0),
              ('stone', 'rock', 'cliff', 'boulder', 'cobble', 'granite', 'slate', 'flagstone',
               'basalt', 'cave', 'stronghold', 'velothi', 'statue')),
    'marble': (0.045, (0.10, 0.45), (0.0, 0.0), ('marble', 'polished', 'alabaster')),
    'sandstone': (0.040, (0.80, 1.00), (0.0, 0.0),
                  ('sandstone', 'plaster', 'stucco', 'concrete', 'cement', 'adobe', 'mortar')),
    'brick': (0.040, (0.75, 1.00), (0.0, 0.0), ('brick', 'terracotta', 'rooftile')),
    'obsidian': (0.040, (0.05, 0.35), (0.0, 0.0), ('obsidian', 'ebony', 'daedric', 'volcanic')),
    'gem': (0.075, (0.00, 0.15), (0.0, 0.0),
            ('gem', 'crystal', 'jewel', 'ruby', 'emerald', 'sapphire', 'amethyst', 'diamond',
             'pearl', 'soulgem')),
    # земля и вода
    'ground': (0.035, (0.85, 1.00), (0.0, 0.0),
               ('dirt', 'ground', 'soil', 'ash', 'dust', 'sand', 'gravel', 'road', 'path',
                'moss', 'terrain')),
    'mud': (0.030, (0.20, 0.50), (0.0, 0.0), ('mud', 'wet', 'swamp', 'bog', 'marsh', 'slime')),
    'water': (0.020, (0.00, 0.10), (0.0, 0.0), ('water', 'river', 'lake', 'sea')),
    'ice': (0.018, (0.02, 0.30), (0.0, 0.0), ('ice', 'frost', 'frozen')),
    'snow': (0.018, (0.60, 0.90), (0.0, 0.0), ('snow',)),
    # дерево и растения
    'wood': (0.040, (0.60, 0.90), (0.0, 0.0),
             ('wood', 'plank', 'log', 'timber', 'beam', 'board', 'crate', 'barrel')),
    'bark': (0.040, (0.80, 1.00), (0.0, 0.0), ('bark', 'trunk', 'root')),
    'lacquer': (0.045, (0.15, 0.45), (0.0, 0.0), ('lacquer', 'varnish', 'lacquered')),
    'leaves': (0.035, (0.45, 0.80), (0.0, 0.0),
               ('leaf', 'leaves', 'foliage', 'grass', 'fern', 'plant', 'flower', 'bush', 'vine',
                'kelp', 'mushroom', 'fungus')),
    # ткань, кожа, тело
    'cloth': (0.040, (0.85, 1.00), (0.0, 0.0),
              ('cloth', 'fabric', 'rope', 'rug', 'carpet', 'tapestry', 'banner', 'linen', 'wool', 'robe',
               'sack', 'curtain', 'bedding', 'shirt', 'pants', 'trousers', 'skirt', 'tunic',
               'dress', 'cloak', 'hood', 'sleeve', 'clothes', 'clothing')),
    'silk': (0.045, (0.30, 0.60), (0.0, 0.0), ('silk', 'satin')),
    'leather': (0.040, (0.40, 0.75), (0.0, 0.0), ('leather', 'hide', 'netch', 'strap')),
    'fur': (0.046, (0.50, 0.85), (0.0, 0.0), ('fur', 'hair', 'pelt', 'wool')),
    'skin': (0.028, (0.40, 0.70), (0.0, 0.0), ('skin', 'flesh', 'body', 'face', 'head')),
    'bone': (0.058, (0.35, 0.70), (0.0, 0.0), ('bone', 'skull', 'ivory', 'horn', 'tusk', 'antler',
                                                'teeth', 'tooth')),
    'chitin': (0.045, (0.20, 0.60), (0.0, 0.0), ('chitin', 'shell', 'carapace', 'insect')),
    # обработанное
    'ceramic': (0.045, (0.05, 0.35), (0.0, 0.0),
                ('ceramic', 'porcelain', 'glazed', 'redware', 'pottery', 'urn', 'vase', 'tile',
                 'limeware', 'crock')),
    # парафин n ≈ 1.43 → f0 0.031; шероховатость — на глаз (в справочных таблицах PBR воска нет)
    'wax': (0.031, (0.25, 0.60), (0.0, 0.0), ('wax', 'candle')),
    'paper': (0.040, (0.75, 1.00), (0.0, 0.0), ('paper', 'parchment', 'book', 'scroll', 'map')),
    'glass': (0.040, (0.00, 0.15), (0.0, 0.0), ('glass', 'window', 'bottle')),
    # металлы: цвет блика — из базового цвета, f0 у них не используется
    'iron': (0.040, (0.40, 0.75), (0.9, 1.0), ('iron', 'metal', 'forged', 'nail', 'hinge')),
    'steel': (0.040, (0.15, 0.40), (0.9, 1.0),
              ('steel', 'blade', 'sword', 'armor', 'armour', 'chain', 'plate', 'weapon')),
    'rust': (0.040, (0.70, 1.00), (0.0, 0.6), ('rust', 'rusty', 'corroded')),
    'gold': (0.040, (0.10, 0.40), (0.9, 1.0), ('gold', 'gilt', 'gilded')),
    'silver': (0.040, (0.10, 0.40), (0.9, 1.0), ('silver',)),
    'copper': (0.040, (0.20, 0.50), (0.9, 1.0), ('copper',)),
    'bronze': (0.040, (0.20, 0.55), (0.9, 1.0), ('bronze', 'brass', 'dwrv', 'dwemer')),
    'patina': (0.040, (0.50, 0.90), (0.2, 0.7), ('patina', 'verdigris', 'tarnish', 'tarnished')),
}

MATERIAL_KEYS = tuple(MATERIALS)


# Подповерхностное рассеяние (SSS) — измеренные свойства. Для узла «SSS».
# Jensen, Marschner, Levoy, Hanrahan, «A Practical Model for Subsurface Light
# Transport», SIGGRAPH 2001, рис. 5(b): приведённое рассеяние σs' и
# поглощение σa, мм⁻¹, для R, G, B. Снег — по глубине затухания света в
# снегу ~10…20 см на 400 нм (France et al., ACP 11, 9787, 2011): берём 100 мм.
SSS_MEASURED = {
    'sss_marble': ((2.19, 2.62, 3.00), (0.0021, 0.0041, 0.0071)),
    'sss_skin2': ((1.09, 1.59, 1.79), (0.013, 0.070, 0.145)),
    'sss_skin1': ((0.74, 0.88, 1.01), (0.032, 0.17, 0.48)),
    'sss_meat': ((0.15, 0.21, 0.38), (0.015, 0.077, 0.19)),       # Chicken1
    'sss_apple': ((2.29, 2.39, 1.97), (0.0030, 0.0034, 0.046)),
    'sss_potato': ((0.68, 0.70, 0.55), (0.0024, 0.0090, 0.12)),
    'sss_wholemilk': ((2.55, 3.21, 3.77), (0.0011, 0.0024, 0.014)),
    'sss_skimmilk': ((0.70, 1.22, 1.90), (0.0014, 0.0025, 0.0142)),
    'sss_cream': ((7.38, 5.47, 3.15), (0.0002, 0.0028, 0.0163)),
    'sss_ketchup': ((0.18, 0.07, 0.03), (0.061, 0.97, 1.45)),
}
SSS_FIXED = {'sss_snow': 100.0, 'sss_opaque': 0.0}


def diffusion_length(sig_s, sig_a):
    """How far light spreads under the surface, mm: 1/σtr with
    σtr = √(3·σa·(σa + σs')) — the distance over which the diffusion
    profile falls e times (the dipole model of Jensen et al. 2001). One
    number for R, G, B: their geometric mean (the shader takes one SSS)."""
    ls = [1.0 / np.sqrt(3.0 * a * (a + s)) for s, a in zip(sig_s, sig_a)]
    return float(np.exp(np.mean(np.log(ls))))


SSS_LENGTH = dict({k: round(diffusion_length(*v), 2) for k, v in SSS_MEASURED.items()},
                  **SSS_FIXED)
SSS_KEYS = tuple(SSS_LENGTH)


def scatter_color(sig_s, sig_a):
    """Colour of the scattering: the diffusion length 1/σtr of each of R, G,
    B divided by the largest of them, 0…1. It says which light goes
    furthest under the surface and so tints the glow at the light/shadow
    boundary — the same meaning as the per-channel «subsurface radius» of
    renderers (skin: red goes furthest, then green, then blue)."""
    ls = [1.0 / np.sqrt(3.0 * a * (a + s)) for s, a in zip(sig_s, sig_a)]
    m = max(ls)
    return tuple(round(float(v / m), 3) for v in ls)


# Снег: σs' у зёрен снега много больше длины волны и от неё не зависит, так
# что длина рассеяния ∝ 1/√σa льда. σa льда — Picard и др., The Cryosphere
# 10, 2655 (2016): 450 нм 0,0191 м⁻¹, 550 нм 0,0636, 600 нм 0,135 (таблица
# кончается на 600 нм, поэтому R взят на 600 — на 650 лёд поглощает ещё
# сильнее, и настоящий цвет немного синее)
_SNOW_A = (0.135, 0.0636, 0.0191)
SSS_COLOR = dict({k: scatter_color(*v) for k, v in SSS_MEASURED.items()},
                 sss_snow=tuple(round(float(min(_SNOW_A) ** 0.5 / a ** 0.5), 3) for a in _SNOW_A),
                 sss_opaque=(1.0, 1.0, 1.0))


def mat(key):
    """Material ``key`` as a dict: f0, rough_lo, rough_hi, metal_lo, metal_hi."""
    f0, r, m, _ = MATERIALS[key]
    return dict(f0=f0, rough_lo=r[0], rough_hi=r[1], metal_lo=m[0], metal_hi=m[1])


def gloss_rank(key):
    """Sort key «from dull to shiny»: the middle of the roughness range,
    descending (metals count as shinier at equal roughness)."""
    f0, r, m, _ = MATERIALS[key]
    return -(r[0] + r[1]) / 2.0 + 0.05 * (m[0] + m[1]) / 2.0


# Слова, которые не ищутся в конце слитного слова: «guarskin», «bearskin» —
# шкура, а не кожа тела; «vivechead» — голова статуи; «thorn», «crust»,
# «displacement» — не рог, не ржавчина, не цемент.
NO_SUFFIX = ('skin', 'head', 'face', 'body', 'horn', 'rust', 'cement')


def detect(name):
    """Materials named in a texture file name, dullest first.

    Returns a list of keys (empty if nothing matched). Parts of the name are
    split at anything that is not a Latin letter; a material matches when a
    part starts with one of its words. A part that starts with none of them
    may end with one of four letters or more — Morrowind glues words
    together: 'workedstone', 'carvedwood', 'doorplank', 'dragonbone' (not
    the words in NO_SUFFIX)."""
    parts = [s for s in re.split(r'[^a-z]+', (name or '').lower()) if s]
    found = set()
    for p in parts:
        hit = [k for k, (_, _, _, words) in MATERIALS.items()
               if any(p.startswith(w) for w in words)]
        if not hit:
            hit = [k for k, (_, _, _, words) in MATERIALS.items()
                   if any(len(w) >= 4 and w not in NO_SUFFIX and len(p) > len(w)
                          and p.endswith(w) for w in words)]
        found.update(hit)
    return sorted(found, key=gloss_rank)


def names(keys):
    """Material keys as words in the interface language, for readouts."""
    import i18n
    return ', '.join(i18n.t('nv.' + k) for k in keys)


def range_params(dull, shiny):
    """Slider values for a texture whose materials go from ``dull`` to
    ``shiny`` (material keys; the same key twice — one material):
    f0 at the dull and the shiny end, the roughness range from the shiny
    material's smoothest to the dull one's roughest, the metalness range
    covering both, and the values used without maps (the dull material's
    middle)."""
    a, b = mat(dull), mat(shiny)
    return dict(f0=a['f0'], f0_shiny=b['f0'],
                rough_lo=min(b['rough_lo'], a['rough_lo']),
                rough_hi=max(a['rough_hi'], b['rough_hi']),
                metal_lo=min(a['metal_lo'], b['metal_lo']),
                metal_hi=max(a['metal_hi'], b['metal_hi']),
                rough_default=round((a['rough_lo'] + a['rough_hi']) / 2, 3),
                metal_default=round((a['metal_lo'] + a['metal_hi']) / 2, 3), fit=1.0)


def preset_update(params, name, val):
    """Inspector hook (Spec.preset_fn): when one of the two material lists
    changes, return the slider values for the new pair, else None."""
    if name not in ('mat_dull', 'mat_shiny') or val == 'custom':
        return None
    dull = val if name == 'mat_dull' else params.get('mat_dull', 'custom')
    shiny = val if name == 'mat_shiny' else params.get('mat_shiny', 'custom')
    if dull == 'custom':
        dull = shiny
    if shiny == 'custom':
        shiny = dull
    if gloss_rank(dull) > gloss_rank(shiny):
        dull, shiny = shiny, dull
    out = range_params(dull, shiny)
    out['mat_dull'], out['mat_shiny'] = dull, shiny
    # «Сила блика» — ручной множитель сверх физики: силу блика у материала уже
    # задают F0 и шероховатость, поэтому при выборе материала он снова 1
    out['spec_gain'] = 1.0
    return out


def effective(p, name):
    """The range parameters actually used: from the file name when
    'auto_name' is on and the name names a material, else the sliders.
    Returns (params dict, list of recognised material keys)."""
    if p.get('auto_name') and name:
        found = detect(name)
        if found:
            q = dict(p)
            q['mats'] = [[k, 1.0] for k in found]       # все найденные, поровну
            q['mats_manual'] = False                    # по блеску: свой порядок — не для них
            q['fit'] = 1.0
            return q, found
    return p, []


def layers(mats, manual=False):
    """The chosen materials as [(key, share), …] in the order of their bands
    on the shininess scale, from the dull end; shares summing to 1. mats —
    the 'mats' parameter: a list of [key, share] (or plain keys); unknown
    keys are dropped.

    manual False (the 'mats_manual' parameter off): dullest material first,
    by its own roughness (gloss_rank). manual True: the order as stored —
    the user has swapped the bands, e.g. when the roughness map came out
    wrong and the wool of the texture is its shiniest place: then the wool
    band is put at the shiny end, so those pixels get wool."""
    out = []
    for m in mats or []:
        k, w = (m, 1.0) if isinstance(m, str) else (m[0], float(m[1]) if len(m) > 1 else 1.0)
        if k in MATERIALS and k not in [a for a, _ in out]:
            out.append((k, max(w, 1e-3)))
    if not manual:
        out.sort(key=lambda kw: gloss_rank(kw[0]))
    tot = sum(w for _, w in out) or 1.0
    return [(k, w / tot) for k, w in out]


def lay(rough, metal, p, shape, xp):
    """Lay roughness and metalness maps into the material range of ``p``.

    rough, metal — H×W maps 0…1 or None (not connected). The map's own
    spread is used, so its pattern stays and only its levels change; the
    spread leaves out clip_dull % of the roughest pixels and clip_shiny %
    of the smoothest (2 and 2 by default) — so that a background which is
    not laid on the model, say, does not become the shiniest place and
    squeeze the rest. Those pixels get the end values. t = how shiny a
    pixel is within the texture, 0 — the dullest, 1 — the shiniest.

    With materials chosen ('mats', see layers) the t range is cut into
    one band per material, in proportion to their shares, dullest first;
    in its band a pixel takes that material's roughness (from its roughest
    to its smoothest along t), its f0 and, without a metal map, its
    metalness. A metal map's spread is laid onto the metalness range of all
    the chosen materials — except with 'mats_manual' on (the bands put in
    an order of one's own): then in each band the metal map is laid onto
    that band's material range, so a band of fur stays non-metal even where
    the map says metal. Without a roughness map the material with the
    largest share gives the middle of its ranges.

    Without materials — the sliders: roughness = rough_hi … rough_lo along
    t, f0 = f0 … f0_shiny along t, metalness = the metal map laid onto
    metal_lo…metal_hi, or without one (or a flat one) metal_lo…metal_hi
    along t; without a roughness map rough_default, metal_default.

    fit 0…1 blends between the incoming maps and the laid ones.

    Returns (rough, metal, f0 map, readouts dict); the dict also carries
    '_hist' — the shininess histogram for the inspector's scale (or None)."""
    fit = float(p.get('fit', 0.0))
    lys = layers(p.get('mats'), bool(p.get('mats_manual')))
    if lys:
        mt = [mat(k) for k, _ in lys]
        bounds = [0.0]
        for _, w in lys:
            bounds.append(bounds[-1] + w)
        bounds[-1] = 1.0
        mlo = min(m['metal_lo'] for m in mt)
        mhi = max(m['metal_hi'] for m in mt)
        big = mt[max(range(len(lys)), key=lambda j: lys[j][1])]
        r_def = 0.5 * (big['rough_lo'] + big['rough_hi'])
        m_def = 0.5 * (big['metal_lo'] + big['metal_hi'])
        f0_def = big['f0']
    else:
        f0a, f0b = float(p['f0']), float(p.get('f0_shiny', p['f0']))
        rlo, rhi = sorted((float(p.get('rough_lo', 0.0)), float(p.get('rough_hi', 1.0))))
        mlo, mhi = sorted((float(p.get('metal_lo', 0.0)), float(p.get('metal_hi', 1.0))))
        r_def, m_def, f0_def = float(p.get('rough_default', 0.7)), \
            float(p.get('metal_default', 0.0)), f0a
    c_dull = min(max(float(p.get('clip_dull', 2.0)), 0.0), 45.0)
    c_shiny = min(max(float(p.get('clip_shiny', 2.0)), 0.0), 45.0)
    info = {'_hist': None}

    def spread(m, tag, lo_pc=2.0, hi_pc=98.0):
        """(lo_pc %, hi_pc %) of a map; noted in the readouts."""
        q = X.to_host(xp.percentile(m, xp.asarray([lo_pc, hi_pc], xp.float32)))
        info[tag + '_in_lo'], info[tag + '_in_hi'] = float(q[0]), float(q[1])
        return float(q[0]), float(q[1])

    if rough is None:
        r = xp.full(shape, r_def, xp.float32)
        t = xp.zeros(shape, xp.float32)
    else:
        r0 = xp.clip(rough, 0, 1)
        a, b = spread(r0, 'rough', c_shiny, 100.0 - c_dull)
        u = xp.clip((r0 - a) / (b - a), 0, 1) if b - a > 1e-3 else xp.full_like(r0, 0.5)
        t = 1.0 - u
        if lys:
            r_lay = xp.zeros_like(r0)
            f0m = xp.zeros_like(r0)
            m_t = xp.zeros_like(r0)
            for j, m_ in enumerate(mt):
                lo_b, hi_b = bounds[j], bounds[j + 1]
                sel = (t >= lo_b) & ((t < hi_b) if j < len(mt) - 1 else (t <= 1.0))
                sl = xp.clip((t - lo_b) / max(hi_b - lo_b, 1e-6), 0, 1)
                r_lay = xp.where(sel, m_['rough_hi'] + sl * (m_['rough_lo'] - m_['rough_hi']), r_lay)
                f0m = xp.where(sel, m_['f0'], f0m)
                m_t = xp.where(sel, m_['metal_lo'] + sl * (m_['metal_hi'] - m_['metal_lo']), m_t)
        else:
            r_lay = rlo + u * (rhi - rlo)
        r = r0 * (1 - fit) + r_lay * fit
        # гистограмма блеска (1 − шероховатость) для шкалы в инспекторе
        sh = X.to_host(1.0 - r0).astype(np.float32).ravel()
        if sh.size > 200000:
            sh = sh[:: max(1, sh.size // 200000)]
        s_lo, s_hi = float(sh.min()), float(sh.max())
        if s_hi - s_lo < 1e-4:
            s_hi = s_lo + 1e-4
        cnt = np.histogram(sh, 96, (s_lo, s_hi))[0].astype(np.float64)
        info['_hist'] = {'h': (cnt / max(cnt.sum(), 1.0)).tolist(), 'lo': s_lo, 'hi': s_hi,
                         'cut_dull': 1.0 - b, 'cut_shiny': 1.0 - a}
    # что пришло: карта, или без карты — металличность по умолчанию
    m0 = (xp.clip(metal, 0, 1) if metal is not None
          else xp.full(shape, m_def, xp.float32))
    flat = True
    if metal is not None:
        a, b = spread(m0, 'metal')
        flat = b - a <= 1e-3    # ровная карта ничего не говорит о том, где металл
    if not flat and lys and rough is not None and p.get('mats_manual'):
        # свой порядок полос: пользователь сам сказал, где какой материал, —
        # карта металличности кладётся в диапазон металличности материала
        # своей полосы (мех остаётся неметаллом, даже если карта ошиблась)
        mn = xp.clip((m0 - a) / (b - a), 0, 1)
        m_b = xp.zeros_like(m0)
        for j, m_ in enumerate(mt):
            lo_b, hi_b = bounds[j], bounds[j + 1]
            sel = (t >= lo_b) & ((t < hi_b) if j < len(mt) - 1 else (t <= 1.0))
            m_b = xp.where(sel, m_['metal_lo'] + mn * (m_['metal_hi'] - m_['metal_lo']), m_b)
        m = m0 * (1 - fit) + m_b * fit
    elif not flat:
        m = m0 * (1 - fit) + (mlo + xp.clip((m0 - a) / (b - a), 0, 1) * (mhi - mlo)) * fit
    elif rough is not None:
        m = m0 * (1 - fit) + ((m_t if lys else mlo + t * (mhi - mlo))) * fit
    else:
        m = m0
    if rough is not None and lys:
        f0 = f0_def * (1 - fit) + f0m * fit
    elif rough is not None:
        f0 = xp.full(shape, f0a, xp.float32) + (f0b - f0a) * t * fit
    else:
        f0 = xp.full(shape, f0_def, xp.float32)
    return (xp.clip(r, 0.02, 1).astype(xp.float32), xp.clip(m, 0, 1).astype(xp.float32),
            xp.asarray(f0, xp.float32), info)


# --- инспектор: группы палитры и шары-превью ----------------------------------
# группы материалов в палитре (порядок — как в таблице выше)
GROUPS = (
    ('mg.stone', ('stone', 'marble', 'sandstone', 'brick', 'obsidian', 'gem')),
    ('mg.ground', ('ground', 'mud', 'water', 'ice', 'snow')),
    ('mg.plant', ('wood', 'bark', 'lacquer', 'leaves')),
    ('mg.body', ('cloth', 'silk', 'leather', 'fur', 'skin', 'bone', 'chitin')),
    ('mg.made', ('ceramic', 'wax', 'paper', 'glass')),
    ('mg.metal', ('iron', 'steel', 'rust', 'gold', 'silver', 'copper', 'bronze', 'patina')),
)

# Цвет шара в палитре — только чтобы узнавать материал глазом; в расчётах не
# участвует. У металлов — их отражение при нормальном падении (линейное, из
# справочных таблиц PBR: железо, золото, серебро, медь); бронза, ржавчина,
# патина и все неметаллы — типичный вид на глаз (sRGB).
_METAL_F0 = {'iron': (0.56, 0.57, 0.58), 'steel': (0.56, 0.57, 0.58),
             'gold': (1.00, 0.71, 0.29), 'silver': (0.95, 0.93, 0.88),
             'copper': (0.95, 0.64, 0.54)}
_LOOK = {'stone': (0.52, 0.50, 0.47), 'marble': (0.88, 0.86, 0.82), 'sandstone': (0.76, 0.66, 0.50),
         'brick': (0.62, 0.30, 0.22), 'obsidian': (0.10, 0.09, 0.12), 'gem': (0.25, 0.55, 0.75),
         'ground': (0.45, 0.36, 0.26), 'mud': (0.30, 0.24, 0.17), 'water': (0.10, 0.18, 0.22),
         'ice': (0.75, 0.85, 0.92), 'snow': (0.93, 0.94, 0.96), 'wood': (0.55, 0.38, 0.22),
         'bark': (0.33, 0.25, 0.18), 'lacquer': (0.45, 0.16, 0.10), 'leaves': (0.28, 0.45, 0.16),
         'cloth': (0.55, 0.50, 0.42), 'silk': (0.70, 0.20, 0.25), 'leather': (0.42, 0.26, 0.15),
         'fur': (0.50, 0.40, 0.30), 'skin': (0.80, 0.60, 0.50), 'bone': (0.85, 0.80, 0.68),
         'chitin': (0.40, 0.33, 0.20), 'ceramic': (0.72, 0.40, 0.25), 'wax': (0.92, 0.86, 0.68),
         'paper': (0.88, 0.84, 0.72), 'glass': (0.55, 0.70, 0.65), 'rust': (0.48, 0.24, 0.12),
         'bronze': (0.80, 0.55, 0.30), 'patina': (0.35, 0.58, 0.48)}
_PREV = {}


def sphere_preview(key, size=36, bg=(0.11, 0.12, 0.13)):
    """A small picture of a sphere of material ``key`` for the palette:
    a studio with a light top and a softbox upper left, the roughness and
    metalness from the middle of the material's ranges, Fresnel by
    Schlick, the reflection blurred by roughness (not an exact BRDF — a
    picture to tell glossy from matt and metal from not). Returns
    size×size×3 uint8; cached."""
    ck = (key, size, bg)
    if ck in _PREV:
        return _PREV[ck]
    m = mat(key)
    rough = 0.5 * (m['rough_lo'] + m['rough_hi'])
    metal = 0.5 * (m['metal_lo'] + m['metal_hi'])
    if key in _METAL_F0:
        base = np.array(_METAL_F0[key], np.float32)
    else:
        base = (np.array(_LOOK.get(key, (0.6, 0.6, 0.6)), np.float32)) ** 2.2
    ss = 3                                      # 3×3 выборки на пиксель — ровный край
    n_ = size * ss
    yy, xx = (np.mgrid[0:n_, 0:n_] + 0.5) / n_ * 2.2 - 1.1
    rr = xx * xx + yy * yy
    inside = rr < 1.0
    z = np.sqrt(np.clip(1.0 - rr, 0, 1))
    N = np.stack([xx, -yy, z], -1)
    V = np.array([0.0, 0.0, 1.0])
    nv = np.clip(N @ V, 0, 1)
    R = 2.0 * nv[..., None] * N - V
    L = np.array([-0.55, 0.6, 0.58]); L /= np.linalg.norm(L)

    def env(d, blur):
        sky = 0.18 + 0.82 * np.clip(0.5 + 0.5 * d[..., 1], 0, 1) ** 1.5
        w = 0.03 + blur * blur * 1.2
        box = np.exp(-(1.0 - np.clip(d @ L, -1, 1)) / w) * (0.08 / w) ** 0.5 * 6.0
        return sky + box
    f0 = np.full(3, float(m['f0']), np.float32) * (1 - metal) + base * metal
    # у шершавых кайма Френеля слабее: отражение по краю размыто и тонет в рассеянном
    F = f0 + (1.0 - f0) * ((1.0 - nv) ** 5)[..., None] * (1.0 - 0.75 * rough)
    e_sh = env(R, rough)[..., None]
    e_mean = 0.55
    spec = F * (e_sh * (1 - rough) + e_mean * rough)
    dif = base * (1 - metal) * (0.25 + 0.75 * np.clip(N @ L, 0, 1) + 0.25 * np.clip(N[..., 1], 0, 1))[..., None]
    col = np.clip((dif * (1 - F) + spec) * 0.8, 0, 1) ** (1 / 2.2)
    bgc = np.array(bg, np.float32)
    img = np.where(inside[..., None], col, bgc)
    img = img.reshape(size, ss, size, ss, 3).mean((1, 3))
    out = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
    _PREV[ck] = out
    return out
