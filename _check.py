# -*- coding: utf-8 -*-
"""Структурная проверка: пустых свёрток нет, все настройки на месте,
сохранение работает, кисти работают, оба языка полные."""
import os, sys, numpy as np, tkinter as tk
from tkinter import ttk
import i18n; i18n.load()
import Texelith as T
T.HAVE_DND = False

root = tk.Tk(); root.geometry('1500x950')
app = T.App(root); root.update()
app.show_classic(True); root.update()   # проверки ниже — классических вкладок

def walk(w, out):
    for c in w.winfo_children():
        if isinstance(c, T.Section):
            out.append((c.title, len(c.body.winfo_children())))
        walk(c, out)
    return out

secs = walk(root, [])
empty = [s for s, n in secs if n == 0]
print('свёрток: %d, пустых: %s' % (len(secs), empty or 'нет'))

NEED = ['s_split','s_area','s_fold','s_prof','s_det','s_seed','s_pad','s_tilt','s_thin',
        's_blade','s_bend','s_clm','s_clb','s_cls','s_domw','s_domh','brush_r','f_tol',
        'd_un','d_unr','d_ao','d_aor','d_lift','d_sat','d_aostr','d_aorad','d_aospread','d_aobulge','d_aotilt',
        'd_aogam','d_aocon','d_sstr','d_sbase','d_sveins','d_slay','d_spig','d_sedge',
        's_bent','m_nbent','d_sbr','d_stint','d_sgam','d_scon','d_spow','d_sbehind','d_sreach','r_hi','r_lo','r_flat','r_grow','r_mix','m_soft','m_hard','m_lrad',
        'm_rest','m_sat','m_hdet','m_wband','m_wmulti','m_wlum','m_wguid','m_fall','m_grad',
        'm_wext','m_wsfs','m_sfsaz','m_sfssm','m_sfschr','m_sfssh','m_sfsscale','m_sfslow','m_sfshigh','m_ndir','m_ndirk','m_ndirfix','m_extcut','dbrush_r','dbrush_s','gbrush_r','g_tol','g_soft','g_amt','g_flat',
        'm_tilt','m_hamp','m_nstr','m_nfine','m_nmid','m_nbig','m_nscale','m_aor','m_aos',
        'm_aowc','m_aown','m_rdamt','m_hgam','m_hcon','m_aogam','m_aocon','m_rgam','m_rcon',
        'm_f0','m_gloss','m_gvar','m_metal','m_tint','z_lo','z_hi','z_soft','z_blur',
        'zbrush_r','m_band','x_minz']
lost = [n for n in NEED if not isinstance(getattr(app, n, None), T.Slider)]
print('пропавших ползунков:', lost or 'нет')

# --- обе локализации без дыр
bad = [k for k, v in i18n.S.items() if not isinstance(v, tuple) or len(v) != 2
       or not v[0] or not v[1]]
print('строк: %d, неполных: %s' % (len(i18n.S), bad or 'нет'))

# --- прогон материала целиком, включая сохранение
app.nb.select(2)
app.load('test/tx_m_ruins_trim_04.dds'); root.update()
print('карт:', len(app.maps))
os.makedirs('out14', exist_ok=True)
outs = app.material_outputs(app.full_maps())
for k in ('_n','_nh','_spec','_ao','_h','_rough','_metal','_fixed'):
    T.save_rgba(outs[k], 'out14/demo%s.dds' % k, k in ('_spec','_ao'))
print('сохранено файлов:', len(os.listdir('out14')))

# --- кисти: подсказка, зона, битые места
app.paint_guide(120, 120, False, True)
app.paint_zone(200, 200, False, True)
app.paint_damage(60, 60, False, True)
root.update()
print('подсказка покрывает %.2f %%, зона %.1f %%, битых %.2f %%' % (
    100*float(np.abs(app.guide).mean()),
    100*float(app.zone_mask(app.mat.src.shape[:2], app.maps).mean()),
    100*float((app.damage_mask()>0.5).mean())))

app.recompute()
# --- все виды рисуются
for _, v in app.MAT_VIEWS:
    a = app.material_picture(v)
    assert a is not None and np.isfinite(np.asarray(a)).all(), v
print('видов материала отрисовано:', len(app.MAT_VIEWS))

# --- листва: все пресеты и все виды
app.nb.select(0); app.load('test/Vurt_PineBranches.dds'); root.update()
for k in T.FOL_PRESET_IDS:
    app.apply_fol_preset(i18n.t(k)); app.recompute()
    assert app.normal is not None
print('пресетов листвы проверено:', len(T.FOL_PRESET_IDS))
for _, v in app.FOL_VIEWS:
    assert app.fol_pic(v) is not None
print('видов листвы отрисовано:', len(app.FOL_VIEWS))

# --- правка нормалей
app.nb.select(4); app.load('test/vurt_FFleavesG2_n.dds'); root.update()
print('разбор нормали:', app.x_text.get('1.0','2.0').strip(), '…', len(app.x_text.get('1.0','end')), 'симв.')

# --- 3D просмотр
app.nb.select(3); app.load('test/tx_colony_stone02.dds'); root.update()
app.v3_redraw(False); root.update()
import time
for cmp_ in (False, True):
    app.v_cmp.set(cmp_)
    t=time.time(); app.v3_redraw(False); root.update()
    print('3D кадр%s: %.2f с' % (' (сравнение)' if cmp_ else '', time.time()-t))
for m in T.V3.MODES:
    app.v_mode.set(m); app.v3_redraw(False)
print('режимов параллакса:', len(T.V3.MODES))

# --- бесшовность: сдвиг картинки должен давать ровно такой же сдвиг карты
import numpy as _np, matgen as _MG
rng=_np.random.default_rng(0); N=256
nz=rng.random((N,N)).astype(_np.float32)
fk=_np.fft.fft2(nz); kk=_np.fft.fftfreq(N)[:,None]**2+_np.fft.fftfreq(N)[None,:]**2
nz=_np.real(_np.fft.ifft2(fk*_np.exp(-kk*400))); nz=(nz-nz.min())/_np.ptp(nz)
tex=_np.stack([nz,nz*0.9,nz*0.8],-1).astype(_np.float32)
pp=app.mat_params()
A=_MG.MaterialJob(tex).all_maps(pp)
B=_MG.MaterialJob(_np.roll(_np.roll(tex,37,0),53,1)).all_maps(pp)
worst=max(float(_np.abs(_np.roll(_np.roll(A[k],37,0),53,1)-B[k]).max())
          for k in ('height','normal','ao','spec','rough','diffuse'))
print('бесшовность карт, худшее расхождение: %.6f' % worst)
assert worst < 1e-5, 'карты не бесшовные'


# --- просвет
app.nb.select(1); app.load('test/vurt_FFleavesG2.dds'); root.update()
app.view_d.set('backlit'); app.recompute(); root.update()
_rgb,_a = app.fol_sss
import numpy as _n2, foliage as _FO
_m = app.src.mask
_v = _FO.vein_map(app.src)
_cov = _FO.fast_blur(app.src.alpha, 30)
blade = float(_a[_m & (_v < 0.08)].mean()); vein = float(_a[_m & (_v > 0.35)].mean())
openl = float(_a[_m & (_cov < 0.45)].mean()); deep = float(_a[_m & (_cov > 0.75)].mean())
print('просвет: пластина %.3f, жилки %.3f, открытый лист %.3f, в глубине %.3f'
      % (blade, vein, openl, deep))
assert vein < blade * 0.85, 'жилки должны быть темнее пластины'
assert deep < openl * 0.92, 'листья в глубине кроны должны быть темнее'
_bk0 = app._behind(app.src, app.norm)
front = float(_a[_m & (_bk0 < 0.1)].mean()); shaded = float(_a[_m & (_bk0 > 0.5)].mean())
print('просвет: лист без соседей позади %.3f, с листом позади %.3f' % (front, shaded))
assert shaded < front * 0.7, 'силуэт листа позади не темнит просвет'
# и главное: просвет не должен быть обводкой контура
_t = app.norm.t
_r = float(_n2.corrcoef(_a[_m], 1.0 - _t[_m])[0, 1])
print('связь просвета с расстоянием до края: %.2f (близко к 1 = светящийся контур)' % _r)
assert _r < 0.5, 'просвет выродился в обводку контура'
_ao = app.fol_ao
# behind > 0.5 означает «подо мной лежит лист», то есть я сверху
_front = float(_ao[_m & (_bk0 > 0.5)].mean())
_under = float(_ao[_m & (_bk0 < 0.1)].mean())
print('AO: среднее %.3f, минимум %.3f; лист сверху %.3f, лист снизу %.3f'
      % (float(_ao[_m].mean()), float(_ao[_m].min()), _front, _under))
assert _ao[_m].mean() > 0.6, 'AO слишком тёмное: запечённое AO листвы светлое с тёмными акцентами'
assert _ao[_m].min() < 0.55, 'AO без контактных теней'
assert _under < _front * 0.95, 'AO не следует за глубиной'
_bk = app._behind(app.src, app.norm)
print('листья позади видны на %.1f %% кроны' % (100 * float((_bk[_m] > 0.3).mean())))
assert 0.05 < (_bk[_m] > 0.3).mean() < 0.9, 'силуэты листьев позади не нашлись'
_z = app._depth(app.src, app.norm)
print('придуманная глубина: размах %.1f пикс., ступенек между листьями %d'
      % (float(_n2.ptp(_z[_m])), int(len(_n2.unique(_n2.round(_z[_m], 2))))))
for v in ('sss','backlit','ao','height','after','pair'):
    app.view_d.set(v); app.redraw()
print('видов на вкладке света:', 6)
o = app.foliage_outputs(['_sss'])
assert o['_sss'].shape[2] == 4
print('_sss сохраняется:', o['_sss'].shape)


# --- bent normal
app.nb.select(0); app.load('test/tx_leaves_03.dds'); root.update()
app.s_bent.set(0.0); app.recompute(); n0 = app.normal.copy()
app.s_bent.set(0.7); app.normal = None; app.recompute(); n1 = app.normal
_d = _n2.degrees(_n2.arccos(_n2.clip((n0*n1).sum(-1), -1, 1)))[app.src.mask]
print('листва, bent повернул нормаль на %.1f° в среднем, 95%% %.1f°'
      % (float(_d.mean()), float(_n2.percentile(_d, 95))))
assert 2 < _d.mean() < 60, 'bent normal не подмешался'
assert abs(float(_n2.sqrt((n1**2).sum(-1)).mean()) - 1) < 1e-3, 'нормаль не единичная'
app.nb.select(2); app.load('test/tx_colony_stone02.dds'); root.update()
app.m_nbent.set(0.0); app.recompute(); m0 = app.maps['normal'].copy()
app.m_nbent.set(0.6); app.recompute(); m1 = app.maps['normal']
_d2 = _n2.degrees(_n2.arccos(_n2.clip((m0*m1).sum(-1), -1, 1)))
print('материал, bent повернул нормаль на %.1f°' % float(_d2.mean()))
assert 2 < _d2.mean() < 70, 'bent normal в материале не подмешался'


# --- починка: отступ маски со знаком, смешивание, кнопка
app.nb.select(2); app.load('test/tx_colony_stone02.dds'); root.update()
app.r_on.set(True); app.r_grow.set(3); app.repair_changed(); app.apply_repair_now()
_mplus = float((app.damage_mask() > 0.5).mean())
app.r_grow.set(-3); app.repair_changed(); app.apply_repair_now()
_mminus = float((app.damage_mask() > 0.5).mean())
print('маска: отступ +3 -> %.1f %%, отступ -3 -> %.1f %%' % (100 * _mplus, 100 * _mminus))
assert _mminus < _mplus, 'отрицательный отступ не сжимает маску'
app.r_grow.set(2); app.r_mix.set(1.0); app._rep_cache = {}
_f1 = app.repaired(app.rgb).copy()
app.r_mix.set(0.4); app._rep_cache = {}
_f2 = app.repaired(app.rgb)
print('смешивание: полное меняет картинку на %.4f, на 0.4 — на %.4f'
      % (float(_n2.abs(_f1 - app.rgb).mean()), float(_n2.abs(_f2 - app.rgb).mean())))
assert _n2.abs(_f2 - app.rgb).mean() < _n2.abs(_f1 - app.rgb).mean(), 'смешивание не работает'
app.r_mix.set(1.0); app.repair_now(); root.update()
assert app.r_on.get() and app.view_m.get() == 'fixed', 'кнопка «починить» не сработала'
app.r_on.set(False); app.repair_changed(); app.recompute()

# --- подробность: по умолчанию всё на виду, скрытое подписано
for _tag, _idx in (('n', 0), ('d', 1), ('m', 2), ('x', 4)):
    app.nb.select(_idx); root.update()
    assert getattr(app, 'all_' + _tag).get(), 'простой режим не должен быть по умолчанию'
def _vis(p):
    n = 0
    def w(x):
        nonlocal n
        for c in x.winfo_children():
            if c.winfo_ismapped():
                n += 1
            w(c)
    w(p)
    return n
app.nb.select(2); root.update()
_full = _vis(app.panel_m)
app.all_m.set(False); app.detail_changed('m'); root.update()
_simple = _vis(app.panel_m)
_notes = len(app.detail_m.notes)
print('колонка материала: всего %d, в простом режиме %d, подписей «ещё N» %d'
      % (_full, _simple, _notes))
assert _simple < _full and _notes > 0, 'скрытое не подписано'
app.all_m.set(True); app.detail_changed('m'); root.update()
assert _vis(app.panel_m) == _full, 'настройки не вернулись'

# --- нормаль в обход высоты
app.nb.select(2); app.load('test/tx_colony_stone02.dds'); root.update()
app.m_wsfs.set(1.0); app.m_sfsaz.set(135); app.m_ndir.set(0.0); app.recompute()
_na = app.maps['normal'].copy()
app.m_ndir.set(1.0); app.m_ndirk.set(0.8); app.recompute()
_nb = app.maps['normal']
print('нормаль: обычная невязка с высотой %.3f, в обход высоты %.3f'
      % (_MG.slope_curl(_na), _MG.slope_curl(_nb)))
assert _MG.slope_curl(_nb) > _MG.slope_curl(_na) + 0.2, 'прямая нормаль не отличается'
assert abs(float(_n2.sqrt((_nb ** 2).sum(-1)).mean()) - 1) < 1e-3, 'вектор не единичный'
_tilt = float(_n2.degrees(_n2.arccos(_n2.clip(_nb[..., 2], -1, 1))).mean())
print('средний наклон прямой нормали: %.0f°' % _tilt)
assert 3 < _tilt < 45, 'крутизна вне разумного'
# подгонка под высоту: ротор должен обнуляться точно
_keepbent = app.m_nbent.get()
app.m_nbent.set(0.0)          # подмешанный bent сам ломает интегрируемость
app.m_ndirfix.set(1.0); app.recompute()
_nc = app.maps['normal']
print('после подгонки под высоту: невязка %.4f' % _MG.slope_curl(_nc))
assert _MG.slope_curl(_nc) < 0.02, 'подгонка не убрала ротор'
app.m_ndirfix.set(0.0); app.m_nbent.set(_keepbent)
app.m_ndir.set(0.0); app.m_wsfs.set(0.0); app.recompute()

# --- все веса высоты в нуле: ровная плита, а не подмена слоем
app.nb.select(2); app.load('test/tx_colony_stone02.dds'); root.update()
_keep = [sl.get() for sl in (app.m_wband, app.m_wmulti, app.m_wlum, app.m_wguid,
                             app.m_wext, app.m_wsfs)]
for sl in (app.m_wband, app.m_wmulti, app.m_wlum, app.m_wguid, app.m_wext, app.m_wsfs):
    sl.set(0.0)
app.recompute()
print('все веса высоты в нуле: размах %.5f, среднее %.3f'
      % (float(app.maps['height'].std()), float(app.maps['height'].mean())))
assert float(app.maps['height'].std()) < 1e-4, 'нулевые веса подменяются каким-то слоем'
for sl, v in zip((app.m_wband, app.m_wmulti, app.m_wlum, app.m_wguid,
                  app.m_wext, app.m_wsfs), _keep):
    sl.set(v)
app.recompute()
assert float(app.maps['height'].std()) > 0.01, 'веса не вернулись'

# --- глубина нейросетью
import depthnet as DN
print('onnxruntime:', DN.runtime_version() or 'нет')
assert app.net_id() in DN.MODEL_IDS
app.net_note(); assert app.m_netinfo.cget('text')
_ready = [k for k in DN.MODEL_IDS if DN.have_model(k)]
print('моделей на месте:', _ready or 'нет')
if _ready and DN.have_runtime():
    _im = T.Image.open('test/tx_colony_stone02.dds').convert('RGB')
    _sm = _n2.asarray(_im.resize((512, 512), T.Image.LANCZOS), _n2.float32) / 255
    for _k in _ready:
        import time as _t
        _t0 = _t.time()
        _raw = DN.compute(_sm, _k, fine=False)
        _dp = DN.shape(_raw, 512, 0.3, 0.4, 1.0)
        assert _dp.shape == (512, 512) and 0 <= _dp.min() and _dp.max() <= 1
        # периодичность: скачок через шов должен быть как у соседних столбцов
        _e = abs(_dp[:, 0] - _dp[:, -1]).mean()
        _i = 0.5 * (abs(_dp[:, 1] - _dp[:, 0]).mean() + abs(_dp[:, -1] - _dp[:, -2]).mean())
        _ey = abs(_dp[0] - _dp[-1]).mean()
        _iy = 0.5 * (abs(_dp[1] - _dp[0]).mean() + abs(_dp[-1] - _dp[-2]).mean())
        print('%-6s %.2f с, шов %.4f против %.4f по X, %.4f против %.4f по Y'
              % (_k, _t.time() - _t0, _e, _i, _ey, _iy))
        assert _e < 3 * _i + 1e-3 and _ey < 3 * _iy + 1e-3, 'нейросетевая глубина не бесшовна'
    # раздельные веса диапазонов: крупное на нуле выравнивает уровни,
    # мелкое отвечает за фактуру — и оба должны оставаться бесшовными
    _b, _h = DN.bands(_raw, DN.macro_sigma(512, 0.3))
    print('диапазоны: крупное %.3f, мелкое %.3f, сумма сходится: %s'
          % (float(_b.std()), float(_h.std()),
             bool(_n2.abs(_b + _h - _raw).max() < 1e-4)))
    assert _b.std() > 0.2 and _h.std() > 0.1, 'разложение вырождено'
    assert _n2.abs(_b + _h - _raw).max() < 1e-4
    _flat = DN.shape(_raw, 512, 0.3, 0.0, 1.0)
    _both = DN.shape(_raw, 512, 0.3, 1.0, 1.0)
    _lowvar = lambda a: float(DN.bands(a, DN.macro_sigma(512, 0.3))[0].std())
    print('крупного в карте: с весом 1 — %.3f, с весом 0 — %.3f'
          % (_lowvar(_both), _lowvar(_flat)))
    assert _lowvar(_flat) < 0.5 * _lowvar(_both), 'вес крупного ни на что не влияет'
    for _nm, _a in (('крупное 0', _flat), ('крупное 1', _both)):
        _e1 = abs(_a[:, 0] - _a[:, -1]).mean()
        _i1 = 0.5 * (abs(_a[:, 1] - _a[:, 0]).mean() + abs(_a[:, -1] - _a[:, -2]).mean())
        assert _e1 < 3 * _i1 + 1e-3, 'диапазон %s не бесшовен' % _nm
    _dt = DN.shape(DN.compute(_sm, _ready[0], fine=True, tile=256), 512, 0.3, 0.4, 1.0)
    _et = abs(_dt[:, 0] - _dt[:, -1]).mean()
    _it = 0.5 * (abs(_dt[:, 1] - _dt[:, 0]).mean() + abs(_dt[:, -1] - _dt[:, -2]).mean())
    print('окнами: шов %.4f против %.4f' % (_et, _it))
    assert _et < 3 * _it + 1e-3, 'нейросетевая глубина окнами не бесшовна'
    # кладётся во внешний слой и доходит до высоты
    app.nb.select(2); app.load('test/tx_colony_stone02.dds'); root.update()
    app.m_wext.set(0.0); app.recompute(); _h0 = app.maps['height'].copy()
    app._net_tex = id(app.rgb); app._net_done(DN.compute(_sm, _ready[0]), _ready[0]); root.update()
    app.m_netlow.set(0.0); app.net_apply(); root.update()
    assert app.extern_full is not None
    app.recompute(); _h1 = app.maps['height']
    print('внешняя глубина от сети меняет высоту на %.3f'
          % float(abs(_h1 - _h0).mean()))
    assert app.extern_full is not None and abs(_h1 - _h0).mean() > 0.01

# --- глубина от сети в листве: порядок листьев и что он меняет
if _ready and DN.have_runtime():
    app.nb.select(1); app.load('test/vurt_FFleavesG2.dds'); root.update()
    app.d_fnet.set(0.0); app.recompute()
    _ao0 = app.fol_ao.copy(); _z0 = app._depth(app.src, app.norm).copy()
    _rgba = app.rgba_full
    _al = _rgba[..., 3:4]
    _card = (_rgba[..., :3] * _al + 0.5 * (1 - _al)).astype(_n2.float32)
    app._net_for = 'foliage'
    app._net_tex = id(app.rgb); app._net_done(DN.compute(_card, _ready[0]), _ready[0]); root.update()
    app.d_fnet.set(0.8); app.recompute()
    _ao1 = app.fol_ao; _z1 = app._depth(app.src, app.norm)
    _mm = app.src.mask
    # порядок листьев должен перестать быть жребием: глубина листа теперь
    # связана с ответом сети, а раньше не была связана ни с чем
    _nt = T.resize_to(app._fnet_raw, _mm.shape)
    _r0 = abs(float(_n2.corrcoef(_z0[_mm], _nt[_mm])[0, 1]))
    _r1 = abs(float(_n2.corrcoef(_z1[_mm], _nt[_mm])[0, 1]))
    print('листва: связь глубины с ответом сети %.2f -> %.2f, AO изменилось на %.3f'
          % (_r0, _r1, float(abs(_ao1 - _ao0)[_mm].mean())))
    assert _r1 > _r0 + 0.2, 'сеть не влияет на порядок листьев'
    assert abs(_ao1 - _ao0)[_mm].mean() > 0.01, 'AO не заметило нового порядка'
    app.d_fnet.set(0.0); app.recompute()

# --- высота из освещения и из затенения
import math as _mt
def _synth(n=256, seed=0):
    rng = _n2.random.default_rng(seed)
    k = _n2.fft.fftfreq(n)
    kk = _n2.sqrt(k[:, None] ** 2 + k[None, :] ** 2)
    F = _n2.fft.fft2(rng.normal(size=(n, n))) * _n2.exp(-(kk * n / 26.0) ** 2)
    h = _n2.real(_n2.fft.ifft2(F)).astype(_n2.float32)
    return (h - h.min()) / max(float(h.max() - h.min()), 1e-6)

def _render(h, az, elev=0.55, amp=18.0):
    gy, gx = _MG.grad_wrap(h * amp)
    nn = _n2.dstack([-gx, -gy, _n2.ones_like(gx)])
    nn /= _n2.sqrt((nn * nn).sum(-1, keepdims=True))
    L = _n2.array([_mt.cos(az) * _mt.cos(elev), -_mt.sin(az) * _mt.cos(elev),
                   _mt.sin(elev)], _n2.float32)
    return _n2.clip((nn * L).sum(-1), 0, 1).astype(_n2.float32)

def _corr(a, b):
    a = (a - a.mean()) / max(float(a.std()), 1e-9)
    b = (b - b.mean()) / max(float(b.std()), 1e-9)
    return float((a * b).mean())

_h = _synth()
_ang = 125.0
_img = _render(_h, _mt.radians(_ang))
_est, _conf = _MG.light_azimuth(_img)
_err = abs((_mt.degrees(_est) - _ang + 90) % 180 - 90)
_rec = _MG.height_from_shading(_img, _mt.radians(_ang))
_naive = _MG.fast_blur(_img, 1.5)
print('свет: истина %.0f°, оценка %.1f° (ошибка по оси %.1f°, уверенность %.2f)'
      % (_ang, _mt.degrees(_est) % 360, _err, _conf))
print('высота: интеграл вдоль света %.3f, яркость-как-высота %.3f'
      % (_corr(_rec, _h), _corr(_naive, _h)))
assert _err < 8, 'направление света определяется неверно'
assert _corr(_rec, _h) > 0.5, 'интеграл вдоль света не восстанавливает рельеф'
assert abs(_corr(_naive, _h)) < 0.2, 'проверка бессмысленна: яркость и так равна высоте'
# бесшовность обоих решателей
for _nm, _a in (('свет', _rec), ('затенение', _MG.height_from_cavity(
        _MG.cavity_ao(_h * 30, radius=28, strength=0.9)))):
    _sh = _n2.roll(_n2.roll(_a, 37, 0), 53, 1)
    _b = (_MG.height_from_shading(_n2.roll(_n2.roll(_img, 37, 0), 53, 1),
                                  _mt.radians(_ang)) if _nm == 'свет'
          else _MG.height_from_cavity(_n2.roll(_n2.roll(
              _MG.cavity_ao(_h * 30, radius=28, strength=0.9), 37, 0), 53, 1)))
    print('  %s: сдвиг картинки даёт тот же сдвиг карты, расхождение %.7f'
          % (_nm, float(_n2.abs(_sh - _b).max())))
    assert float(_n2.abs(_sh - _b).max()) < 1e-4, '%s не бесшовно' % _nm
# краска и падающие тени: обе поправки должны помогать там, где мешают
def _paintfield(n, amp, seed=7):
    rng = _n2.random.default_rng(seed)
    k = _n2.fft.fftfreq(n)
    kk = _n2.sqrt(k[:, None] ** 2 + k[None, :] ** 2)
    ch = []
    for _ in range(3):
        F = _n2.fft.fft2(rng.normal(size=(n, n))) * _n2.exp(-(kk * n / 18.0) ** 2)
        a = _n2.real(_n2.fft.ifft2(F))
        ch.append((a - a.mean()) / max(float(a.std()), 1e-6))
    return _n2.clip(_n2.array([0.45, 0.40, 0.34], _n2.float32)
                    * _n2.exp(amp * _n2.dstack(ch)), 0.05, 0.95)

def _shadowfield(n, seed=3):
    rng = _n2.random.default_rng(seed)
    k = _n2.fft.fftfreq(n)
    kk = _n2.sqrt(k[:, None] ** 2 + k[None, :] ** 2)
    F = _n2.fft.fft2(rng.normal(size=(n, n))) * _n2.exp(-(kk * n / 6.0) ** 2)
    a = _n2.real(_n2.fft.ifft2(F))
    a = (a - a.mean()) / max(float(a.std()), 1e-6)
    return (0.45 + 0.55 / (1.0 + _n2.exp(-a * 8.0))).astype(_n2.float32)

_N = 256
_h2 = _synth(_N, seed=1)
_gy, _gx = _MG.grad_wrap(_h2 * 18.0)
_nn = _n2.dstack([-_gx, -_gy, _n2.ones_like(_gx)])
_nn /= _n2.sqrt((_nn * _nn).sum(-1, keepdims=True))
_Lv = _n2.array([_mt.cos(_mt.radians(125)) * _mt.cos(0.55),
                 -_mt.sin(_mt.radians(125)) * _mt.cos(0.55), _mt.sin(0.55)], _n2.float32)
_sh = _n2.clip((_nn * _Lv).sum(-1), 0, 1) * _shadowfield(_N)
_rgbp = _MG.lin_to_srgb(_n2.clip(_paintfield(_N, 1.0) * _sh[..., None], 0, 1)).astype(_n2.float32)
_Lp = _MG.luminance(_MG.srgb_to_lin(_rgbp))
_az2 = _mt.radians(125)
_plain = _corr(_MG.height_from_shading(_Lp, _az2), _h2)
_fix = _corr(_MG.height_from_shading(_Lp, _az2, g=_MG.shading_signal(
    _rgbp, chroma=1.0, shadow=1.0)), _h2)
print('краска и падающая тень: без поправок %.3f, с поправками %.3f' % (_plain, _fix))
assert _fix > _plain + 0.05, 'поправки на краску и тени не помогают'
# на одноцветной картинке поправка по цвету не должна ничего портить
_rgbm = _n2.dstack([_MG.lin_to_srgb(_n2.clip(0.42 * _sh, 0, 1))] * 3).astype(_n2.float32)
_a = _MG.chroma_albedo(_rgbm, 24.0)
print('одноцветная текстура: поправка по цвету даёт размах %.5f' % float(_a.std()))
assert float(_a.std()) < 1e-3, 'на монохроме поправка по цвету должна молчать'
# частотный фильтр слоя
_b1 = _MG.weigh_bands(_h2, _MG.band_radius(_N, 0.3), 0.0, 1.0)
_b2 = _MG.weigh_bands(_h2, _MG.band_radius(_N, 0.3), 1.0, 1.0)
print('фильтр слоя: крупного осталось %.3f из %.3f'
      % (float(_MG.split_bands(_b1, _MG.band_radius(_N, 0.3))[0].std()),
         float(_MG.split_bands(_b2, _MG.band_radius(_N, 0.3))[0].std())))
assert _MG.split_bands(_b1, _MG.band_radius(_N, 0.3))[0].std() < 0.6 * \
    _MG.split_bands(_b2, _MG.band_radius(_N, 0.3))[0].std(), 'фильтр слоя не работает'

_cav = _MG.cavity_ao(_h * 30, radius=28, strength=0.9)
print('из затенения: связь с истиной %.3f' % _corr(_MG.height_from_cavity(_cav, 0.1), _h))
assert _corr(_MG.height_from_cavity(_cav, 0.1), _h) > 0.5

# --- значок
_ic = T.app_icon(64)
print('значок: %s, непрозрачных пикселей %.0f %%'
      % (_ic.size, 100 * float((_n2.asarray(_ic)[..., 3] > 128).mean())))
assert _ic.size == (64, 64) and (_n2.asarray(_ic)[..., 3] > 128).mean() > 0.4

# --- узлы
import nodes as ND
app.load('test/tx_colony_stone02.dds'); root.update()
print('вкладок: %d, узлов в реестре: %d, в графе по умолчанию: %d'
      % (app.nb.index('end'), len(ND.REG), len(app.graph.nodes)))
assert app.nb.index('end') == 5          # вкладка узлов заменена узловым окном
app.g_eval(); root.update()
# узел считается в отдельном потоке — ждём, пока окно заберёт ответ
import time as _tm
_t0 = _tm.time()
while getattr(app, '_g_busy', False) and _tm.time() - _t0 < 120:
    root.update(); _tm.sleep(0.05)
root.update()
assert 'узел' in app.g_status.cget('text') or 'node' in app.g_status.cget('text')
# у каждого типа узла есть название и подписи портов на обоих языках
for _k, _sp in ND.REG.items():
    assert ('nd.' + _k) in i18n.S, 'нет названия узла %s' % _k
    for _p, _ in list(_sp.ins) + list(_sp.outs):
        assert ('np.' + _p) in i18n.S, 'нет подписи порта %s' % _p
    for _pr in _sp.params:
        assert ('np.' + _pr[0]) in i18n.S, 'нет подписи параметра %s' % _pr[0]
# кольца запрещены
_g = ND.Graph()
_a = _g.add('source'); _b = _g.add('height_lum'); _c = _g.add('tone')
assert _g.connect(_a, 'rgb', _b, 'rgb') and _g.connect(_b, 'h', _c, 'h')
assert not _g.connect(_c, 'h', _b, 'rgb'), 'кольцо должно запрещаться'
assert not _g.connect(_a, 'rgb', _c, 'h'), 'разные типы не должны соединяться'
# сохранение и чтение графа
_txt = app.graph.to_json()
_g2 = ND.Graph(); _g2.from_json(_txt)
assert len(_g2.nodes) == len(app.graph.nodes) and len(_g2.links) == len(app.graph.links)
print('граф сохраняется и читается: узлов %d, связей %d' % (len(_g2.nodes), len(_g2.links)))
# каждый тип узла считается
_sm = _n2.asarray(T.Image.open('test/tx_colony_stone02.dds').convert('RGB')
                  .resize((256, 256)), _n2.float32) / 255
_ctx = {'src': _n2.ascontiguousarray(_sm), 'key': 'chk'}
_skip = set() if (_ready and DN.have_runtime()) else {'depth_net'}
import esrgan as _ES
_skip |= {k for k, m in (('esrgan_normal', 'normal'), ('esrgan_franken', 'franken'))
          if not _ES.have_model(m)}
import deepbump as _DB, supermat_bridge as _SB
if not (_DB.have_model() and DN.have_runtime()):
    _skip.add('deepbump')
if _SB.problem():
    _skip.add('supermat')
_ok = 0
for _k in ND.REG:
    if _k in _skip:
        continue
    _gg = ND.Graph()
    _src = _gg.add('source')
    _nid = _gg.add(_k) if _k != 'source' else _src
    if _k != 'source':
        for _port, _kind in ND.REG[_k].ins:
            if _kind == ND.RGB:
                _gg.connect(_src, 'rgb', _nid, _port)
            elif _kind in (ND.GRAY, ND.ANY):
                _h = _gg.add('height_lum')
                _gg.connect(_src, 'rgb', _h, 'rgb')
                _gg.connect(_h, 'h', _nid, _port)
            elif _kind == ND.NRM:
                _h = _gg.add('height_lum'); _nn = _gg.add('normal')
                _gg.connect(_src, 'rgb', _h, 'rgb')
                _gg.connect(_h, 'h', _nn, 'h')
                _gg.connect(_nn, 'n', _nid, _port)
    _out = _gg.eval(_nid, _ctx)
    assert any(isinstance(v, _n2.ndarray) for v in _out.values()) or _k == 'out'
    _ok += 1
print('узлов посчиталось: %d из %d' % (_ok, len(ND.REG) - len(_skip)))

# ======================================================= сборка 21 ======
import esrgan as _ES, view3d as _V3, normalfix as _NX
from PIL import Image as _Im

def _load01(p):
    return _n2.asarray(_Im.open(p).convert('RGB'), _n2.float32) / 255

# --- нормаль по освещению (R и G) повторяет файл, собранный автором в Фотошопе
_wall = _load01('test/tx_ex_stronghold_wall12.dds')
_his = _load01('test/ps_fix.dds')
_L = _MG.luminance(_MG.srgb_to_lin(_wall))
_g = _MG.shading_signal(_wall, 40.0, 1.5, 0.3, 48.0, 8, None)
_rg = _MG.normal_from_shading(_L, None, 0.74, 1.0, _g, True, False, 0.6, 0.82, 2.0) * 0.5 + 0.5
_H, _W = _his.shape[:2]
_rgs = _n2.stack([_n2.asarray(_Im.fromarray(_rg[..., i]).resize((1237, 1237), _Im.BILINEAR))
                  for i in range(3)], -1)[:_H, :_W]
_cr = [float(_n2.corrcoef(_rgs[..., i].ravel(), _his[..., i].ravel())[0, 1]) for i in range(2)]
print('нормаль R и G против файла автора: R %.3f, G %.3f' % tuple(_cr))
assert min(_cr) > 0.9, 'нормаль по освещению не повторяет способ автора'

# --- превью на половинной копии совпадает с тем, что сохранится
_half = _n2.asarray(_Im.fromarray((_wall * 255).astype(_n2.uint8)).resize(
    (1024, 1024), _Im.LANCZOS), _n2.float32) / 255
_p = app.mat_params()
_F = _MG.MaterialJob(_wall).all_maps(_p)
_Hm = _MG.MaterialJob(_half, 0.5).all_maps(_MG.at_scale(_p, 0.5))
def _dn(a):
    return _n2.asarray(_Im.fromarray(a.astype(_n2.float32)).resize((1024, 1024), _Im.BILINEAR))
_ch = float(_n2.corrcoef(_dn(_F['height']).ravel(), _Hm['height'].ravel())[0, 1])
_ca = float(_n2.corrcoef(_dn(_F['ao']).ravel(), _Hm['ao'].ravel())[0, 1])
_avg = _F['normal'].reshape(1024, 2, 1024, 2, 3).mean(axis=(1, 3))
_rms = float(_n2.sqrt(((_Hm['normal'][..., :2] - _avg[..., :2]) ** 2).mean()))
print('превью против сохранённого: высота %.3f, AO %.3f, нормаль СКО %.3f' % (_ch, _ca, _rms))
assert _ch > 0.99 and _ca > 0.98 and _rms < 0.05, 'превью показывает не то, что сохранится'

# --- нормаль ↔ высота
_hh = _MG.fast_blur(_n2.random.default_rng(1).standard_normal((256, 256)).astype(_n2.float32), 4) * 60
for _dx in (True, False):
    _nn = _MG.normal_from_heightmap(_hh, _dx)
    _back = _MG.normal_from_heightmap(_MG.height_from_normal(_nn, _dx), _dx)
    assert _MG.angle_between(_nn, _back) < 1.0, 'туда-обратно расходится'
print('нормаль → высота → нормаль: %.2f°' % _MG.angle_between(_nn, _back))
_Im.fromarray((_n2.linspace(0, 1, 256 * 256).reshape(256, 256) * 65535).astype(_n2.uint16)
              ).save('test/_h16.png')
app.nb.select(4); app.load('test/_h16.png'); root.update()
_lev = len(_n2.unique(app.rgb[..., 0]))
print('16-битная высота: уровней после загрузки %d' % _lev)
assert _lev > 10000, '16 бит обрезаны до 8'
app.x_src.set(T.t('x.src.height')); app.x_src_changed(); app.recompute(); root.update()
assert app.x_conv is not None and app.x_conv.shape[-1] == 3, 'высота не перевелась в нормаль'
app.x_src.set(T.t('x.src.normal')); os.remove('test/_h16.png')

# --- параллакс OpenMW
assert abs(_MG.parallax_range_cm(128) - 7.06) < 0.01
_a, _i = _MG.parallax_alpha(_n2.full((64, 64), 0.37, _n2.float32), 128, 3.0, dither=False)
assert int(round(float(_a.mean()) * 255)) == 128, 'ровная высота не легла в 128'
_a, _i = _MG.parallax_alpha(_hh, 128, 3.5, 'median', dither=False)
_span = (float(_n2.percentile(_a, 99.5)) - float(_n2.percentile(_a, 0.5))) * _i['avail_cm']
print('параллакс: запрошено 3.5 см, в альфе %.2f см, занято уровней %d' % (_span, _i['used']))
assert abs(_span - 3.5) < 0.1
_a2, _i2 = _MG.parallax_alpha(_hh, 128, 20.0, 'median', dither=False)
assert _i2['depth_cm'] <= _i2['avail_cm'] + 1e-6 and _i2['clipped'] < 0.05, 'не урезано до предела'
# направление смещения в 3D совпадает с обычным смещением
_N = 128
_hq = _n2.zeros((_N, _N), _n2.float32); _hq[48:80, 48:80] = 1
_dq = _n2.full((_N, _N, 3), 0.3, _n2.float32); _dq[48:80, 48:80] = [1, 0.2, 0.2]
_mp = dict(height=_hq, normal=_n2.dstack([_n2.zeros((_N, _N)), _n2.zeros((_N, _N)),
                                           _n2.ones((_N, _N))]).astype(_n2.float32),
           diffuse=_dq, ao=_n2.ones((_N, _N), _n2.float32),
           spec=_n2.zeros((_N, _N, 3), _n2.float32), shininess=_n2.zeros((_N, _N), _n2.float32))
def _cy(im):
    m = (im[..., 0] > 0.6) & (im[..., 1] < 0.35)
    return float(_n2.nonzero(m)[0].mean())
_y0 = _cy(_V3.render(dict(_mp, nh_alpha=_n2.full((_N, _N), 0.5, _n2.float32)), 300, 220,
                     yaw=0.0, pitch=1.0, dist=2.2, tiles=1.0, mode='openmw', shadow=False))
_yf = _cy(_V3.render(_mp, 300, 220, yaw=0.0, pitch=1.0, dist=2.2, tiles=1.0, mode='flat'))
_ys = _cy(_V3.render(dict(_mp, nh_alpha=_n2.zeros((_N, _N), _n2.float32)), 300, 220,
                     yaw=0.0, pitch=1.0, dist=2.2, tiles=1.0, mode='openmw', shadow=False))
_yo = _cy(_V3.render(dict(_mp, height=_n2.zeros((_N, _N), _n2.float32)), 300, 220, yaw=0.0,
                     pitch=1.0, dist=2.2, tiles=1.0, depth=0.04, mode='offset', shadow=False))
print('3D: 128 не сдвигает (%.1f против %.1f), утопленное уходит туда же, куда в смещении'
      % (_y0, _yf))
assert abs(_y0 - _yf) < 0.5 and (_ys - _yf) * (_yo - _yf) > 0
# нормаль, согласованная с глубиной: превью и полный размер одинаковы
_q = dict(_p, plx_normal=True, plx_depth=3.0)
_F2 = _MG.MaterialJob(_wall).all_maps(_q)
_H2 = _MG.MaterialJob(_half, 0.5).all_maps(_MG.at_scale(_q, 0.5))
def _tl(n):
    return float(_n2.degrees(_n2.arccos(_n2.clip(n[..., 2], -1, 1))).mean())
print('нормаль под глубину 3 см: наклон %.1f° полный, %.1f° превью'
      % (_tl(_F2['normal']), _tl(_H2['normal'])))
assert abs(_tl(_F2['normal']) - _tl(_H2['normal'])) < 1.5

# --- рельеф в диффуз
_d = _n2.random.default_rng(0).random((64, 64, 3)).astype(_n2.float32)
for _m in _MG.RELIEF_MODES:
    assert _n2.abs(_MG.relief_into_diffuse(_d, _hh[:64, :64] / 60 + 0.5, _m, 0.0) - _d).max() == 0
for _m in ('overlay', 'soft'):
    assert _n2.abs(_MG.relief_into_diffuse(_d, _n2.full((64, 64), 0.5, _n2.float32), _m, 2.0,
                                           sat_keep=0) - _d).max() < 1e-6
_r1 = _MG.MaterialJob(_half).all_maps(dict(_p, rd_amt=1.0, rd_mode='overlay', rd_src='height'))
_dd = float(_n2.abs(_r1['diffuse'] - _r1['diffuse_base']).max(-1).mean() * 255)
print('рельеф в диффуз, перекрытие силой 1: в среднем %.1f ступени из 255' % _dd)
assert 3 < _dd < 30

# --- ESRGAN: веса читаются без PyTorch, граф считает как PyTorch, окна бесшовны
if _ES.have_onnx_lib():
    import onnxruntime as _ort, tempfile as _tf
    _xin = _n2.load('test/esr_tiny_in.npy'); _yref = _n2.load('test/esr_tiny_out.npy')
    for _f in ('test/esr_tiny_zip.pth', 'test/esr_tiny_legacy.pth'):
        _op = os.path.join(_tf.gettempdir(), 'texelith_tiny.onnx')
        _ES.to_onnx(_ES.load_pth(_f), _op)
        _s = _ort.InferenceSession(_op, providers=['CPUExecutionProvider'])
        _err = float(_n2.abs(_s.run(None, {'x': _xin})[0] - _yref).max())
        assert _err < 1e-4, 'перевод весов ESRGAN расходится с PyTorch'
    _img = _n2.random.default_rng(2).random((96, 128, 3)).astype(_n2.float32)
    _o1 = _ES.run(_s, _img, tile=40, pad=24)
    _o2 = _ES.run(_s, _n2.roll(_n2.roll(_img, 17, 0), 29, 1), tile=40, pad=24)
    _seam = float(_n2.abs(_n2.roll(_n2.roll(_o1, 17, 0), 29, 1) - _o2).max())
    print('ESRGAN: расхождение с PyTorch %.1e, сдвиг картинки даёт тот же сдвиг карты %.1e'
          % (_err, _seam))
    assert _seam < 1e-4
else:
    print('ESRGAN: пакет onnx не установлен, перевод весов не проверен')

# --- запись: расширение решает формат, у нормали мипы нормируются
_tmpd = __import__('tempfile').mkdtemp()
T.save_rgba(T.rgb_rgba(_n2.full((8, 8, 3), 0.5, _n2.float32)), os.path.join(_tmpd, 'a.png'), False)
assert open(os.path.join(_tmpd, 'a.png'), 'rb').read(4) == b'\x89PNG', 'в .png записан не PNG'
_nm = _MG.normal_from_heightmap(_hh[:64, :64], True) * 0.5 + 0.5
T.save_rgba(T.rgb_rgba(_nm), os.path.join(_tmpd, 'n.dds'), False, normal=True)
_mip = _n2.frombuffer(open(os.path.join(_tmpd, 'n.dds'), 'rb').read()[128 + 64 * 64 * 4:
                      128 + 64 * 64 * 4 + 32 * 32 * 4], _n2.uint8).reshape(32, 32, 4)
_v = _mip[..., [2, 1, 0]].astype(_n2.float32) / 127.5 - 1
print('мип нормали: длина вектора %.3f' % float(_n2.sqrt((_v ** 2).sum(-1)).mean()))
assert abs(float(_n2.sqrt((_v ** 2).sum(-1)).mean()) - 1) < 0.02, 'мипы нормали укорочены'

# --- пакет: работает в своём потоке и не трогает окно; граф — тоже
import shutil as _sh
_sh.rmtree(_tmpd)
_tmpd = __import__('tempfile').mkdtemp()
for _f in ('tx_colony_stone02.dds', 'tx_cointop_01.dds'):
    _sh.copy('test/' + _f, _tmpd)
app.nb.select(2); app.load('test/tx_colony_stone02.dds'); root.update()
for _k, _v in app.saveflags.items():
    _v.set(_k in ('_nh', '_ao'))
app.m_rdamt.set(1.0)
T.filedialog.askdirectory = lambda **k: _tmpd
T.messagebox.showinfo = lambda *a, **k: None
app.batch('material')
_t0 = _tm.time()
while getattr(app, '_bat_out', None) is None and _tm.time() - _t0 < 600:
    root.update(); _tm.sleep(0.05)
root.update()
_made = sorted(f for f in os.listdir(_tmpd) if f.endswith(('_nh.dds', '_ao.dds')))
print('пакет материала: %s' % _made)
assert len(_made) == 4 and app._bat_out[0] == 2
app.m_rdamt.set(0.0)
app.g_reset(); root.update()
_t0 = _tm.time()
while getattr(app, '_g_busy', False) and _tm.time() - _t0 < 120:
    root.update(); _tm.sleep(0.05)
app._bat_out = None
app.g_batch()
_t0 = _tm.time()
while app._bat_out is None and _tm.time() - _t0 < 600:
    root.update(); _tm.sleep(0.05)
_gm = sorted(f for f in os.listdir(_tmpd) if f.endswith('_n.dds'))
print('пакет по графу: %s' % _gm)
assert len(_gm) == 2
_sh.rmtree(_tmpd)

# --- превью по умолчанию освещает рассеянным светом как OpenMW (ровно)
assert app.amb_hemi.get() is False


# --- исправления по ревью
import repair as _RP
_v = _n2.zeros((4, 4, 3), _n2.float32); _v[..., 0] = 0.8 * _n2.sin(_n2.radians(36.87))
_v[..., 2] = 0.8 * _n2.cos(_n2.radians(36.87))
_fx = _NX.fix(_v * 0.5 + 0.5) * 2 - 1
_ang = float(_n2.degrees(_n2.arccos(_fx[0, 0, 2] / _n2.linalg.norm(_fx[0, 0]))))
print('ремонт нормали: вектор длиной 0.8 под 36.9° остался под %.1f°' % _ang)
assert abs(_ang - 36.87) < 0.5
_img = _n2.clip(_n2.random.default_rng(0).random((96, 96, 3)) * 0.4 + 0.3, 0, 1).astype(_n2.float32)
_img[:, :8] = 1.0; _img[:, -5:] = 1.0
_b1, _ = _RP.exposure_masks(_img)
_b2, _ = _RP.exposure_masks(_n2.roll(_img, (11, 23), (0, 1)))
assert float(_n2.abs(_n2.roll(_b1, (11, 23), (0, 1)) - _b2).max()) < 1e-5, 'маска битых мест с швом'
_gd = _n2.zeros((64, 64), _n2.float32); _gd[10:20, 10:20] = -1
assert float(T.resize_signed(_gd, (32, 32)).min()) < -0.5, 'подсказка «ниже» пропадает'
_j = _MG.MaterialJob(_half[:256, :256].copy())
_q = dict(_p, ao_w_norm=1.0, ao_w_cav=0.0)
_ao1 = _j.all_maps(dict(_q, n_strength=0.5))['ao'].copy()
_ao2 = _j.all_maps(dict(_q, n_strength=2.5))['ao']
assert float(_n2.abs(_ao1 - _ao2).mean()) > 1e-3, 'AO не обновляется от силы нормали'
_q = dict(_p, h_from_delit=True)
_ha = _j.all_maps(dict(_q, restore=0.0))['height'].copy()
_hb = _j.all_maps(dict(_q, restore=1.0))['height']
assert float(_n2.abs(_ha - _hb).mean()) > 1e-4, 'высота не обновляется от «вернуть зерно»'
print('ревью: кэш AO и высоты, подсказка «ниже», бесшовная маска — в порядке')

# --- мелкие битые пятна чинятся и заплаткой; у вкладки «Узлы» есть свой вид
_im = _n2.clip(_n2.random.default_rng(5).random((128, 128, 3)) * 0.3 + 0.35, 0, 1).astype(_n2.float32)
_mk = _n2.zeros((128, 128), _n2.float32); _mk[60:65, 60:64] = 1.0     # 20 пикселей
_im2 = _im.copy(); _im2[60:65, 60:64] = 1.0
_fx = _RP.repair(_im2, _mk, 'patch')
print('мелкое пятно 20 px: после заплатки яркость %.2f (было 1.00)' % float(_fx[60:65, 60:64].mean()))
assert float(_fx[60:65, 60:64].mean()) < 0.6, 'мелкие пятна не чинятся'
# + вид скрытой вкладки узлов
assert len(app.views()) == app.nb.index('end') + 1, 'у вкладки нет своего вида'

# ======================================================= сборка 22 ======
import matgen as _MG22, nodes as _ND22, project as _PR22, texio as _TX22
import deepbump as _DB22, supermat_bridge as _SB22, editor as _ED22
import json as _js, shutil as _sh, tempfile as _tf
_rng22 = _n2.random.default_rng(22)

# --- баланс нормаль ↔ высота: точное восстановление, масштаб, знак, раскладка
_N = 192
_Ht = (_MG22.fast_blur(_rng22.normal(size=(_N, _N)).astype(_n2.float32), 3) * 40
       + _MG22.fast_blur(_rng22.normal(size=(_N, _N)).astype(_n2.float32), 25) * 300)
_nt = _MG22.normal_from_heightmap(_Ht, True)
for _b in (0.0, 0.4, 1.0):
    _H, _nn, _inf = _MG22.balance_height_normal(_nt, 0.01 * _Ht + 3.0, _b, 0.0, True)
    _e = float(_n2.abs((_H - _H.mean()) - (_Ht - _Ht.mean())).mean() / _Ht.std())
    assert _e < 1e-3 and abs(_inf['scale'] - 100.0) < 0.5 and _inf['curl_out'] < 1e-4, (_b, _e, _inf)
_H, _nn, _inf = _MG22.balance_height_normal(_nt, -_Ht, 0.5, 0.0, True)
assert _inf['scale'] < 0 and _inf['corr'] < -0.99, 'вывернутая высота должна давать минус'
_H1, _n1, _ = _MG22.balance_height_normal(_nt, _Ht * 0.3, 0.6, 0.0, True)
_H2, _n2b, _ = _MG22.balance_height_normal(_nt * [1, -1, 1], _Ht * 0.3, 0.6, 0.0, False)
assert _n2.abs(_H1 - _H2).max() < 1e-4 and _n2.abs(_n1 - _n2b * [1, -1, 1]).max() < 1e-4
# keep=1 возвращает входную нормаль; и сдвиг картинки даёт сдвиг результата
_noisy = _nt + _rng22.normal(size=_nt.shape).astype(_n2.float32) * 0.05
_noisy /= _n2.linalg.norm(_noisy, axis=-1, keepdims=True)
_, _nk, _ik = _MG22.balance_height_normal(_noisy, _Ht, 0.5, 1.0, True)
_Hs, _, _ = _MG22.balance_height_normal(_n2.roll(_noisy, (17, 41), (0, 1)),
                                        _n2.roll(_Ht, (17, 41), (0, 1)), 0.5, 0.0, True)
_H0, _, _ = _MG22.balance_height_normal(_noisy, _Ht, 0.5, 0.0, True)
print('баланс: восстановление точное, масштаб 100, keep=1 отличие %.3f°, сдвиг %.2g'
      % (_ik['angle_n'], float(_n2.abs(_n2.roll(_H0, (17, 41), (0, 1)) - _Hs).max())))
assert _ik['angle_n'] < 0.2
assert float(_n2.abs(_n2.roll(_H0, (17, 41), (0, 1)) - _Hs).max()) < 1e-3

# --- раскладка зелёного по ротору
_isdx, _cdx, _cgl = _MG22.guess_green(_nt)
_isdx2, _, _ = _MG22.guess_green(_nt * [1, -1, 1])
print('зелёный по ротору: DirectX %.3f / %.3f, OpenGL-карта распознана: %s'
      % (_cdx, _cgl, not _isdx2))
assert _isdx and not _isdx2

# --- DeepBump: раскладка OpenGL, шов не хуже середины
_wall = _TX22.read_rgba('test/tx_ex_stronghold_wall12.dds')[..., :3]
_w512 = _n2.clip(_TX22.resize_wrap(_wall, 512, 512), 0, 1)
if _DB22.have_model() and DN.have_runtime():
    _nb = _DB22.run_gray(_w512.mean(-1), 0.5)
    _L = _MG22.luminance(_MG22.srgb_to_lin(_w512))
    _hl = _L - _MG22.fast_blur(_L, 12)
    _c = {}
    for _dx in (True, False):
        _hh = _MG22.height_from_normal(_nb, _dx)
        _c[_dx] = _corr(_hh - _MG22.fast_blur(_hh, 12), _hl)
    _ej = float(_n2.abs(_nb[0] - _nb[-1]).mean()); _ij = float(_n2.abs(_nb[1:] - _nb[:-1]).mean())
    print('DeepBump: связь с яркостью как OpenGL %.2f, как DirectX %.2f; шов %.4f, внутри %.4f'
          % (_c[False], _c[True], _ej, _ij))
    assert _c[False] > 0.2 > _c[True] and not _MG22.guess_green(_nb)[0]
    assert _ej < 1.5 * _ij
else:
    print('DeepBump: модели нет — пропуск')

# --- ESRGAN на настоящих весах: сверка с независимой реализацией в torch
if _ES.have_model('normal'):
    _en = _ES.run('normal', _w512)
    _nv = _en * 2 - 1
    _nv /= _n2.linalg.norm(_nv, axis=-1, keepdims=True)
    print('ESRGAN NormalMapGenerator: раскладка по ротору — %s'
          % ('DirectX' if _MG22.guess_green(_nv)[0] else 'OpenGL'))
    assert _MG22.guess_green(_nv)[0], 'NormalMapGenerator должен выдавать DirectX'
    try:
        import torch as _tc, torch.nn.functional as _F
        _sd = {k: _tc.from_numpy(_n2.asarray(v, _n2.float32))
               for k, v in _ES.load_pth(_ES.pth_path('normal')).items()}
        _cv = lambda x, p: _F.conv2d(x, _sd[p + '.weight'], _sd[p + '.bias'], padding=1)
        _lr = lambda x: _F.leaky_relu(x, 0.2)

        def _rdb(x, p):
            a = _lr(_cv(x, p + '.conv1.0')); b = _lr(_cv(_tc.cat([x, a], 1), p + '.conv2.0'))
            c = _lr(_cv(_tc.cat([x, a, b], 1), p + '.conv3.0'))
            d = _lr(_cv(_tc.cat([x, a, b, c], 1), p + '.conv4.0'))
            return _cv(_tc.cat([x, a, b, c, d], 1), p + '.conv5.0') * 0.2 + x
        _x = _F.pad(_tc.from_numpy(_w512[:64, :64]).permute(2, 0, 1)[None], (48,) * 4,
                    mode='circular')
        _f = _cv(_x, 'model.0'); _hq = _f
        for _i in range(12):
            _p = 'model.1.sub.%d' % _i
            _hq = _rdb(_rdb(_rdb(_hq, _p + '.RDB1'), _p + '.RDB2'), _p + '.RDB3') * 0.2 + _hq
        _hq = _cv(_hq, 'model.1.sub.12') + _f
        _ref = _cv(_lr(_cv(_hq, 'model.2')), 'model.4')[0].permute(1, 2, 0).numpy()[48:-48, 48:-48]
        _our = _ES.run('normal', _w512[:64, :64], tile=64, pad=48)
        print('ESRGAN против torch: расхождение %.2g' % float(_n2.abs(_n2.clip(_ref, 0, 1) - _our).max()))
        assert float(_n2.abs(_n2.clip(_ref, 0, 1) - _our).max()) < 1e-4
    except ImportError:
        print('ESRGAN против torch: torch нет — пропуск')

# --- PBR → OpenMW
_base = _n2.full((8, 8, 3), 0.6, _n2.float32)
_o = _ND22.REG['pbr_openmw'].fn({'base': _base, 'metal': _n2.ones((8, 8), _n2.float32),
                                 'rough': _n2.full((8, 8), 0.3, _n2.float32)},
                                dict(f0=0.04, metal_dark=1.0, ao_diffuse=1.0, ao_spec=1.0,
                                     spec_gain=1.0, rough_default=0.7), {})
_exp = 2.0 / (0.09 ** 2) - 2.0
print('PBR → OpenMW: металл — блик %.2f (база 0.60), диффуз %.2f, показатель %.0f (ждём %.0f)'
      % (float(_o['spec'][..., 0].mean()), float(_o['diffuse'].mean()),
         float(_o['spec'][..., 3].mean()) * 255, min(_exp, 255)))
assert abs(float(_o['spec'][..., 0].mean()) - 0.6) < 0.02 and float(_o['diffuse'].mean()) < 0.01
assert abs(float(_o['spec'][..., 3].mean()) * 255 - min(_exp, 255)) < 1.0

# --- смешение нормалей
_na = _MG22.normal_from_heightmap(_Ht * 0.02, True)
_mm = _ND22.REG['normal_mix'].fn({'a': _na, 'b': _nt}, {'amount': 0.0, 'mode': 'detail'}, {})['n']
assert float(_n2.abs(_mm - _na).max()) < 1e-4, 'деталь с силой 0 должна отдавать А'

# --- кэш на диске
_calls = []
_arr = _n2.random.default_rng().random((32, 32, 3)).astype(_n2.float32)   # новый на каждом прогоне
_v1 = _ND22.disk_cached('chk22', _arr, {'x': 1}, lambda: (_calls.append(1), _arr * 2)[1])
_v2 = _ND22.disk_cached('chk22', _arr, {'x': 1}, lambda: (_calls.append(1), _arr * 3)[1])
assert len(_calls) == 1 and _n2.allclose(_v1, _v2), 'кэш не сработал'

# --- растяжение на торе
_pat = _n2.sin(_n2.linspace(0, 4 * _n2.pi, 64, endpoint=False))[None, :].repeat(64, 0).astype(_n2.float32)
_rz = _TX22.resize_wrap(_pat, 64, 100)
print('растяжение на торе: шов %.4f, внутри %.4f'
      % (float(_n2.abs(_rz[:, 0] - _rz[:, -1]).mean()), float(_n2.abs(_n2.diff(_rz, axis=1)).mean())))
assert float(_n2.abs(_rz[:, 0] - _rz[:, -1]).mean()) < 1.5 * float(_n2.abs(_n2.diff(_rz, axis=1)).mean())

# --- превью на уменьшенной копии = файл (радиусы пересчитываются)
_ctxf = {'src': _wall, 'key': 'f22', 'px': 1.0}
_ctxh = {'src': _n2.clip(_TX22.resize_wrap(_wall, 1024, 1024), 0, 1), 'key': 'h22', 'px': 0.5}
_gc = _PR22.template('classic')
_hn = [k for k in _gc.nodes if _gc.nodes[k]['type'] == 'normal'][0]
_ho = [k for k in _gc.nodes if _gc.nodes[k]['type'] == 'ao'][0]
_nf = _gc.eval(_hn, _ctxf)['n']; _nh_ = _gc.eval(_hn, _ctxh)['n']
_af = _gc.eval(_ho, _ctxf)['ao']; _ah = _gc.eval(_ho, _ctxh)['ao']
_nfd = _TX22.resize_wrap(_nf, 1024, 1024)
_nfd /= _n2.linalg.norm(_nfd, axis=-1, keepdims=True)
_rms = float(_n2.sqrt(((_nfd - _nh_) ** 2).sum(-1).mean()))
_cao = _corr(_TX22.resize_wrap(_af, 1024, 1024), _ah)
print('превью = файл (узлы): нормаль RMS %.3f, AO связь %.3f' % (_rms, _cao))
assert _rms < 0.12 and _cao > 0.9

# --- проекты: шаблоны, запись и чтение, пакет
_sm22 = _n2.clip(_TX22.resize_wrap(_wall, 128, 128), 0, 1)
for _k, _ in _PR22.TEMPLATES:
    if _k == 'supermat' and _SB22.problem():
        continue
    if _k == 'neural' and not (_DB22.have_model() and _ready and DN.have_runtime()):
        continue
    _g = _PR22.template(_k)
    for _o in _g.outputs():
        assert _g.eval(_o, {'src': _sm22, 'key': 't' + _k})['any'] is not None, (_k, _o)
_td = _tf.mkdtemp()
for _sub in ('', 'sub', 'icons'):
    os.makedirs(os.path.join(_td, 'in', _sub), exist_ok=True)
for _f in ('a.png', 'sub/b.png', 'icons/c.png', 'a_n.png'):
    T.Image.fromarray((_sm22 * 255).astype(_n2.uint8)).save(os.path.join(_td, 'in', _f))
_g = _PR22.template('classic')
_PR22.save(os.path.join(_td, 'p.texelith'), _g, os.path.join(_td, 'in', 'a.png'),
           {'in': os.path.join(_td, 'in'), 'out': os.path.join(_td, 'out')}, {'zoom': 1.2})
_pr = _PR22.load(os.path.join(_td, 'p.texelith'))
assert _pr['graph'].to_json() == _g.to_json() and _pr['view']['zoom'] == 1.2
_r = _PR22.run_batch(_pr['graph'], _pr['batch'])
_made = sorted(os.path.relpath(os.path.join(a, f), os.path.join(_td, 'out')).replace('\\', '/')
               for a, _, fs in os.walk(os.path.join(_td, 'out')) for f in fs if f.endswith('.dds'))
_r2 = _PR22.run_batch(_pr['graph'], _pr['batch'])
_same = _PR22.targets(_g, 'a.png', os.path.join(_td, 'in'), os.path.join(_td, 'in'))
print('пакет проекта: %s; повтор — сделано %d, пропущено %d' % (_made, _r2['done'], _r2['skipped']))
assert _made == ['a.dds', 'a_nh.dds', 'a_spec.dds', 'sub/b.dds', 'sub/b_nh.dds', 'sub/b_spec.dds']
assert _r2['done'] == 0 and _r2['skipped'] == 2
assert all(not p.endswith('a.dds') for _, p in _same), 'диффуз не пишется поверх исходника'

# мост SuperMat здесь не проверяется: ему нужно отдельное окружение с PyTorch
_sh.rmtree(_td, ignore_errors=True)

# --- узловое окно: провод мышью, отмена, проект, язык
app.show_classic(False); root.update()
_ed = app.editor
_ed.open_texture('test/tx_colony_stone02.dds')
_t0 = _tm.time()
while (_ed._pending or _ed._busy) and _tm.time() - _t0 < 300:
    root.update(); _tm.sleep(0.05)
root.update()
assert not _ed.errors, _ed.errors
assert all(k in _ed.thumbs for k in _ed.graph.nodes if _ed.graph.nodes[k]['type'] != 'out'
           or _ed.results.get(k, {}).get('any') is not None)
_ed.frame_all(); root.update()
_g = _ed.graph
_s = [k for k in _g.nodes if _g.nodes[k]['type'] == 'source'][0]
_x0, _y0 = _ed.w2s(*_ed.port_w(_s, 'rgb', False))
_ed.add_node('tone', _x0 + 200, _y0 + 260)
_tn = _ed.st['sel'][0]
_ed.add_node('height_lum', _x0 + 60, _y0 + 260)
_hl = _ed.st['sel'][0]
_g.connect(_s, 'rgb', _hl, 'rgb')
_xa, _ya = _ed.w2s(*_ed.port_w(_hl, 'h', False)); _xb, _yb = _ed.w2s(*_ed.port_w(_tn, 'h', True))
for _ev, _xx, _yy in (('<Button-1>', _xa, _ya), ('<B1-Motion>', _xb, _yb),
                      ('<ButtonRelease-1>', _xb, _yb)):
    _ed.cv.event_generate(_ev, x=int(_xx), y=int(_yy))
assert _g.inputs_of(_tn).get('h') == (_hl, 'h'), 'провод мышью не соединил'
_ed.undo(); assert 'h' not in _g.inputs_of(_tn)
_ed.redo(); assert _g.inputs_of(_tn).get('h') == (_hl, 'h')
_pp = os.path.join(_tf.mkdtemp(), 'chk.texelith')
_ED22.filedialog.asksaveasfilename = lambda **k: _pp
assert _ed.save_project(True) and not _ed.st['dirty']
_j = _g.to_json()
_ed.open_project(_pp)
assert _ed.graph.to_json() == _j
app.set_lang('English'); root.update()
assert app.editor.graph.to_json() == _j
app.set_lang('Русский'); root.update()
print('узловое окно: провод мышью, отмена/возврат, проект, смена языка — работают')

# --- видеокарта: все узлы и 3D на строгой подмене CuPy дают то же, что numpy
import xpu as _X
_srcg = _n2.clip(_TX22.resize_wrap(_wall, 192, 192), 0, 1).astype(_n2.float32)
_gbad = []
for _k in _ND22.REG:
    if _k in _ND22.HOST_NODES:
        continue
    def _bld(k=_k):
        g = _ND22.Graph(); s = g.add('source'); nid = g.add(k) if k != 'source' else s
        if k != 'source':
            for port, kind in _ND22.REG[k].ins:
                if kind == _ND22.RGB:
                    g.connect(s, 'rgb', nid, port)
                elif kind in (_ND22.GRAY, _ND22.ANY):
                    h = g.add('height_lum'); g.connect(s, 'rgb', h, 'rgb'); g.connect(h, 'h', nid, port)
                elif kind == _ND22.NRM:
                    h = g.add('height_lum'); nn = g.add('normal')
                    g.connect(s, 'rgb', h, 'rgb'); g.connect(h, 'h', nn, 'h'); g.connect(nn, 'n', nid, port)
        return g, nid
    _X.use(False); _g, _nid = _bld(); _ref = _g.eval(_nid, {'src': _srcg, 'key': 'c'})
    _X.use(True, fake=True); _g, _nid = _bld()
    try:
        _o = _g.eval(_nid, {'src': _X.to_dev(_srcg), 'key': 'g'})
        for _p, _v in _ref.items():
            if not _p.startswith('_') and isinstance(_v, _n2.ndarray):
                assert _n2.allclose(_X.to_host(_o[_p]), _v, atol=1e-5), (_k, _p)
                assert _X.is_dev(_o[_p]) or _k in ('out',), (_k, _p, 'осталось не на устройстве')
    except Exception as _e:
        _gbad.append('%s: %s' % (_k, _e))
    finally:
        _X.use(False)
_H3 = 96
_m3 = {'diffuse': _srcg[:_H3, :_H3], 'normal': _MG22.normal_from_heightmap(_srcg[:_H3, :_H3].mean(-1) * 8, True),
       'height': _srcg[:_H3, :_H3].mean(-1), 'spec': _srcg[:_H3, :_H3] * 0.2,
       'shininess': _srcg[:_H3, :_H3, 0], 'ao': _srcg[:_H3, :_H3, 1]}
for _mode in ('openmw', 'occlusion', 'offset'):
    _a = _V3.render(_m3, 120, 90, mode=_mode, hemi=True)
    _X.use(True, fake=True)
    try:
        _b = _X.to_host(_V3.render(_X.dev_tree(_m3), 120, 90, mode=_mode, hemi=True))
        if not _n2.allclose(_a, _b, atol=1e-5):
            _gbad.append('3D ' + _mode)
    except Exception as _e:
        _gbad.append('3D %s: %s' % (_mode, _e))
    finally:
        _X.use(False)
print('видеокарта (строгая подмена CuPy): узлов %d, 3D-режимов 3, расхождений %d %s'
      % (len([k for k in _ND22.REG if k not in _ND22.HOST_NODES]), len(_gbad), _gbad))
assert not _gbad

# --- починка: маска монотонна по порогу, тонкая трещина выделяется, заливки без чёрного
_rng = _n2.random.default_rng(3)
_im = _n2.clip(_rng.random((160, 160, 3)) * 0.3 + 0.4, 0, 1).astype(_n2.float32)
_yy, _xx = _n2.mgrid[:160, :160]
_hole = (_yy - 50) ** 2 + (_xx - 50) ** 2 <= 20 ** 2
_crack = (abs(_yy - 120) <= 1) & (_xx > 20) & (_xx < 140)
_im[_hole | _crack] = 0.02
_cov = [float((_RP.exposure_masks(_im, lo=_lo)[1] > 0.5).mean()) for _lo in (0.05, 0.1, 0.2, 0.3)]
assert all(b >= a - 1e-9 for a, b in zip(_cov, _cov[1:])), 'маска не растёт с порогом: %s' % _cov
_c = _RP.exposure_masks(_im)[1]
assert (_c > 0.5)[_crack].mean() > 0.95, 'трещина толщиной 3 px не выделилась'
_c2 = _RP.exposure_masks(_im, grow=-3)[1]
assert (_c2 > 0.5)[_crack].mean() < 0.05 and (_c2 > 0.5)[_hole].mean() > 0.95, \
    'отрицательный отступ должен убирать тонкое и оставлять широкое'
_fl = {}
for _m in ('mean', 'gradient', 'spots'):
    _r = _ND22.REG['repair'].fn({'rgb': _im}, dict(
        {pr[0]: pr[2] for pr in _ND22.REG['repair'].params}, method=_m), {'px': 1.0})
    _fl[_m] = float(_RP.luminance(_r['rgb'])[_hole].mean())
    assert _fl[_m] > 0.3, 'заливка «%s» осталась тёмной' % _m
    assert _r['overlay'].shape == _im.shape
print('починка: маска по порогу %s, трещина и отступ — в порядке, заливки %s'
      % (['%.3f' % v for v in _cov], {k: round(v, 2) for k, v in _fl.items()}))
_pp = _ND22.REG['pbr_openmw']
assert set(_pp.presets) <= set(dict((pr[0], pr[3]) for pr in _pp.params)['preset'])
_rg = _pp.fn({'base': _im}, dict({pr[0]: pr[2] for pr in _pp.params}, **_pp.presets['gold']), {})
assert abs(dict(_rg['_info'])['metal'] - _pp.presets['gold']['metal_default']) < 1e-6, \
    'металличность из пресета не применилась'
_rr = (_rng.random((160, 160)) * 0.3 + 0.1).astype(_n2.float32)
_rs = _pp.fn({'base': _im, 'rough': _rr}, dict({pr[0]: pr[2] for pr in _pp.params},
                                              **_pp.presets['stone']), {})
_ri = dict(_rs['_info'])
assert 0.65 <= _ri['rough'] <= 0.95, 'карта шероховатости не вписалась в камень: %.3f' % _ri['rough']
print('PBR: шероховатость %.2f…%.2f вписана в камень, средняя %.3f'
      % (_ri['rough_in_lo'], _ri['rough_in_hi'], _ri['rough']))
print('PBR → OpenMW: пресетов материалов %d' % len(_pp.presets))

# --- фильтры: high-pass убирает крупное, всё замыкается через край, превью = файл
_N = 256
_yy, _xx = _n2.mgrid[:_N, :_N] / _N
_lowf = 0.5 + 0.4 * _n2.sin(2 * _n2.pi * _xx) * _n2.cos(2 * _n2.pi * _yy)
_finef = _n2.random.default_rng(7).standard_normal((_N, _N)).astype(_n2.float32) * 0.05
_hf = _n2.clip(_lowf + _finef, 0, 1).astype(_n2.float32)
def _fl(k, h, ctx=None, **kw):
    return _ND22.REG[k].fn({'h': h}, dict({pr[0]: pr[2] for pr in _ND22.REG[k].params}, **kw),
                           ctx or {'px': 1.0})
_hp = _fl('highpass', _hf, lo_cut=0.02)
_cl = abs(_n2.corrcoef(_hp['h'].ravel(), _lowf.ravel())[0, 1])
assert _cl < 0.2, 'high-pass не убрал крупное: корреляция %.3f' % _cl
assert _n2.abs(_n2.roll(_hp['h'], (40, 9), (0, 1))
               - _fl('highpass', _n2.roll(_hf, (40, 9), (0, 1)), lo_cut=0.02)['h']).max() < 1e-4, \
    'high-pass с швом'
assert abs(dict(_fl('highpass', _hf[::2, ::2], {'px': 0.5}, lo_cut=0.02)['_info'])['size_px']
           - dict(_hp['_info'])['size_px']) < 1e-6, 'размер фильтра в превью и в файле разный'
for _k in ('lowpass', 'bandpass', 'median', 'sharpen'):
    _o = _fl(_k, _hf)['h']
    assert _o.shape == _hf.shape and _n2.isfinite(_o).all(), _k
_rr = _n2.hypot(*(_n2.mgrid[:_N, :_N] - _N // 2))
_bs = (0.3 + 0.1 * _n2.sin(2 * _n2.pi * _xx)).astype(_n2.float32)
_sp = _bs + 0.5 * (_rr <= 15)
_ring = (_rr >= 17) & (_rr <= 45)
_halo = {}
for _ke in (False, True):
    _d = (_fl('highpass', _sp.astype(_n2.float32), stretch=False, keep_edges=_ke)['h']
          - _fl('highpass', _bs, stretch=False, keep_edges=_ke)['h'])
    _halo[_ke] = float(-_d[_ring].mean())
assert _halo[True] < 0.5 * _halo[False], 'край не спасает от ореола: %s' % _halo
assert _fl('edge_smooth', _hf)['h'].shape == _hf.shape
print('фильтры: после high-pass связь с крупным %.3f; ореол %.4f -> %.4f с сохранением краёв; '
      'размыть, полоса, медиана, резкость — в порядке' % (_cl, _halo[False], _halo[True]))

# --- переименование: старые настройки ~/.tessera переезжают, проекты .tessera открываются
import tempfile as _tf, subprocess as _sp
_home = _tf.mkdtemp()
os.makedirs(os.path.join(_home, '.tessera', 'models'))
open(os.path.join(_home, '.tessera', 'models', 'm.onnx'), 'w').write('x')
_r = _sp.run([sys.executable, '-c', 'import i18n; print(i18n.APP_DIR)'], capture_output=True, text=True,
             env=dict(os.environ, HOME=_home, USERPROFILE=_home), cwd=os.path.dirname(os.path.abspath(__file__)))
assert _r.stdout.strip().endswith('.texelith') and os.path.exists(
    os.path.join(_home, '.texelith', 'models', 'm.onnx')), 'старые настройки не переехали: ' + _r.stdout + _r.stderr
assert '.tessera' in _PR22.EXTS
print('переименование: ~/.tessera переехала в ~/.texelith, проекты .tessera открываются')

# --- глубина: «сцена» сети у краёв кадра не оставляет гребня по шву текстуры
import depthnet as _DNs
_orun = _DNs._run
def _fake_run(sess, img):
    """Как сеть: рельеф картинки + купол «сцены» по всему кадру."""
    h, w = img.shape[:2]
    yy, xx = _n2.mgrid[:h, :w].astype(_n2.float32)
    dome = -(((yy / h - 0.5) ** 2 + (xx / w - 0.5) ** 2)) * 4.0
    return (img.mean(-1) + dome).astype(_n2.float32)
_DNs._run = _fake_run
try:
    _tex = _n2.random.default_rng(11).random((128, 160, 3)).astype(_n2.float32)
    _dd = _DNs.depth_fast(None, _tex)
finally:
    _DNs._run = _orun
# гребень по шву = «шатёр»: высоко у краёв тайла, низко в середине
_ty = _n2.abs(_n2.arange(128) / 128.0 - 0.5)[:, None]
_tx = _n2.abs(_n2.arange(160) / 160.0 - 0.5)[None, :]
_tent = abs(float(_n2.corrcoef(_dd.ravel(), (_ty + _tx + 0 * _dd).ravel())[0, 1]))
assert _tent < 0.3, 'гребень по шву в глубине: связь с «шатром» %.2f' % _tent
print('глубина: связь с гребнем по шву %.2f (до исправления 0.72)' % _tent)

print('ВСЁ ХОРОШО')
