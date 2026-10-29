# -*- coding: utf-8 -*-
"""
Texelith — graphics card self-test.

Runs every node that can run on the card twice — on the processor and on the
card — on a sample texture, compares the results and the times, renders the
3D view both ways, and writes everything into gpu_test.log next to this file.
Run it with GPU-test.bat (or ``python gpu_test.py``) after installing CuPy.

The numbers are what decides whether the card path is right: the same maps
(differences at the level of float rounding, about 1e-5) and how much faster.

Texelith — самопроверка видеокарты: каждый узел считается на процессоре и
на видеокарте, результаты и время сравниваются, всё пишется в gpu_test.log.
"""

import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np          # noqa: E402

import xpu as X             # noqa: E402

LOG = os.path.join(HERE, 'gpu_test.log')
_out = open(LOG, 'w', encoding='utf-8')


def say(*a):
    """Print and write a line of the log."""
    s = ' '.join(str(x) for x in a)
    print(s)
    _out.write(s + '\n')
    _out.flush()


def timed(fn, sync=False):
    """(result, seconds) of fn(); waits for the card to finish when sync."""
    t = time.perf_counter()
    r = fn()
    if sync and X.active() and X._state['cp'] is not None:
        try:
            X._state['cp'].cuda.Device().synchronize()
        except Exception:
            pass
    return r, time.perf_counter() - t


def maxdiff(a, b):
    """Largest absolute difference of two arrays (after moving to host)."""
    a, b = X.to_host(a), X.to_host(b)
    return float(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)).max())


def share(a, b, tol=1e-2):
    """Share of values that differ by more than tol. A yes/no decision
    (threshold, posterize, a mask by a cut-off, a step of the parallax
    search) flips where the input sits right at the boundary, and rounding
    of 1e-5 is enough for that: the largest difference is then a whole
    step, but only on a handful of pixels. So a result passes when the
    largest difference is at rounding level OR such pixels are under 0.1 %."""
    a, b = X.to_host(a), X.to_host(b)
    return float((np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)) > tol).mean())


def passed(d, sh):
    """The verdict for a largest difference d and a share sh (see share)."""
    return d < 2e-3 or sh < 1e-3


def main():
    """Run the checks; returns the process exit code."""
    say('Texelith GPU self-test,', time.strftime('%Y-%m-%d %H:%M:%S'))
    say('python', sys.version.split()[0], 'numpy', np.__version__)
    try:
        import cupy
        say('cupy', cupy.__version__)
    except Exception as e:                           # noqa: BLE001
        say('cupy: NOT importable —', e)
    on, info = X.use(True)
    say('card:', info if on else 'NOT USED — ' + info)
    if not on:
        return 1
    import nodes as ND
    import project as PR
    import texio as TX
    import view3d as V3
    src = TX.read_rgba(os.path.join(HERE, 'samples', 'Rock064.jpg'))[..., :3]
    src = np.ascontiguousarray(src, np.float32)
    say('texture', src.shape)

    # --- каждый узел: процессор против видеокарты
    bad = 0
    for k in ND.REG:
        if k in ND.HOST_NODES:
            continue
        g = ND.Graph()
        s = g.add('source')
        nid = g.add(k) if k != 'source' else s
        if k != 'source':
            for port, kind in ND.REG[k].ins:
                if kind == ND.RGB:
                    g.connect(s, 'rgb', nid, port)
                elif kind in (ND.GRAY, ND.ANY):
                    h = g.add('height_lum')
                    g.connect(s, 'rgb', h, 'rgb')
                    g.connect(h, 'h', nid, port)
                elif kind == ND.NRM:
                    h, n = g.add('height_lum'), g.add('normal')
                    g.connect(s, 'rgb', h, 'rgb')
                    g.connect(h, 'h', n, 'h')
                    g.connect(n, 'n', nid, port)
        try:
            X.use(False)
            ref, tc = timed(lambda: g.eval(nid, {'src': src, 'key': 'cpu'}))
            X.use(True)
            g._cache.clear()
            dsrc = X.to_dev(src)
            g.eval(nid, {'src': dsrc, 'key': 'warm'})       # первый запуск компилирует ядра
            g._cache.clear()
            out, tg = timed(lambda: g.eval(nid, {'src': dsrc, 'key': 'gpu'}), True)
            ports = [p for p in ref if not p.startswith('_') and isinstance(ref[p], np.ndarray)]
            d = max([maxdiff(ref[p], out[p]) for p in ports] or [0.0])
            sh = max([share(ref[p], out[p]) for p in ports] or [0.0])
            ok = passed(d, sh)
            bad += not ok
            say('%-4s %-18s diff %.1e (%.4f %% px)   cpu %.3f s   gpu %.3f s' % (
                'ok' if ok else 'DIFF', k, d, sh * 100, tc, tg))
        except Exception as e:                       # noqa: BLE001
            bad += 1
            X.use(True)
            say('ERR  %-18s %s: %s' % (k, e.__class__.__name__, e))
            say(''.join(traceback.format_exc().splitlines(True)[-6:]))

    # --- узлы с особыми настройками: то, что при настройках по умолчанию не считается
    poly = [[0.3, 0.3], [0.45, 0.32], [0.42, 0.5], [0.28, 0.46]]
    extra = [('patch', {'patches': [{'poly': poly, 'dx': 0.2, 'dy': 0.1}]}, 'rgb'),
             ('patch', {'patches': [{'poly': poly, 'dx': 0.2, 'dy': 0.1}],
                        'seamless': False}, 'rgb'),
             ('normal_flatfix', {'level': 0.3, 'min_area': 200}, 'h'),
             ('repair', {'hi': 0.7, 'lo': 0.2, 'flat': 0.08}, 'rgb')]
    for k, upd, feed in extra:
        g = ND.Graph()
        s = g.add('source')
        nid = g.add(k)
        g.nodes[nid]['params'].update(upd)
        port = ND.REG[k].ins[0][0]
        if feed == 'rgb':
            g.connect(s, 'rgb', nid, port)
        else:
            h = g.add('height_lum')
            g.connect(s, 'rgb', h, 'rgb')
            g.connect(h, 'h', nid, port)
        name = '%s %s' % (k, ','.join(sorted(upd))[:12])
        try:
            X.use(False)
            ref, tc = timed(lambda: g.eval(nid, {'src': src, 'key': 'cpu'}))
            X.use(True)
            g._cache.clear()
            dsrc = X.to_dev(src)
            g.eval(nid, {'src': dsrc, 'key': 'warm'})
            g._cache.clear()
            out, tg = timed(lambda: g.eval(nid, {'src': dsrc, 'key': 'gpu'}), True)
            ports = [p for p in ref if not p.startswith('_') and isinstance(ref[p], np.ndarray)]
            d = max([maxdiff(ref[p], out[p]) for p in ports] or [0.0])
            sh = max([share(ref[p], out[p]) for p in ports] or [0.0])
            ok = passed(d, sh)
            bad += not ok
            say('%-4s %-22s diff %.1e (%.4f %% px)   cpu %.3f s   gpu %.3f s' % (
                'ok' if ok else 'DIFF', name, d, sh * 100, tc, tg))
        except Exception as e:                       # noqa: BLE001
            bad += 1
            X.use(True)
            say('ERR  %-22s %s: %s' % (name, e.__class__.__name__, e))
            say(''.join(traceback.format_exc().splitlines(True)[-6:]))

    # --- «Листва»: входы у неё на процессоре, а прожилки (Sato) и AO кроны
    # (HBAO) она сама считает на видеокарте. Карточка — из той же текстуры:
    # альфа = светлые места, так что силуэт рваный, как у листвы
    try:
        rgb = np.ascontiguousarray(src[:512, :512])
        lum = rgb.mean(-1)
        alpha = (lum > float(np.median(lum))).astype(np.float32)
        S = ND.REG['foliage']
        p = {q[0]: q[2] for q in S.params}
        keep = ND.disk_cached
        ND.disk_cached = lambda name, arr, params, fn: fn()   # без кэша на диске
        try:
            X.use(False)
            ref, tc = timed(lambda: S.fn({'rgb': rgb, 'a': alpha}, p, {}))
            X.use(True)
            out, tg = timed(lambda: S.fn({'rgb': rgb, 'a': alpha}, p, {}), True)
        finally:
            ND.disk_cached = keep
        ports = [q for q in ref if not q.startswith('_') and isinstance(ref[q], np.ndarray)]
        d = max([maxdiff(ref[q], out[q]) for q in ports] or [0.0])
        sh = max([share(ref[q], out[q]) for q in ports] or [0.0])
        ok = passed(d, sh)
        bad += not ok
        say('%-4s %-22s diff %.1e (%.4f %% px)   cpu %.3f s   gpu %.3f s' % (
            'ok' if ok else 'DIFF', 'foliage 512', d, sh * 100, tc, tg))
    except Exception as e:                       # noqa: BLE001
        bad += 1
        X.use(True)
        say('ERR  %-22s %s: %s' % ('foliage', e.__class__.__name__, e))
        say(''.join(traceback.format_exc().splitlines(True)[-6:]))

    # --- AO по линиям (LSAO, GTAO): своё ядро видеокарты
    import imgfilters as F
    for meth in ('lsao', 'gtao'):
        g = ND.Graph()
        s = g.add('source')
        h = g.add('height_lum')
        a = g.add('ao')
        g.connect(s, 'rgb', h, 'rgb')
        g.connect(h, 'h', a, 'h')
        g.nodes[a]['params'].update(method=meth)
        try:
            X.use(False)
            ref, tc = timed(lambda: g.eval(a, {'src': src, 'key': 'cpu'}))
            X.use(True)
            g._cache.clear()
            dsrc = X.to_dev(src)
            g.eval(a, {'src': dsrc, 'key': 'warm'})
            g._cache.clear()
            out, tg = timed(lambda: g.eval(a, {'src': dsrc, 'key': 'gpu'}), True)
            used = X.kernel(F._LSAO_CU, 'lsao_sweep') is not None
            d = maxdiff(ref['ao'], out['ao'])
            ok = passed(d, share(ref['ao'], out['ao'])) and used
            bad += not ok
            say('%-4s ao %-14s diff %.1e   cpu %.3f s   gpu %.3f s   kernel %s' % (
                'ok' if ok else 'DIFF', meth, d, tc, tg,
                'used' if used else 'NOT compiled — ' + X.last_error()))
        except Exception as e:                       # noqa: BLE001
            bad += 1
            X.use(True)
            say('ERR  ao %-14s %s: %s' % (meth, e.__class__.__name__, e))
            say(''.join(traceback.format_exc().splitlines(True)[-6:]))

    # --- целый шаблон
    for tpl in ('classic',):
        g = PR.template(tpl)
        X.use(False)
        _, tc = timed(lambda: [g.eval(o, {'src': src, 'key': 'c'}) for o in g.outputs()])
        X.use(True)
        g._cache.clear()
        dsrc = X.to_dev(src)
        [g.eval(o, {'src': dsrc, 'key': 'w'}) for o in g.outputs()]
        g._cache.clear()
        _, tg = timed(lambda: [g.eval(o, {'src': dsrc, 'key': 'g'}) for o in g.outputs()], True)
        say('template %-10s cpu %.2f s   gpu %.2f s' % (tpl, tc, tg))

    # --- 3D
    H, W = src.shape[:2]
    m = {'diffuse': src, 'normal': np.dstack([np.zeros((H, W)), np.zeros((H, W)),
                                              np.ones((H, W))]).astype(np.float32),
         'height': src.mean(-1), 'spec': np.zeros_like(src),
         'shininess': np.zeros((H, W), np.float32), 'ao': np.ones((H, W), np.float32)}
    m['nh_alpha'] = m['height']
    for mode in ('openmw', 'occlusion'):
        X.use(False)
        a, tc = timed(lambda: V3.render(m, 900, 600, mode=mode, depth=0.04))
        X.use(True)
        dm = X.dev_tree(m)
        V3.render(dm, 900, 600, mode=mode, depth=0.04)
        b, tg = timed(lambda: X.to_host(V3.render(dm, 900, 600, mode=mode, depth=0.04)), True)
        d, sh = maxdiff(a, b), share(a, b)
        bad += not passed(d, sh)
        say('%-4s 3D %-10s diff %.1e (%.4f %% px)   cpu %.3f s   gpu %.3f s' % (
            'ok' if passed(d, sh) else 'DIFF', mode, d, sh * 100, tc, tg))
    say('RESULT:', 'all good' if not bad else '%d problems' % bad)
    return 0 if not bad else 2


if __name__ == '__main__':
    try:
        code = main()
    except Exception:                                # noqa: BLE001
        say(traceback.format_exc())
        code = 3
    _out.close()
    print('\nLog written to', LOG)
    sys.exit(code)
