# -*- coding: utf-8 -*-
"""Albedo Fix: put back into ready textures what the albedo chain flattened
out — without running SuperMat, the depth network or DeepBump again.

Works on the results of a Texelith batch (the PBR albedo `<name>_fixed`,
the vanilla diffuse `<name>`, the ORM map `<name>_orm`) plus the source
textures the batch was run on. For every texture it rebuilds the albedo and
then the diffuse for plain OpenMW:

1. Large patches (what is seen from afar). «Remove shading» takes large light
   and dark patches for shadows, and «Fit albedo» brings back only the mean
   lightness, cast and saturation — so the spread of lightness on a large
   scale drops (on the stronghold textures by 17…50 %), and the texture looks
   milky from a distance. Here the albedo's OWN large-scale lightness
   variation (blur of radius width/256) is scaled by k, so that its spread
   equals the source's, measured outside the source's defects. Only that one
   number is taken from the source, not its pattern: the shading of the
   source does not come back. «weight» 0…1 — how far to go (1 — the whole
   way); k is kept within 0.5…3.

2. Fine texture. SuperMat works at its own resolution (512 by default): on a
   2048 texture everything finer than 4 px comes out soft, while «Fit
   albedo» puts back the source's detail only up to its radius (2 px). The
   band between that radius and width / SuperMat resolution is added here
   from the source's lightness, × «detail» — the same thing «Fit albedo»
   would have done with the larger radius, as its detail is simply added to
   OKLab lightness. Black defects of the source (dark below the threshold,
   with a margin) give nothing.

3. AO. The diffuse is the new albedo × AO^strength in linear light (AO from
   the R channel of `_orm`), as «PBR → OpenMW» bakes it.

Without a `_fixed` the albedo is recovered from the diffuse by dividing out
the AO it already has (strength «AO already in the diffuse»). Without a
source texture steps 1 and 2 are skipped (the report says so).

Defects: pixels of the source darker than the threshold (sRGB luminance),
grown by a margin of max(2, width/512) px, are left out of the
measurement in step 1 and give no detail in step 2.

Nothing in the input folders is written to. Results go to a separate
folder with the same sub-folders (`diffuse/`, `fixed/` when the batch used
the «by kind of map» layout); a report `albedo_fix_report.txt` is written
there. Works on the video card when CuPy and a card are available (xpu);
DXT compression is on the processor.

Run Albedo-Fix.bat for the window (with a source / now / after preview),
or: python albedo_fix.py <sources> <done textures> <output> [--recursive]

Доводка готовых текстур: вернуть крупные пятна и мелкую фактуру альбедо и
заново запечь AO в диффуз — без повторного запуска нейросетей."""

import os
import sys
import time
import threading

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))         # модули Texelith — в папке выше

import texio as TX                                # noqa: E402
import matgen as MG                               # noqa: E402
import imgfilters as F                            # noqa: E402
import xpu as X                                   # noqa: E402

BUILD = 25          # версия набора файлов, проверяется при запуске

EXTS = ('.dds', '.png', '.tga', '.bmp')

# настройки по умолчанию: как в пакете, которым делались текстуры
DEFAULTS = dict(spread=True, weight=1.0, detail_on=True, detail=0.84, r_old=2.0,
                sm_res=512, ao=0.99, ao_old=0.99, thr=0.02)


def _t(key, *args):
    """Interface text (i18n, the language chosen in Texelith)."""
    import i18n
    if not getattr(_t, 'ready', False):
        i18n.load()
        _t.ready = True
    return i18n.t(key, *args)


# ------------------------------------------------------------ поиск файлов --
def _index(root, recursive=True):
    """name in lower case (with extension) → full path, for every picture."""
    out = {}
    for d, dirs, files in os.walk(root):
        if not recursive:
            dirs[:] = []
        for fn in files:
            if os.path.splitext(fn)[1].lower() in EXTS:
                out.setdefault(fn.lower(), os.path.join(d, fn))
    return out


def _by_stem(index):
    """stem in lower case → path (first found), from an _index result."""
    out = {}
    for name, p in index.items():
        out.setdefault(os.path.splitext(name)[0], p)
    return out


def find_jobs(src_root, done_root):
    """What to process: list of dicts with 'name' (stem), 'diffuse', 'fixed',
    'orm', 'source' (full paths; fixed/orm/source may be None) and 'rel' (the
    diffuse's path relative to done_root). A diffuse is a picture under
    done_root whose stem does not end with a map suffix
    (texio.MAP_SUFFIXES); the rest are found by name anywhere under their
    roots (sources: in all sub-folders, the batch's sorting folders do not
    matter)."""
    done = _by_stem(_index(done_root))
    src = _by_stem(_index(src_root)) if src_root and os.path.isdir(src_root) else {}
    jobs = []
    for stem, p in sorted(done.items()):
        if stem.endswith(TX.MAP_SUFFIXES):
            continue
        jobs.append(dict(name=os.path.splitext(os.path.basename(p))[0], diffuse=p,
                         fixed=done.get(stem + '_fixed'), orm=done.get(stem + '_orm'),
                         source=src.get(stem), rel=os.path.relpath(p, done_root)))
    return jobs


def check_folders(srcs, dst):
    """i18n key of an error if the output folder is one of the input folders
    or lies inside one, else None."""
    b = os.path.normcase(os.path.abspath(dst))
    for s in srcs:
        if not s:
            continue
        a = os.path.normcase(os.path.abspath(s))
        if a == b or b.startswith(a.rstrip('\\/') + os.sep):
            return 'af.same'
    return None


# -------------------------------------------------------------- расчёт ---
def load(job, ao_old=1.0):
    """Read one texture: dict with 'albedo' (sRGB H×W×3), 'alpha' (H×W or
    None), 'ao' (H×W, ones without an ORM), 'src' (sRGB H×W×3 resized to the
    albedo, or None) and 'from_diffuse' (True if the albedo was recovered from
    the diffuse because there is no _fixed). Host arrays."""
    dif = TX.read_rgba(job['diffuse'])
    H, W = dif.shape[:2]
    alpha = dif[..., 3] if float(dif[..., 3].min()) < 0.999 else None
    ao = np.ones((H, W), np.float32)
    if job['orm']:
        a = TX.read_rgba(job['orm'])[..., 0]
        ao = a if a.shape == (H, W) else TX.resize_to(a, (H, W))
    if job['fixed']:
        alb = TX.read_rgba(job['fixed'])[..., :3]
        if alb.shape[:2] != (H, W):
            alb = np.stack([TX.resize_to(alb[..., c], (H, W)) for c in range(3)], -1)
        frm = False
    else:
        # альбедо = диффуз ÷ AO, который в нём уже есть (линейный свет)
        lin = MG.srgb_to_lin(dif[..., :3]) / np.power(np.maximum(ao, 0.02), ao_old)[..., None]
        alb, frm = MG.lin_to_srgb(np.clip(lin, 0, 1)), True
    src = None
    if job['source']:
        s = TX.read_rgba(job['source'])[..., :3]
        src = s if s.shape[:2] == (H, W) else np.stack(
            [TX.resize_to(s[..., c], (H, W)) for c in range(3)], -1)
    return dict(albedo=np.asarray(alb, np.float32), alpha=alpha, ao=np.asarray(ao, np.float32),
                src=None if src is None else np.asarray(src, np.float32), from_diffuse=frm)


def fix(d, opt, full_width=None):
    """The new albedo and diffuse for one loaded texture (see the module
    docstring). d: result of load (host or device arrays); opt: settings as in
    DEFAULTS; full_width: width of the full-size texture when d is a reduced
    preview copy (radii are set by the full width and scaled down).
    Returns (albedo, diffuse, info) — sRGB H×W×3 and a dict of readouts."""
    xp = X.xp(d['albedo'])
    alb = xp.asarray(d['albedo'], xp.float32)
    H, W = alb.shape[:2]
    fw = float(full_width or W)
    sc = W / fw                                     # уменьшение превью
    L, a, b = F.to_oklab(alb)
    info = {'k': 1.0, 'band': 0.0, 'defects': 0.0}
    src = d.get('src')
    if src is not None and (opt['spread'] or opt['detail_on']):
        src = xp.asarray(src, xp.float32)
        Ls = F.to_oklab(src)[0]
        lum = (src * xp.asarray(np.array([0.2126, 0.7152, 0.0722], np.float32))).sum(-1)
        nd = X.ndi(lum)
        grow = max(1, int(round(max(2.0, fw / 512.0) * sc)))
        bad = nd.maximum_filter((lum < float(opt['thr'])).astype(xp.float32),
                                size=2 * grow + 1, mode='wrap') > 0.5
        good = ~bad
        info['defects'] = float(bad.mean()) * 100.0
        if opt['spread'] and int(good.sum()) > 64:
            R = max(1.0, fw / 256.0 * sc)
            lo_s = MG.fast_blur(Ls, R, mode='wrap')
            lo_a = MG.fast_blur(L, R, mode='wrap')
            ss, sa = float(lo_s[good].std()), float(lo_a[good].std())
            k = ss / sa if sa > 1e-6 else 1.0
            k = 1.0 + float(opt['weight']) * (k - 1.0)
            k = min(max(k, 0.5), 3.0)
            L = L + (k - 1.0) * (lo_a - float(lo_a[good].mean()))
            info['k'] = k
        if opt['detail_on']:
            r_old = float(opt['r_old'])
            r_new = fw / float(opt['sm_res'])
            if r_new > r_old + 0.25:
                keep = MG.fast_blur((~bad).astype(xp.float32), max(1.0, 2.0 * sc), mode='wrap')
                band = (MG.fast_blur(Ls, r_old * sc, mode='wrap')
                        - MG.fast_blur(Ls, r_new * sc, mode='wrap'))
                L = L + float(opt['detail']) * keep * band
                info['band'] = r_new
    alb2 = F.from_oklab(xp.clip(L, 0, 1), a, b)
    ao = xp.asarray(d['ao'], xp.float32)
    lin = MG.srgb_to_lin(alb2) * xp.power(xp.clip(ao, 0, 1), float(opt['ao']))[..., None]
    dif = MG.lin_to_srgb(xp.clip(lin, 0, 1)).astype(xp.float32)
    return alb2, dif, info


def process(job, done_root, dst, opt, compress=('quicktex', 14)):
    """One texture, full size: writes the new diffuse (and _fixed when the
    batch had one) under dst at the same relative places. Returns a note for
    the report."""
    d = load(job, float(opt['ao_old']))
    if X.active():
        d = {k: (X.to_dev(v) if isinstance(v, np.ndarray) else v) for k, v in d.items()}
    alb, dif, info = fix(d, opt)
    alb, dif = np.asarray(X.to_host(alb)), np.asarray(X.to_host(dif))
    alpha = d['alpha'] if d['alpha'] is None else np.asarray(X.to_host(d['alpha']))
    out_d = os.path.join(dst, job['rel'])
    os.makedirs(os.path.dirname(out_d) or '.', exist_ok=True)
    TX.write_value(dif, out_d, compress, alpha=alpha)
    if job['fixed']:
        out_f = os.path.join(dst, os.path.relpath(job['fixed'], done_root))
        os.makedirs(os.path.dirname(out_f) or '.', exist_ok=True)
        TX.write_value(alb, out_f, compress, alpha=alpha)
    parts = []
    if job['source'] is None:
        parts.append(_t('af.r_nosrc'))
    else:
        parts.append(_t('af.r_k', info['k']))
        if info['band']:
            parts.append(_t('af.r_band', float(opt['r_old']), info['band']))
        parts.append(_t('af.r_def', info['defects']))
    if d['from_diffuse']:
        parts.append(_t('af.r_nofixed'))
    if not job['orm']:
        parts.append(_t('af.r_noorm'))
    return ', '.join(parts)


def run(src_root, done_root, dst, opt, say=print, stop=lambda: False):
    """The whole folder: every diffuse under done_root (see find_jobs), into
    dst. say(text) gets one line per file; stop() → True ends early. Writes
    albedo_fix_report.txt into dst. Returns counts dict."""
    jobs = find_jobs(src_root, done_root)
    os.makedirs(dst, exist_ok=True)
    cnt = dict(found=len(jobs), done=0, failed=0)
    lines = [_t('af.title') + '  ' + time.strftime('%Y-%m-%d %H:%M'),
             _t('af.r_src', src_root), _t('af.r_done', done_root), _t('af.r_dst', dst),
             _t('af.r_opt', ', '.join('%s=%s' % kv for kv in sorted(opt.items()))),
             _t('af.r_dev', X.device_name() if X.active() else _t('af.cpu')), '']
    for k, job in enumerate(jobs):
        if stop():
            lines.append(_t('af.r_stopped'))
            break
        try:
            note = process(job, done_root, dst, opt)
            cnt['done'] += 1
            line = '%s  %s' % (job['rel'], note)
        except Exception as e:               # noqa: BLE001 — один файл не валит пакет
            cnt['failed'] += 1
            line = '%s  %s %s: %s' % (job['rel'], _t('af.r_error'), e.__class__.__name__, e)
        lines.append(line)
        say('[%d/%d] %s' % (k + 1, len(jobs), line))
    lines += ['', _t('af.done', cnt['found'], cnt['done'], cnt['failed'])]
    with open(os.path.join(dst, 'albedo_fix_report.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    return cnt


# ----------------------------------------------------------------- окно ---
def window():
    """The tool's window: three folders, the settings, a list of textures
    with a source / now / after preview, and the run in a thread."""
    import queue
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from PIL import Image, ImageTk
    t = _t
    root = tk.Tk()
    root.title(t('af.title'))
    root.geometry('1180x1000')
    frm = ttk.Frame(root, padding=10)
    frm.pack(fill='both', expand=True)
    ttk.Label(frm, text=t('af.intro'), wraplength=1140, justify='left').grid(
        row=0, column=0, columnspan=4, sticky='w', pady=(0, 8))
    src_v, done_v, dst_v = tk.StringVar(), tk.StringVar(), tk.StringVar()
    st = {'jobs': [], 'busy': False, 'stop': False, 'cache': None, 'after': None}

    def pick(var, follow=False):
        """Folder dialog; a newly picked results folder also sets the output."""
        d = filedialog.askdirectory()
        if d:
            var.set(d)
            if follow and not dst_v.get():
                dst_v.set(d.rstrip('\\/') + '_fixed')
            rescan()
    for r, (key, var, fol) in enumerate((('af.src', src_v, False), ('af.done_dir', done_v, True),
                                         ('af.dst', dst_v, False)), 1):
        ttk.Label(frm, text=t(key)).grid(row=r, column=0, sticky='w')
        ttk.Entry(frm, textvariable=var, width=90).grid(row=r, column=1, columnspan=2, sticky='ew')
        ttk.Button(frm, text='…', width=3,
                   command=lambda v=var, f=fol: pick(v, f)).grid(row=r, column=3)
    opts = ttk.Frame(frm)
    opts.grid(row=4, column=0, columnspan=4, sticky='ew', pady=6)
    opts.columnconfigure(1, weight=1)
    v = {k: (tk.BooleanVar(value=val) if isinstance(val, bool) else tk.DoubleVar(value=val))
         for k, val in DEFAULTS.items()}
    row = [0]

    def check(key, text):
        """A check box bound to v[key]."""
        ttk.Checkbutton(opts, text=t(text), variable=v[key],
                        command=lambda: schedule()).grid(row=row[0], column=0, columnspan=3,
                                                         sticky='w')
        row[0] += 1

    def slider(key, text, lo, hi, fmt='%.2f', hint=None):
        """A slider with its value and an optional grey hint."""
        ttk.Label(opts, text=t(text)).grid(row=row[0], column=0, sticky='w', padx=(20, 0))
        lab = ttk.Label(opts, width=7)
        lab.grid(row=row[0], column=2, sticky='w')

        def moved(_=None):
            lab.config(text=fmt % v[key].get())
            schedule()
        ttk.Scale(opts, from_=lo, to=hi, variable=v[key], command=moved).grid(
            row=row[0], column=1, sticky='ew')
        lab.config(text=fmt % v[key].get())
        row[0] += 1
        if hint:
            ttk.Label(opts, text=t(hint), foreground='#777', wraplength=1000,
                      justify='left').grid(row=row[0], column=0, columnspan=3, sticky='w',
                                           padx=(20, 0))
            row[0] += 1
    check('spread', 'af.spread')
    slider('weight', 'af.weight', 0.0, 1.0, hint='af.spread_hint')
    check('detail_on', 'af.detail_on')
    slider('detail', 'af.detail', 0.0, 2.0)
    slider('r_old', 'af.r_old', 0.5, 8.0, '%.1f')
    ttk.Label(opts, text=t('af.sm_res')).grid(row=row[0], column=0, sticky='w', padx=(20, 0))
    res_cb = ttk.Combobox(opts, values=('256', '512', '768', '1024', '2048'), state='readonly',
                          width=8)
    res_cb.set(str(int(DEFAULTS['sm_res'])))
    res_cb.grid(row=row[0], column=1, sticky='w')
    res_cb.bind('<<ComboboxSelected>>', lambda e: (v['sm_res'].set(float(res_cb.get())),
                                                   schedule()))
    row[0] += 1
    ttk.Label(opts, text=t('af.detail_hint'), foreground='#777', wraplength=1000,
              justify='left').grid(row=row[0], column=0, columnspan=3, sticky='w', padx=(20, 0))
    row[0] += 1
    slider('thr', 'af.thr', 0.0, 0.2, '%.3f', hint='af.thr_hint')
    slider('ao', 'af.ao', 0.0, 3.0)
    slider('ao_old', 'af.ao_old', 0.0, 3.0, hint='af.ao_old_hint')
    mid = ttk.Frame(frm)
    mid.grid(row=5, column=0, columnspan=4, sticky='nsew')
    lst = tk.Listbox(mid, width=34, exportselection=False)
    lst.pack(side='left', fill='y')
    views = ttk.Frame(mid)
    views.pack(side='left', fill='both', expand=True, padx=8)
    cap = ttk.Frame(views)
    cap.pack(fill='x')
    for key in ('af.v_src', 'af.v_now', 'af.v_new'):
        ttk.Label(cap, text=t(key)).pack(side='left', expand=True)
    imgs = ttk.Frame(views)
    imgs.pack(fill='both', expand=True)
    pics = [ttk.Label(imgs) for _ in range(3)]
    for p in pics:
        p.pack(side='left', expand=True)
    readout = ttk.Label(views, text='', foreground='#777')
    readout.pack(anchor='w')
    bar = ttk.Frame(frm)
    bar.grid(row=6, column=0, columnspan=4, sticky='w', pady=4)
    log = tk.Text(frm, height=5, wrap='none')
    log.grid(row=7, column=0, columnspan=4, sticky='nsew')
    frm.columnconfigure(1, weight=1)
    frm.rowconfigure(5, weight=1)
    lines = queue.Queue()

    def say(s):
        """Queue a line for the log (called from the worker thread)."""
        lines.put(s)

    def pump():
        """Move queued lines into the log; Tk is touched only from here."""
        while not lines.empty():
            log.insert('end', lines.get() + '\n')
            log.see('end')
        root.after(100, pump)
    pump()

    def opt():
        """Current settings as a plain dict."""
        o = {k: var.get() for k, var in v.items()}
        o['sm_res'] = float(res_cb.get())
        return o

    def rescan():
        """Find the textures again and fill the list."""
        lst.delete(0, 'end')
        d = done_v.get().strip()
        st['jobs'] = find_jobs(src_v.get().strip(), d) if d and os.path.isdir(d) else []
        for j in st['jobs']:
            lst.insert('end', j['rel'] + ('' if j['source'] else '  ·'))
        if st['jobs']:
            lst.selection_set(0)
            show()

    def show(_=None):
        """Load the selected texture as a reduced (≤ 256 px) preview copy."""
        sel = lst.curselection()
        if not sel:
            return
        job = st['jobs'][sel[0]]
        try:
            d = load(job, float(v['ao_old'].get()))
        except Exception as e:                # noqa: BLE001
            say('%s: %s' % (job['rel'], e))
            return
        W = d['albedo'].shape[1]
        k = max(1, int(np.ceil(max(d["albedo"].shape[:2]) / 256.0)))
        small = {kk: (vv[::k, ::k].copy() if isinstance(vv, np.ndarray) else vv)
                 for kk, vv in d.items()}
        now = TX.read_rgba(job['diffuse'])[::k, ::k, :3]
        st['cache'] = (small, W, now)
        draw()
    lst.bind('<<ListboxSelect>>', show)

    def schedule():
        """Redraw the preview shortly after the settings stop changing."""
        if st['after']:
            root.after_cancel(st['after'])
        st['after'] = root.after(150, draw)

    def draw():
        """Source, current diffuse and the new diffuse of the selected texture."""
        st['after'] = None
        if st['cache'] is None:
            return
        small, W, now = st['cache']
        _, dif, info = fix(small, opt(), full_width=W)
        src = small['src'] if small['src'] is not None else np.zeros_like(now)
        for lab, a in zip(pics, (src, now, X.to_host(dif))):
            im = Image.fromarray((np.clip(np.asarray(a)[..., :3], 0, 1) * 255 + .5).astype(np.uint8))
            ph = ImageTk.PhotoImage(im)
            lab.config(image=ph)
            lab.image = ph
        readout.config(text=t('af.readout', info['k'], info['defects']) if small['src'] is not None
                       else t('af.r_nosrc'))

    def start():
        """Check the folders and run the batch in a thread."""
        if st['busy']:
            return
        s, d, out = src_v.get().strip(), done_v.get().strip(), dst_v.get().strip()
        if not d or not os.path.isdir(d):
            messagebox.showerror(t('af.title'), t('af.nodone'))
            return
        if not out:
            out = d.rstrip('\\/') + '_fixed'
            dst_v.set(out)
        err = check_folders((s, d), out)
        if err:
            messagebox.showerror(t('af.title'), t(err))
            return
        if not st['jobs']:
            rescan()
        if not st['jobs']:
            messagebox.showinfo(t('af.title'), t('af.none'))
            return
        if os.path.isdir(out) and os.listdir(out) and not messagebox.askyesno(
                t('af.title'), t('nf.overwrite', out)):
            return
        o = opt()
        st.update(busy=True, stop=False)
        log.delete('1.0', 'end')
        say(t('af.found', len(st['jobs'])))

        def work():
            """Worker: the batch, then the summary."""
            try:
                c = run(s, d, out, o, say, lambda: st['stop'])
                say(t('af.done', c['found'], c['done'], c['failed']))
                say(t('nf.report', os.path.join(out, 'albedo_fix_report.txt')))
            except Exception as e:            # noqa: BLE001
                say('ERROR: %s' % e)
            finally:
                st['busy'] = False
        threading.Thread(target=work, daemon=True).start()
    ttk.Button(bar, text=t('nf.start'), command=start).pack(side='left')
    ttk.Button(bar, text=t('nf.stop'), command=lambda: st.update(stop=True)).pack(
        side='left', padx=6)
    ttk.Button(bar, text=t('ao_s.rescan'), command=rescan).pack(side='left')
    root.mainloop()


def main(argv):
    """Command line: <sources> <done textures> <output>; no arguments — the
    window. Settings are the defaults (DEFAULTS)."""
    X.use(True)                                    # видеокарта, если есть CuPy
    args = [a for a in argv if not a.startswith('--')]
    if not args:
        window()
        return 0
    if len(args) < 3:
        print(__doc__)
        return 2
    if check_folders(args[:2], args[2]):
        print(_t('af.same'))
        return 2
    c = run(args[0], args[1], args[2], dict(DEFAULTS))
    print(_t('af.done', c['found'], c['done'], c['failed']))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
