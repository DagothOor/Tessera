# -*- coding: utf-8 -*-
"""AO Strength: change how strongly ambient occlusion darkens ready diffuse
textures — a whole folder at once, without running the Texelith graph again.

The occlusion is taken from the texture's ORM map (R = AO, G = roughness,
B = metalness — the 'orm' output of Texelith's «PBR» node, glTF, Unreal);
an ARM map (Poly Haven and others) has the same layout and is taken too.
The pair is found by name: tx_name.dds and tx_name_orm.dds (or _arm), in
the same folder or in a separate ORM folder with the same sub-folders.

The diffuse is changed in linear light, the way «PBR → OpenMW» bakes AO in:
    diffuse = colour × AO^k
so going from the strength that is already baked in (k_old) to a new one:
    new = old × AO^(k_new − k_old)
k_old = 0 means the diffuse has no AO yet (a pack from elsewhere); 1 is
Texelith's default «AO in the diffuse». Lowering the strength divides by
AO, so where AO was near zero the old diffuse kept almost nothing and the
place stays dark (AO is floored at 0.02 there so noise is not blown up);
a colour that was clipped at white when baked cannot come back either.

The alpha (foliage, grilles) is kept; DDS files are written the way Texelith
writes a diffuse (texio.write_value: DXT5 with mips, coverage-keeping mips
for alpha-tested textures, uncompressed if the source was uncompressed).
Results go into a separate output folder with the same sub-folders; the
source folders are never written to. A report, ao_strength_report.txt, is
written next to the results.

The arithmetic is one power and one multiplication per pixel; the time goes
into DXT compression (quicktex, on the processor), so nothing here is worth
sending to the video card.

Run AO-Strength.bat for the window (with a before/after preview), or:
    python ao_strength.py <diffuse folder> <output folder> --old 1 --new 0.5
        [--orm <ORM folder>] [--recursive]

Сила AO в готовых диффузах пачкой: AO берётся из R-канала карты ORM."""

import os
import sys
import time
import threading

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))         # модули Texelith — в папке выше

import texio as TX                                # noqa: E402
import matgen as MG                               # noqa: E402

BUILD = 25          # версия набора файлов, проверяется при запуске

EXTS = ('.dds', '.png', '.tga', '.bmp')
ORM_SUFFIXES = ('_orm', '_arm')
AO_FLOOR = 0.02     # ниже AO не делим: там от старого диффуза почти ничего не осталось


def _t(key, *args):
    """Interface text (i18n, the language chosen in Texelith)."""
    import i18n
    if not getattr(_t, 'ready', False):
        i18n.load()
        _t.ready = True
    return i18n.t(key, *args)


# ------------------------------------------------------------ поиск пар ---
def _files(root, recursive):
    """Relative paths of picture files under root (sub-folders only when
    recursive)."""
    out = []
    for d, dirs, files in os.walk(root):
        if not recursive:
            dirs[:] = []
        for fn in files:
            if os.path.splitext(fn)[1].lower() in EXTS:
                out.append(os.path.relpath(os.path.join(d, fn), root))
    return sorted(out)


def find_pairs(dif_root, orm_root=None, recursive=False):
    """Diffuse textures that have an ORM (or ARM) map: list of (relative
    path of the diffuse, full path of the ORM). A diffuse is any picture
    whose name does not end with a map suffix (texio.MAP_SUFFIXES); its ORM
    is <name>_orm.<any ext> (or _arm) at the same relative place under
    orm_root (default: dif_root). Files without a partner are left out."""
    orm_root = orm_root or dif_root
    orms = {}
    for rel in _files(orm_root, recursive):
        stem = os.path.splitext(rel)[0]
        for suf in ORM_SUFFIXES:
            if stem.lower().endswith(suf):
                orms.setdefault(os.path.normcase(stem[:-len(suf)]),
                                os.path.join(orm_root, rel))
    pairs = []
    for rel in _files(dif_root, recursive):
        stem = os.path.splitext(rel)[0]
        if stem.lower().endswith(TX.MAP_SUFFIXES):
            continue
        o = orms.get(os.path.normcase(stem))
        if o is not None:
            pairs.append((rel, o))
    return pairs


def check_folders(srcs, dst):
    """i18n key of an error if the output folder is one of the source
    folders or lies inside one (results would mix with the originals), else
    None."""
    b = os.path.normcase(os.path.abspath(dst))
    for s in srcs:
        if not s:
            continue
        a = os.path.normcase(os.path.abspath(s))
        if a == b or b.startswith(a.rstrip('\\/') + os.sep):
            return 'ao_s.same'
    return None


# -------------------------------------------------------------- расчёт ---
def load_pair(dif_path, orm_path):
    """(diffuse RGBA float32 0…1, AO H×W float32 0…1 at the diffuse's size).
    AO is the red channel of the ORM map, resized if the sizes differ."""
    rgba = TX.read_rgba(dif_path)
    ao = TX.read_rgba(orm_path)[..., 0]
    if ao.shape != rgba.shape[:2]:
        ao = TX.resize_to(ao, rgba.shape[:2])
    return rgba, np.clip(ao, 0.0, 1.0).astype(np.float32)


def apply(rgba, ao, k_old, k_new):
    """The diffuse with the AO strength changed from k_old to k_new.

    rgba: H×W×4 (or ×3) sRGB 0…1; ao: H×W 0…1. In linear light (the 2.2
    power Texelith uses) the colour is multiplied by AO^(k_new − k_old); when
    that exponent is negative, AO is floored at AO_FLOOR. Alpha is kept.
    Returns float32 like rgba."""
    e = float(k_new) - float(k_old)
    out = np.array(rgba, np.float32, copy=True)
    if abs(e) < 1e-6:
        return out
    a = np.maximum(ao, AO_FLOOR) if e < 0 else ao
    lin = MG.srgb_to_lin(out[..., :3]) * np.power(a, e)[..., None]
    out[..., :3] = MG.lin_to_srgb(np.clip(lin, 0.0, 1.0))
    return out


def process(dif_path, orm_path, out_path, k_old, k_new, compress=('quicktex', 14)):
    """Change one texture and write it to out_path; DDS stays compressed
    unless the source was uncompressed. Returns a note for the report."""
    rgba, ao = load_pair(dif_path, orm_path)
    res = apply(rgba, ao, k_old, k_new)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    alpha = res[..., 3] if rgba.shape[2] == 4 and float(rgba[..., 3].min()) < 0.999 else None
    comp = compress
    if os.path.splitext(dif_path)[1].lower() == '.dds':
        try:
            import normal_flip as NF          # разбор заголовка DDS
            with open(dif_path, 'rb') as f:
                info = NF.dds_info(f.read(160))
            if info['kind'] in ('rgb', 'rgba8'):
                comp = False                  # исходник без сжатия — и результат без
        except Exception:                     # noqa: BLE001 — не разобрали, пишем как обычно
            pass
    TX.write_value(res[..., :3], out_path, comp, alpha=alpha)
    darker = float((MG.srgb_to_lin(res[..., :3]).mean() + 1e-6) /
                   (MG.srgb_to_lin(rgba[..., :3]).mean() + 1e-6))
    return _t('ao_s.r_note', float(ao.mean()), (darker - 1.0) * 100.0)


def run(dif_root, dst, k_old, k_new, orm_root=None, recursive=False, say=print,
        stop=lambda: False):
    """The whole folder: every diffuse that has an ORM map, from k_old to
    k_new, into dst with the same sub-folders. say(text) gets one line per
    file; stop() → True ends early. Writes ao_strength_report.txt into dst.
    Returns counts dict."""
    pairs = find_pairs(dif_root, orm_root, recursive)
    os.makedirs(dst, exist_ok=True)
    cnt = dict(found=len(pairs), done=0, failed=0)
    lines = [_t('ao_s.title') + '  ' + time.strftime('%Y-%m-%d %H:%M'),
             _t('ao_s.r_src', dif_root), _t('ao_s.r_orm', orm_root or dif_root),
             _t('ao_s.r_dst', dst), _t('ao_s.r_k', k_old, k_new), '']
    for k, (rel, orm) in enumerate(pairs):
        if stop():
            lines.append(_t('ao_s.r_stopped'))
            break
        try:
            note = process(os.path.join(dif_root, rel), orm, os.path.join(dst, rel),
                           k_old, k_new)
            cnt['done'] += 1
            line = '%s  %s' % (rel, note)
        except Exception as e:               # noqa: BLE001 — один файл не валит пакет
            cnt['failed'] += 1
            line = '%s  %s %s: %s' % (rel, _t('ao_s.r_error'), e.__class__.__name__, e)
        lines.append(line)
        say('[%d/%d] %s' % (k + 1, len(pairs), line))
    lines += ['', _t('ao_s.done', cnt['found'], cnt['done'], cnt['failed'])]
    with open(os.path.join(dst, 'ao_strength_report.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    return cnt


# ----------------------------------------------------------------- окно ---
def window():
    """The tool's window: folders, the two strengths, a list of the pairs
    found with a before/after preview of the selected one, and the run."""
    import queue
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from PIL import Image, ImageTk
    t = _t
    root = tk.Tk()
    root.title(t('ao_s.title'))
    root.geometry('1000x720')
    frm = ttk.Frame(root, padding=10)
    frm.pack(fill='both', expand=True)
    dif_v, orm_v, dst_v = tk.StringVar(), tk.StringVar(), tk.StringVar()
    rec_v = tk.BooleanVar(value=False)
    old_v, new_v = tk.DoubleVar(value=1.0), tk.DoubleVar(value=1.0)
    ttk.Label(frm, text=t('ao_s.intro'), wraplength=960, justify='left').grid(
        row=0, column=0, columnspan=4, sticky='w', pady=(0, 8))
    st = {'pairs': [], 'busy': False, 'stop': False, 'cache': None, 'after': None}

    def pick(var, follow=False):
        """Folder dialog; a newly picked diffuse folder also sets the output."""
        d = filedialog.askdirectory()
        if d:
            var.set(d)
            if follow and not dst_v.get():
                dst_v.set(d.rstrip('\\/') + '_AO')
            rescan()
    for r, (key, var, fol) in enumerate((('ao_s.dif', dif_v, True), ('ao_s.orm', orm_v, False),
                                         ('ao_s.dst', dst_v, False)), 1):
        ttk.Label(frm, text=t(key)).grid(row=r, column=0, sticky='w')
        ttk.Entry(frm, textvariable=var, width=80).grid(row=r, column=1, columnspan=2,
                                                       sticky='ew')
        ttk.Button(frm, text='…', width=3,
                   command=lambda v=var, f=fol: pick(v, f)).grid(row=r, column=3)
    ttk.Checkbutton(frm, text=t('ao_s.recursive'), variable=rec_v,
                    command=lambda: rescan()).grid(row=4, column=0, columnspan=4, sticky='w')

    def slider(row, key, var):
        """A strength slider 0…3 with its value."""
        ttk.Label(frm, text=t(key)).grid(row=row, column=0, sticky='w')
        lab = ttk.Label(frm, width=6)
        lab.grid(row=row, column=3, sticky='w')

        def moved(_=None):
            lab.config(text='%.2f' % var.get())
            schedule_preview()
        ttk.Scale(frm, from_=0.0, to=3.0, variable=var, command=moved).grid(
            row=row, column=1, columnspan=2, sticky='ew')
        lab.config(text='%.2f' % var.get())
    slider(5, 'ao_s.old', old_v)
    slider(6, 'ao_s.new', new_v)
    ttk.Label(frm, text=t('ao_s.old_hint'), wraplength=960, justify='left',
              foreground='#777').grid(row=7, column=0, columnspan=4, sticky='w')
    mid = ttk.Frame(frm)
    mid.grid(row=8, column=0, columnspan=4, sticky='nsew', pady=6)
    lst = tk.Listbox(mid, width=34, exportselection=False)
    lst.pack(side='left', fill='y')
    views = ttk.Frame(mid)
    views.pack(side='left', fill='both', expand=True, padx=8)
    cap = ttk.Frame(views)
    cap.pack(fill='x')
    ttk.Label(cap, text=t('ao_s.before')).pack(side='left', expand=True)
    ttk.Label(cap, text=t('ao_s.after')).pack(side='left', expand=True)
    imgs = ttk.Frame(views)
    imgs.pack(fill='both', expand=True)
    before, after = ttk.Label(imgs), ttk.Label(imgs)
    before.pack(side='left', expand=True)
    after.pack(side='left', expand=True)
    bar = ttk.Frame(frm)
    bar.grid(row=9, column=0, columnspan=4, sticky='w')
    log = tk.Text(frm, height=8, wrap='none')
    log.grid(row=10, column=0, columnspan=4, sticky='nsew')
    frm.columnconfigure(1, weight=1)
    frm.rowconfigure(8, weight=1)
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

    def rescan():
        """Find the pairs again and fill the list."""
        lst.delete(0, 'end')
        d = dif_v.get().strip()
        st['pairs'] = find_pairs(d, orm_v.get().strip() or None, rec_v.get()) \
            if d and os.path.isdir(d) else []
        for rel, _ in st['pairs']:
            lst.insert('end', rel)
        if st['pairs']:
            lst.selection_set(0)
            show_pair()

    def show_pair(_=None):
        """Load the selected pair (reduced to ≤ 400 px) for the preview."""
        sel = lst.curselection()
        if not sel:
            return
        rel, orm = st['pairs'][sel[0]]
        try:
            rgba, ao = load_pair(os.path.join(dif_v.get().strip(), rel), orm)
        except Exception as e:                # noqa: BLE001
            say('%s: %s' % (rel, e))
            return
        k = max(1, int(np.ceil(max(rgba.shape[:2]) / 400.0)))
        st['cache'] = (rgba[::k, ::k].copy(), ao[::k, ::k].copy())
        draw_preview()
    lst.bind('<<ListboxSelect>>', show_pair)

    def schedule_preview():
        """Redraw the preview shortly after the sliders stop moving."""
        if st['after']:
            root.after_cancel(st['after'])
        st['after'] = root.after(120, draw_preview)

    def draw_preview():
        """Before and after of the selected texture with the current strengths."""
        st['after'] = None
        if st['cache'] is None:
            return
        rgba, ao = st['cache']
        res = apply(rgba, ao, old_v.get(), new_v.get())
        for lab, a in ((before, rgba), (after, res)):
            im = Image.fromarray((np.clip(a[..., :3], 0, 1) * 255 + 0.5).astype(np.uint8))
            ph = ImageTk.PhotoImage(im)
            lab.config(image=ph)
            lab.image = ph

    def start():
        """Check the folders and run the batch in a thread."""
        if st['busy']:
            return
        d, o, out = dif_v.get().strip(), orm_v.get().strip(), dst_v.get().strip()
        if not d or not os.path.isdir(d):
            messagebox.showerror(t('ao_s.title'), t('ao_s.nosrc'))
            return
        if not out:
            out = d.rstrip('\\/') + '_AO'
            dst_v.set(out)
        err = check_folders((d, o), out)
        if err:
            messagebox.showerror(t('ao_s.title'), t(err))
            return
        if not st['pairs']:
            rescan()
        if not st['pairs']:
            messagebox.showinfo(t('ao_s.title'), t('ao_s.none'))
            return
        if os.path.isdir(out) and os.listdir(out) and not messagebox.askyesno(
                t('ao_s.title'), t('nf.overwrite', out)):
            return
        k_old, k_new, rec = old_v.get(), new_v.get(), rec_v.get()
        st.update(busy=True, stop=False)
        log.delete('1.0', 'end')
        say(t('ao_s.found', len(st['pairs'])))

        def work():
            """Worker: the batch, then the summary."""
            try:
                c = run(d, out, k_old, k_new, o or None, rec, say, lambda: st['stop'])
                say(t('ao_s.done', c['found'], c['done'], c['failed']))
                say(t('nf.report', os.path.join(out, 'ao_strength_report.txt')))
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
    """Command line (see the module docstring); no arguments — the window."""
    args = [a for a in argv if not a.startswith('--')]

    def opt(name, default):
        """Value after --name, or default."""
        if name in argv:
            i = argv.index(name)
            if i + 1 < len(argv):
                return argv[i + 1]
        return default
    if not args:
        window()
        return 0
    vals = {opt('--old', None), opt('--new', None), opt('--orm', None)}
    args = [a for a in args if a not in vals]
    if len(args) < 2:
        print(__doc__)
        return 2
    orm = opt('--orm', None)
    if check_folders((args[0], orm), args[1]):
        print(_t('ao_s.same'))
        return 2
    c = run(args[0], args[1], float(opt('--old', 1.0)), float(opt('--new', 1.0)), orm,
            '--recursive' in argv)
    print(_t('ao_s.done', c['found'], c['done'], c['failed']))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
