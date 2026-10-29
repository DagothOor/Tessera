# -*- coding: utf-8 -*-
"""Normal Flip: turn ready normal maps (_n, _nh) from the OpenGL green layout
into the DirectX one that OpenMW expects — a whole folder at once.

OpenMW reads normal maps in the DirectX layout (green: y down the image).
Maps made for OpenGL have the green channel the other way round, and the
relief in the game looks lit from the wrong side. This tool flips green,
G' = 255 − G, and leaves everything else alone.

How the flip is done:
  * DDS — right inside the compressed blocks, every mip level, without
    decoding and encoding again: DXT1 (BC1), DXT3 (BC2), DXT5 (BC3), ATI2
    (BC5) and uncompressed 16/24/32-bit, also with the DX10 header. Red,
    blue and the alpha (the parallax height of _nh) stay bit for bit; green
    is exactly 255 − G at the block end colours and within ±1 at the
    in-between ones (rounding of the decoder). The format and the mips stay
    the file's own.
  * other DDS that Pillow can read (BC7 and the like): decoded, flipped and
    written as DXT5 with new mips (texio.save_rgba) — the log says so;
  * PNG, TGA, BMP: the picture with green inverted, in the same format.

Which files to flip:
  * 'auto' (default) — each map is checked (matgen.guess_green: a normal
    that came from a surface has almost no curl in its true layout, and a
    lot in the wrong one); only the OpenGL ones are flipped. When the two
    readings are too close to tell (flat or noisy maps) the file is left
    alone and the log says «unclear»;
  * 'all' — every map found is flipped, nothing is checked.

Files are found by name: the part before the extension ends with _n or _nh
(any case). Results go into a separate output folder with the same
sub-folder layout; the source folder is never written to. A report
(normal_flip_report.txt) is written next to the results.

Run Normal-Flip.bat for the window, or from the command line:
    python normal_flip.py <source folder> [<output folder>] [--all] [--recursive]

Нормали OpenGL → DirectX для OpenMW пачкой: зелёный канал переворачивается
прямо в сжатых блоках DDS, без пересжатия."""

import os
import sys
import struct
import threading
import time

import numpy as np

BUILD = 25          # версия набора файлов, проверяется при запуске

EXTS = ('.dds', '.png', '.tga', '.bmp')
SUFFIXES = ('_n', '_nh')

# DDS: смещения полей заголовка от начала файла (после 'DDS ')
_FLAGS, _H, _W, _MIPS = 8, 12, 16, 28
_PF_FLAGS, _FOURCC, _BITS, _GMASK = 80, 84, 88, 96
_CAPS2 = 112
_DDSD_MIPMAPCOUNT = 0x20000
_DDPF_FOURCC, _DDPF_RGB = 0x4, 0x40

# формат DXGI (заголовок DX10) → наш вид блоков; None — не умеем на месте
_DXGI = {70: 'bc1', 71: 'bc1', 72: 'bc1', 73: 'bc2', 74: 'bc2', 75: 'bc2',
         76: 'bc3', 77: 'bc3', 78: 'bc3', 82: 'bc5', 83: 'bc5',
         27: 'rgba8', 28: 'rgba8', 29: 'rgba8', 87: 'rgba8', 88: 'rgba8', 90: 'rgba8',
         91: 'rgba8', 92: 'rgba8', 93: 'rgba8'}     # BC5 SNORM (84) — знаковый, не трогаем
_FOURCCS = {b'DXT1': 'bc1', b'DXT2': 'bc2', b'DXT3': 'bc2', b'DXT4': 'bc3', b'DXT5': 'bc3',
            b'ATI2': 'bc5', b'BC5U': 'bc5'}
_BLOCK = {'bc1': 8, 'bc2': 16, 'bc3': 16, 'bc5': 16}


class Unsupported(Exception):
    """The file's format cannot be flipped in place (the caller may fall back
    to decoding it)."""


def _t(key, *args):
    """Interface text (i18n, the language chosen in Texelith)."""
    import i18n
    if not getattr(_t, 'ready', False):
        i18n.load()
        _t.ready = True
    return i18n.t(key, *args)


# ------------------------------------------------------------ блоки DDS ---
def _flip_565(c):
    """Invert the 6-bit green of RGB565 colours (uint16 array)."""
    g = (c >> 5) & 63
    return ((c & ~np.uint16(63 << 5)) | ((63 - g) << 5)).astype(np.uint16)


def _flip_color_blocks(b, bc1):
    """Flip green in the 8-byte colour part of BC1/BC2/BC3 blocks.

    b: uint8 array (n, 8) — c0, c1 (RGB565, little-endian), 32 bits of 2-bit
    indices. Green of both end colours is inverted; the in-between colours
    follow, being mixes of the two. In BC1 the order c0 > c1 picks the mode
    (4 colours, or 3 and transparent black); when inverting green changes
    that order, the end colours are swapped back and the indices remapped
    (4-colour: 0↔1, 2↔3; 3-colour: 0↔1, 2 and 3 stay), so the mode and the
    transparent texels stay. BC2/BC3 colour parts are always 4-colour, so
    nothing else is needed there. Returns the new (n, 8) array."""
    v = b.view('<u2').reshape(-1, 4).copy()
    c0, c1 = v[:, 0].copy(), v[:, 1].copy()
    idx = b.view('<u4').reshape(-1, 2)[:, 1].copy()
    n0, n1 = _flip_565(c0), _flip_565(c1)
    if bc1:
        swap = (c0 > c1) != (n0 > n1)
        four = c0 > c1
        # у 4-цветных — XOR 1 у каждого индекса; у 3-цветных — только у 0 и 1
        hi = (idx >> 1) & np.uint32(0x55555555)
        m3 = (~hi) & np.uint32(0x55555555)
        new_idx = np.where(four, idx ^ np.uint32(0x55555555), idx ^ m3)
        idx = np.where(swap, new_idx, idx).astype(np.uint32)
        n0, n1 = np.where(swap, n1, n0), np.where(swap, n0, n1)
    out = np.empty((len(c0), 8), np.uint8)
    out.view('<u2').reshape(-1, 4)[:, 0] = n0
    out.view('<u2').reshape(-1, 4)[:, 1] = n1
    out.view('<u4').reshape(-1, 2)[:, 1] = idx
    return out


# новые номера оттенков при обращении канала в блоке BC4/BC5 (концы
# переставляются, чтобы режим блока остался тем же)
_MAP8 = np.array([1, 0, 7, 6, 5, 4, 3, 2], np.uint64)   # e0 > e1: 8 значений
_MAP6 = np.array([1, 0, 5, 4, 3, 2, 7, 6], np.uint64)   # e0 ≤ e1: 6 значений, 0 и 255


def _flip_channel_blocks(b):
    """Invert one channel stored as BC4-style blocks (the green half of BC5).

    b: uint8 array (n, 8) — two end values and 16 three-bit indices. The ends
    become 255 − e and swap places, which keeps the block's mode (e0 > e1:
    eight values; otherwise six and 0, 255), and every index is remapped to
    the value that is now its mirror. Exact: the mirrored palette is the old
    one inverted. Returns the new (n, 8) array."""
    e0, e1 = b[:, 0].astype(np.int32), b[:, 1].astype(np.int32)
    bits = np.zeros(len(b), np.uint64)
    for k in range(6):
        bits |= b[:, 2 + k].astype(np.uint64) << np.uint64(8 * k)
    eight = e0 > e1
    nb = np.zeros_like(bits)
    for t in range(16):
        sh = np.uint64(3 * t)
        i = (bits >> sh) & np.uint64(7)
        j = np.where(eight, _MAP8[i.astype(np.int64)], _MAP6[i.astype(np.int64)])
        nb |= j.astype(np.uint64) << sh
    out = np.empty_like(b)
    out[:, 0] = (255 - e1).astype(np.uint8)
    out[:, 1] = (255 - e0).astype(np.uint8)
    for k in range(6):
        out[:, 2 + k] = ((nb >> np.uint64(8 * k)) & np.uint64(255)).astype(np.uint8)
    return out


def _level_sizes(w, h, mips, kind, bpp):
    """Bytes of every mip level, top first."""
    out = []
    for lv in range(mips):
        lw, lh = max(1, w >> lv), max(1, h >> lv)
        if kind in _BLOCK:
            out.append(((lw + 3) // 4) * ((lh + 3) // 4) * _BLOCK[kind])
        else:
            out.append(lw * lh * bpp)
    return out


def dds_info(data):
    """What the DDS header says: dict with kind ('bc1', 'bc2', 'bc3',
    'bc5', 'rgba8', 'rgb' or None), width, height, mips, data offset and,
    for uncompressed files, bytes per pixel and the green bit mask; 'name'
    — the format as text for the report. Raises Unsupported for cube maps,
    volumes and texture arrays."""
    if data[:4] != b'DDS ' or len(data) < 128:
        raise Unsupported('not a DDS file')
    flags, h, w = struct.unpack_from('<3I', data, _FLAGS)
    mips = struct.unpack_from('<I', data, _MIPS)[0]
    mips = mips if (flags & _DDSD_MIPMAPCOUNT or mips > 1) and mips > 0 else 1
    pff, fourcc, bits = struct.unpack_from('<I4sI', data, _PF_FLAGS)
    gmask = struct.unpack_from('<I', data, _GMASK)[0]
    caps2 = struct.unpack_from('<I', data, _CAPS2)[0]
    if caps2 & 0x200 or caps2 & 0x200000:
        raise Unsupported('cube map or volume texture')
    info = dict(width=w, height=h, mips=mips, offset=128, bpp=0, gmask=0, kind=None,
                name=fourcc.decode('latin-1') if pff & _DDPF_FOURCC else '')
    if pff & _DDPF_FOURCC and fourcc == b'DX10':
        dxgi, _dim, _misc, arr = struct.unpack_from('<4I', data, 128)
        if arr > 1:
            raise Unsupported('texture array')
        info.update(offset=148, kind=_DXGI.get(dxgi), name='DX10/%d' % dxgi, bpp=4)
        if info['kind'] == 'rgba8':
            info['gmask'] = 0x0000FF00          # и у RGBA, и у BGRA зелёный — второй байт
    elif pff & _DDPF_FOURCC:
        info['kind'] = _FOURCCS.get(fourcc)
    elif pff & _DDPF_RGB and bits in (16, 24, 32) and gmask:
        info.update(kind='rgb', bpp=bits // 8, gmask=gmask, name='RGB%d' % bits)
    return info


def flip_dds_bytes(data):
    """The DDS file ``data`` (bytes) with green inverted in place, every
    mip level, same format. Returns (new bytes, format name). Raises
    Unsupported when the format cannot be done in place."""
    info = dds_info(data)
    kind = info['kind']
    if kind is None:
        raise Unsupported('format %s' % (info['name'] or '?'))
    sizes = _level_sizes(info['width'], info['height'], info['mips'], kind, info['bpp'])
    off = info['offset']
    total = sum(sizes)
    if off + total > len(data):
        # уровней в файле меньше, чем в заголовке, — берём сколько целиком есть
        acc, keep = 0, 0
        for s in sizes:
            if off + acc + s > len(data):
                break
            acc += s
            keep += 1
        if keep == 0:
            raise Unsupported('file is shorter than its header says')
        total = acc
    buf = bytearray(data)
    body = np.frombuffer(bytes(buf[off:off + total]), np.uint8)
    if kind in _BLOCK:
        bs = _BLOCK[kind]
        blocks = body.reshape(-1, bs).copy()
        if kind == 'bc1':
            blocks = _flip_color_blocks(blocks, True)
        elif kind in ('bc2', 'bc3'):
            blocks[:, 8:] = _flip_color_blocks(np.ascontiguousarray(blocks[:, 8:]), False)
        else:                                         # bc5: R — первый блок, G — второй
            blocks[:, 8:] = _flip_channel_blocks(np.ascontiguousarray(blocks[:, 8:]))
        new = blocks.tobytes()
    else:
        bpp, gm = info['bpp'], info['gmask']
        if bpp == 4:
            px = body.view('<u4') ^ np.uint32(gm)
        elif bpp == 2:
            px = body.view('<u2') ^ np.uint16(gm)
        else:                                         # 24 бита: маска по байтам
            mb = np.frombuffer(struct.pack('<I', gm)[:3], np.uint8)
            px = (body.reshape(-1, 3) ^ mb).reshape(-1)
        new = px.tobytes()
    buf[off:off + total] = new
    name = {'bc1': 'DXT1', 'bc2': 'DXT3', 'bc3': 'DXT5', 'bc5': 'BC5/ATI2'}.get(kind) \
        or info['name']
    return bytes(buf), name


# ------------------------------------------------------- чтение и проверка --
def read_normal(path):
    """Top level of a map as float32 RGBA 0…1 (Pillow; texio.read_rgba)."""
    import texio as TX
    return TX.read_rgba(path)


def _half(a):
    """2×2 box average of an (H, W, C) picture."""
    h, w = a.shape[0] // 2 * 2, a.shape[1] // 2 * 2
    return a[:h, :w].reshape(h // 2, 2, w // 2, 2, a.shape[2]).mean(axis=(1, 3))


def detect(rgba):
    """Green layout of a normal map: (verdict, curl as DirectX, curl as
    OpenGL); verdict is 'dx', 'gl' or 'unclear' (the smaller reading more
    than 0.6 of the larger).

    The curl is measured on a copy reduced to ≤ 512 px (half size at least
    when the side is 256 or more) and blurred by 1 px. Block compression
    adds noise of its own that has curl in either layout and evens the two
    readings out: on five photo textures turned into DXT1 normals the ratio
    of the readings was 0.62…0.74 at full size and 0.15…0.26 this way. Blue
    near zero everywhere (BC5 keeps only x and y) — z is rebuilt as
    √(1 − x² − y²)."""
    import matgen as MG
    from scipy import ndimage as ndi
    a = np.asarray(rgba, np.float32)[..., :3]
    if min(a.shape[:2]) >= 256:
        a = _half(a)                    # шум сжатия — в мелочи, половинный размер его гасит
    while max(a.shape[:2]) > 512:
        a = _half(a)
    a = ndi.gaussian_filter(a, (1.0, 1.0, 0), mode='wrap')
    n = a * 2.0 - 1.0
    if float(a[..., 2].mean()) < 0.05:
        n[..., 2] = np.sqrt(np.clip(1.0 - n[..., 0] ** 2 - n[..., 1] ** 2, 0.0, 1.0))
    n = n / np.maximum(np.sqrt((n * n).sum(-1, keepdims=True)), 1e-6)
    is_dx, c_dx, c_gl = MG.guess_green(n.astype(np.float32))
    lo, hi = sorted((c_dx, c_gl))
    if hi <= 0 or lo / hi > 0.6:
        return 'unclear', c_dx, c_gl
    return ('dx' if is_dx else 'gl'), c_dx, c_gl


def flip_file(src, dst):
    """Flip green of the map ``src`` into ``dst``. Returns a note for the
    report: how it was done."""
    ext = os.path.splitext(src)[1].lower()
    os.makedirs(os.path.dirname(dst) or '.', exist_ok=True)
    if ext == '.dds':
        with open(src, 'rb') as f:
            data = f.read()
        try:
            new, fmt = flip_dds_bytes(data)
            with open(dst, 'wb') as f:
                f.write(new)
            return _t('nf.r_inplace', fmt)
        except Unsupported as e:
            # не умеем на месте — раскрываем, переворачиваем, пишем DXT5
            import texio as TX
            rgba = read_normal(src)
            rgba[..., 1] = 1.0 - rgba[..., 1]
            TX.save_rgba(np.clip(rgba * 255.0 + 0.5, 0, 255).astype(np.uint8), dst,
                         ('quicktex', 14), normal=True)
            return _t('nf.r_decoded', e)
    from PIL import Image
    im = Image.open(src)
    im.load()
    mode = im.mode
    im = im.convert('RGBA' if 'A' in mode else 'RGB')
    ch = list(im.split())
    ch[1] = ch[1].point(lambda v: 255 - v)
    out = Image.merge(im.mode, ch)
    out.save(dst)
    return _t('nf.r_picture')


# ---------------------------------------------------------------- пакет ---
def find_maps(root, recursive):
    """Normal maps under root: relative paths of files whose name ends with
    _n or _nh before a known extension. Sub-folders only with recursive."""
    out = []
    for d, dirs, files in os.walk(root):
        if not recursive:
            dirs[:] = []
        for fn in sorted(files):
            stem, ext = os.path.splitext(fn)
            if ext.lower() in EXTS and stem.lower().endswith(SUFFIXES):
                out.append(os.path.relpath(os.path.join(d, fn), root))
    return sorted(out)


def check_folders(src, dst):
    """Error text if the output folder is the source folder or lies inside
    it (results would mix with the originals), else None."""
    a, b = os.path.normcase(os.path.abspath(src)), os.path.normcase(os.path.abspath(dst))
    if a == b or b.startswith(a.rstrip('\\/') + os.sep):
        return 'nf.same'
    return None


def run(src, dst, mode='auto', recursive=False, copy_dx=False, say=print,
        stop=lambda: False):
    """Process the folder: flip the OpenGL maps (mode 'auto') or all of them
    ('all') from src into dst, keeping the sub-folder layout. copy_dx: also
    copy maps that are already DirectX (so dst is a complete set). say(text)
    gets one line per file; stop() → True ends early. Writes
    normal_flip_report.txt into dst. Returns counts dict."""
    import shutil
    maps = find_maps(src, recursive)
    os.makedirs(dst, exist_ok=True)
    cnt = dict(found=len(maps), flipped=0, kept=0, unclear=0, failed=0)
    lines = [_t('nf.title') + '  ' + time.strftime('%Y-%m-%d %H:%M'),
             _t('nf.r_src', src), _t('nf.r_dst', dst),
             _t('nf.r_mode', _t('nf.auto' if mode == 'auto' else 'nf.all'),
                _t('ni.yes' if recursive else 'ni.no')), '']
    for k, rel in enumerate(maps):
        if stop():
            lines.append(_t('nf.r_stopped'))
            break
        s, d = os.path.join(src, rel), os.path.join(dst, rel)
        try:
            verdict, cdx, cgl = ('-', 0.0, 0.0)
            if mode == 'auto':
                verdict, cdx, cgl = detect(read_normal(s))
            if mode == 'all' or verdict == 'gl':
                note = flip_file(s, d)
                cnt['flipped'] += 1
                what = _t('nf.r_flipped')
            elif verdict == 'dx':
                cnt['kept'] += 1
                what = _t('nf.r_dx')
                note = ''
                if copy_dx:
                    os.makedirs(os.path.dirname(d) or '.', exist_ok=True)
                    shutil.copy2(s, d)
                    note = _t('nf.r_copied')
            else:
                cnt['unclear'] += 1
                what = _t('nf.r_unclear')
                note = ''
            curl = '  ' + _t('nf.r_curl', cdx, cgl) if mode == 'auto' else ''
            line = '%s  %s%s%s' % (rel, what, curl, ('  (' + note + ')') if note else '')
        except Exception as e:                   # noqa: BLE001 — один файл не валит пакет
            cnt['failed'] += 1
            line = '%s  %s %s: %s' % (rel, _t('nf.r_error'), e.__class__.__name__, e)
        lines.append(line)
        say('[%d/%d] %s' % (k + 1, len(maps), line))
    lines += ['', _t('nf.done', cnt['found'], cnt['flipped'], cnt['kept'], cnt['unclear'],
                     cnt['failed'])]
    with open(os.path.join(dst, 'normal_flip_report.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    return cnt


# ----------------------------------------------------------------- окно ---
def window():
    """The tool's window: source and output folders, sub-folders, mode,
    a log; the work runs in a thread so the window stays alive."""
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    import i18n
    i18n.load()
    t = i18n.t
    root = tk.Tk()
    root.title(t('nf.title'))
    root.geometry('760x520')
    frm = ttk.Frame(root, padding=10)
    frm.pack(fill='both', expand=True)
    src_v, dst_v = tk.StringVar(), tk.StringVar()
    rec_v, copy_v = tk.BooleanVar(value=False), tk.BooleanVar(value=False)
    mode_v = tk.StringVar(value='auto')
    # входы окна берутся до запуска потока: Tk-переменные читаются только здесь
    ttk.Label(frm, text=t('nf.intro'), wraplength=720, justify='left').grid(
        row=0, column=0, columnspan=3, sticky='w', pady=(0, 8))

    def pick(var, other=None):
        """Folder dialog; the output folder follows a newly picked source."""
        d = filedialog.askdirectory()
        if d:
            var.set(d)
            if other is not None and not other.get():
                other.set(d.rstrip('\\/') + '_DX')
    ttk.Label(frm, text=t('nf.src')).grid(row=1, column=0, sticky='w')
    ttk.Entry(frm, textvariable=src_v, width=70).grid(row=1, column=1, sticky='ew')
    ttk.Button(frm, text='…', width=3, command=lambda: pick(src_v, dst_v)).grid(row=1, column=2)
    ttk.Label(frm, text=t('nf.dst')).grid(row=2, column=0, sticky='w')
    ttk.Entry(frm, textvariable=dst_v, width=70).grid(row=2, column=1, sticky='ew')
    ttk.Button(frm, text='…', width=3, command=lambda: pick(dst_v)).grid(row=2, column=2)
    ttk.Checkbutton(frm, text=t('nf.recursive'), variable=rec_v).grid(
        row=3, column=0, columnspan=3, sticky='w', pady=(6, 0))
    ttk.Radiobutton(frm, text=t('nf.auto'), value='auto', variable=mode_v).grid(
        row=4, column=0, columnspan=3, sticky='w')
    ttk.Radiobutton(frm, text=t('nf.all'), value='all', variable=mode_v).grid(
        row=5, column=0, columnspan=3, sticky='w')
    ttk.Checkbutton(frm, text=t('nf.copy_dx'), variable=copy_v).grid(
        row=6, column=0, columnspan=3, sticky='w')
    bar = ttk.Frame(frm)
    bar.grid(row=7, column=0, columnspan=3, sticky='w', pady=6)
    log = tk.Text(frm, height=16, wrap='none')
    log.grid(row=8, column=0, columnspan=3, sticky='nsew')
    frm.columnconfigure(1, weight=1)
    frm.rowconfigure(8, weight=1)
    state = {'stop': False, 'busy': False}
    import queue
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

    def start():
        """Check the folders and run the batch in a thread."""
        if state['busy']:
            return
        s, d = src_v.get().strip(), dst_v.get().strip()
        if not s or not os.path.isdir(s):
            messagebox.showerror(t('nf.title'), t('nf.nosrc'))
            return
        if not d:
            d = s.rstrip('\\/') + '_DX'
            dst_v.set(d)
        err = check_folders(s, d)
        if err:
            messagebox.showerror(t('nf.title'), t(err))
            return
        n = len(find_maps(s, rec_v.get()))
        if n == 0:
            messagebox.showinfo(t('nf.title'), t('nf.none'))
            return
        if os.path.isdir(d) and os.listdir(d) and not messagebox.askyesno(
                t('nf.title'), t('nf.overwrite', d)):
            return
        state.update(stop=False, busy=True)
        log.delete('1.0', 'end')
        say(t('nf.found', n))

        def work():
            """Worker: the batch, then the summary."""
            try:
                c = run(s, d, mode_v.get(), rec_v.get(), copy_v.get(), say,
                        lambda: state['stop'])
                say(t('nf.done', c['found'], c['flipped'], c['kept'], c['unclear'],
                      c['failed']))
                say(t('nf.report', os.path.join(d, 'normal_flip_report.txt')))
            except Exception as e:               # noqa: BLE001
                say('ERROR: %s' % e)
            finally:
                state['busy'] = False
        threading.Thread(target=work, daemon=True).start()
    ttk.Button(bar, text=t('nf.start'), command=start).pack(side='left')
    ttk.Button(bar, text=t('nf.stop'), command=lambda: state.update(stop=True)).pack(
        side='left', padx=6)
    root.mainloop()


def main(argv):
    """Command line: folders and --all / --recursive / --copy-dx; no
    arguments — the window."""
    args = [a for a in argv if not a.startswith('--')]
    if not args:
        window()
        return 0
    src = args[0]
    dst = args[1] if len(args) > 1 else src.rstrip('\\/') + '_DX'
    if check_folders(src, dst):
        print(_t('nf.same'))
        return 2
    c = run(src, dst, 'all' if '--all' in argv else 'auto', '--recursive' in argv,
            '--copy-dx' in argv)
    print(_t('nf.done', c['found'], c['flipped'], c['kept'], c['unclear'], c['failed']))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
