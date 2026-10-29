# -*- coding: utf-8 -*-
"""
Texelith — texture map authoring for game engines.
Texelith — карты для игровых текстур.

Four workspaces:
  Foliage: normals   — per-leaf shape from the alpha silhouette, from veins
                       or from bright tufts. Leaves, needles, bare branches,
                       moss, grass and groundcover.
  Foliage: light, AO — remove painted lighting, build occlusion.
  Material           — opaque textures: damaged-area repair, de-lighting,
                       height (including depth estimated by a neural network),
                       normal, AO, specular/roughness, tiling, and several
                       materials on one texture by mask.
  Normal map repair  — inspect and fix finished normal maps.

Run it by double-clicking Texelith.bat, or `python Texelith.py`.
Keep i18n.py, foliage.py, matgen.py, normalfix.py, repair.py, view3d.py,
depthnet.py, nodes.py and esrgan.py beside it.

F11 — full screen.  Wheel — zoom.  Drag — pan.  Double click — fit.

This file is the whole user interface: widgets, the four workspaces, the node
graph tab, the 3D view, file input/output and batch processing. The image math
lives in the modules listed above; this file only wires it to Tk.

Conventions a fork must keep:
  * BUILD must be equal in every module of the distribution; check_build()
    compares them at start-up, because mixed file versions are the most common
    cause of strange behaviour.
  * Heavy work runs in worker threads. A worker thread never touches Tk (not
    even root.after); it writes plain attributes that the main thread polls
    with after(). Breaking this gives
    "RuntimeError: main thread is not in main loop".
  * Interactive previews are computed on a half-size copy of the texture, with
    size-dependent parameters rescaled by matgen.at_scale; saving, batch and
    the 3D view always use full-size maps.
  * Every visible string goes through i18n.t('key').
"""

import sys
import subprocess

REQUIRED = (('numpy', 'numpy'), ('scipy', 'scipy'),
            ('skimage', 'scikit-image'), ('PIL', 'pillow'))
OPTIONAL = (('tkinterdnd2', 'tkinterdnd2'),)
# кодировщик DXT получше Pillow (меньше полос на плавных переходах); готовые
# сборки есть для Python 3.9…3.13 — на более новом его не ставим, а то pip
# при каждом запуске пытался бы собирать его из исходников
if sys.version_info < (3, 14):
    OPTIONAL += (('quicktex', 'quicktex'),)


def _missing(pairs):
    """Return the pip package names whose modules cannot be imported.

    pairs: iterable of (import_name, pip_package_name), e.g. REQUIRED.
    Returns a list of pip package names (empty if everything imports).
    """
    out = []
    for mod, pkg in pairs:
        try:
            __import__(mod)
        except ImportError:
            out.append(pkg)
    return out


def _pip(pkgs, line=None):
    """Install packages with the running interpreter's pip.

    Tries a plain install, then --user, then --break-system-packages (needed on
    distributions with an externally managed Python), stopping at the first
    success. line: optional callable(text), called with every line pip prints
    (from the calling thread) so a window can show what is going on. Returns
    None on success, otherwise the last 1500 characters of pip's output as the
    error text.
    """
    cmd = [sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
           '--progress-bar', 'off']
    out = 'pip did not start'
    flags = 0x08000000 if sys.platform == 'win32' else 0      # без окна консоли
    for extra in ([], ['--user'], ['--break-system-packages']):
        try:
            p = subprocess.Popen(cmd + extra + pkgs, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, errors='replace',
                                 creationflags=flags)
        except Exception as e:                      # noqa: BLE001
            out = str(e)
            continue
        buf = []
        for ln in p.stdout:
            buf.append(ln)
            if line:
                line(ln.strip())
        p.wait()
        if p.returncode == 0:
            return None
        out = ''.join(buf)
    return out[-1500:]


def _ensure_packages():
    """Make sure the required libraries are installed before anything imports them.

    Runs at import time, before numpy/scipy/PIL are imported. Missing optional
    packages (tkinterdnd2) are installed silently. If required ones are missing,
    asks the user in a bare Tk dialog, installs them and restarts the process
    with os.execv so the fresh packages are importable; exits if the user
    declines or pip fails.

    pip runs in a separate thread and the window shows its last line and the
    elapsed time: with pip in the window's own thread the window stopped
    answering for the minutes the download takes, and Windows marked it
    "Not responding", as if the program had hung.
    """
    need, extra = _missing(REQUIRED), _missing(OPTIONAL)
    if not need:
        if extra:
            _pip(['--only-binary=:all:'] + extra)
        return
    import os
    import threading
    import time
    import tkinter as tk
    from tkinter import messagebox
    root = tk.Tk()
    root.withdraw()
    if not messagebox.askyesno('Texelith — first run',
                               'Missing libraries:\n\n    %s\n\n'
                               'Install them now? A few minutes, about 100 MB.\n\n'
                               'Python: %s' % (',  '.join(need), sys.version.split()[0])):
        sys.exit()
    win = tk.Toplevel(root)
    win.title('Texelith — installing')
    win.protocol('WM_DELETE_WINDOW', lambda: None)       # закрытие посреди установки
    tk.Label(win, text='Installing: %s' % ', '.join(need), padx=30,
             pady=(14)).pack(anchor='w')
    last = tk.Label(win, text='starting pip…', padx=30, width=70, anchor='w',
                    foreground='#666')
    last.pack(anchor='w')
    clock = tk.Label(win, text='', padx=30, pady=10, anchor='w')
    clock.pack(anchor='w')
    state = {'line': '', 'err': None, 'done': False}

    def work():
        """Worker thread: run pip; only plain values are written here."""
        state['err'] = _pip(need, lambda s: state.__setitem__('line', s) if s else None)
        if state['err'] is None and extra:
            _pip(['--only-binary=:all:'] + extra)
        state['done'] = True
    t0 = time.time()
    threading.Thread(target=work, daemon=True).start()

    def poll():
        """Main thread: show pip's last line and the time; finish when done."""
        if not state['done']:
            last.config(text=state['line'][:110] or 'starting pip…')
            clock.config(text='%d s' % (time.time() - t0))
            root.after(300, poll)
            return
        root.quit()
    poll()
    root.mainloop()
    if state['err']:
        messagebox.showerror('Installation failed',
                             'python -m pip install %s\n\n%s' % (' '.join(need), state['err']))
        sys.exit()
    root.destroy()
    os.execv(sys.executable, [sys.executable] + sys.argv)


_ensure_packages()


import math
import os
import threading
import webbrowser
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

import foliage as FO
import matgen as MG
import normalfix as NX
import repair as RP
import view3d as V3
import depthnet as DN
import nodes as ND
import esrgan as ES
import deepbump as DB
import supermat_bridge as SB
import project as PR
import editor as EDM
import texio as TXIO
import xpu as XPU
from i18n import t, Lang, LANGS, name_of
import i18n

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    HAVE_DND = True
except Exception:
    HAVE_DND = False

APP = 'Texelith'
BUILD = 25          # все файлы поставки должны быть одной версии

BG, PANEL, EDGE = '#232629', '#2b3035', '#3a4046'
FG, DIM, ACC = '#d8dbde', '#8a9298', '#6ea8dc'


# ============================================================== виджеты =====
class ZoomView(tk.Canvas):
    """Image preview canvas with wheel zoom and mouse-drag panning.

    Also hosts the brush for mask painting: when paint_on is set, left button
    paints and right button erases, and the coordinates are passed to paint_cb
    in full-size image pixels (see _to_img).
    """

    def __init__(self, parent):
        """Create the canvas and bind mouse, wheel and resize events.

        Wheel is bound both as <MouseWheel> (Windows/macOS) and as
        <Button-4>/<Button-5> (X11). paint_cb, when set, is called as
        paint_cb(x, y, erase, final) in image coordinates.
        """
        super().__init__(parent, background='#16181a', highlightthickness=0)
        self.img = None
        self.scale = 1.0
        self.src_w = self.src_h = 1
        self.ox = self.oy = 0.0
        self._drag = None
        self._fit_pending = True
        self.paint_cb = None            # (x, y, erase, final) в координатах картинки
        self.paint_on = False
        self.overlay_cb = None          # рисует поверх картинки после render()
        self.bind('<Configure>', self._configured)
        self.bind('<ButtonPress-1>', self._down)
        self.bind('<B1-Motion>', self._move)
        self.bind('<ButtonRelease-1>', self._up)
        self.bind('<ButtonPress-3>', lambda e: self._paint(e, True))
        self.bind('<B3-Motion>', lambda e: self._paint(e, True))
        self.bind('<ButtonRelease-3>', lambda e: self._paint(e, True, True))
        self.bind('<Double-Button-1>', lambda e: self.fit())
        # средняя кнопка — двигать вид и тогда, когда левая занята кистью
        self.bind('<ButtonPress-2>', lambda e: setattr(self, '_drag', (e.x, e.y, self.ox, self.oy)))
        self.bind('<B2-Motion>', self._pan)
        self.bind('<ButtonRelease-2>', lambda e: setattr(self, '_drag', None))
        self.bind('<MouseWheel>', self._wheel)                 # Windows / macOS
        self.bind('<Button-4>', lambda e: self._zoom(1.25, e.x, e.y))   # X11
        self.bind('<Button-5>', lambda e: self._zoom(0.8, e.x, e.y))

    def _configured(self, _e=None):
        """<Configure> handler: fit the first image once the canvas has a real
        size (a canvas not yet mapped reports width 1), otherwise re-render."""
        if self._fit_pending and self.winfo_width() > 10 and self.img is not None:
            self._fit_pending = False
            self.fit()
        else:
            self.render()

    def set_image(self, arr, keep_view=True, ref=None):
        """Show an image, optionally keeping the current zoom and pan.

        arr: float array (H, W) or (H, W, 3), values 0…1 (clipped here).
        keep_view: keep zoom/pan; if False (or no view yet) the image is fitted.
        ref: (height, width) of the full-size version of this same picture.
            The zoom scale is measured relative to it, so switching between the
            full and the reduced (half-size) preview does not make the zoom jump.
        """
        a = np.clip(arr, 0, 1)
        if a.ndim == 2:
            a = np.stack([a] * 3, -1)
        self.img = Image.fromarray((a * 255 + 0.5).astype(np.uint8))
        self.src_h, self.src_w = ref if ref else self.img.size[::-1]
        if not keep_view or self.scale <= 0 or self._fit_pending:
            if self.winfo_width() > 10:
                self._fit_pending = False
                self.fit()
            else:
                self._fit_pending = True
        else:
            self.render()

    @property
    def bscale(self):
        """Screen pixels per pixel of the shown image itself (not of the ref size)."""
        return self.scale * max(self.src_w, 1) / self.img.width

    def fit(self):
        """Zoom and centre the image to fit the canvas (also on double click)."""
        if self.img is None:
            return
        w, h = max(self.winfo_width(), 1), max(self.winfo_height(), 1)
        self.scale = min(w / max(self.src_w, 1), h / max(self.src_h, 1))
        self.ox = (w - self.src_w * self.scale) / 2
        self.oy = (h - self.src_h * self.scale) / 2
        self.render()

    def one_to_one(self):
        """Set 100 % zoom (one full-size pixel per screen pixel), keeping the
        point at the canvas centre in place."""
        if self.img is None:
            return
        w, h = self.winfo_width(), self.winfo_height()
        cx = (w / 2 - self.ox) / max(self.scale, 1e-6)
        cy = (h / 2 - self.oy) / max(self.scale, 1e-6)
        self.scale = 1.0
        self.ox, self.oy = w / 2 - cx, h / 2 - cy
        self.render()

    def _to_img(self, e):
        """Convert an event position to image pixel coordinates (x, y), ints.

        The coordinates are in the reference size (ref) passed to set_image,
        not in the size of the shown copy: the lit view may be shown
        decimated, but the brush must land where it would on the image the
        tab computes on. For the material tab that is the (possibly half-size)
        working copy; _mfull then scales to the full size.
        """
        s = max(self.scale, 1e-6)
        return int((e.x - self.ox) / s), int((e.y - self.oy) / s)

    def _paint(self, e, erase=False, final=False):
        """Forward a brush event to paint_cb in image coordinates.

        erase: True for the right mouse button. final: True on button release,
        so the owner can commit the stroke and recompute.
        """
        if self.paint_cb is None or self.img is None:
            return
        x, y = self._to_img(e)
        self.paint_cb(x, y, erase, final)

    def _down(self, e):
        """<ButtonPress-1>: start a brush stroke in paint mode, else start panning."""
        if self.paint_on:
            self._paint(e)
            return
        self._drag = (e.x, e.y, self.ox, self.oy)

    def _up(self, e):
        """<ButtonRelease-1>: finish the brush stroke or the pan."""
        if self.paint_on:
            self._paint(e, final=True)
        self._drag = None

    def _move(self, e):
        """<B1-Motion>: continue the brush stroke, or pan the view."""
        if self.paint_on:
            self._paint(e)
            return
        if self._drag:
            x0, y0, ox, oy = self._drag
            self.ox, self.oy = ox + e.x - x0, oy + e.y - y0
            self.render()

    def _pan(self, e):
        """<B2-Motion>: pan the view with the middle button."""
        if self._drag:
            x0, y0, ox, oy = self._drag
            self.ox, self.oy = ox + e.x - x0, oy + e.y - y0
            self.render()

    def _wheel(self, e):
        """<MouseWheel> (Windows/macOS): zoom in or out by 1.25× around the cursor."""
        self._zoom(1.25 if e.delta > 0 else 0.8, e.x, e.y)

    def _zoom(self, k, mx, my):
        """Multiply the zoom by k around canvas point (mx, my), in pixels.

        The resulting scale is clamped to 0.02…32 (2 % … 3200 %).
        """
        if self.img is None:
            return
        ns = min(max(self.scale * k, 0.02), 32.0)
        k = ns / self.scale
        self.ox, self.oy = mx - (mx - self.ox) * k, my - (my - self.oy) * k
        self.scale = ns
        self.render()

    def render(self):
        """Redraw the visible part of the image and the zoom hint.

        Only the visible crop of the source is resized and converted to a
        PhotoImage; resizing the whole picture makes a zoomed-in 2048 texture
        sluggish. Nearest-neighbour is used above 1.5× so pixels stay crisp,
        Lanczos below.
        """
        self.delete('all')
        if self.img is None:
            return
        cw, ch = max(self.winfo_width(), 1), max(self.winfo_height(), 1)
        b = max(self.bscale, 1e-6)
        # берём только видимый кусок исходника — иначе 2048 в зуме тормозит
        sx0 = max(0, int(-self.ox / b))
        sy0 = max(0, int(-self.oy / b))
        sx1 = min(self.img.width, int((cw - self.ox) / b) + 2)
        sy1 = min(self.img.height, int((ch - self.oy) / b) + 2)
        if sx1 <= sx0 or sy1 <= sy0:
            return
        crop = self.img.crop((sx0, sy0, sx1, sy1))
        tw = max(1, int((sx1 - sx0) * b))
        th = max(1, int((sy1 - sy0) * b))
        rs = Image.NEAREST if b >= 1.5 else Image.LANCZOS
        self._photo = ImageTk.PhotoImage(crop.resize((tw, th), rs))
        self.create_image(self.ox + sx0 * b, self.oy + sy0 * b,
                          anchor='nw', image=self._photo)
        self.create_text(8, ch - 8, anchor='sw', fill=DIM, font=('TkDefaultFont', 8),
                         text=t('com.zoomhint', round(self.scale * 100)))
        if self.overlay_cb:
            self.overlay_cb(self)


class Stage3D(tk.Canvas):
    """3D view canvas. Dragging the mouse tilts the plane, the wheel zooms.

    The canvas only holds the camera state (yaw, pitch, dist) and shows frames;
    rendering is done by the owner through on_change.
    """

    def __init__(self, parent, on_change):
        """Create the canvas and bind mouse, wheel and resize events.

        on_change(dragging: bool) is called whenever the camera or the canvas
        size changes; dragging=True asks for a cheap draft frame. Initial
        camera: yaw 0.30 rad, pitch 1.00 rad, distance 2.5 plane sizes.
        """
        super().__init__(parent, background='#0e1012', highlightthickness=0)
        self.on_change = on_change
        self.yaw, self.pitch, self.dist = 0.30, 1.00, 2.5
        self.dragging = False
        self._img = None
        self._drag = None
        self.bind('<ButtonPress-1>', self._down)
        self.bind('<B1-Motion>', self._move)
        self.bind('<ButtonRelease-1>', self._up)
        self.bind('<MouseWheel>', lambda e: self._zoom(0.88 if e.delta > 0 else 1.14))
        self.bind('<Button-4>', lambda e: self._zoom(0.88))
        self.bind('<Button-5>', lambda e: self._zoom(1.14))
        self.bind('<Configure>', lambda e: self.on_change(False))

    def _down(self, e):
        """<ButtonPress-1>: remember the start point and camera for dragging."""
        self._drag = (e.x, e.y, self.yaw, self.pitch)
        self.dragging = True

    def _move(self, e):
        """<B1-Motion>: rotate the camera, 0.006 rad per pixel of mouse travel.

        Yaw is clamped to ±1.2 rad, pitch to 0…1.45 rad (never below the plane).
        Requests a draft frame (on_change(True)).
        """
        if not self._drag:
            return
        x0, y0, y, p = self._drag
        self.yaw = max(-1.2, min(1.2, y + (e.x - x0) * 0.006))
        self.pitch = max(0.0, min(1.45, p + (e.y - y0) * 0.006))
        self.on_change(True)

    def _up(self, _e):
        """<ButtonRelease-1>: end the drag and request a full-quality frame."""
        self._drag = None
        self.dragging = False
        self.on_change(False)

    def _zoom(self, k):
        """Wheel: multiply the camera distance by k, clamped to 1.15…9.0."""
        self.dist = max(1.15, min(9.0, self.dist * k))
        self.on_change(False)

    def show(self, arr):
        """Show a rendered frame stretched over the whole canvas.

        arr: float array (h, w, 3), values 0…1. The frame is computed at its own
        resolution and stretched to fill the window: "frame size" is a quality
        slider, not the size of the picture.
        """
        self.delete('all')
        cw, ch = max(self.winfo_width(), 1), max(self.winfo_height(), 1)
        im = Image.fromarray((np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8))
        if im.size != (cw, ch):
            im = im.resize((cw, ch), Image.LANCZOS if im.width < cw else Image.BILINEAR)
        self._img = ImageTk.PhotoImage(im)
        self.create_image(cw // 2, ch // 2, image=self._img)
        self.create_text(8, ch - 8, anchor='sw', fill=DIM,
                         font=('TkDefaultFont', 8), text=t('v3.hint'))

    def note(self, text):
        """Clear the canvas and show a centred text message instead of a frame."""
        self.delete('all')
        self._img = None
        cw, ch = max(self.winfo_width(), 1), max(self.winfo_height(), 1)
        self.create_text(cw // 2, ch // 2, fill=DIM, justify='center', width=420,
                         font=('TkDefaultFont', 10), text=text)


class LightRing(tk.Canvas):
    """Round light-direction picker: angle around the ring is the azimuth,
    distance from the centre is the elevation (centre = 90°, rim = 0°).

    Class-level flags are shared by all rings: BUSY drops mouse events that
    arrive while a recompute is still running (the mouse is faster than the
    math), DRAG tells previews that the light is being dragged or spun, so a
    coarser preview is acceptable.
    """
    SIZE = 150
    FRONT = 0.62        # с зоной «сзади»: доля радиуса под свет спереди
    BUSY = False
    DRAG = False        # мышь на кольце или свет крутится — превью можно грубее

    def __init__(self, parent, on_change, back=False):
        """Create a SIZE×SIZE ring; on_change() is called when the light moves.

        Initial light: azimuth 125°, elevation 35°. back: the light may also
        go behind the texture (to see translucency): then the inner disc is
        the front (centre 90°, its rim 0°) and the outer band behind it, down
        to −90° (straight behind) at the outer edge; then spinning can also go
        round the card's x or y axis (``axis``), over it and behind it.
        """
        super().__init__(parent, width=self.SIZE, height=self.SIZE,
                         background=BG, highlightthickness=0)
        self.on_change = on_change
        self.back = bool(back)
        self.axis = 'z'      # вокруг чего крутится свет: z — нормаль, x / y — оси карточки
        self.az, self.el = 125.0, 35.0
        self.spin = False
        self._job = None
        self.bind('<Button-1>', self._grab)
        self.bind('<B1-Motion>', self._grab)
        self.bind('<ButtonRelease-1>', self._done)
        self.draw()

    def _done(self, _e=None):
        """<ButtonRelease-1>: clear DRAG and fire once more for a full-quality
        preview."""
        LightRing.DRAG = False
        self._fire()                # отпустили — перерисовать честно

    def _grab(self, e):
        """<Button-1>/<B1-Motion>: set the light from the mouse position.

        Azimuth in degrees 0…360 (screen angle, y down); elevation in degrees,
        90 at the centre falling linearly to 0 at the ring, clamped to 5…85.
        """
        LightRing.DRAG = True
        c, r = self.SIZE / 2, self.SIZE / 2 - 12
        dx, dy = e.x - c, e.y - c
        self.az = math.degrees(math.atan2(dy, dx)) % 360
        d = math.hypot(dx, dy)
        if self.back:
            ri = r * self.FRONT
            if d <= ri:
                self.el = max(0.0, min(90.0, 90.0 * (1.0 - d / ri)))
            else:
                # внешнее кольцо — свет за карточкой, до −90° (прямо сзади) у края
                self.el = -min(90.0, 90.0 * (d - ri) / (r - ri))
        else:
            self.el = max(5.0, min(85.0, 90.0 * (1.0 - min(d, r) / r)))
        self.draw()
        self._fire()

    def _fire(self):
        """Call on_change unless a previous call is still running (then the
        event is simply dropped: the mouse is faster than the recompute)."""
        if LightRing.BUSY:          # мышь быстрее расчёта — лишние события выбрасываем
            return
        LightRing.BUSY = True
        try:
            self.on_change()
        finally:
            LightRing.BUSY = False

    def draw(self):
        """Redraw the ring, guide circles, the light handle and the angle label."""
        self.delete('all')
        c, r = self.SIZE / 2, self.SIZE / 2 - 12
        if self.back:
            ri = r * self.FRONT
            # внешнее кольцо — свет за карточкой
            self.create_oval(c - r, c - r, c + r, c + r, outline=EDGE, fill='#251f22')
            self.create_oval(c - ri, c - ri, c + ri, c + ri, outline='#6b5a48', fill=BG)
            for k in (0.33, 0.66):
                self.create_oval(c - ri * k, c - ri * k, c + ri * k, c + ri * k,
                                 outline='#31363b')
            self.create_text(c, c - (r + ri) / 2, fill='#7d6a5a', font=('TkDefaultFont', 7),
                             text=t('com.light.back'))
        else:
            self.create_oval(c - r, c - r, c + r, c + r, outline=EDGE)
            for k in (0.33, 0.66):
                self.create_oval(c - r * k, c - r * k, c + r * k, c + r * k, outline='#31363b')
        self.create_line(c - r, c, c + r, c, fill='#31363b')
        self.create_line(c, c - r, c, c + r, fill='#31363b')
        if self.back:
            ri = r * self.FRONT
            d = (ri * (1 - self.el / 90) if self.el >= 0
                 else ri + (r - ri) * min(1.0, -self.el / 90.0))
        else:
            d = r * (1 - self.el / 90)
        x = c + d * math.cos(math.radians(self.az))
        y = c + d * math.sin(math.radians(self.az))
        self.create_line(c, c, x, y, fill=EDGE)
        self.create_oval(x - 6, y - 6, x + 6, y + 6, fill=ACC, outline='#eaf2fb')
        self.create_text(c, self.SIZE - 3, anchor='s', fill=DIM,
                         font=('TkDefaultFont', 8),
                         text=t('com.light', self.az, self.el))

    def vector(self):
        """Return the unit light direction as float32 array (3,): (x, y, z),
        z pointing out of the texture."""
        a, e = math.radians(self.az), math.radians(self.el)
        return np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a),
                         math.sin(e)], np.float32)

    def toggle_spin(self, on):
        """Start or stop spinning the light around the ring (checkbox)."""
        self.spin = on
        if on and self._job is None:
            self._tick()
        elif not on and self._job is not None:
            self.after_cancel(self._job)
            self._job = None

    def _tick(self):
        """One spin step: +4° azimuth every 60 ms via after(), with DRAG set so
        previews stay coarse while spinning."""
        if not self.spin:
            self._job = None
            return
        if self.axis == 'z' or not self.back:
            self.az = (self.az + 4) % 360
        else:
            # поворот вектора света на 4° вокруг оси карточки x или y: свет
            # проходит над ней, уходит за неё и возвращается снизу (сбоку)
            x, y, z = (float(q) for q in self.vector())
            c, s_ = math.cos(math.radians(4)), math.sin(math.radians(4))
            if self.axis == 'x':
                y, z = y * c - z * s_, y * s_ + z * c
            else:
                x, z = x * c - z * s_, x * s_ + z * c
            self.el = math.degrees(math.asin(max(-1.0, min(1.0, z))))
            if math.hypot(x, y) > 1e-4:
                self.az = math.degrees(math.atan2(y, x)) % 360
        LightRing.DRAG = True
        self.draw()
        self._fire()
        self._job = self.after(60, self._tick)


class Info(ttk.Label):
    """An "ⓘ" icon with a tooltip. The text pops up on hover instead of taking
    up half of the panel."""

    def __init__(self, parent, text):
        """Create the icon; text is the tooltip (empty text shows no tooltip)."""
        super().__init__(parent, text='ⓘ', style='Info.TLabel', cursor='question_arrow')
        self.msg = text
        self.tip = None
        self.bind('<Enter>', self._show)
        self.bind('<Leave>', self._hide)

    def _show(self, _e=None):
        """<Enter>: open a borderless tooltip window 18 px below-right of the icon."""
        if self.tip or not self.msg:
            return
        self.tip = tk.Toplevel(self)
        self.tip.wm_overrideredirect(True)
        self.tip.configure(background=EDGE)
        tk.Label(self.tip, text=self.msg, justify='left', wraplength=380,
                 background=PANEL, foreground=FG, padx=10, pady=7,
                 font=('TkDefaultFont', 9)).pack(padx=1, pady=1)
        # не за край экрана: у правого края — левее значка, у нижнего — выше
        self.tip.update_idletasks()
        w, h = self.tip.winfo_reqwidth(), self.tip.winfo_reqheight()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        x, y = self.winfo_rootx() + 18, self.winfo_rooty() + 18
        if x + w > sw - 4:
            x = max(4, self.winfo_rootx() - w - 4)
        if y + h > sh - 4:
            y = max(4, self.winfo_rooty() - h - 4)
        self.tip.wm_geometry('+%d+%d' % (x, y))

    def _hide(self, _e=None):
        """<Leave>: close the tooltip window."""
        if self.tip:
            self.tip.destroy()
            self.tip = None


class Slider:
    """A labelled slider with a field for typing the value and an optional
    tooltip icon.

    Not a widget itself: the widgets live in self.box, which carries _adv (for
    Detail) and _slider (so Section.mark can find it and compare with init).
    """
    def __init__(self, parent, title, lo, hi, init, hint, on_change,
                 integer=False, fmt='%.2f', adv=False):
        """Create and pack the slider into parent.

        title: visible label. lo, hi: slider range, in the units of the setting.
        init: initial value; kept to mark sections whose sliders were touched.
        hint: tooltip text or empty. on_change(): called on every move (usually
        App.schedule, i.e. a delayed recompute). integer: round the value and
        show it without decimals. fmt: printf format of the shown value.
        adv: "advanced" setting, hidden in the simple mode by Detail.
        """
        box = ttk.Frame(parent)
        box.pack(fill='x', pady=(5, 0))
        box._adv = bool(adv)          # прятать ли в простом режиме
        box._slider = self
        self.box = box
        self.init = init              # чтобы отмечать тронутые разделы
        self.integer, self.fmt, self.title = integer, fmt, title
        self.var = tk.DoubleVar(value=init)
        self.lo, self.hi, self.on_change = lo, hi, on_change
        row = ttk.Frame(box)
        row.pack(fill='x')
        # значок подсказки ставится первым: длинное название его не вытолкнет
        if hint:
            Info(row, hint).pack(side='right')
        self.lab = ttk.Label(row, text=title)
        self.lab.pack(side='left')
        # поле справа: значение можно вписать с клавиатуры (Enter или уход из
        # поля — применить, Esc — вернуть прежнее)
        self.txt = tk.StringVar(value=self._num(init))
        srow = ttk.Frame(box)
        srow.pack(fill='x')
        self.ent = ttk.Entry(srow, textvariable=self.txt, width=7, justify='right')
        self.ent.pack(side='right', padx=(4, 2))
        for ev in ('<Return>', '<KP_Enter>', '<FocusOut>'):
            self.ent.bind(ev, self._typed)
        self.ent.bind('<Escape>', lambda e: self.txt.set(self._num(self.var.get())))

        def cb(_=None):
            """ttk.Scale command: refresh the shown value and notify the owner."""
            self.txt.set(self._num(self.var.get()))
            on_change()

        self._cb = cb
        self.scale_w = ttk.Scale(srow, from_=lo, to=hi, variable=self.var, command=cb)
        self.scale_w.pack(side='left', fill='x', expand=True)

    def _typed(self, e=None):
        """Value typed into the field: comma or point, kept within the
        slider's range; anything that is not a number restores the old value."""
        try:
            v = float(self.txt.get().strip().replace(',', '.'))
        except ValueError:
            self.txt.set(self._num(self.var.get()))
            return
        v = min(max(v, self.lo), self.hi)
        if self.integer:
            v = int(round(v))
        if abs(v - float(self.var.get())) > 1e-9:
            self.var.set(v)
            self._cb()
        else:
            self.txt.set(self._num(v))

    def enable(self, on=True):
        """Enable or grey out the slider.

        Grey it out rather than hide it: a slider that currently has no effect
        stays in place and is shown grey, so it is clear that it exists and why
        it does not work.
        """
        self.scale_w.state(['!disabled'] if on else ['disabled'])
        self.ent.state(['!disabled'] if on else ['disabled'])
        self.lab.configure(foreground=FG if on else DIM)

    def _num(self, v):
        """The value as shown in the field."""
        return str(int(round(v))) if self.integer else self.fmt % v

    def _text(self, v):
        """Return the text "Title: value" for value v."""
        return '%s: %s' % (self.title, self._num(v))

    def get(self):
        """Return the current value: int if integer=True, else float."""
        v = float(self.var.get())
        return int(round(v)) if self.integer else v

    def set(self, v):
        """Set the value and the field without calling on_change."""
        self.var.set(v)
        self.txt.set(self._num(v))


class Section(ttk.Frame):
    """Collapsible block of settings: a clickable header and a body below it."""

    def __init__(self, parent, title, opened=True):
        """Create and pack the section; widgets go into self.body.

        title: header text. opened: whether the body is shown initially.
        """
        super().__init__(parent)
        self.pack(fill='x', pady=(8, 0))
        self.title = title
        self.opened = bool(opened)
        row = ttk.Frame(self)
        row.pack(fill='x')
        self.head = ttk.Label(row, style='Head.TLabel', cursor='hand2')
        self.head.pack(side='left')
        self.head.bind('<Button-1>', self.toggle)
        self._hrow = row
        self.body = ttk.Frame(self)
        if self.opened:
            self.body.pack(fill='x')
        self._label()

    def add_info(self, text):
        """Add a tooltip icon to the header row; returns self for chaining."""
        Info(self._hrow, text).pack(side='right')
        return self

    def _label(self):
        """Refresh the header: ▾/▸ arrow, title and the "•" touched mark."""
        mark = '  •' if getattr(self, 'touched', False) and not self.opened else ''
        self.head.config(text=('▾  ' if self.opened else '▸  ') + self.title + mark)

    def mark(self):
        """Put a dot on a collapsed header if anything inside differs from its
        initial value. Otherwise a collapsed panel does not show where you have
        already done some work.

        A slider counts as changed when it differs from Slider.init by more
        than 1e-6.
        """

        def any_changed(w):
            """Return True if any Slider under widget w (recursively) is off
            its initial value."""
            for c in w.winfo_children():
                sl = getattr(c, '_slider', None)
                if sl is not None and abs(float(sl.get()) - float(sl.init)) > 1e-6:
                    return True
                if any_changed(c):
                    return True
            return False

        was = getattr(self, 'touched', False)
        self.touched = any_changed(self.body)
        if self.touched != was:
            self._label()

    def toggle(self, _=None):
        """<Button-1> on the header: collapse or expand the body."""
        self.opened = not self.opened
        if self.opened:
            self.body.pack(fill='x')
        else:
            self.body.forget()
        self._label()


def sec(parent, key, opened=True, info=True):
    """Create a Section titled t(key), with the tooltip t(key + '.info') if the
    string table has it, and return its body frame.

    opened: initial state; info=False suppresses the tooltip.
    """
    s = Section(parent, t(key), opened)
    if info and (key + '.info') in i18n.S:
        s.add_info(t(key + '.info'))
    return s.body


def check(parent, key, var, cmd=None, info=True, adv=False):
    """Checkbox with a tooltip on the right.

    key: string key for the label (tooltip from key + '.info' if present).
    var: tk variable bound to the checkbox. cmd: called on toggle.
    adv: mark the row as an advanced setting for Detail. Returns the row frame.
    """
    row = ttk.Frame(parent)
    row.pack(fill='x')
    row._adv = bool(adv)
    ttk.Checkbutton(row, text=t(key), variable=var,
                    command=cmd or (lambda: None)).pack(side='left')
    if info and (key + '.info') in i18n.S:
        Info(row, t(key + '.info')).pack(side='right')
    return row


class Detail:
    """The "essentials / all settings" switch for one settings column.

    Widgets marked adv=True are removed in the simple mode, and a column of
    sixty sliders turns into about fifteen. They cannot be hidden one by one:
    when shown again they would be packed at the end. So for every frame that
    holds advanced settings the full order of its children is remembered, and
    on switching the frame is re-packed as a whole.

    By default ALL settings are shown (the "all" checkbox in panel_head starts
    checked). Hiding them by default was tried, and 85 sliders disappeared
    without any trace for the user. That is also why every section with hidden
    widgets gets a clickable "…and N more settings are hidden here" line (see
    _notes): the simple mode must never look like sliders went missing.

    Must be built after the panel is fully populated and packed, because it
    records pack_info() of the existing children.
    """

    def __init__(self, panel, on_click=None):
        """Scan panel for advanced widgets and create the "N more" notes.

        panel: the settings column (a frame). on_click(): called when a
        "N more" note is clicked; App passes a function that switches the
        column back to "all settings".
        """
        self.groups = []
        self.notes = []
        self._scan(panel)
        self._notes(panel, on_click)

    def _notes(self, w, on_click):
        """Create an "N more settings" line in every section where something
        is hidden. Without it the simple mode looks like missing sliders.

        Walks widget w recursively; the labels are created but not packed
        (apply packs them only in the simple mode). on_click is bound to them.
        """
        for c in w.winfo_children():
            if isinstance(c, Section):
                n = self._count(c.body)
                if n:
                    lab = ttk.Label(c.body, style='Hint.TLabel', cursor='hand2',
                                    text=t('ui.more', n))
                    if on_click is not None:
                        lab.bind('<Button-1>', lambda e: on_click())
                    self.notes.append((lab, n))
            self._notes(c, on_click)

    def _count(self, w):
        """Return the number of widgets marked _adv under w, recursively."""
        n = 0
        for c in w.winfo_children():
            if getattr(c, '_adv', False):
                n += 1
            n += self._count(c)
        return n

    def _scan(self, w):
        """Record, for every frame under w that has at least one advanced child,
        the full ordered list of its packed children as (widget, pack_info,
        is_adv). Unpacked children are skipped."""
        rec = []
        for c in w.winfo_children():
            try:
                info = c.pack_info()
            except Exception:
                info = None
            if info:
                rec.append((c, info, bool(getattr(c, '_adv', False))))
            self._scan(c)
        if any(a for _, _, a in rec):
            self.groups.append(rec)

    def apply(self, show_all):
        """Show all settings (show_all=True) or only the essentials.

        Every recorded frame is unpacked and re-packed in its original order,
        skipping advanced widgets in the simple mode; the "N more" notes are
        shown only in the simple mode.
        """
        for rec in self.groups:
            for c, _, _ in rec:
                c.pack_forget()
            for c, info, adv in rec:
                if show_all or not adv:
                    try:
                        c.pack(**info)
                    except Exception:
                        pass
        for lab, n in self.notes:
            if show_all:
                lab.pack_forget()
            else:
                lab.pack(anchor='w', pady=(4, 0))


def stage(parent, text):
    """Add a stage heading (separator + small bold caption) and return its row.

    It does not collapse; it just splits the column into parts so the order of
    work reads from top to bottom. It is not an advanced widget, so Detail
    always keeps it visible, together with the settings that are shown by
    default (all of them, see Detail).
    """
    row = ttk.Frame(parent)
    row.pack(fill='x', pady=(12, 2))
    ttk.Separator(row).pack(fill='x')
    ttk.Label(row, text=text, style='Stage.TLabel').pack(anchor='w', pady=(4, 0))
    return row


class ScrollPanel(ttk.Frame):
    """Scrollable settings column; otherwise not everything fits when the window
    is not full screen."""

    def __init__(self, parent, width=306):
        """Create the column; put widgets into self.inner.

        width: column width in pixels. The mouse wheel scrolls it when the
        pointer is over the canvas or the inner frame itself.
        """
        super().__init__(parent)
        self.canvas = tk.Canvas(self, background=BG, highlightthickness=0, width=width)
        sb = ttk.Scrollbar(self, orient='vertical', command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self.inner.bind('<Configure>',
                        lambda e: self.canvas.configure(scrollregion=self.canvas.bbox('all')))
        self.canvas.create_window((0, 0), window=self.inner, anchor='nw', width=width - 4)
        self.canvas.configure(yscrollcommand=sb.set)
        self.canvas.pack(side='left', fill='both', expand=True)
        sb.pack(side='right', fill='y')
        for w in (self.canvas, self.inner):
            w.bind('<MouseWheel>', lambda e: self.canvas.yview_scroll(-e.delta // 120, 'units'))
            w.bind('<Button-4>', lambda e: self.canvas.yview_scroll(-1, 'units'))
            w.bind('<Button-5>', lambda e: self.canvas.yview_scroll(1, 'units'))


def dark_theme(root):
    """Configure the dark ttk theme ('clam' based) and all custom styles
    (Hint, Head, Stage, Info labels, Small buttons, notebook tabs, combobox)."""
    root.configure(background=BG)
    st = ttk.Style(root)
    st.theme_use('clam')
    st.configure('.', background=BG, foreground=FG, fieldbackground=PANEL,
                 bordercolor=EDGE, lightcolor=PANEL, darkcolor=PANEL,
                 focuscolor=BG, troughcolor='#191c1f', arrowcolor=FG)
    st.configure('TFrame', background=BG)
    st.configure('TLabel', background=BG, foreground=FG)
    st.configure('Hint.TLabel', background=BG, foreground=DIM, font=('TkDefaultFont', 8))
    st.configure('Head.TLabel', background=BG, foreground=ACC)
    st.configure('Stage.TLabel', background=BG, foreground=DIM,
                 font=('TkDefaultFont', 8, 'bold'))
    st.configure('Info.TLabel', background=BG, foreground=DIM)
    st.configure('TButton', background=PANEL, foreground=FG, borderwidth=1, padding=5)
    st.configure('Small.TButton', background=PANEL, foreground=FG, borderwidth=1,
                 padding=2, font=('TkDefaultFont', 8))
    st.map('Small.TButton', background=[('active', EDGE)])
    st.map('TButton', background=[('active', EDGE)])
    for w in ('TCheckbutton', 'TRadiobutton'):
        st.configure(w, background=BG, foreground=FG, indicatorcolor=PANEL)
        st.map(w, background=[('active', BG)], indicatorcolor=[('selected', ACC)])
    st.configure('Horizontal.TScale', background=ACC, troughcolor='#191c1f',
                 bordercolor=EDGE, lightcolor=ACC, darkcolor=ACC, gripcount=0)
    st.configure('TNotebook', background=BG, bordercolor=EDGE, tabmargins=(2, 4, 2, 0))
    st.configure('TNotebook.Tab', background=BG, foreground=DIM, padding=(16, 7), borderwidth=0)
    st.map('TNotebook.Tab', background=[('selected', PANEL)], foreground=[('selected', FG)])
    st.configure('TSeparator', background=EDGE)
    st.configure('TCombobox', fieldbackground=PANEL, background=PANEL, foreground=FG,
                 arrowcolor=FG, selectbackground=PANEL, selectforeground=FG)
    st.map('TCombobox', fieldbackground=[('readonly', PANEL)],
           foreground=[('readonly', FG)], selectbackground=[('readonly', PANEL)],
           selectforeground=[('readonly', FG)])
    root.option_add('*TCombobox*Listbox.background', PANEL)
    root.option_add('*TCombobox*Listbox.foreground', FG)


# чтение и запись файлов вынесены в texio.py (им пользуется и пакетная
# обработка без окна)
from texio import (gray_rgba, rgb_rgba, PLAIN_EXT, save_rgba, save_gray16,  # noqa: E402
                   resize_to3, read_rgba, resize_signed, resize_to, MAP_SUFFIXES)


# ---------------------------------------------------------- пресеты листвы -
# mode, split, min_area, thin, blade, tilt, bend, fold, profile, detail,
# clump_mid, clump_big, clump_scale
FOL_PRESETS = {
    'f.p.leaves':   dict(mode='silh', split=0.35, area=60,  thin=3.5, blade=0.0,
                         tilt=0.45, bend=0.10, fold=0.90, prof=2.0, det=0.22,
                         clm=0.50, clb=0.25, cls=24),
    'f.p.needles':  dict(mode='silh', split=0.20, area=40,  thin=5.0, blade=0.6,
                         tilt=0.55, bend=0.25, fold=1.00, prof=1.6, det=0.25,
                         clm=0.45, clb=0.25, cls=20),
    'f.p.branches': dict(mode='silh', split=0.55, area=400, thin=2.5, blade=0.0,
                         tilt=0.20, bend=0.45, fold=1.05, prof=2.2, det=0.30,
                         clm=0.35, clb=0.20, cls=32),
    'f.p.moss':     dict(mode='lum',  split=0.30, area=40,  thin=4.0, blade=0.5,
                         tilt=0.50, bend=0.00, fold=0.85, prof=2.5, det=0.35,
                         clm=0.70, clb=0.35, cls=14),
    'f.p.grass':    dict(mode='silh', split=0.25, area=180, thin=6.0, blade=1.0,
                         tilt=0.55, bend=0.35, fold=1.00, prof=1.6, det=0.20,
                         clm=0.30, clb=0.15, cls=20),
    'f.p.dense':    dict(mode='dense', split=0.35, area=60, thin=3.5, blade=0.3,
                         tilt=0.45, bend=0.10, fold=0.95, prof=2.0, det=0.25,
                         clm=0.60, clb=0.30, cls=26),
}
FOL_PRESET_IDS = list(FOL_PRESETS)
CHANNELS = ('luma', 'R', 'G', 'B', 'A', 'sat', 'hue', 'height', 'ao')
EXT_CHANNELS = ('luma', 'R', 'G', 'B', 'A')
NDIR_BLUE = ('unit', 'ao')       # откуда синий у нормали по освещению
X_SRC = ('normal', 'height')     # чем считать открытую картинку на правке нормалей


# ==================================================================== окно ==
# ============================================================== значок ======
def app_icon(size=256):
    """Draw the program icon: a mosaic tile, lit on the left, its own normal map
    on the right.

    It is drawn with the same code as everything else: height, the normal from
    the height, lighting and occlusion from those. So there is no separate
    image file, and the icon cannot be lost when the program is copied.

    A texelith is exactly that: a single small stone of a mosaic.

    size: icon edge in pixels (drawn at 4× with a minimum of 64×4, then reduced
    with Lanczos). Returns a PIL RGBA image of size×size.
    """
    n = max(64, int(size)) * 4                 # рисуем крупно, потом уменьшаем
    y, x = np.mgrid[0:n, 0:n].astype(np.float32)
    u, v = (x + 0.5) / n * 2 - 1, (y + 0.5) / n * 2 - 1

    def rsq(half, r):
        """Signed distance to a rounded square: negative inside.

        half: half-size, r: corner radius, both in the −1…1 icon coordinates.
        """
        ax, ay = np.abs(u) - (half - r), np.abs(v) - (half - r)
        d = np.sqrt(np.maximum(ax, 0) ** 2 + np.maximum(ay, 0) ** 2)
        return d + np.minimum(np.maximum(ax, ay), 0) - r

    d = rsq(0.80, 0.16)
    edge = np.clip(-d * n / 4 / max(size * 0.02, 1e-3), 0, 1)
    h = np.clip(-d / 0.10, 0, 1)
    h = h * np.clip(np.minimum(np.abs(u), np.abs(v)) / 0.09, 0.0, 1.0) ** 0.6
    hh = MG.fast_blur(np.clip(h, 0, 1).astype(np.float32) ** 0.8,
                      n / 150.0, mode='nearest')
    nrm = MG.normal_from_height(hh * (n / 900.0), strength=1.0, directx=False)

    L = np.array([-0.45, -0.55, 0.70], np.float32)
    L /= float(np.sqrt((L * L).sum()))
    dot = np.clip((nrm * L).sum(-1), 0, 1)
    ao = MG.cavity_ao(hh, radius=n / 40.0, strength=1.0)
    lit = np.array([0.66, 0.70, 0.74], np.float32) * (0.22 + dot)[..., None]
    lit = lit * (0.45 + 0.55 * ao)[..., None]
    nmap = (nrm * 0.5 + 0.5) * (0.80 + 0.20 * ao)[..., None]

    face = np.where((u > 0.0)[..., None], nmap, lit)
    bg = np.array([0.137, 0.149, 0.161], np.float32)
    rgba = np.concatenate([np.clip(bg + (face - bg) * edge[..., None], 0, 1),
                           np.clip(edge * 1.2, 0, 1)[..., None]], -1)
    im = Image.fromarray((rgba * 255 + 0.5).astype(np.uint8))
    return im.resize((int(size), int(size)), Image.LANCZOS)


def set_icon(root):
    """Set the window icon, and also write Texelith.ico next to the program (if
    missing) so a shortcut can point to it. Errors here do not matter and are
    ignored: the program works without an icon."""
    try:
        root._icons = [ImageTk.PhotoImage(app_icon(s)) for s in (64, 32, 16)]
        root.iconphoto(True, *root._icons)
    except Exception:
        pass
    try:
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Texelith.ico')
        if not os.path.exists(p):
            app_icon(256).save(p, sizes=[(16, 16), (32, 32), (48, 48),
                                         (64, 64), (128, 128), (256, 256)])
    except Exception:
        pass


def taskbar_id():
    """Give the process its own Windows AppUserModelID.

    Windows groups taskbar windows by program, and by default Python lumps
    everything into one pile together with other people's scripts. An own ID
    unties that pile, and the icon then shows as a separate program. This must
    be done BEFORE the first window appears. Does nothing (silently) outside
    Windows.
    """
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            'Texelith.TextureMaps')
    except Exception:
        pass


class App:
    """The main window: all tabs, their state, recomputation and file I/O.

    One large class by design (see REVIEW.md). State is kept in attributes:
    the loaded texture (rgb, alpha, path), per-workspace jobs (src/norm for
    foliage, mat = matgen.MaterialJob for the material tab), the caches of
    computed maps, and the tk variables and Slider objects of every setting.
    Settings changes go through schedule() → recompute() with a delay; heavy
    work runs in worker threads that never touch Tk (see _net_poll).
    """
    def __init__(self, root):
        """Set up the window (theme, icon, 1440×920, F11/Escape full screen),
        initialise the state attributes, build the UI and register the
        drag-and-drop target if tkinterdnd2 is available."""
        self.root = root
        dark_theme(root)
        set_icon(root)
        root.geometry('1440x920')
        root.bind('<F11>', self.toggle_full)
        root.bind('<Escape>', lambda e: root.attributes('-fullscreen', False))
        self.full = False

        self.src = None          # foliage.Source (если есть альфа)
        self.norm = None         # foliage.Normals
        self.mat = None          # matgen.MaterialJob
        self.rgb = None          # исходный RGB
        self.normal = self.diffuse = None
        self.maps = self.fixed = self.alpha = None
        self.fol_ao = self.fol_sss = None
        self.branch = None
        self.pending = None
        self.g_pending = None
        self.fine_job = None
        self._coarse = False
        self._pk_key = None
        self.path = None
        # ручные пометки материала
        self.g_pos = self.g_neg = None      # области «поднять» и «опустить»
        self.guide = None
        self.z_hand = self.z_kill = None    # маска второй зоны, дописанная кистью
        self.dmg_hand = self.dmg_kill = None
        self.dmg = None
        self._rep_key = None
        self._rep_cache = {}
        # узловой редактор — главное окно; его состояние переживает пересборку
        # окна при смене языка
        self.ed = {}
        self.editor = None
        self.classic_on = False
        EDM.bind_ui(ZoomView=ZoomView, Stage3D=Stage3D, LightRing=LightRing, Slider=Slider,
                    Info=Info, ScrollPanel=ScrollPanel, sec=sec, BG=BG, PANEL=PANEL,
                    EDGE=EDGE, FG=FG, DIM=DIM, ACC=ACC, APP=APP)
        root.protocol('WM_DELETE_WINDOW', self.quit)

        self.build_ui()
        if HAVE_DND:
            root.drop_target_register(DND_FILES)
            root.dnd_bind('<<Drop>>', self.on_drop)

    # ------------------------------------------------------- сборка окна ---
    def build_ui(self):
        """Build the top bar and the notebook with all six tabs.

        Called at start-up and again on language change (set_lang destroys and
        rebuilds everything). Switching tabs schedules a recompute, because each
        tab shows its own maps. Ends with apply_detail_all so every column
        starts in the "all settings" mode.
        """
        r = self.root
        r.title('%s — %s' % (APP, t('app.sub')))
        # два рабочих места: узловой редактор (главное) и прежние вкладки
        self.ws = ttk.Frame(r)
        self.cl = ttk.Frame(r)
        self.editor = EDM.Editor(self.ws, self)
        EDM.install_menu(self)

        self.top = ttk.Frame(self.cl, padding=(10, 8, 10, 2))
        self.top.pack(fill='x')
        top = self.top
        ttk.Button(top, text=t('ed.back'), command=lambda: self.show_classic(False),
                   style='Small.TButton').pack(side='left', padx=(0, 10))
        ttk.Button(top, text=t('top.open'), command=self.open_file).pack(side='left')
        self.name = ttk.Label(top, style='Hint.TLabel',
                              text=t('top.drop') if HAVE_DND else t('top.nofile'))
        self.name.pack(side='left', padx=12)

        lang = ttk.Frame(top)
        lang.pack(side='right', padx=(10, 0))
        self.lang_var = tk.StringVar(value=name_of(Lang.cur))
        cbl = ttk.Combobox(lang, values=[name_of(c) for c in LANGS], width=9,
                           textvariable=self.lang_var, state='readonly')
        cbl.pack(side='left')
        cbl.bind('<<ComboboxSelected>>', lambda e: self.set_lang(cbl.get()))
        gb = ttk.Button(top, text=t('top.guide'), command=self.open_guide)
        gb.pack(side='right', padx=2)
        Info(top, t('top.guide.info')).pack(side='right', padx=(0, 4))
        ttk.Button(top, text='1:1', width=4,
                   command=lambda: self.view().one_to_one()).pack(side='right', padx=2)
        ttk.Button(top, text=t('top.fit'),
                   command=lambda: self.view().fit()).pack(side='right', padx=2)
        self.flip = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text=t('top.flip'), variable=self.flip,
                        command=self.redraw).pack(side='right', padx=8)

        nb = ttk.Notebook(self.cl, padding=6)
        nb.pack(fill='both', expand=True)
        self.tab_n, self.tab_d = ttk.Frame(nb), ttk.Frame(nb)
        self.tab_m, self.tab_x = ttk.Frame(nb), ttk.Frame(nb)
        self.tab_v = ttk.Frame(nb)
        self.tab_g = ttk.Frame(nb)
        nb.add(self.tab_n, text=t('tab.foliage'))
        nb.add(self.tab_d, text=t('tab.leafdiff'))
        nb.add(self.tab_m, text=t('tab.material'))
        nb.add(self.tab_v, text=t('tab.v3'))
        nb.add(self.tab_x, text=t('tab.fix'))
        # вкладка узлов заменена узловым окном; сама вкладка строится, но не
        # показывается (её функции используют проверки)
        nb.bind('<<NotebookTabChanged>>', lambda e: self.schedule())
        self.nb = nb

        # переменные блока нейросети общие для двух вкладок: модель выбирается
        # один раз, а кнопок и весов у каждой вкладки свои
        self.m_netmodel = tk.StringVar(value=t('nn.' + DN.MODEL_IDS[0]))
        self.m_netfine = tk.BooleanVar(value=False)

        self.build_foliage_tab()
        self.build_leafdiff_tab()
        self.build_material_tab()
        self.build_view3d_tab()
        self.build_fix_tab()
        self.build_nodes_tab()
        self.apply_detail_all()
        (self.cl if self.classic_on else self.ws).pack(fill='both', expand=True)

    def lang_widgets(self, parent):
        """Language list and guide button, packed at the right of a toolbar."""
        self._lang_ed = tk.StringVar(value=name_of(Lang.cur))
        cbl = ttk.Combobox(parent, values=[name_of(c) for c in LANGS], width=9,
                           textvariable=self._lang_ed, state='readonly')
        cbl.pack(side='right')
        cbl.bind('<<ComboboxSelected>>', lambda e: self.set_lang(cbl.get()))
        ttk.Button(parent, text=t('top.guide'), command=self.open_guide,
                   style='Small.TButton').pack(side='right', padx=4)

    def show_classic(self, on):
        """Switch between the node workspace and the classic tabs.

        The classic tabs work on their own loaded texture; when switching to
        them, the texture open in the node workspace is loaded there too.
        """
        self.classic_on = bool(on)
        if on:
            self.ws.pack_forget()
            self.cl.pack(fill='both', expand=True)
            tex = self.ed.get('tex')
            if tex and tex != self.path and os.path.isfile(tex):
                self.load(tex)
            else:
                self.schedule()
        else:
            self.cl.pack_forget()
            self.ws.pack(fill='both', expand=True)
            self.editor.update_title()

    def quit(self):
        """Close the window, offering to save an unsaved project first."""
        if self.editor is not None and not self.editor.ask_discard():
            return
        try:
            SB.stop()
        except Exception:
            pass
        self.root.destroy()

    def toggle_full(self, _=None):
        """<F11>: toggle full-screen mode."""
        self.full = not self.full
        self.root.attributes('-fullscreen', self.full)

    # ---------------------------------------------------------- язык ------
    VAR_TYPES = (tk.BooleanVar, tk.StringVar, tk.DoubleVar, tk.IntVar)

    def combo_vars(self):
        """Return {attribute name: (list of ids, string-key prefix)} for every
        combobox variable.

        Comboboxes store the TRANSLATED name. On language change it must be
        translated anew, not carried over as is; snapshot/restore use this
        table to go through the language-independent id.
        """
        return {'fpreset': (FOL_PRESET_IDS, ''),
                'preset': (MG.PRESET_IDS, 'p.'),
                'z_preset': (MG.PRESET_IDS, 'p.'),
                'm_extch': (EXT_CHANNELS, 'ch.'),
                'm_ndirb': (NDIR_BLUE, 'm.ndir.b.'),
                'x_src': (X_SRC, 'x.src.'),
                'm_plxref': (MG.PARALLAX_REFS, 'm.plx.r.'),
                'm_rdsrc': (MG.RELIEF_SOURCES, 'm.rd.s.'),
                'm_rdmode': (MG.RELIEF_MODES, 'm.rd.m.'),
                'm_netmodel': (DN.MODEL_IDS, 'nn.'),
                'z_chan': (CHANNELS, 'ch.')}

    def snapshot(self):
        """Return all setting values keyed by attribute name.

        Needed on language change: the interface is rebuilt from scratch, while
        the user's work must stay. Collects Slider objects, tk variables and
        dicts of tk variables; combobox values are stored as ('#id', id) so they
        can be re-translated by restore.
        """
        d = {}
        for k, v in list(self.__dict__.items()):
            if isinstance(v, Slider) or isinstance(v, self.VAR_TYPES):
                try:
                    d[k] = v.get()
                except Exception:
                    pass
            elif isinstance(v, dict) and v and all(
                    isinstance(x, self.VAR_TYPES) for x in v.values()):
                d[k] = {kk: xx.get() for kk, xx in v.items()}
        for name, (ids, pref) in self.combo_vars().items():
            if name in d:
                cur = d[name]
                d[name] = ('#id', next((i for i in ids if t(pref + i) == cur), ids[0]))
        return d

    def restore(self, d):
        """Put back values saved by snapshot into the (rebuilt) widgets.

        Combobox ids are translated into the current language; values that no
        longer fit are silently skipped.
        """
        combo = self.combo_vars()
        for k, val in d.items():
            if k in combo and isinstance(val, tuple) and val[:1] == ('#id',):
                ids, pref = combo[k]
                val = t(pref + val[1])
            v = self.__dict__.get(k)
            try:
                if isinstance(v, Slider) or isinstance(v, self.VAR_TYPES):
                    v.set(val)
                elif isinstance(v, dict) and isinstance(val, dict):
                    for kk, xx in v.items():
                        if kk in val:
                            xx.set(val[kk])
            except Exception:
                pass

    def set_lang(self, display_name):
        """Language combobox (<<ComboboxSelected>>): switch the interface language.

        Saves the choice, snapshots all settings, destroys and rebuilds the
        whole UI, restores the settings and the current tab, and, if a texture
        is loaded, refits the views and recomputes.
        """
        code = next((c for c in LANGS if name_of(c) == display_name), 'ru')
        if code == Lang.cur:
            return
        keep = self.snapshot()
        keep.pop('lang_var', None)
        keep.pop('_lang_ed', None)
        i18n.save(code)
        cur = self.nb.index('current')
        self.editor.close()
        self.ws.destroy()
        self.cl.destroy()
        self.build_ui()
        self.restore(keep)
        self.apply_detail_all()
        self.nb.select(cur)
        self.net_note()          # подпись под кнопкой зависит от выбранной модели
        if self.rgb is not None:
            self.set_name_line()
            for v in self.views():
                v.scale = 0
                v._fit_pending = True
            self.rebuild_job(recompute=False)
            self.recompute()

    def open_guide(self):
        """Guide button: open guide.html in the browser at the current
        language's anchor, or say that the file is missing."""
        here = os.path.dirname(os.path.abspath(__file__))
        p = os.path.join(here, 'guide.html')
        if not os.path.exists(p):
            messagebox.showinfo(APP, t('sys.noguide'))
            return
        webbrowser.open('file:///' + p.replace('\\', '/') + '#' + Lang.cur)

    # ------------------------------------------------------- вкладка листвы
    def build_foliage_tab(self):
        """Build the "Foliage: normals" tab: preset, leaf detection, shape,
        orientation, clumps, card, hand mask, output, and the view column with
        the light ring. Every slider schedules a recompute."""
        sp = ScrollPanel(self.tab_n)
        sp.pack(side='left', fill='y')
        panel = sp.inner
        self.panel_head(panel, 'n')
        f = self.schedule

        left = sec(panel, 'f.preset')
        self.fpreset = tk.StringVar(value=t(FOL_PRESET_IDS[0]))
        cbp = ttk.Combobox(left, values=[t(k) for k in FOL_PRESET_IDS],
                           textvariable=self.fpreset, state='readonly')
        cbp.pack(fill='x', pady=(2, 2))
        cbp.bind('<<ComboboxSelected>>', lambda e: self.apply_fol_preset(cbp.get()))

        left = sec(panel, 'f.noalpha', False)
        self.f_tol = Slider(left, t('f.tol'), 0.02, 0.5, 0.12, t('f.tol.info'), lambda: None, adv=True)
        ttk.Button(left, text=t('f.makealpha'),
                   command=self.make_alpha).pack(fill='x', pady=(4, 0))

        left = sec(panel, 'f.detect')
        self.mode = tk.StringVar(value='silh')
        for key, val in (('f.m.silh', 'silh'), ('f.m.dense', 'dense'), ('f.m.lum', 'lum')):
            row = ttk.Frame(left)
            row.pack(fill='x')
            ttk.Radiobutton(row, text=t(key), value=val, variable=self.mode,
                            command=f).pack(side='left')
            Info(row, t(key + '.info')).pack(side='right')
        self.s_split = Slider(left, t('f.split'), 0, 1, 0.35, t('f.split.info'), f)
        self.s_area = Slider(left, t('f.minarea'), 0, 1200, 60, t('f.minarea.info'),
                             f, integer=True, adv=True)

        left = sec(panel, 'f.shape')
        self.s_fold = Slider(left, t('f.fold'), -1.5, 1.5, 0.9, t('f.fold.info'), f, adv=True)
        self.s_prof = Slider(left, t('f.prof'), 1.0, 5.0, 2.0, t('f.prof.info'), f, fmt='%.1f', adv=True)
        self.s_det = Slider(left, t('f.det'), 0, 0.6, 0.22, t('f.det.info'), f, adv=True)
        self.s_seed = Slider(left, t('f.seed'), 1, 50, 3, '', f, integer=True, adv=True)
        self.s_pad = Slider(left, t('f.pad'), 0, 24, 8, t('f.pad.info'), f, integer=True, adv=True)

        left = sec(panel, 'f.orient')
        self.s_tilt = Slider(left, t('f.tilt'), 0, 1.5, 0.45, t('f.tilt.info'), f)
        self.s_thin = Slider(left, t('f.thin'), 1.5, 12.0, 3.5, t('f.thin.info'), f, fmt='%.1f', adv=True)
        self.s_blade = Slider(left, t('f.blade'), 0, 1.0, 0.0, t('f.blade.info'), f)
        self.s_bend = Slider(left, t('f.bend'), -1.0, 1.0, 0.10, t('f.bend.info'), f)

        left = sec(panel, 'f.clump')
        self.s_clm = Slider(left, t('f.clm'), -1.5, 1.5, 0.5, '', f, adv=True)
        self.s_clb = Slider(left, t('f.clb'), -1.5, 1.5, 0.25, t('f.clb.info'), f, adv=True)
        self.s_cls = Slider(left, t('f.cls'), 6, 90, 24, '', f, integer=True, adv=True)

        left = sec(panel, 'f.card', False)
        self.s_domw = Slider(left, t('f.domw'), -1.2, 1.2, 0.0, '', f, adv=True)
        self.s_domh = Slider(left, t('f.domh'), -1.2, 1.2, 0.0, '', f, adv=True)

        left = sec(panel, 'f.hand', False)
        self.brush_on = tk.BooleanVar(value=False)
        check(left, 'f.brush', self.brush_on, self.set_brush)
        self.mask_vis = tk.BooleanVar(value=True)
        row = ttk.Frame(left)
        row.pack(fill='x')
        ttk.Checkbutton(row, text=t('com.mask.vis'), variable=self.mask_vis,
                        command=self.redraw).pack(side='left')
        Info(row, t('f.mask.info')).pack(side='right')
        self.brush_r = Slider(left, t('com.brush.size'), 4, 120, 24, '', lambda: None,
                              integer=True, adv=True)
        row = ttk.Frame(left)
        row.pack(fill='x', pady=(4, 0))
        for key, cmd in (('com.clear', self.clear_brush), ('com.load', self.load_brush),
                         ('com.save', self.save_brush)):
            ttk.Button(row, text=t(key), width=9, command=cmd,
                       style='Small.TButton').pack(side='left', expand=True, fill='x')

        self.s_bent = Slider(left, t('bent'), 0, 1.0, 0.0, t('bent.info'), f)

        left = sec(panel, 'f.out')
        self.f_fast = tk.BooleanVar(value=True)
        check(left, 'f.fast', self.f_fast, self.reload_foliage)
        self.n_compress = tk.BooleanVar(value=False)
        ttk.Checkbutton(left, text=t('com.dxt'), variable=self.n_compress).pack(anchor='w')
        ttk.Button(left, text=t('f.savenormal'),
                   command=self.save_normal).pack(fill='x', pady=(6, 0))
        ttk.Button(left, text=t('com.batch'),
                   command=lambda: self.batch('normal')).pack(fill='x', pady=(6, 0))
        self.n_status = ttk.Label(left, style='Hint.TLabel', wraplength=280)
        self.n_status.pack(anchor='w', pady=8)

        right = ttk.Frame(self.tab_n)
        right.pack(side='left', fill='both', expand=True)
        self.n_view = ZoomView(right)
        self.n_view.pack(side='left', fill='both', expand=True)
        side = ttk.Frame(right, padding=(8, 0, 0, 0))
        side.pack(side='left', fill='y')
        self.FOL_VIEWS = (('f.v.light', 'light'), ('f.v.orig', 'orig'),
                          ('f.v.map', 'map'), ('f.v.seg', 'seg'),
                          ('f.v.branch', 'branch'))
        self.view_n = tk.StringVar(value='light')
        self.view_n2 = tk.StringVar(value='orig')
        self.split_n = tk.BooleanVar(value=False)
        self.view_side(side, self.FOL_VIEWS, self.view_n, self.view_n2, self.split_n)
        self.ring = LightRing(side, self.redraw)
        self.ring.pack(pady=4)
        self.spin = tk.BooleanVar(value=False)
        ttk.Checkbutton(side, text=t('com.spin'), variable=self.spin,
                        command=lambda: self.ring.toggle_spin(self.spin.get())).pack(anchor='w')

    def view_side(self, side, views, var, var2, splitvar):
        """Build the right-hand column choosing what to show: the list of views,
        the second (split) panel toggle and its view.

        views: sequence of (string key, view id). var: tk variable of the main
        view; var2: of the second panel; splitvar: BooleanVar for split mode.
        """
        ttk.Label(side, text=t('com.show'), style='Head.TLabel').pack(anchor='w')
        for key, val in views:
            ttk.Radiobutton(side, text=t(key), value=val, variable=var,
                            command=self.redraw).pack(anchor='w')
        ttk.Separator(side).pack(fill='x', pady=6)
        row = ttk.Frame(side)
        row.pack(fill='x')
        ttk.Checkbutton(row, text=t('com.split'), variable=splitvar,
                        command=self.split_toggled).pack(side='left')
        Info(row, t('com.split.info')).pack(side='right')
        names = [t(k) for k, _ in views]
        back = dict((t(k), v) for k, v in views)
        cb = ttk.Combobox(side, values=names, width=18, state='readonly')
        cb.set(next(n for n, v in back.items() if v == var2.get()))
        cb.pack(fill='x', pady=(2, 0))
        cb.bind('<<ComboboxSelected>>',
                lambda e: (var2.set(back[cb.get()]), self.redraw()))
        ttk.Separator(side).pack(fill='x', pady=8)

    def split_toggled(self):
        """Split checkbox: redraw and refit. The picture width doubles, so it is
        fitted again; otherwise only half of it is visible."""
        self.redraw()
        self.view().fit()

    def apply_fol_preset(self, display):
        """Foliage preset combobox: set the mode and sliders from FOL_PRESETS.

        display: the translated preset name as shown in the combobox. Drops the
        cached normal and schedules a recompute in 30 ms.
        """
        key = next((k for k in FOL_PRESET_IDS if t(k) == display), None)
        if key is None:
            return
        p = FOL_PRESETS[key]
        self.mode.set(p['mode'])
        for sl, v in ((self.s_split, p['split']), (self.s_area, p['area']),
                      (self.s_thin, p['thin']), (self.s_blade, p['blade']),
                      (self.s_tilt, p['tilt']), (self.s_bend, p['bend']),
                      (self.s_fold, p['fold']), (self.s_prof, p['prof']),
                      (self.s_det, p['det']), (self.s_clm, p['clm']),
                      (self.s_clb, p['clb']), (self.s_cls, p['cls'])):
            sl.set(v)
        self.normal = None
        self.schedule(30)

    def fol_args(self, force):
        """Return the foliage-normal settings as the positional argument tuple
        of foliage.Normals.run (mode, split, seed, …, min_area, blade, bend).

        force: the hand-painted branch/leaf mask (self.branch) or None; passed
        through in its slot. The order must match Normals.run exactly.
        """
        return (self.mode.get(), self.s_split.get(), self.s_seed.get(),
                self.s_thin.get(), self.s_tilt.get(), self.s_fold.get(),
                self.s_det.get(), self.s_prof.get(), self.s_pad.get(),
                self.s_domw.get(), self.s_domh.get(), force,
                self.s_clm.get(), self.s_clb.get(), float(self.s_cls.get()),
                int(self.s_area.get()), self.s_blade.get(), self.s_bend.get())

    # -------------------------------------------- вкладка света листвы -----
    def build_leafdiff_tab(self):
        """Build the "Foliage: light, AO" tab: de-lighting, AO map, leaf order
        from a depth network, subsurface/translucency (SSS), what to save, and
        the view column with its own light ring (elevation 75°, see below)."""
        sp = ScrollPanel(self.tab_d)
        sp.pack(side='left', fill='y')
        panel = sp.inner
        self.panel_head(panel, 'd')
        f = self.schedule

        left = sec(panel, 'd.delight')
        self.d_un = Slider(left, t('d.un'), 0, 1.0, 0.8, '', f)
        self.d_unr = Slider(left, t('d.unr'), 8, 160, 40, '', f, integer=True, adv=True)
        self.d_ao = Slider(left, t('d.ao'), 0, 1.0, 0.5, '', f)
        self.d_aor = Slider(left, t('d.aor'), 4, 96, 24, '', f, integer=True, adv=True)
        self.d_lift = Slider(left, t('d.lift'), 0.5, 2.0, 1.0, '', f)
        self.d_sat = Slider(left, t('d.sat'), 0, 1.0, 0.7, t('d.sat.info'), f)

        left = sec(panel, 'd.aomap')
        self.d_aostr = Slider(left, t('d.aostr'), 0, 1.5, 1.0, '', f)
        self.d_aospread = Slider(left, t('d.aospread'), 0, 48, 12,
                                 t('d.aospread.info'), f, integer=True)
        self.d_aobulge = Slider(left, t('d.aobulge'), 0, 30, 6, t('d.aobulge.info'), f,
                                integer=True, adv=True)
        self.d_aorad = Slider(left, t('d.aorad2'), 4, 80, 24, t('d.aorad2.info'), f,
                              integer=True)
        self.d_aotilt = Slider(left, t('d.aotilt'), 0, 1.0, 0.4, t('d.aotilt.info'), f, adv=True)
        self.d_aogam = Slider(left, t('d.aogam'), 0.2, 3.0, 1.0, '', f, adv=True)
        self.d_aocon = Slider(left, t('d.aocon'), 0.2, 3.0, 1.0, '', f, adv=True)

        # ---- порядок листьев от сети ----
        ttk.Separator(left).pack(fill='x', pady=6)
        ttk.Label(left, text=t('d.net'), style='Head.TLabel').pack(anchor='w')
        nr = ttk.Frame(left)
        nr.pack(fill='x', pady=(4, 0))
        ttk.Label(nr, text=t('m.nn.model')).pack(side='left')
        cbn = ttk.Combobox(nr, values=[t('nn.' + k) for k in DN.MODEL_IDS], width=14,
                           textvariable=self.m_netmodel, state='readonly')
        cbn.pack(side='left', padx=6)
        Info(nr, t('d.net.info')).pack(side='right')
        self.d_fnet = Slider(left, t('d.net.w'), 0, 1.0, 0.0, t('d.net.w.info'), f)
        self.d_netbtn = ttk.Button(left, text=t('m.nn.run'),
                                   command=lambda: self.net_depth(True))
        self.d_netbtn.pack(fill='x', pady=(6, 0))
        self.d_netbar = ttk.Progressbar(left, mode='determinate', maximum=1000)
        self.d_netinfo = ttk.Label(left, style='Hint.TLabel', wraplength=280)
        self.d_netinfo.pack(anchor='w', pady=(4, 0))

        left = sec(panel, 'd.sss', False)
        self.d_sstr = Slider(left, t('d.sss.str'), 0, 2.0, 1.0, '', f, adv=True)
        self.d_sbase = Slider(left, t('d.sss.base'), 0, 1.5, 0.15, t('d.sss.base.info'), f, adv=True)
        self.d_sveins = Slider(left, t('d.sss.veins'), 0, 2.5, 0.9, t('d.sss.veins.info'), f, adv=True)
        self.d_sbehind = Slider(left, t('d.sss.behind'), 0, 2.0, 1.0,
                                t('d.sss.behind.info'), f, adv=True)
        self.d_sreach = Slider(left, t('d.sss.reach'), 2, 60, 14, t('d.sss.reach.info'), f,
                               integer=True, adv=True)
        self.d_slay = Slider(left, t('d.sss.lay'), 0, 2.5, 0.5, t('d.sss.lay.info'), f, adv=True)
        self.d_spig = Slider(left, t('d.sss.pig'), 0, 2.0, 0.7, t('d.sss.pig.info'), f, adv=True)
        self.d_sedge = Slider(left, t('d.sss.edge'), 0, 1.5, 0.15, t('d.sss.edge.info'), f, adv=True)
        self.d_sbr = Slider(left, t('d.sss.branch'), 0, 1.0, 1.0, t('d.sss.branch.info'), f, adv=True)
        self.d_stint = Slider(left, t('d.sss.tint'), 0, 1.0, 0.35, t('d.sss.tint.info'), f, adv=True)
        self.d_sgam = Slider(left, t('d.sss.gam'), 0.2, 3.0, 1.0, '', f, adv=True)
        self.d_scon = Slider(left, t('d.sss.con'), 0.2, 3.0, 1.0, '', f, adv=True)
        ttk.Separator(left).pack(fill='x', pady=6)
        self.d_spow = Slider(left, t('d.sss.pow'), 0.5, 8.0, 2.0, t('d.sss.pow.info'),
                             self.redraw, fmt='%.1f', adv=True)

        left = sec(panel, 'd.what')
        self.fsave = {}
        for key in ('_fixed', '_ao', '_sss', '_n', '_nh', '_h', '_spec', '_rough'):
            v = tk.BooleanVar(value=key in ('_fixed', '_ao'))
            ttk.Checkbutton(left, text=t('sv.' + key), variable=v).pack(anchor='w')
            self.fsave[key] = v
        self.d_compress = tk.BooleanVar(value=True)
        ttk.Checkbutton(left, text=t('com.dxt'), variable=self.d_compress).pack(anchor='w')
        self.d_cov = tk.BooleanVar(value=True)
        check(left, 'd.cov', self.d_cov)
        ttk.Button(left, text=t('com.save.sel'),
                   command=self.save_foliage_maps).pack(fill='x', pady=(6, 0))
        ttk.Button(left, text=t('com.batch'),
                   command=lambda: self.batch('diffuse')).pack(fill='x', pady=(6, 0))
        self.d_status = ttk.Label(left, style='Hint.TLabel', wraplength=280,
                                  text=t('d.hint'))
        self.d_status.pack(anchor='w', pady=8)

        dright = ttk.Frame(self.tab_d)
        dright.pack(side='left', fill='both', expand=True)
        self.d_view = ZoomView(dright)
        self.d_view.pack(side='left', fill='both', expand=True)
        dside = ttk.Frame(dright, padding=(8, 0, 0, 0))
        dside.pack(side='left', fill='y')
        ttk.Label(dside, text=t('com.show'), style='Head.TLabel').pack(anchor='w')
        self.view_d = tk.StringVar(value='pair')
        for key, val in (('d.v.pair', 'pair'), ('d.v.after', 'after'),
                         ('d.v.ao', 'ao'), ('d.v.height', 'height'),
                         ('d.v.sss', 'sss'), ('d.v.backlit', 'backlit')):
            row = ttk.Frame(dside)
            row.pack(fill='x')
            ttk.Radiobutton(row, text=t(key), value=val, variable=self.view_d,
                            command=self.redraw).pack(side='left')
            if (key + '.info') in i18n.S:
                Info(row, t(key + '.info')).pack(side='right')
        self.ring_d = LightRing(dside, self.redraw)
        # почти прямо за карточкой: на пологом свете просвет почти не виден,
        # и вид «против света» с настройками по умолчанию выглядел бы пустым
        self.ring_d.el = 75.0
        self.ring_d.draw()
        self.ring_d.pack(pady=6)

    # -------------------------------------------------- вкладка материала --
    def panel_head(self, panel, tag):
        """Build the header of a settings column: the detail switch and "fold all".

        There are many settings now, and without this the column reads like one
        long sheet. panel: the column frame; tag: column id ('n', 'd', 'm',
        'x'), used to name the attributes all_<tag> (BooleanVar of the "all
        settings" checkbox) and panel_<tag>. The checkbox starts CHECKED: all
        settings are shown by default (see Detail for why). Returns the row.
        """
        row = ttk.Frame(panel)
        row.pack(fill='x', pady=(0, 2))
        var = tk.BooleanVar(value=True)
        setattr(self, 'all_' + tag, var)
        ttk.Checkbutton(row, text=t('ui.all'), variable=var,
                        command=lambda: self.detail_changed(tag)).pack(side='left')
        Info(row, t('ui.all.info')).pack(side='right')
        ttk.Button(row, text=t('ui.fold'), width=9,
                   command=lambda: self.fold_all(panel)).pack(side='right', padx=4)
        setattr(self, 'panel_' + tag, panel)
        return row

    def apply_detail_all(self):
        """(Re)create the Detail switch for every settings column and apply the
        state of its "all settings" checkbox.

        Called after the UI is built or rebuilt, because Detail records the
        packing of the finished panel. With the default checkbox state this
        shows every setting: hiding them by default was tried, and 85 sliders
        disappeared without a trace for the user.
        """
        for tag in ('n', 'd', 'm', 'x'):
            if hasattr(self, 'panel_' + tag):
                setattr(self, 'detail_' + tag, None)
                self.detail_changed(tag)

    def detail_changed(self, tag):
        """The "all settings" checkbox of column tag: show all or the essentials.

        Creates the column's Detail on first use; its "N more settings" notes
        call show(), which ticks the checkbox and shows everything again.
        """
        d = getattr(self, 'detail_' + tag, None)
        if d is None:
            var = getattr(self, 'all_' + tag)

            def show():
                """Click on an "N more settings" note: switch back to all."""
                var.set(True)
                self.detail_changed(tag)
            d = Detail(getattr(self, 'panel_' + tag), show)
            setattr(self, 'detail_' + tag, d)
        d.apply(getattr(self, 'all_' + tag).get())

    def fold_all(self, panel, want=None):
        """The "fold all" button: collapse or expand every Section in panel.

        want: True to open, False to close; None toggles (close all if any is
        open, otherwise open all).
        """
        secs = []

        def walk(w):
            """Collect all Section widgets under w, recursively."""
            for c in w.winfo_children():
                if isinstance(c, Section):
                    secs.append(c)
                walk(c)
        walk(panel)
        if want is None:
            want = not any(x.opened for x in secs)
        for x in secs:
            if x.opened != want:
                x.toggle()

    def mark_sections(self):
        """Put dots on collapsed sections in which something was changed
        (Section.mark for every section in the window; errors are ignored)."""
        def walk(w):
            """Call mark() on every Section under w, recursively."""
            for c in w.winfo_children():
                if isinstance(c, Section):
                    c.mark()
                walk(c)
        try:
            walk(self.root)
        except Exception:
            pass

    def build_material_tab(self):
        """Build the Material tab, in the order of work (the stage headings):

        source (preset, damaged-area repair, de-lighting); relief (height
        layers, shape from shading, external depth, network depth, ESRGAN,
        height hint); maps (height shape, normal incl. the normal from
        lighting, OpenMW parallax, AO, relief into diffuse, tone, specular);
        zones (second material by mask, tiling); output. Then the view column
        with the light ring and the hemispherical-ambient preview switch.
        Most sliders schedule a recompute; repair, network and hint sliders
        call their own handlers.
        """
        sp = ScrollPanel(self.tab_m)
        sp.pack(side='left', fill='y')
        panel = sp.inner
        f = self.schedule
        self.panel_head(panel, 'm')

        stage(panel, t('ui.stage.src'))
        left = sec(panel, 'm.preset')
        self.preset = tk.StringVar(value=t('p.' + MG.PRESET_IDS[0]))
        cb = ttk.Combobox(left, values=[t('p.' + k) for k in MG.PRESET_IDS],
                          textvariable=self.preset, state='readonly')
        cb.pack(fill='x', pady=(2, 2))
        cb.bind('<<ComboboxSelected>>', lambda e: self.apply_preset(cb.get()))

        # ------------------------------------------------ битые участки ---
        left = sec(panel, 'm.repair', False)
        self.r_on = tk.BooleanVar(value=False)
        check(left, 'm.rep.on', self.r_on, self.repair_changed, info=False)
        self.r_auto = tk.BooleanVar(value=True)
        check(left, 'm.rep.auto', self.r_auto, self.repair_changed, info=False)
        self.r_hi = Slider(left, t('m.rep.hi'), 0.6, 1.0, 0.90, '', self.repair_changed, adv=True)
        self.r_lo = Slider(left, t('m.rep.lo'), 0.0, 0.4, 0.08, '', self.repair_changed, adv=True)
        self.r_flat = Slider(left, t('m.rep.flat'), 0.005, 0.12, 0.030,
                             t('m.rep.flat.info'), self.repair_changed, fmt='%.3f', adv=True)
        self.r_grow = Slider(left, t('m.rep.grow'), -16, 16, 2, t('m.rep.grow.info'),
                             self.repair_changed, integer=True)
        self.r_mix = Slider(left, t('m.rep.mix'), 0, 1.0, 1.0, t('m.rep.mix.info'),
                            self.repair_changed)
        ttk.Label(left, text=t('m.rep.method'), style='Head.TLabel').pack(anchor='w',
                                                                         pady=(6, 0))
        self.r_method = tk.StringVar(value='patch')
        for key, val in (('m.rep.patch', 'patch'), ('m.rep.tone', 'tone'),
                         ('m.rep.smooth', 'smooth')):
            row = ttk.Frame(left)
            row.pack(fill='x')
            ttk.Radiobutton(row, text=t(key), value=val, variable=self.r_method,
                            command=self.repair_changed).pack(side='left')
            Info(row, t(key + '.info')).pack(side='right')
        self.rbrush_on = tk.BooleanVar(value=False)
        check(left, 'm.rep.hand', self.rbrush_on,
              lambda: self.pick_brush('rbrush_on'), info=False)
        rb = ttk.Frame(left)
        rb.pack(fill='x', pady=(6, 0))
        ttk.Button(rb, text=t('m.rep.now'), command=self.repair_now).pack(
            side='left', expand=True, fill='x')
        ttk.Button(rb, text=t('m.rep.savedif'), command=self.save_delit).pack(
            side='left', expand=True, fill='x', padx=(4, 0))
        self.r_status = ttk.Label(left, style='Hint.TLabel', wraplength=280)
        self.r_status.pack(anchor='w', pady=(4, 0))

        # ------------------------------------------------- снятие света ---
        left = sec(panel, 'm.delight')
        self.m_soft = Slider(left, t('m.soft'), 0, 1, 0.5, t('m.soft.info'), f)
        self.m_hard = Slider(left, t('m.hard'), 0, 1, 0.0, t('m.hard.info'), f)
        self.m_lrad = Slider(left, t('m.lrad'), 8, 160, 48, '', f, integer=True)
        self.m_rest = Slider(left, t('m.rest'), 0, 1.5, 0.0, t('m.rest.info'), f, adv=True)
        self.m_sat = Slider(left, t('d.sat'), 0, 1.0, 0.7, t('d.sat.info'), f)

        # --------------------------------------------------- слои высоты ---
        stage(panel, t('ui.stage.relief'))
        left = sec(panel, 'm.hsrc')
        self.m_hdet = Slider(left, t('m.hdet'), 0.05, 1.0, 0.45, t('m.hdet.info'), f)
        self.m_wband = Slider(left, t('m.wband'), 0, 1.5, 0.5, t('m.wband.info'), f)
        self.m_wmulti = Slider(left, t('m.wmulti'), 0, 1.5, 1.0, t('m.wmulti.info'), f)
        self.m_wlum = Slider(left, t('m.wlum'), 0, 1.5, 0.0, t('m.wlum.info'), f)
        self.m_wguid = Slider(left, t('m.wguid'), 0, 1.5, 0.4, t('m.wguid.info'), f)
        self.m_fall = Slider(left, t('m.fall'), 0.2, 0.95, 0.7, '', f, adv=True)
        self.m_grad = Slider(left, t('m.grad'), 2, 40, 8, '', f, integer=True, adv=True)

        # ---- по освещению, снятому на самой текстуре ----
        ttk.Separator(left).pack(fill='x', pady=6)
        row = ttk.Frame(left)
        row.pack(fill='x')
        ttk.Label(row, text=t('m.sfs'), style='Head.TLabel').pack(side='left')
        Info(row, t('m.sfs.info')).pack(side='right')
        self.m_wsfs = Slider(left, t('m.sfs.w'), 0, 1.5, 0.0, t('m.sfs.w.info'), f)
        self.m_sfsaz = Slider(left, t('m.sfs.az'), 0, 359, 135, t('m.sfs.az.info'), f,
                              integer=True)
        self.m_sfssm = Slider(left, t('m.sfs.smooth'), 0.02, 1.5, 0.25,
                              t('m.sfs.smooth.info'), f, adv=True)
        self.m_sfschr = Slider(left, t('m.sfs.chroma'), 0, 1.5, 0.6,
                               t('m.sfs.chroma.info'), f, adv=True)
        self.m_sfssh = Slider(left, t('m.sfs.shadow'), 0, 1.5, 1.0,
                              t('m.sfs.shadow.info'), f, adv=True)
        self.m_sfsscale = Slider(left, t('m.sfs.scale'), 0.05, 1.0, 0.3,
                                 t('m.sfs.scale.info'), f, adv=True)
        self.m_sfslow = Slider(left, t('m.sfs.low'), 0, 1.5, 1.0,
                               t('m.sfs.low.info'), f, adv=True)
        self.m_sfshigh = Slider(left, t('m.sfs.high'), 0, 2.0, 1.0,
                                t('m.sfs.high.info'), f, adv=True)
        ttk.Button(left, text=t('m.sfs.auto'),
                   command=self.guess_light).pack(fill='x', pady=(6, 0))
        self.m_sfsinfo = ttk.Label(left, style='Hint.TLabel', wraplength=280)
        self.m_sfsinfo.pack(anchor='w', pady=(4, 0))

        # ------------------------------------------- внешняя глубина -------
        left = sec(panel, 'm.ext', False)
        self.m_extname = ttk.Label(left, style='Hint.TLabel', wraplength=280,
                                   text=t('m.ext.none'))
        self.m_extname.pack(anchor='w')
        r1 = ttk.Frame(left)
        r1.pack(fill='x', pady=(4, 0))
        ttk.Button(r1, text=t('m.ext.load'), width=11,
                   command=self.load_depth).pack(side='left', expand=True, fill='x')
        b2 = ttk.Button(r1, text=t('m.ext.fromh'), width=11, command=self.depth_from_height)
        b2.pack(side='left', expand=True, fill='x')
        r2 = ttk.Frame(left)
        r2.pack(fill='x', pady=(3, 0))
        ttk.Button(r2, text=t('m.ext.save'), width=11,
                   command=self.save_depth).pack(side='left', expand=True, fill='x')
        ttk.Button(r2, text=t('m.ext.drop'), width=11,
                   command=lambda: self.load_depth(None)).pack(side='left', expand=True, fill='x')
        ch = ttk.Frame(left)
        ch.pack(fill='x', pady=(6, 0))
        ttk.Label(ch, text=t('m.ext.chan')).pack(side='left')
        self.m_extch = tk.StringVar(value=t('ch.luma'))
        cb2 = ttk.Combobox(ch, values=[t('ch.' + c) for c in EXT_CHANNELS], width=12,
                           textvariable=self.m_extch, state='readonly')
        cb2.pack(side='left', padx=6)
        cb2.bind('<<ComboboxSelected>>', lambda e: self.apply_depth_channel())
        Info(ch, t('m.ext.chan.info')).pack(side='right')
        self.m_wext = Slider(left, t('m.ext.w'), 0, 1.5, 0.0, '', f)
        self.m_extinv = tk.BooleanVar(value=False)
        ttk.Checkbutton(left, text=t('com.invert'), variable=self.m_extinv,
                        command=f).pack(anchor='w')
        self.m_extcav = tk.BooleanVar(value=False)
        check(left, 'm.ext.cav', self.m_extcav, f, adv=True)
        self.m_extcut = Slider(left, t('m.ext.cut'), 0.05, 0.8, 0.25,
                               t('m.ext.cut.info'), f, adv=True)

        ttk.Separator(left).pack(fill='x', pady=6)
        self.dbrush_on = tk.BooleanVar(value=False)
        check(left, 'm.ext.brush', self.dbrush_on, lambda: self.pick_brush('dbrush_on'))
        self.dbrush_r = Slider(left, t('com.brush.size'), 4, 160, 40, '', lambda: None,
                               integer=True, adv=True)
        self.dbrush_s = Slider(left, t('m.ext.force'), 0.02, 1.0, 0.25, '', lambda: None, adv=True)

        # ------------------------------------------- глубина нейросетью ----
        left = sec(panel, 'm.nn', False)
        nr = ttk.Frame(left)
        nr.pack(fill='x', pady=(4, 0))
        ttk.Label(nr, text=t('m.nn.model')).pack(side='left')
        cb3 = ttk.Combobox(nr, values=[t('nn.' + k) for k in DN.MODEL_IDS], width=14,
                           textvariable=self.m_netmodel, state='readonly')
        cb3.pack(side='left', padx=6)
        cb3.bind('<<ComboboxSelected>>', lambda e: self.net_note())
        Info(nr, t('m.nn.model.info')).pack(side='right')
        check(left, 'm.nn.fine', self.m_netfine, self.net_note, adv=True)
        self.m_nettile = Slider(left, t('m.nn.tile'), 0.15, 0.75, 0.5,
                                t('m.nn.tile.info'), self.net_note)
        self.m_netscale = Slider(left, t('m.nn.scale'), 0.05, 1.0, 0.3,
                                 t('m.nn.scale.info'), self.net_apply)
        self.m_netlow = Slider(left, t('m.nn.low'), 0, 1.5, 0.4,
                               t('m.nn.low.info'), self.net_apply)
        self.m_nethigh = Slider(left, t('m.nn.high'), 0, 2.0, 1.0,
                                t('m.nn.high.info'), self.net_apply)
        self.m_netcon = Slider(left, t('m.nn.curve'), 0.2, 6.0, 1.0,
                               t('m.nn.curve.info'), self.net_apply)
        self.m_netbtn = ttk.Button(left, text=t('m.nn.run'), command=self.net_depth)
        self.m_netbtn.pack(fill='x', pady=(6, 0))
        self.m_netbar = ttk.Progressbar(left, mode='determinate', maximum=1000)
        self.m_netinfo = ttk.Label(left, style='Hint.TLabel', wraplength=280)
        self.m_netinfo.pack(anchor='w', pady=(4, 0))
        self.net_note()

        # ------------------------------------------------ ESRGAN ----------
        left = sec(panel, 'm.esr', False)
        self.m_esrlic = ttk.Label(left, style='Hint.TLabel', wraplength=280,
                                  text=t('m.esr.lic'))
        self.m_esrlic.pack(anchor='w')
        self.m_esrn_on = tk.BooleanVar(value=True)
        check(left, 'm.esr.n.on', self.m_esrn_on)
        self.m_esrf_on = tk.BooleanVar(value=True)
        check(left, 'm.esr.f.on', self.m_esrf_on)
        br = ttk.Frame(left)
        br.pack(fill='x', pady=(6, 0))
        self.m_esrbtn = ttk.Button(br, text=t('m.esr.run'), command=self.esr_run)
        self.m_esrbtn.pack(side='left', fill='x', expand=True)
        ttk.Button(br, text=t('m.esr.file'), command=self.esr_file).pack(side='left',
                                                                      padx=(4, 0))
        self.m_esrbar = ttk.Progressbar(left, mode='determinate', maximum=1000)
        self.m_esrinfo = ttk.Label(left, style='Hint.TLabel', wraplength=280)
        self.m_esrinfo.pack(anchor='w', pady=(4, 0))
        ttk.Separator(left).pack(fill='x', pady=6)
        self.m_esrnmix = Slider(left, t('m.esr.nmix'), 0, 1.0, 0.0, t('m.esr.nmix.info'), f)
        self.m_esrnk = Slider(left, t('m.esr.nk'), 0.2, 3.0, 1.0, t('m.esr.nk.info'), f)
        self.m_esrnflip = tk.BooleanVar(value=False)
        check(left, 'm.esr.nflip', self.m_esrnflip, f)
        self.m_wesr = Slider(left, t('m.esr.wh'), 0, 1.5, 0.0, t('m.esr.wh.info'), f)
        self.m_esrrough = Slider(left, t('m.esr.rough'), 0, 1.0, 0.0,
                                 t('m.esr.rough.info'), f)
        self.esr_note()

        # ------------------------------------------- подсказка по высоте ---
        left = sec(panel, 'm.guide', False)
        self.gbrush_on = tk.BooleanVar(value=False)
        check(left, 'm.guide.brush', self.gbrush_on, lambda: self.pick_brush('gbrush_on'))
        self.gbrush_r = Slider(left, t('com.brush.size'), 2, 80, 12, '', lambda: None,
                               integer=True, adv=True)
        self.g_tol = Slider(left, t('m.guide.tol'), 0.01, 0.5, 0.10,
                            t('m.guide.tol.info'), self.rebuild_guide)
        self.g_cont = tk.BooleanVar(value=True)
        check(left, 'm.guide.cont', self.g_cont, self.rebuild_guide, adv=True)
        self.g_soft = Slider(left, t('m.guide.soft'), 0, 40, 1, '', self.rebuild_guide,
                             integer=True, adv=True)
        self.g_amt = Slider(left, t('m.guide.amt'), -0.6, 0.6, 0.25, '', f)
        self.g_flat = Slider(left, t('m.guide.flat'), 0, 1.0, 0.0, t('m.guide.flat.info'), f, adv=True)
        self.g_inv = tk.BooleanVar(value=False)
        check(left, 'm.guide.inv', self.g_inv, self.rebuild_guide, info=False)
        row = ttk.Frame(left)
        row.pack(fill='x', pady=(4, 0))
        ttk.Button(row, text=t('com.clear'), style='Small.TButton',
                   command=self.clear_guide).pack(side='left', expand=True, fill='x')

        # ------------------------------------------------ форма высоты -----
        stage(panel, t('ui.stage.maps'))
        left = sec(panel, 'm.hshape', False)
        self.m_tilt = Slider(left, t('m.tilt'), 30, 85, 75, t('m.tilt.info'), f, integer=True)
        self.m_hinv = tk.BooleanVar(value=False)
        ttk.Checkbutton(left, text=t('m.hinv'), variable=self.m_hinv,
                        command=f).pack(anchor='w')
        self.m_hsrc = tk.BooleanVar(value=False)
        check(left, 'm.hdelit', self.m_hsrc, f)
        self.m_hamp = Slider(left, t('m.hamp'), 2, 60, 20, t('m.hamp.info'), f, integer=True)

        # ---------------------------------------------------- нормаль ------
        left = sec(panel, 'm.normal', False)
        self.m_nstr = Slider(left, t('m.nstr'), 0.1, 3.0, 1.0, '', f)
        Info(left, t('m.nscales.info')).pack(anchor='w')
        self.m_nfine = Slider(left, t('m.nfine'), 0, 2.0, 1.0, '', f, adv=True)
        self.m_nmid = Slider(left, t('m.nmid'), 0, 2.0, 0.5, '', f, adv=True)
        self.m_nbig = Slider(left, t('m.nbig'), 0, 2.0, 0.2, '', f, adv=True)
        self.m_nscale = Slider(left, t('m.nscale'), 2, 40, 8, t('m.nscale.info'), f,
                               integer=True, adv=True)
        self.m_nbent = Slider(left, t('bent'), 0, 1.0, 0.0, t('bent.info'), f)
        self.m_dx = tk.BooleanVar(value=True)
        check(left, 'm.dx', self.m_dx, f)

        # ---- нормаль по освещению: R и G из слоя освещения ----
        ttk.Separator(left).pack(fill='x', pady=6)
        row = ttk.Frame(left)
        row.pack(fill='x')
        ttk.Label(row, text=t('m.ndir.head'), style='Head.TLabel').pack(side='left')
        Info(row, t('m.ndir.head.info')).pack(side='right')
        self.m_ndir = Slider(left, t('m.sfs.direct'), 0, 1.0, 0.0,
                             t('m.sfs.direct.info'), f)
        self.m_ndirk = Slider(left, t('m.sfs.direct.k'), 0.2, 3.0, 1.0,
                              t('m.sfs.direct.k.info'), f)
        br = ttk.Frame(left)
        br.pack(fill='x', pady=(4, 0))
        ttk.Label(br, text=t('m.ndir.blue')).pack(side='left')
        self.m_ndirb = tk.StringVar(value=t('m.ndir.b.unit'))
        cbb = ttk.Combobox(br, values=[t('m.ndir.b.' + k) for k in NDIR_BLUE], width=12,
                           textvariable=self.m_ndirb, state='readonly')
        cbb.pack(side='left', padx=6)
        cbb.bind('<<ComboboxSelected>>', lambda e: f())
        Info(br, t('m.ndir.blue.info')).pack(side='right')
        self.m_ndirinv = tk.BooleanVar(value=False)
        check(left, 'm.ndir.inv', self.m_ndirinv, f)
        self.m_ndirsw = tk.BooleanVar(value=False)
        check(left, 'm.sfs.direct.swap', self.m_ndirsw, f, adv=True)
        self.m_ndirfix = Slider(left, t('m.sfs.fix'), 0, 1.0, 0.0,
                                t('m.sfs.fix.info'), f)
        self.m_ndirnote = ttk.Label(left, style='Hint.TLabel', wraplength=280)
        self.m_ndirnote.pack(anchor='w', pady=(4, 0))

        # ------------------------------------------- параллакс OpenMW -----
        left = sec(panel, 'm.plx', False)
        self.m_plxrep = Slider(left, t('m.plx.rep'), 16, 1024, 128, t('m.plx.rep.info'), f,
                               integer=True)
        self.m_plxdep = Slider(left, t('m.plx.dep'), 0.1, 30.0, 7.0, t('m.plx.dep.info'), f,
                               fmt='%.1f')
        ttk.Label(left, text=t('m.plx.ref'), wraplength=280).pack(anchor='w', pady=(4, 0))
        rr = ttk.Frame(left)
        rr.pack(fill='x')
        self.m_plxref = tk.StringVar(value=t('m.plx.r.median'))
        cbr = ttk.Combobox(rr, values=[t('m.plx.r.' + k) for k in MG.PARALLAX_REFS],
                           textvariable=self.m_plxref, state='readonly')
        cbr.pack(side='left', fill='x', expand=True)
        cbr.bind('<<ComboboxSelected>>', lambda e: f())
        Info(rr, t('m.plx.ref.info')).pack(side='right')
        self.m_plxlev = Slider(left, t('m.plx.lev'), 0.0, 1.0, 0.5, t('m.plx.lev.info'), f)
        self.m_plxdith = tk.BooleanVar(value=True)
        check(left, 'm.plx.dith', self.m_plxdith, f)
        self.m_plxnorm = tk.BooleanVar(value=False)
        check(left, 'm.plx.norm', self.m_plxnorm, f)
        self.m_plxnote = ttk.Label(left, style='Hint.TLabel', wraplength=280)
        self.m_plxnote.pack(anchor='w', pady=(4, 0))

        # ------------------------------------------------------- AO --------
        left = sec(panel, 'm.ao', False)
        self.m_aor = Slider(left, t('m.aor'), 4, 80, 28, '', f, integer=True)
        self.m_aos = Slider(left, t('m.aos'), 0, 1.5, 0.9, '', f)
        self.m_aowc = Slider(left, t('m.aowc'), 0, 1.5, 1.0, t('m.aowc.info'), f)
        self.m_aown = Slider(left, t('m.aown'), 0, 1.5, 0.35, t('m.aown.info'), f)

        # ---------------------------------------- рельеф в диффуз ----------
        left = sec(panel, 'm.rd', False)
        for lab, var, ids, pref, init in (
                ('m.rd.src', 'm_rdsrc', MG.RELIEF_SOURCES, 'm.rd.s.', 'height'),
                ('m.rd.mode', 'm_rdmode', MG.RELIEF_MODES, 'm.rd.m.', 'overlay')):
            ttk.Label(left, text=t(lab)).pack(anchor='w', pady=(4, 0))
            rr = ttk.Frame(left)
            rr.pack(fill='x')
            v = tk.StringVar(value=t(pref + init))
            setattr(self, var, v)
            cb = ttk.Combobox(rr, values=[t(pref + k) for k in ids], textvariable=v,
                              state='readonly')
            cb.pack(side='left', fill='x', expand=True)
            cb.bind('<<ComboboxSelected>>', lambda e: f())
            Info(rr, t(lab + '.info')).pack(side='right')
        self.m_rdamt = Slider(left, t('m.rd.amt'), 0, 3.0, 0.0, t('m.rd.amt.info'), f)
        self.m_rdscale = Slider(left, t('m.sfs.scale'), 0.05, 1.0, 0.3,
                                t('m.rd.scale.info'), f)
        self.m_rdlow = Slider(left, t('m.sfs.low'), 0, 1.5, 1.0, '', f)
        self.m_rdhigh = Slider(left, t('m.sfs.high'), 0, 3.0, 1.0, '', f)
        self.m_rdsat = Slider(left, t('d.sat'), 0, 1.0, 1.0, t('m.rd.sat.info'), f)
        self.m_rdnote = ttk.Label(left, style='Hint.TLabel', wraplength=280)
        self.m_rdnote.pack(anchor='w', pady=(4, 0))

        # ------------------------------------------------------ тон --------
        left = sec(panel, 'm.tone', False)
        self.m_hgam = Slider(left, t('m.hgam'), 0.2, 3.0, 1.0, '', f, adv=True)
        self.m_hcon = Slider(left, t('m.hcon'), 0.2, 3.0, 1.0, '', f, adv=True)
        ttk.Separator(left).pack(fill='x', pady=6)
        self.m_aogam = Slider(left, t('m.aogam'), 0.2, 3.0, 1.0, '', f, adv=True)
        self.m_aocon = Slider(left, t('m.aocon'), 0.2, 3.0, 1.0, '', f, adv=True)
        ttk.Separator(left).pack(fill='x', pady=6)
        self.m_rgam = Slider(left, t('m.rgam'), 0.2, 3.0, 1.0, '', f, adv=True)
        self.m_rcon = Slider(left, t('m.rcon'), 0.2, 3.0, 1.0, '', f, adv=True)

        # ------------------------------------------------- отражение -------
        left = sec(panel, 'm.spec', False)
        self.m_f0 = Slider(left, t('m.f0'), 0.01, 1.0, 0.045, t('m.f0.info'), f, fmt='%.3f')
        self.m_gloss = Slider(left, t('m.gloss'), 0, 1, 0.28, '', f)
        self.m_gvar = Slider(left, t('m.gvar'), 0, 1, 0.35, '', f, adv=True)
        self.m_metal = Slider(left, t('m.metal'), 0, 1, 0.0, '', f, adv=True)
        self.m_tint = Slider(left, t('m.tint'), 0, 1, 0.0, '', f, adv=True)

        # ------------------------------------------------ вторая зона ------
        stage(panel, t('ui.stage.zones'))
        left = sec(panel, 'm.zone', False)
        self.z_on = tk.BooleanVar(value=False)
        check(left, 'm.zone.on', self.z_on, f)
        ttk.Label(left, text=t('m.zone.preset')).pack(anchor='w', pady=(6, 0))
        self.z_preset = tk.StringVar(value=t('p.' + MG.PRESET_IDS[6]))
        cbz = ttk.Combobox(left, values=[t('p.' + k) for k in MG.PRESET_IDS],
                           textvariable=self.z_preset, state='readonly')
        cbz.pack(fill='x', pady=(2, 2))
        cbz.bind('<<ComboboxSelected>>', lambda e: self.schedule())
        row = ttk.Frame(left)
        row.pack(fill='x', pady=(6, 0))
        ttk.Label(row, text=t('m.zone.chan')).pack(side='left')
        self.z_chan = tk.StringVar(value=t('ch.luma'))
        cbc = ttk.Combobox(row, values=[t('ch.' + c) for c in CHANNELS], width=13,
                           textvariable=self.z_chan, state='readonly')
        cbc.pack(side='left', padx=6)
        cbc.bind('<<ComboboxSelected>>', lambda e: self.schedule())
        Info(row, t('m.zone.src.info')).pack(side='right')
        self.z_lo = Slider(left, t('m.zone.lo'), 0, 1, 0.0, '', f, adv=True)
        self.z_hi = Slider(left, t('m.zone.hi'), 0, 1, 0.35, '', f, adv=True)
        self.z_soft = Slider(left, t('m.zone.soft'), 0.005, 0.4, 0.05, '', f, fmt='%.3f', adv=True)
        self.z_blur = Slider(left, t('m.zone.blur'), 0, 40, 2, '', f, integer=True, adv=True)
        self.z_inv = tk.BooleanVar(value=False)
        ttk.Checkbutton(left, text=t('m.zone.inv'), variable=self.z_inv,
                        command=f).pack(anchor='w')
        self.zbrush_on = tk.BooleanVar(value=False)
        check(left, 'm.zone.brush', self.zbrush_on, lambda: self.pick_brush('zbrush_on'))
        self.zbrush_r = Slider(left, t('com.brush.size'), 2, 80, 12, '', lambda: None,
                               integer=True, adv=True)
        row = ttk.Frame(left)
        row.pack(fill='x', pady=(4, 0))
        ttk.Button(row, text=t('com.clear'), style='Small.TButton',
                   command=self.clear_zone).pack(side='left', expand=True, fill='x')

        # ------------------------------------------ бесшовность и вывод ----
        left = sec(panel, 'm.seam', False)
        self.m_fast = tk.BooleanVar(value=True)
        check(left, 'm.fast', self.m_fast, self.rebuild_job)
        self.m_seam = tk.BooleanVar(value=False)
        check(left, 'm.seamon', self.m_seam, f)
        self.m_band = Slider(left, t('m.band'), 4, 128, 24, t('m.band.info'), f, integer=True, adv=True)

        stage(panel, t('ui.stage.out'))
        left = sec(panel, 'm.what')
        self.saveflags = {}
        for key in ('_n', '_nh', '_spec', '_ao', '_h', '_rough', '_metal', '_fixed'):
            v = tk.BooleanVar(value=key in ('_nh', '_spec', '_ao'))
            ttk.Checkbutton(left, text=t('ms.' + key), variable=v).pack(anchor='w')
            self.saveflags[key] = v
        self.m_compress = tk.BooleanVar(value=True)
        ttk.Checkbutton(left, text=t('com.dxt'), variable=self.m_compress).pack(anchor='w')
        ttk.Button(left, text=t('com.save.sel'),
                   command=self.save_material).pack(fill='x', pady=(6, 0))
        ttk.Button(left, text=t('com.batch'),
                   command=lambda: self.batch('material')).pack(fill='x', pady=(6, 0))
        self.m_status = ttk.Label(left, style='Hint.TLabel', wraplength=280)
        self.m_status.pack(anchor='w', pady=8)

        right = ttk.Frame(self.tab_m)
        right.pack(side='left', fill='both', expand=True)
        self.m_view = ZoomView(right)
        self.m_view.pack(side='left', fill='both', expand=True)
        side = ttk.Frame(right, padding=(8, 0, 0, 0))
        side.pack(side='left', fill='y')
        self.MAT_VIEWS = (('m.v.lit', 'lit'), ('m.v.orig', 'orig'),
                          ('m.v.diffuse', 'diffuse'), ('m.v.dbase', 'dbase'),
                          ('m.v.height', 'height'),
                          ('m.v.normal', 'normal'), ('m.v.nh', 'nh'), ('m.v.ao', 'ao'),
                          ('m.v.spec', 'spec'), ('m.v.rough', 'rough'),
                          ('m.v.ext', 'ext'), ('m.v.esrn', 'esrn'),
                          ('m.v.esrh', 'esrh'), ('m.v.esrr', 'esrr'),
                          ('m.v.guide', 'guide'),
                          ('m.v.zone', 'zone'), ('m.v.damage', 'damage'),
                          ('m.v.fixed', 'fixed'))
        self.view_m = tk.StringVar(value='lit')
        self.view_m2 = tk.StringVar(value='height')
        self.split_m = tk.BooleanVar(value=False)
        self.view_side(side, self.MAT_VIEWS, self.view_m, self.view_m2, self.split_m)
        self.ring_m = LightRing(side, self.redraw)
        self.ring_m.pack(pady=4)
        # рассеянный свет в превью: ровный, как у OpenMW, или полусферой, как
        # в современных движках. От этого зависит, виден ли bent normal вообще
        self.amb_hemi = tk.BooleanVar(value=False)
        check(side, 'com.hemi', self.amb_hemi, self.redraw)
        self.spin_m = tk.BooleanVar(value=False)
        ttk.Checkbutton(side, text=t('com.spin'), variable=self.spin_m,
                        command=lambda: self.ring_m.toggle_spin(self.spin_m.get())).pack(anchor='w')



    # ------------------------------------------------------ вкладка узлов --
    def build_nodes_tab(self):
        """Build the Nodes tab: node add/delete/reset, graph load/save, save
        outputs, batch; a parameter section for the selected node; the graph
        canvas (mouse bindings below) and a preview view under it. Starts from
        nodes.default_graph()."""
        sp = ScrollPanel(self.tab_g, width=300)
        sp.pack(side='left', fill='y')
        panel = sp.inner
        self.panel_head(panel, 'g')

        left = sec(panel, 'g.add')
        self.g_pick = tk.StringVar(value=t('nd.source'))
        cb = ttk.Combobox(left, values=[t('nd.' + k) for k, spec in ND.REG.items() if not spec.hidden],
                          textvariable=self.g_pick, state='readonly')
        cb.pack(fill='x', pady=(2, 4))
        ttk.Button(left, text=t('g.add.btn'), command=self.g_add).pack(fill='x')
        r = ttk.Frame(left)
        r.pack(fill='x', pady=(6, 0))
        ttk.Button(r, text=t('g.del'), command=self.g_del).pack(
            side='left', expand=True, fill='x')
        ttk.Button(r, text=t('g.reset'), command=self.g_reset).pack(
            side='left', expand=True, fill='x', padx=(4, 0))
        r2 = ttk.Frame(left)
        r2.pack(fill='x', pady=(4, 0))
        ttk.Button(r2, text=t('g.load'), command=self.g_load).pack(
            side='left', expand=True, fill='x')
        ttk.Button(r2, text=t('g.save'), command=self.g_save).pack(
            side='left', expand=True, fill='x', padx=(4, 0))
        ttk.Button(left, text=t('g.saveout'), command=self.g_save_maps).pack(
            fill='x', pady=(6, 0))
        ttk.Button(left, text=t('g.batch'), command=self.g_batch).pack(
            fill='x', pady=(6, 0))

        self.g_params = sec(panel, 'g.params')
        self.g_status = ttk.Label(panel, style='Hint.TLabel', wraplength=280,
                                  text=t('g.hint'))
        self.g_status.pack(anchor='w', pady=8)

        right = ttk.Frame(self.tab_g)
        right.pack(side='left', fill='both', expand=True)
        self.g_canvas = tk.Canvas(right, background=BG, highlightthickness=0)
        self.g_canvas.pack(side='top', fill='both', expand=True)
        gv = ttk.Frame(right, height=300)
        gv.pack(side='bottom', fill='x')
        gv.pack_propagate(False)
        self.g_view = ZoomView(gv)
        self.g_view.pack(fill='both', expand=True)
        for ev, fn in (('<Button-1>', self.g_press), ('<B1-Motion>', self.g_drag),
                       ('<ButtonRelease-1>', self.g_release),
                       ('<Button-3>', self.g_rclick),
                       ('<Double-Button-1>', self.g_dbl)):
            self.g_canvas.bind(ev, fn)
        self.graph = ND.default_graph()
        self.g_sel = None
        self.g_move = None
        self.g_wire = None
        self.g_cache_key = None
        self.root.after(80, self.g_redraw)

    # --- геометрия узла
    GW, GH0, GROW = 148, 22, 15

    def g_box(self, nid):
        """Return the node's box on the canvas as (x0, y0, x1, y1) in pixels;
        the height grows with the number of ports."""
        n = self.graph.nodes[nid]
        sp = ND.REG[n['type']]
        h = self.GH0 + self.GROW * max(len(sp.ins), len(sp.outs), 1) + 6
        return n['x'], n['y'], n['x'] + self.GW, n['y'] + h

    def g_port_xy(self, nid, port, is_in):
        """Return the canvas position (x, y) of a port: inputs on the left
        edge of the node, outputs on the right."""
        n = self.graph.nodes[nid]
        sp = ND.REG[n['type']]
        lst = sp.ins if is_in else sp.outs
        idx = [p for p, _ in lst].index(port)
        y = n['y'] + self.GH0 + self.GROW * idx + 8
        return (n['x'] if is_in else n['x'] + self.GW), y

    def g_hit_port(self, x, y):
        """Return (node id, port, is_input, kind) of the port under canvas point
        (x, y), within ±9×±8 px, or None."""
        for nid in self.graph.nodes:
            sp = ND.REG[self.graph.nodes[nid]['type']]
            for is_in, lst in ((True, sp.ins), (False, sp.outs)):
                for port, kind in lst:
                    px, py = self.g_port_xy(nid, port, is_in)
                    if abs(px - x) < 9 and abs(py - y) < 8:
                        return nid, port, is_in, kind
        return None

    def g_hit_node(self, x, y):
        """Return the id of the topmost node whose box contains (x, y), or None."""
        for nid in reversed(list(self.graph.nodes)):
            x0, y0, x1, y1 = self.g_box(nid)
            if x0 <= x <= x1 and y0 <= y <= y1:
                return nid
        return None

    # --- рисование
    def g_redraw(self):
        """Redraw the graph canvas: links coloured by data kind, node boxes
        (selected one highlighted), ports and the wire being dragged."""
        c = self.g_canvas
        c.delete('all')
        for a, ap, b, bp in self.graph.links:
            if a not in self.graph.nodes or b not in self.graph.nodes:
                continue
            x1, y1 = self.g_port_xy(a, ap, False)
            x2, y2 = self.g_port_xy(b, bp, True)
            kind = dict(ND.REG[self.graph.nodes[a]['type']].outs)[ap]
            c.create_line(x1, y1, x1 + 30, y1, x2 - 30, y2, x2, y2, smooth=True,
                          width=2, fill=ND.KIND_COLOR.get(kind, DIM))
        for nid, n in self.graph.nodes.items():
            sp = ND.REG[n['type']]
            x0, y0, x1, y1 = self.g_box(nid)
            act = (nid == self.g_sel)
            c.create_rectangle(x0, y0, x1, y1, fill=PANEL,
                               outline=ACC if act else EDGE, width=2 if act else 1)
            c.create_rectangle(x0, y0, x1, y0 + self.GH0 - 4, fill=EDGE, outline='')
            c.create_text(x0 + 7, y0 + 9, text=t('nd.' + n['type']), anchor='w',
                          fill=FG, font=('TkDefaultFont', 8, 'bold'))
            for is_in, lst in ((True, sp.ins), (False, sp.outs)):
                for port, kind in lst:
                    px, py = self.g_port_xy(nid, port, is_in)
                    col = ND.KIND_COLOR.get(kind, DIM)
                    c.create_oval(px - 5, py - 5, px + 5, py + 5, fill=col, outline=BG)
                    c.create_text(px + (10 if is_in else -10), py,
                                  text=ND.label(n['type'], port), anchor='w' if is_in else 'e',
                                  fill=DIM, font=('TkDefaultFont', 7))
        if self.g_wire:
            x1, y1, x2, y2 = self.g_wire
            c.create_line(x1, y1, x2, y2, fill=ACC, width=2, dash=(4, 3))
        c.configure(scrollregion=c.bbox('all'))

    # --- мышь
    def g_press(self, e):
        """<Button-1> on the graph canvas.

        On an input port: disconnect it. On an output port: start dragging a
        new wire. On a node: select it, show its parameters and start moving
        it. On empty space: clear the selection.
        """
        hit = self.g_hit_port(e.x, e.y)
        if hit:
            nid, port, is_in, kind = hit
            if is_in:
                self.graph.disconnect(nid, port)
                self.g_redraw()
                return
            self.g_from = (nid, port)
            self.g_wire = (e.x, e.y, e.x, e.y)
            return
        nid = self.g_hit_node(e.x, e.y)
        if nid:
            self.g_sel = nid
            self.g_move = (e.x - self.graph.nodes[nid]['x'],
                           e.y - self.graph.nodes[nid]['y'])
            self.g_show_params(nid)
        else:
            self.g_sel = None
            self.g_show_params(None)
        self.g_redraw()

    def g_drag(self, e):
        """<B1-Motion> on the graph canvas: extend the wire or move the node
        (coordinates kept non-negative)."""
        if self.g_wire:
            self.g_wire = (self.g_wire[0], self.g_wire[1], e.x, e.y)
            self.g_redraw()
        elif self.g_move and self.g_sel:
            n = self.graph.nodes[self.g_sel]
            n['x'] = max(0, e.x - self.g_move[0])
            n['y'] = max(0, e.y - self.g_move[1])
            self.g_redraw()

    def g_release(self, e):
        """<ButtonRelease-1> on the graph canvas: finish a wire on an input port.

        Graph.connect refuses cycles and kind mismatches; then a status message
        is shown, otherwise the graph is evaluated.
        """
        if self.g_wire:
            hit = self.g_hit_port(e.x, e.y)
            if hit and hit[2]:
                a, ap = self.g_from
                if not self.graph.connect(a, ap, hit[0], hit[1]):
                    self.g_status.config(text=t('g.badlink'))
                else:
                    self.g_status.config(text='')
                    self.g_eval()
            self.g_wire = None
            self.g_redraw()
        self.g_move = None

    def g_rclick(self, e):
        """<Button-3> on the graph canvas: delete the node under the cursor."""
        nid = self.g_hit_node(e.x, e.y)
        if nid:
            self.graph.remove(nid)
            if self.g_sel == nid:
                self.g_sel = None
                self.g_show_params(None)
            self.g_redraw()

    def g_dbl(self, e):
        """<Double-Button-1> on a node: select it and preview its output."""
        nid = self.g_hit_node(e.x, e.y)
        if nid:
            self.g_sel = nid
            self.g_eval(nid)

    # --- параметры выбранного узла
    def g_show_params(self, nid):
        """Fill the parameter panel (self.g_params) with controls for node nid.

        Called when a node is selected or the selection is cleared (nid None:
        the panel is just emptied). Each parameter from the node's spec in
        ND.REG gets a widget by kind: 'f'/'i' a Slider with the spec's range,
        'b' a check box, anything else a read-only combobox of the allowed
        values. Every change goes through g_set, which re-evaluates the graph
        with a delay.
        """
        for w in self.g_params.winfo_children():
            w.destroy()
        if nid is None:
            return
        n = self.graph.nodes[nid]
        sp = ND.REG[n['type']]
        ttk.Label(self.g_params, text=t('nd.' + n['type']),
                  style='Head.TLabel').pack(anchor='w')
        info = 'nd.' + n['type'] + '.info'
        if info in i18n.S:
            Info(self.g_params, t(info)).pack(anchor='w')
        for pr in sp.params:
            name, kind = pr[0], pr[1]
            if kind == 'd':
                continue            # данные правятся на холсте, не ползунком
            if kind in ('f', 'i'):
                s = Slider(self.g_params, ND.label(n['type'], name), pr[3], pr[4],
                           n['params'][name], '', lambda: None, integer=(kind == 'i'))
                s.var.trace_add('write', lambda *a, k=name, sl=s:
                                self.g_set(nid, k, sl.get()))
            elif kind == 'b':
                v = tk.BooleanVar(value=bool(n['params'][name]))
                ttk.Checkbutton(self.g_params, text=ND.label(n['type'], name), variable=v,
                                command=lambda k=name, vv=v:
                                self.g_set(nid, k, bool(vv.get()))).pack(anchor='w')
            else:
                row = ttk.Frame(self.g_params)
                row.pack(fill='x', pady=(4, 0))
                ttk.Label(row, text=ND.label(n['type'], name)).pack(side='left')
                v = tk.StringVar(value=str(n['params'][name]))
                cb = ttk.Combobox(row, values=list(pr[3]), textvariable=v,
                                  state='readonly', width=10)
                cb.pack(side='right')
                cb.bind('<<ComboboxSelected>>',
                        lambda e, k=name, vv=v: self.g_set(nid, k, vv.get()))

    def g_set(self, nid, key, val):
        """Store parameter key = val of node nid and schedule g_eval in 250 ms.

        Called by the parameter widgets built in g_show_params. An unchanged
        value is ignored; repeated changes within the delay restart the timer
        (self.g_pending), so dragging a slider evaluates the graph once.
        """
        if self.graph.nodes.get(nid, {}).get('params', {}).get(key) == val:
            return
        self.graph.nodes[nid]['params'][key] = val
        if self.g_pending:
            self.root.after_cancel(self.g_pending)
        self.g_pending = self.root.after(250, self.g_eval)

    # --- счёт и просмотр
    def g_eval(self, nid=None):
        """Evaluate a node of the graph and show its output in the graph view.

        nid: node id; default is the selected node, else the first output
        node. Evaluation runs in a separate thread: the graph may contain
        neural networks, and the window must not freeze for a minute. The
        source is the loaded texture's RGB, halved (LANCZOS) when its longer
        side exceeds 1024 px, to keep the preview fast. If an evaluation is
        already running, the request is remembered in self._g_again and done
        as soon as the current one finishes. The worker thread never touches
        Tk: it only writes self._g_msg (progress text) and self._g_out
        (('ok', nid, outputs) or ('err', nid, message)), which _g_poll reads
        from the main thread.
        """
        self.g_pending = None
        if self.rgb is None:
            self.g_status.config(text=t('g.nosrc'))
            return
        nid = nid or self.g_sel or (self.graph.outputs() or [None])[0]
        if nid is None:
            return
        if getattr(self, '_g_busy', False):
            self._g_again = nid            # досчитаем, как только освободится
            return
        src = self.rgb[..., :3]
        if max(src.shape[:2]) > 1024:
            im = Image.fromarray((np.clip(src, 0, 1) * 255).astype(np.uint8))
            im = im.resize((src.shape[1] // 2, src.shape[0] // 2), Image.LANCZOS)
            src = np.asarray(im, np.float32) / 255
        self._g_busy, self._g_again, self._g_out = True, None, None
        self._g_msg = t('com.computing')

        def sub(i, n):
            """Progress callback for long nodes: step i of n (worker thread;
            only sets the status text attribute)."""
            self._g_msg = t('g.sub', i, n)
        ctx = {'src': np.ascontiguousarray(src), 'sub': sub,
               'key': str(id(self.rgb)) + str(src.shape)}
        graph = self.graph

        def work():
            """Worker thread body: evaluate the graph and put the result or
            the error text into self._g_out. Never touches Tk."""
            try:
                self._g_out = ('ok', nid, XPU.host_tree(graph.eval(
                    nid, ctx, lambda k: setattr(self, '_g_msg', t(
                        'g.node', t('nd.' + graph.nodes[k]['type']))))))
            except Exception as ex:
                self._g_out = ('err', nid, str(ex))
        threading.Thread(target=work, daemon=True).start()
        self._g_poll()

    def _g_poll(self):
        """Main-thread poll (every 200 ms via after) of the graph worker.

        While self._g_out is None it shows the progress text self._g_msg.
        When the worker is done it shows the error text, or puts the first
        array output into the graph view: RGBA (_nh) is shown as its normal
        (RGB), a 3-channel array with values below -0.01 is taken as a signed
        normal and mapped to 0…1, a grey map is repeated into three channels.
        Then a request queued in self._g_again (see g_eval) is started.
        """
        out = self._g_out
        if out is None:
            self.g_status.config(text=self._g_msg or '')
            self.root.after(200, self._g_poll)
            return
        self._g_busy = False
        kind, nid, res = out
        if kind == 'err':
            self.g_status.config(text=res)
        else:
            vals = [v for v in res.values() if isinstance(v, np.ndarray)]
            if not vals:
                self.g_status.config(text=t('g.novalue'))
            else:
                a = vals[0]
                if a.ndim == 3 and a.shape[2] == 4:
                    a = a[..., :3]                        # _nh: показываем нормаль
                elif a.ndim == 3 and a.shape[2] == 3 and float(a.min()) < -0.01:
                    a = a * 0.5 + 0.5                     # нормаль
                self.g_view.set_image(np.clip(a if a.ndim == 3 else np.dstack([a] * 3), 0, 1),
                                      keep_view=False)
                self.g_status.config(text=t('g.ok', t('nd.' + self.graph.nodes[nid]['type'])
                                            if nid in self.graph.nodes else ''))
        if getattr(self, '_g_again', None):
            nxt, self._g_again = self._g_again, None
            self.g_eval(nxt)

    def g_add(self):
        """Button "Add node": add a node of the type chosen in the self.g_pick
        combobox (matched by translated name, 'tone' if nothing matches) at
        (40, 40), select it and show its parameters."""
        key = next((k for k in ND.REG if t('nd.' + k) == self.g_pick.get()), 'levels')
        nid = self.graph.add(key, 40, 40)
        self.g_sel = nid
        self.g_show_params(nid)
        self.g_redraw()

    def g_del(self):
        """Button "Delete node": remove the selected node with its links."""
        if self.g_sel:
            self.graph.remove(self.g_sel)
            self.g_sel = None
            self.g_show_params(None)
            self.g_redraw()

    def g_reset(self):
        """Button "Reset": replace the graph with ND.default_graph() (the same
        chain the tabs run) and evaluate it."""
        self.graph = ND.default_graph()
        self.g_sel = None
        self.g_show_params(None)
        self.g_redraw()
        self.g_eval()

    def g_save(self):
        """Button "Save graph": write the graph (nodes, links, positions,
        parameters) to a JSON file chosen by the user."""
        p = filedialog.asksaveasfilename(title=t('g.save'), defaultextension='.json',
                                         initialfile='graph.json',
                                         filetypes=[('JSON', '*.json'), ('*', '*.*')])
        if not p:
            return
        with open(p, 'w', encoding='utf-8') as f:
            f.write(self.graph.to_json())
        self.g_status.config(text=t('m.ext.saved', os.path.basename(p)))

    def g_load(self):
        """Button "Load graph": read a graph from a JSON file, replacing the
        current one, and evaluate it. A file that cannot be read or parsed
        gives an error box and leaves the current graph as it was."""
        p = filedialog.askopenfilename(title=t('g.load'),
                                       filetypes=[('JSON', '*.json'), ('*', '*.*')])
        if not p:
            return
        try:
            with open(p, encoding='utf-8') as f:
                self.graph.from_json(f.read())
        except Exception as ex:
            messagebox.showerror(t('com.notopen'), str(ex))
            return
        self.g_sel = None
        self.g_show_params(None)
        self.g_redraw()
        self.g_eval()

    @staticmethod
    def g_write(v, p, compress):
        """Write the value of an output node to a file: normal, _nh (RGBA),
        colour or grey.

        v: float32 array. H×W×4 is written as RGBA 0…1 (normal + height in
        alpha, the _nh map); H×W×3 with values below -0.01 is taken as a signed
        normal in -1…1 and written by FO.save_normal_dds; other H×W×3 as colour
        0…1; H×W as grey 0…1. Values are clipped to 0…1 where applicable.
        p: output path; compress: bool, DDS compression.
        """
        if v.ndim == 3 and v.shape[2] == 4:
            save_rgba((np.clip(v, 0, 1) * 255 + 0.5).astype(np.uint8), p, compress,
                      normal=True)
        elif v.ndim == 3 and float(v.min()) < -0.01:
            FO.save_normal_dds(v, p, compress)
        elif v.ndim == 3:
            save_rgba(rgb_rgba(np.clip(v, 0, 1)), p, compress)
        else:
            save_rgba(gray_rgba(np.clip(v, 0, 1)), p, compress)

    def g_save_maps(self):
        """Save everything connected to the output nodes, under their names.

        "Save outputs" button. The user picks a base file name; each output
        node writes <stem><node name>.dds via g_write. Unlike the preview in
        g_eval this evaluates at full size and in the main thread. Failures
        are collected and shown together in one error box.
        """
        outs = self.graph.outputs()
        if not outs or self.rgb is None:
            return
        base = filedialog.asksaveasfilename(
            title=t('g.saveout'), defaultextension='.dds',
            initialfile=os.path.splitext(os.path.basename(self.path or 'texture'))[0],
            filetypes=[('DirectDraw Surface', '*.dds'), ('*', '*.*')])
        if not base:
            return
        stem = os.path.splitext(base)[0]
        src = self.rgb[..., :3]
        ctx = {'src': np.ascontiguousarray(src), 'key': 'full' + str(id(self.rgb))}
        done, bad = [], []
        for nid in outs:
            name = self.graph.nodes[nid]['params']['name']
            try:
                v = XPU.to_host(self.graph.eval(nid, ctx).get('any'))
                if v is None:
                    bad.append('%s — %s' % (name, t('g.novalue')))
                    continue
                self.g_write(v, stem + name + '.dds', self.m_compress.get())
                done.append(stem + name + '.dds')
            except Exception as ex:
                bad.append('%s — %s' % (name, ex))
        self.g_status.config(text=t('com.saved.n', len(done)))
        if bad:
            messagebox.showerror(t('g.saveout'), '\n'.join(bad))

    def g_batch(self):
        """Run the graph over every texture in a folder.

        "Batch" button of the graph tab. The graph is copied (through JSON), so
        edits on screen do not leak into a running batch; neural networks in
        it are recomputed for each texture. Sources are .dds/.png/.tga files
        whose names do not end with an output-node name or a known map suffix
        (MAP_SUFFIXES). Each output node writes <stem><name>.dds next to the
        source. Runs in a worker thread that only sets self._bat_msg,
        self._bat_prog and self._bat_out; _bat_start/_bat_poll show them.
        """
        if not self.graph.outputs():
            messagebox.showinfo(t('com.nothing'), t('g.noout'))
            return
        d = filedialog.askdirectory(title=t('com.folder'))
        if not d:
            return
        g = ND.Graph()
        g.from_json(self.graph.to_json())
        comp = self.m_compress.get()
        self._bat_start(self.g_status)

        def work():
            """Worker thread body: process the files one by one and store
            (number done, list of 'file — error') in self._bat_out."""
            # готовые карты (свои и чужие) исходниками не считаются, иначе
            # второй прогон по той же папке обрабатывал бы собственный вывод
            names = {g.nodes[n]['params']['name'] for n in g.outputs()} | set(MAP_SUFFIXES)
            files = [f for f in sorted(os.listdir(d))
                     if f.lower().endswith(('.dds', '.png', '.tga'))
                     and not any(os.path.splitext(f)[0].lower().endswith(x.lower())
                                 for x in names)]
            done, bad = 0, []
            nf = max(len(files), 1)
            for i, f in enumerate(files):
                self._bat_msg = '%d/%d  %s' % (i + 1, len(files), f)
                self._bat_prog = i / nf
                try:
                    im = Image.open(os.path.join(d, f))
                    im.load()
                    rgb = np.array(im.convert('RGB')).astype(np.float32) / 255

                    def sub(a, b, i=i, f=f):
                        """Progress callback: step a of b within file i."""
                        self._bat_prog = (i + a / max(b, 1) * 0.9) / nf
                    ctx = {'src': rgb, 'sub': sub, 'key': 'batch:' + f,
                           'name': os.path.basename(f)}
                    stem = os.path.join(d, os.path.splitext(f)[0])
                    for nid in g.outputs():
                        v = g.eval(nid, ctx, lambda k, i=i, f=f: setattr(
                            self, '_bat_msg', '%d/%d  %s  %s' % (
                                i + 1, len(files), f, t('nd.' + g.nodes[k]['type']))))
                        v = XPU.to_host(v.get('any'))
                        if v is not None:
                            self.g_write(v, stem + g.nodes[nid]['params']['name'] + '.dds',
                                         comp)
                    done += 1
                except Exception as e:
                    bad.append('%s — %s' % (f, e))
            self._bat_prog = 1.0
            self._bat_out = (done, bad)
        threading.Thread(target=work, daemon=True).start()

    # ------------------------------------------------------ вкладка 3D ----
    def build_view3d_tab(self):
        """Build the 3D tab: camera (pitch 0…83°, yaw ±68°, distance 1.2…9 plane
        sizes, 1…8 tiles), parallax mode (OpenMW by default — exactly what the
        engine will show — occlusion, offset, flat) with depth in % of a tile
        and ray steps, self-shadowing, grid, render resolution (240…2400 px)
        and flat-vs-mode comparison; then the Stage3D canvas and the view
        column with the light ring, hemispherical ambient and light spin.
        The maps come from the Material tab (see v3_maps).
        """
        sp = ScrollPanel(self.tab_v)
        sp.pack(side='left', fill='y')
        panel = sp.inner
        f = lambda: self.v3_redraw(False)

        left = sec(panel, 'v3.cam')
        self.v_pitch = Slider(left, t('v3.pitch'), 0, 83, 57, '', self.v3_from_sliders,
                              integer=True)
        self.v_yaw = Slider(left, t('v3.yaw'), -68, 68, 17, '', self.v3_from_sliders,
                            integer=True)
        self.v_dist = Slider(left, t('v3.dist'), 1.2, 9.0, 2.5, '', self.v3_from_sliders,
                             fmt='%.1f')
        self.v_tiles = Slider(left, t('v3.tiles'), 1, 8, 3, t('v3.tiles.info'), f,
                              integer=True)

        left = sec(panel, 'v3.par')
        # по умолчанию — ровно то, что покажет OpenMW
        self.v_mode = tk.StringVar(value='openmw')
        for key, val in (('v3.m.openmw', 'openmw'), ('v3.m.occlusion', 'occlusion'),
                         ('v3.m.offset', 'offset'), ('v3.m.flat', 'flat')):
            row = ttk.Frame(left)
            row.pack(fill='x')
            ttk.Radiobutton(row, text=t(key), value=val, variable=self.v_mode,
                            command=f).pack(side='left')
            Info(row, t(key + '.info')).pack(side='right')
        self.v_depth = Slider(left, t('v3.depth'), 0, 25, 3, t('v3.depth.info'), f,
                              integer=True)
        self.v_steps = Slider(left, t('v3.steps'), 6, 64, 24, t('v3.steps.info'), f,
                              integer=True)
        self.v_shadow = tk.BooleanVar(value=True)
        self.v_shadowrow = check(left, 'v3.shadow', self.v_shadow, f)
        self.v_grid = tk.BooleanVar(value=False)
        check(left, 'v3.grid', self.v_grid, f)

        left = sec(panel, 'v3.q')
        self.v_res = Slider(left, t('v3.res'), 240, 2400, 1600, t('v3.res.info'), f,
                            integer=True)
        self.v_cmp = tk.BooleanVar(value=False)
        check(left, 'v3.cmp', self.v_cmp, f)

        right = ttk.Frame(self.tab_v)
        right.pack(side='left', fill='both', expand=True)
        self.v_view = Stage3D(right, self.v3_dragged)
        self.v_view.pack(side='left', fill='both', expand=True)
        side = ttk.Frame(right, padding=(8, 0, 0, 0))
        side.pack(side='left', fill='y')
        ttk.Label(side, text=t('com.show'), style='Head.TLabel').pack(anchor='w')
        self.ring_v = LightRing(side, lambda: self.v3_redraw(True))
        self.ring_v.pack(pady=4)
        check(side, 'com.hemi', self.amb_hemi, lambda: self.v3_redraw(False))
        self.spin_v = tk.BooleanVar(value=False)
        ttk.Checkbutton(side, text=t('com.spin'), variable=self.spin_v,
                        command=lambda: self.ring_v.toggle_spin(self.spin_v.get())).pack(anchor='w')

    def v3_from_sliders(self):
        """Camera sliders changed: copy pitch/yaw (degrees → radians) and
        distance into the Stage3D camera and redraw.

        The sliders and the mouse move the same camera, so they are kept in
        agreement (the reverse direction is v3_dragged).
        """
        self.v_view.pitch = math.radians(self.v_pitch.get())
        self.v_view.yaw = math.radians(self.v_yaw.get())
        self.v_view.dist = self.v_dist.get()
        self.v3_redraw(False)

    def v3_dragged(self, live):
        """Stage3D on_change callback (mouse drag, wheel, canvas resize): copy
        the camera back into the sliders and redraw; live=True while dragging
        asks for a draft frame (see v3_redraw)."""
        self.v_pitch.set(round(math.degrees(self.v_view.pitch)))
        self.v_yaw.set(round(math.degrees(self.v_view.yaw)))
        self.v_dist.set(self.v_view.dist)
        self.v3_redraw(live)

    def v3_maps(self, full=False):
        """Return the maps for the 3D view, taken from the Material tab; if
        there are none yet, compute them.

        full: bool, want full-size maps. The material preview is computed on
        a half-size copy, and close up the 3D view looks blurry because of it.
        The full size is computed in a separate thread, once, after the
        sliders stop, and kept until the settings change; until then the
        preview maps are returned. The cache key covers the texture, all
        material parameters, tiling, external depth, height hint, repair
        state and ESRGAN results. Everything that reads the window (Tk
        variables, masks) is gathered here in the main thread; the thread only
        runs numpy (MaterialJob.all_maps) and writes self._v3_full/_v3_key,
        and _v3_wait polls for it. With the second zone on, only the preview is
        used, because the zone mask is read from the window.
        Returns the maps dict of MaterialJob.all_maps, or None when no texture
        or no material job is loaded.
        """
        if self.rgb is None or self.mat is None:
            return None
        if self.maps is None:
            band = self.m_band.get() if self.m_seam.get() else 0
            self.maps = self.build_maps(self.mat, band)
        if not full or self.mat.src.shape[:2] == self.rgb.shape[:2]:
            return self.maps
        key = (id(self.rgb), repr(sorted(self.mat_params().items())),
               self.m_seam.get(), self.m_band.get(), self.z_on.get(),
               id(getattr(self, 'extern_full', None)),
               float(self.extern_full.sum()) if getattr(self, 'extern_full', None)
               is not None else 0.0,
               float(np.abs(self.guide).sum()) if self.guide is not None else 0.0,
               self.r_on.get(), getattr(self, '_dmg_key', None),
               0 if self.dmg_hand is None else int(self.dmg_hand.sum()),
               0 if self.dmg_kill is None else int(self.dmg_kill.sum()),
               tuple(id(v) for v in (getattr(self, '_esr_raw', None) or {}).values()))
        if getattr(self, '_v3_key', None) == key and getattr(self, '_v3_full', None):
            return self._v3_full
        if self.z_on.get():
            return self.maps          # маска второй зоны читает окно — только превью
        if getattr(self, '_v3_job', None) != key:
            self._v3_job = key
            self._v3_full = None
            # всё, что читает окно, — здесь, в главном потоке; поток считает
            # только numpy
            p = self.mat_params()
            band = self.m_band.get() if self.m_seam.get() else 0
            shape = self.rgb.shape[:2]
            src = self.repaired(self.rgb)
            ext, esr, guide = self._extern_for(shape), self._esr_for(shape), self.guide

            def work(key=key):
                """Worker thread: compute full-size maps (None on error) and
                store them only if key is still the current job."""
                try:
                    j = MG.MaterialJob(src)
                    j.set_extern(ext)
                    j.set_guide(guide)
                    j.set_esr(*esr)
                    m = j.all_maps(p, band)
                except Exception:
                    m = None
                if self._v3_job == key:
                    self._v3_full, self._v3_key = m, key
            threading.Thread(target=work, daemon=True).start()
            self._v3_wait()
        return self.maps

    def _v3_wait(self):
        """Wait for the full-size maps and redraw.

        Polled from the main thread every 300 ms via after() until the worker
        started by v3_maps has stored the maps for the current job key.
        """
        if getattr(self, '_v3_key', None) != getattr(self, '_v3_job', None):
            self.root.after(300, self._v3_wait)
            return
        self.v3_redraw(False)

    def v3_frame(self, w, h, mode, coarse):
        """Render one 3D frame with the current camera, light and settings.

        w, h: frame size in pixels; mode: 'openmw', 'occlusion', 'offset' or
        'flat'; coarse: draft frame — preview maps instead of full-size, half
        the ray steps (at least 6) and no self-shadowing. Depth slider % is
        passed as a fraction of a tile. Returns V3.render's h×w×3 float32
        image in 0…1.
        """
        return V3.render(self.v3_maps(full=not coarse), w, h,
                         yaw=self.v_view.yaw, pitch=self.v_view.pitch,
                         dist=self.v_view.dist, tiles=float(self.v_tiles.get()),
                         depth=self.v_depth.get() / 100.0,
                         steps=max(6, self.v_steps.get() // (2 if coarse else 1)),
                         mode=mode, shadow=self.v_shadow.get() and not coarse,
                         hemi=self.amb_hemi.get(),
                         light=tuple(self.ring_v.vector()), grid=self.v_grid.get())

    def v3_redraw(self, live=False):
        """Redraw the 3D view (only when the 3D tab is the current one).

        live: True while rotating — the frame is computed smaller and without
        shadow, and the proper frame is drawn a fraction of a second (220 ms)
        after the motion stops. The same draft mode is used while the camera
        or the light ring is being dragged. Frame width is the resolution
        slider capped by the canvas width, height follows the canvas aspect.
        With comparison on, the left half is 'flat' and the right half the
        chosen mode. Also greys out the depth and steps sliders in OpenMW mode
        (there the depth comes from the _nh alpha, not the slider) and the
        shadow box outside occlusion mode.
        """
        if self.nb.index('current') != 3:
            return
        if self.v3_maps() is None:
            self.v_view.note(t('v3.nomaps'))
            return
        coarse = bool(live or self.v_view.dragging or LightRing.DRAG)
        cw = max(self.v_view.winfo_width(), 120)
        ch = max(self.v_view.winfo_height(), 90)
        w = min(int(self.v_res.get()), cw)
        h = max(90, int(round(w * ch / max(cw, 1))))
        if coarse:
            w, h = max(160, w // 2), max(120, h // 2)
        if self.v_cmp.get():
            a = self.v3_frame(w // 2, h, 'flat', coarse)
            b = self.v3_frame(w // 2, h, self.v_mode.get(), coarse)
            img = np.concatenate([a, b], axis=1)
            img[:, w // 2 - 1:w // 2 + 1] = 0.35
        else:
            img = self.v3_frame(w, h, self.v_mode.get(), coarse)
        self.v_view.show(img)
        # в режиме движка глубину задаёт альфа _nh, а не ползунок
        om = self.v_mode.get() == 'openmw'
        self.v_depth.enable(not om)
        self.v_steps.enable(not om)
        for c in self.v_shadowrow.winfo_children():
            if isinstance(c, ttk.Checkbutton):
                c.state(['disabled'] if self.v_mode.get() != 'occlusion' else ['!disabled'])
        if getattr(self, 'v_fine', None):
            self.root.after_cancel(self.v_fine)
            self.v_fine = None
        if coarse and not self.v_view.dragging:
            self.v_fine = self.root.after(220, lambda: self.v3_redraw(False))

    # ------------------------------------------- вкладка правки нормалей ---
    def build_fix_tab(self):
        """Build the "Normal map repair" tab: what to fix (rebuild, renormalise,
        clamp with minimum Z 0.05…0.6, flip green), normal ↔ height conversion
        (source kind, DirectX, low-frequency cut 0…0.8, amplitude 1…200, save
        converted), output (DXT compression, keep alpha, overwrite, save,
        batch); then the view and the side column with the text report and the
        view mode (before/after, after, converted, back-converted). The fix
        check boxes are kept in self.x_fixboxes so they can be greyed out when
        a height map rather than a normal is open.
        """
        sp = ScrollPanel(self.tab_x)
        sp.pack(side='left', fill='y')
        panel = sp.inner
        self.panel_head(panel, 'x')
        f = self.schedule

        left = sec(panel, 'x.what')
        self.x_fixboxes = []            # гасятся, когда открыта высота, а не нормаль
        self.x_rebuild = tk.BooleanVar(value=False)
        self.x_fixboxes.append(check(left, 'x.rebuild', self.x_rebuild, f))
        self.x_renorm = tk.BooleanVar(value=True)
        self.x_fixboxes.append(check(left, 'x.renorm', self.x_renorm, f))
        self.x_clamp = tk.BooleanVar(value=True)
        cb = ttk.Checkbutton(left, text=t('x.clamp'), variable=self.x_clamp, command=f)
        cb.pack(anchor='w', pady=(6, 0))
        self.x_fixboxes.append(cb)
        self.x_minz = Slider(left, t('x.minz'), 0.05, 0.6, 0.2, t('x.minz.info'), f, adv=True)
        self.x_flip = tk.BooleanVar(value=False)
        cb = ttk.Checkbutton(left, text=t('x.flip'), variable=self.x_flip, command=f)
        cb.pack(anchor='w', pady=(6, 0))
        self.x_fixboxes.append(cb)

        # ---- нормаль ↔ высота ----
        left = sec(panel, 'x.conv')
        ttk.Label(left, text=t('x.conv.src')).pack(anchor='w')
        self.x_src = tk.StringVar(value=t('x.src.normal'))
        cbs = ttk.Combobox(left, values=[t('x.src.' + k) for k in X_SRC],
                           textvariable=self.x_src, state='readonly')
        cbs.pack(fill='x', pady=(2, 0))
        cbs.bind('<<ComboboxSelected>>', lambda e: self.x_src_changed())
        self.x_dx = tk.BooleanVar(value=True)
        check(left, 'x.conv.dx', self.x_dx, f)
        self.x_cut = Slider(left, t('x.conv.cut'), 0.0, 0.8, 0.0, t('x.conv.cut.info'), f)
        self.x_amp = Slider(left, t('x.conv.amp'), 1, 200, 20, t('x.conv.amp.info'), f,
                            integer=True)
        self.x_convnote = ttk.Label(left, style='Hint.TLabel', wraplength=280)
        self.x_convnote.pack(anchor='w', pady=(4, 0))
        ttk.Button(left, text=t('x.conv.save'),
                   command=self.save_conv).pack(fill='x', pady=(6, 0))

        left = sec(panel, 'f.out')
        self.x_compress = tk.BooleanVar(value=False)
        ttk.Checkbutton(left, text=t('com.dxt'), variable=self.x_compress).pack(anchor='w')
        self.x_keepalpha = tk.BooleanVar(value=True)
        ttk.Checkbutton(left, text=t('x.keepalpha'),
                        variable=self.x_keepalpha).pack(anchor='w')
        self.x_over = tk.BooleanVar(value=False)
        check(left, 'x.over', self.x_over)
        ttk.Button(left, text=t('x.savefixed'),
                   command=self.save_fixed_normal).pack(fill='x', pady=(6, 0))
        ttk.Button(left, text=t('com.batch'),
                   command=lambda: self.batch('fixnormal')).pack(fill='x', pady=(6, 0))
        self.x_status = ttk.Label(left, style='Hint.TLabel', wraplength=280)
        self.x_status.pack(anchor='w', pady=8)

        right = ttk.Frame(self.tab_x)
        right.pack(side='left', fill='both', expand=True)
        self.x_view = ZoomView(right)
        self.x_view.pack(side='left', fill='both', expand=True)
        side = ttk.Frame(right, padding=(8, 0, 0, 0))
        side.pack(side='left', fill='y')
        ttk.Label(side, text=t('x.report'), style='Head.TLabel').pack(anchor='w')
        self.x_text = tk.Text(side, width=44, height=26, background='#1b1e21',
                              foreground=FG, insertbackground=FG, relief='flat',
                              font=('TkFixedFont', 8), wrap='word')
        self.x_text.pack(fill='y', expand=True, pady=4)
        self.x_mode = tk.StringVar(value='both')
        for key, val in (('x.v.both', 'both'), ('x.v.after', 'after'),
                         ('x.v.conv', 'conv'), ('x.v.back', 'back')):
            ttk.Radiobutton(side, text=t(key), value=val, variable=self.x_mode,
                            command=self.x_mode_changed).pack(anchor='w')

    # =================================================== параметры и карты ==
    def apply_preset(self, display, target='base'):
        """Material preset combobox: set the height, normal, AO, specular and
        de-lighting sliders from MG.PRESETS and schedule a recompute.

        display: the translated preset name as shown in the combobox (an
        unknown name does nothing). Maximum tilt is always reset to 75.
        target: accepted but not used; the second zone's preset is read
        separately, in zone_params.
        """
        pid = next((k for k in MG.PRESET_IDS if t('p.' + k) == display), None)
        if pid is None:
            return
        p = MG.PRESETS[pid]
        self.m_hdet.set(p['height'][0])
        self.m_hcon.set(p['height'][1])
        self.m_tilt.set(75)
        self.m_hinv.set(p['height'][2])
        self.m_nstr.set(p['nstr'])
        self.m_aor.set(p['ao'][0])
        self.m_aos.set(p['ao'][1])
        f0, gl, gv, me, ti = p['spec']
        self.m_f0.set(f0)
        self.m_gloss.set(gl)
        self.m_gvar.set(gv)
        self.m_metal.set(me)
        self.m_tint.set(ti)
        self.m_soft.set(p['soft'])
        self.m_hard.set(p['hard'])
        self.schedule()

    def mat_params(self):
        """Collect every Material-tab setting into the parameter dict of
        matgen.MaterialJob.all_maps.

        Values are in slider units, at full-size scale (callers rescale
        size-dependent ones with MG.at_scale for the half-size preview); the
        light azimuth is converted from degrees to radians, and comboboxes are
        translated back from their displayed names to language-independent ids
        (with a fallback id if nothing matches). Reads Tk variables, so it must
        be called in the main thread.
        """
        return dict(soft=self.m_soft.get(), hard=self.m_hard.get(),
                    light_radius=self.m_lrad.get(), keep=1.0,
                    restore=self.m_rest.get(), sat_keep=self.m_sat.get(),
                    h_detail=self.m_hdet.get(), h_contrast=self.m_hcon.get(),
                    h_invert=self.m_hinv.get(), h_from_delit=self.m_hsrc.get(),
                    h_amp=float(self.m_hamp.get()), max_tilt=float(self.m_tilt.get()),
                    guided_radius=float(self.m_grad.get()), macro_falloff=self.m_fall.get(),
                    w_band=self.m_wband.get(), w_multi=self.m_wmulti.get(),
                    w_lum=self.m_wlum.get(), w_guided=self.m_wguid.get(),
                    w_ext=self.m_wext.get(), ext_invert=self.m_extinv.get(),
                    ext_cavity=self.m_extcav.get(), ext_cut=self.m_extcut.get(),
                    w_sfs=self.m_wsfs.get(),
                    sfs_az=math.radians(self.m_sfsaz.get()),
                    sfs_smooth=self.m_sfssm.get(),
                    sfs_chroma=self.m_sfschr.get(), sfs_shadow=self.m_sfssh.get(),
                    n_direct=self.m_ndirk.get(),
                    n_direct_mix=self.m_ndir.get(),
                    n_direct_swap=self.m_ndirsw.get(),
                    n_direct_inv=self.m_ndirinv.get(),
                    n_direct_b=next((k for k in NDIR_BLUE
                                     if t('m.ndir.b.' + k) == self.m_ndirb.get()), 'unit'),
                    n_direct_fix=self.m_ndirfix.get(),
                    plx_repeat=float(self.m_plxrep.get()), plx_depth=self.m_plxdep.get(),
                    plx_ref=next((k for k in MG.PARALLAX_REFS
                                  if t('m.plx.r.' + k) == self.m_plxref.get()), 'median'),
                    plx_level=self.m_plxlev.get(), plx_dither=self.m_plxdith.get(),
                    plx_normal=self.m_plxnorm.get(),
                    esr_nmix=self.m_esrnmix.get(), esr_nk=self.m_esrnk.get(),
                    esr_nflip=self.m_esrnflip.get(), w_esr=self.m_wesr.get(),
                    esr_rough=self.m_esrrough.get(),
                    sfs_scale=self.m_sfsscale.get(), sfs_low=self.m_sfslow.get(),
                    sfs_high=self.m_sfshigh.get(),
                    guide_amt=self.g_amt.get(), guide_flat=self.g_flat.get(),
                    ao_radius=float(self.m_aor.get()), ao_strength=self.m_aos.get(),
                    ao_w_cav=self.m_aowc.get(), ao_w_norm=self.m_aown.get(),
                    rd_amt=self.m_rdamt.get(),
                    rd_src=next((k for k in MG.RELIEF_SOURCES
                                 if t('m.rd.s.' + k) == self.m_rdsrc.get()), 'height'),
                    rd_mode=next((k for k in MG.RELIEF_MODES
                                  if t('m.rd.m.' + k) == self.m_rdmode.get()), 'overlay'),
                    rd_scale=self.m_rdscale.get(), rd_low=self.m_rdlow.get(),
                    rd_high=self.m_rdhigh.get(), rd_sat=self.m_rdsat.get(),
                    n_strength=self.m_nstr.get(), directx=self.m_dx.get(),
                    n_fine=self.m_nfine.get(), n_mid=self.m_nmid.get(),
                    n_big=self.m_nbig.get(), n_scale=float(self.m_nscale.get()),
                    n_bent=self.m_nbent.get(),
                    h_gamma=self.m_hgam.get(), ao_gamma=self.m_aogam.get(),
                    ao_contrast=self.m_aocon.get(), rough_gamma=self.m_rgam.get(),
                    rough_contrast=self.m_rcon.get(),
                    f0=self.m_f0.get(), gloss=self.m_gloss.get(),
                    gloss_var=self.m_gvar.get(), metal=self.m_metal.get(),
                    tint=self.m_tint.get())

    def zone_params(self):
        """Parameters of the second zone: the same base, but the material part
        comes from the zone's own preset.

        Replaced from the preset: specular (f0, gloss, gloss variation, metal,
        tint), normal strength, AO radius and strength, height detail and
        contrast. The relief geometry is shared: it is about the surface
        itself, not about what the surface is made of. Without a recognised
        zone preset this is just mat_params().
        """
        p = dict(self.mat_params())
        pid = next((k for k in MG.PRESET_IDS if t('p.' + k) == self.z_preset.get()), None)
        if pid is None:
            return p
        z = MG.PRESETS[pid]
        f0, gl, gv, me, ti = z['spec']
        p.update(f0=f0, gloss=gl, gloss_var=gv, metal=me, tint=ti,
                 n_strength=z['nstr'], ao_radius=float(z['ao'][0]),
                 ao_strength=z['ao'][1], h_detail=z['height'][0],
                 h_contrast=z['height'][1])
        return p

    # ------------------------------------------------------ битые участки --
    def blank(self, shape=None):
        """Return an all-False bool mask of shape (default: the texture's H×W)."""
        shape = shape or self.rgb.shape[:2]
        return np.zeros(shape, bool)

    def damage_mask(self):
        """Mask of damaged areas at full size: what is found automatically, plus
        what is painted in with the brush, minus what is erased.

        The automatic part (RP.exposure_masks: blown-out plus crushed areas,
        thresholds hi/lo, flatness, grow in px — plus widens, minus shrinks)
        is cached under a key of those settings, together with the found
        fractions (self._auto_bc) for the status line. Returns H×W float32 in
        0…1 (also stored in self.dmg), or None without a texture.
        """
        if self.rgb is None:
            return None
        key = (self.r_auto.get(), round(self.r_hi.get(), 3), round(self.r_lo.get(), 3),
               round(self.r_flat.get(), 4), int(self.r_grow.get()))
        if getattr(self, '_dmg_key', None) != key or self.dmg is None:
            if self.r_auto.get():
                b, c = RP.exposure_masks(self.rgb, hi=self.r_hi.get(), lo=self.r_lo.get(),
                                         flat=self.r_flat.get(),
                                         grow=float(self.r_grow.get()))
                self._auto = np.clip(b + c, 0, 1)
                self._auto_bc = (float((b > 0.5).mean()), float((c > 0.5).mean()))
            else:
                self._auto = np.zeros(self.rgb.shape[:2], np.float32)
                self._auto_bc = (0.0, 0.0)
            self._dmg_key = key
        m = self._auto.copy()
        if self.dmg_hand is not None:
            m = np.maximum(m, self.dmg_hand.astype(np.float32))
        if self.dmg_kill is not None:
            m = np.where(self.dmg_kill, 0.0, m)
        self.dmg = m
        return m

    def repaired(self, rgb):
        """The image with its damaged areas repaired, at the requested size.

        rgb: H×W×3 float32 0…1, full size or the half-size preview copy; the
        damage mask is resized to it. Returns rgb unchanged when repair is off
        or nothing is marked. The mix slider blends original and repaired
        (below 1 the area is only toned down, not replaced). Cached by size,
        method, mix and mask state, since repair is slow.
        """
        if not self.r_on.get():
            return rgb
        m = self.damage_mask()
        if m is None or not (m > 0.5).any():
            return rgb
        key = (rgb.shape[:2], self.r_method.get(), round(self.r_mix.get(), 3),
               self._dmg_key,
               0 if self.dmg_hand is None else int(self.dmg_hand.sum()),
               0 if self.dmg_kill is None else int(self.dmg_kill.sum()))
        if self._rep_cache.get('key') == key:
            return self._rep_cache['img']
        mm = resize_to(m, rgb.shape[:2])
        out = RP.repair(rgb, mm, self.r_method.get())
        k = float(self.r_mix.get())
        if k < 1.0:
            # смесь оригинала и починенного: иногда нужно не заменить
            # битое место, а только притушить
            out = np.clip(rgb * (1.0 - k) + out * k, 0, 1).astype(np.float32)
        self._rep_cache = {'key': key, 'img': out}
        return out

    def repair_now(self):
        """Button "Repair": turn repair on, repair and show the result at once
        (switches the view to the repaired image): otherwise it is unclear
        whether anything happened."""
        if self.rgb is None:
            return
        self.r_on.set(True)
        self._rep_cache = {}
        self.apply_repair_now()
        self.view_m.set('fixed')
        self.recompute()

    def save_delit(self):
        """Save the de-lit diffuse — after repair and mixing: that is the
        order in the program, the light is removed from what came out.

        Asks for a path (default <name>_fixed.dds next to the source; DDS or
        PNG) and writes the full-size diffuse from full_maps().
        """
        if self.maps is None:
            self.recompute()
        if self.maps is None:
            return
        p = filedialog.asksaveasfilename(
            title=t('m.rep.savedif'), defaultextension='.dds',
            initialfile=(os.path.splitext(os.path.basename(self.path or 'texture'))[0]
                         + '_fixed.dds'),
            initialdir=os.path.dirname(self.path or ''),
            filetypes=[('DirectDraw Surface', '*.dds'), ('PNG', '*.png'), ('*', '*.*')])
        if not p:
            return
        d = self.full_maps()['diffuse']
        save_rgba(rgb_rgba(d), p, self.m_compress.get())
        self.r_status.config(text=t('m.ext.saved', os.path.basename(p)))

    def repair_changed(self):
        """Callback of the repair settings.

        Finding damaged areas and repairing them takes noticeable time, so
        they are not run while a slider is dragged — only a flag is set that
        they need recomputing (self._need_rep), and the recompute itself runs
        after a pause (schedule(200), which calls apply_repair_now).
        """
        self._rep_cache = {}
        self._need_rep = True
        self.schedule(200)

    def apply_repair_now(self):
        """Recompute the damage mask, show the found percentages (blown-out,
        crushed) in the repair status line — with a note when repair itself is
        off, since detection always runs — and rebuild the material job with
        the repaired source (without recomputing the maps yet)."""
        self._need_rep = False
        if self.rgb is None:
            return
        self.damage_mask()
        b, c = getattr(self, '_auto_bc', (0.0, 0.0))
        txt = t('m.rep.found', 100 * b, 100 * c)
        if not self.r_on.get():
            # самая частая жалоба: «находит, а не чинит». Находит всегда,
            # чинит только с этой галочкой
            txt += '   ' + t('m.rep.off')
        self.r_status.config(text=txt)
        self.rebuild_job(recompute=False)

    # ------------------------------------------------ подсказка по высоте --
    def rebuild_guide(self):
        """Turn the height-hint strokes into areas: the program finds their
        borders by colour itself, a stroke only shows where to look.

        "Raise" strokes (self.g_pos) add +1, "lower" strokes (self.g_neg) add
        -1, each grown into an area by RP.wand with the tolerance and
        contiguity settings; then optionally blurred (softness in px) and
        inverted. The result, H×W float32 in -1…1 (None without strokes), is
        stored in self.guide and passed to the material job at its size.
        """
        if self.rgb is None:
            return
        if (self.g_pos is None or not self.g_pos.any()) and \
           (self.g_neg is None or not self.g_neg.any()):
            self.guide = None
        else:
            tol, cont = self.g_tol.get(), self.g_cont.get()
            g = np.zeros(self.rgb.shape[:2], np.float32)
            if self.g_pos is not None and self.g_pos.any():
                g += RP.wand(self.rgb, self.g_pos, tol, cont, 0.0)
            if self.g_neg is not None and self.g_neg.any():
                g -= RP.wand(self.rgb, self.g_neg, tol, cont, 0.0)
            s = float(self.g_soft.get())
            if s > 0:
                g = RP._blur(g, s)
            self.guide = np.clip(-g if self.g_inv.get() else g, -1, 1)
        if self.mat is not None:
            self.mat.set_guide(resize_to(self.guide, self.mat.src.shape[:2])
                               if self.guide is not None else None)
        self.schedule(30)

    def clear_guide(self):
        """Button "Clear" of the height hint: drop all strokes and the hint."""
        self.g_pos = self.g_neg = self.guide = None
        if self.mat is not None:
            self.mat.set_guide(None)
        self.schedule(30)

    # ------------------------------------------------------- маска зоны ----
    def zone_mask(self, shape, maps=None):
        """Mask of the second material zone at the given size.

        shape: (H, W) wanted. The base is a soft range lo…hi of the chosen
        channel (luma, R/G/B, alpha, saturation, hue… of the texture, or the
        computed height or AO when maps is given). Brush strokes are grown
        into areas by RP.wand and added; erased strokes are cut out; then
        optional inversion and blur (px). Returns H×W float32 in 0…1.
        Reads Tk variables — main thread only.
        """
        chan = next((c for c in CHANNELS if t('ch.' + c) == self.z_chan.get()), 'luma')
        rgb = resize_to3(self.rgb, shape)
        if chan == 'height' and maps is not None:
            x = resize_to(maps['height'], shape)
            m = RP.channel_mask(np.stack([x] * 3, -1), 'luma', self.z_lo.get(),
                                self.z_hi.get(), self.z_soft.get())
        elif chan == 'ao' and maps is not None:
            x = resize_to(maps['ao'], shape)
            m = RP.channel_mask(np.stack([x] * 3, -1), 'luma', self.z_lo.get(),
                                self.z_hi.get(), self.z_soft.get())
        else:
            a = resize_to(self.alpha, shape) if self.alpha is not None else None
            m = RP.channel_mask(rgb, chan, self.z_lo.get(), self.z_hi.get(),
                                self.z_soft.get(), a)
        if self.z_hand is not None and self.z_hand.any():
            region = RP.wand(self.rgb, self.z_hand, self.g_tol.get(), self.g_cont.get(), 0.0)
            m = np.maximum(m, resize_to(region, shape))
        if self.z_kill is not None and self.z_kill.any():
            m = np.where(resize_to(self.z_kill, shape, nearest=True), 0.0, m)
        if self.z_inv.get():
            m = 1.0 - m
        b = float(self.z_blur.get())
        if b > 0:
            m = RP._blur(m, b)
        return np.clip(m, 0, 1)

    def clear_zone(self):
        """Button "Clear" of the zone mask: drop the painted and erased strokes."""
        self.z_hand = self.z_kill = None
        self.schedule(30)

    # ---------------------------------------------------------- кисти ------
    BRUSHES = ('dbrush_on', 'gbrush_on', 'zbrush_on', 'rbrush_on')

    def pick_brush(self, which):
        """Brush check box callback: there is one brush in this tab, so turning
        one on turns the others off.

        which: attribute name from BRUSHES (depth, height hint, zone, damage).
        Connects the matching paint_* handler to the material view, or
        disconnects painting when the box was turned off.
        """
        on = getattr(self, which).get()
        for name in self.BRUSHES:
            if name != which:
                getattr(self, name).set(False)
        cb = {'dbrush_on': self.paint_depth, 'gbrush_on': self.paint_guide,
              'zbrush_on': self.paint_zone, 'rbrush_on': self.paint_damage}[which]
        self.m_view.paint_on = bool(on)
        self.m_view.paint_cb = cb if on else None

    def _mfull(self, x, y):
        """Convert preview pixel coordinates to full-size coordinates.

        The preview may be the half-size copy; returns (x, y) as ints, or the
        input unchanged when nothing is loaded.
        """
        if self.mat is None or self.rgb is None:
            return x, y
        k = self.rgb.shape[0] / max(self.mat.src.shape[0], 1)
        return int(x * k), int(y * k)

    @staticmethod
    def _disc(arr, x, y, r, value):
        """Set a filled disc of radius r px centred at (x, y) in the 2-D mask
        arr to value, in place, clipped at the borders (no wrap-around).
        Returns False if the centre is outside arr, else True."""
        h, w = arr.shape
        if not (0 <= x < w and 0 <= y < h):
            return False
        y0, y1 = max(0, y - r), min(h, y + r + 1)
        x0, x1 = max(0, x - r), min(w, x + r + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        arr[y0:y1, x0:x1][(xx - x) ** 2 + (yy - y) ** 2 <= r * r] = value
        return True

    def paint_guide(self, x, y, erase=False, final=False):
        """Height-hint brush (paint_cb of the material view).

        x, y: brush position from the view, converted to full size by _mfull;
        left button paints "raise" (self.g_pos), right button (erase=True)
        paints "lower" (self.g_neg). The radius slider is in preview pixels
        and is scaled to full size. On release (final) the hint is rebuilt.
        """
        if self.rgb is None:
            return
        if self.g_pos is None:
            self.g_pos, self.g_neg = self.blank(), self.blank()
        x, y = self._mfull(x, y)
        r = max(1, int(self.gbrush_r.get() * self.rgb.shape[0] /
                       max(self.mat.src.shape[0], 1)))
        self._disc(self.g_neg if erase else self.g_pos, x, y, r, True)
        if final:
            self.rebuild_guide()

    def paint_zone(self, x, y, erase=False, final=False):
        """Second-zone brush (paint_cb of the material view): left button adds
        to the zone (self.z_hand), right button (erase) cuts out of it
        (self.z_kill). Coordinates and radius as in paint_guide; on release a
        recompute is scheduled."""
        if self.rgb is None:
            return
        if self.z_hand is None:
            self.z_hand, self.z_kill = self.blank(), self.blank()
        x, y = self._mfull(x, y)
        r = max(1, int(self.zbrush_r.get() * self.rgb.shape[0] /
                       max(self.mat.src.shape[0], 1)))
        self._disc(self.z_kill if erase else self.z_hand, x, y, r, True)
        if final:
            self.schedule(30)

    def paint_damage(self, x, y, erase=False, final=False):
        """Damage brush (paint_cb of the material view): left button marks
        damaged areas (self.dmg_hand), right button (erase) unmarks them
        (self.dmg_kill), overriding the automatic detection. Coordinates and
        radius as in paint_guide. While painting only the view is redrawn;
        on release repair_changed schedules the (slow) repair."""
        if self.rgb is None:
            return
        if self.dmg_hand is None:
            self.dmg_hand, self.dmg_kill = self.blank(), self.blank()
        x, y = self._mfull(x, y)
        r = max(1, int(self.dbrush_r.get() * self.rgb.shape[0] /
                       max(self.mat.src.shape[0], 1)))
        self._disc(self.dmg_kill if erase else self.dmg_hand, x, y, r, True)
        if final:
            self.repair_changed()
        else:
            self.redraw()

    # ------------------------------------------------- внешняя глубина -----
    def _set_extern(self, arr, label):
        """Set the external depth layer and schedule a recompute.

        arr: full-size H×W depth, clipped to 0…1 as float32 and kept in
        self.extern_full (None removes the layer). label: text shown in the
        external-depth section. If the layer's weight is 0 it is raised to 1,
        otherwise a loaded depth would seem to do nothing. The material job
        gets the layer at its own (preview) size.
        """
        self.extern_full = None if arr is None else np.clip(arr.astype(np.float32), 0, 1)
        self.m_extname.config(text=label)
        if self.extern_full is not None and self.m_wext.get() <= 0:
            self.m_wext.set(1.0)
        if self.rgb is not None and self.mat is not None:
            self.mat.set_extern(self._extern_for(self.mat.src.shape[:2]))
        self.schedule(30)

    def apply_depth_channel(self):
        """Channel combobox of the external depth: take the chosen channel of
        the loaded image (self.extern_raw) as depth — luma (Rec. 709 weights),
        R, G, B or A (all ones if the image has no alpha) — and set it via
        _set_extern. Does nothing when no external image is loaded."""
        raw = getattr(self, 'extern_raw', None)
        if raw is None:
            return
        ch = next((c for c in EXT_CHANNELS if t('ch.' + c) == self.m_extch.get()), 'luma')
        if ch == 'luma':
            a = 0.2126 * raw[..., 0] + 0.7152 * raw[..., 1] + 0.0722 * raw[..., 2]
        elif ch == 'A':
            a = raw[..., 3] if raw.shape[2] > 3 else np.ones(raw.shape[:2], np.float32)
        else:
            a = raw[..., 'RGB'.index(ch)]
        self._set_extern(a, '%s   %s %s' % (getattr(self, 'extern_name', ''),
                                            t('m.ext.chan'), t('ch.' + ch)))

    def depth_from_height(self):
        """Take the current computed height into the external layer, so it can
        be edited further with the brush without touching what it was derived
        from.

        The height is resized to full size and set as external depth with
        weight 1; the weights of the luminance-derived layers (band, multi,
        luminance, guided) are set to 0, otherwise the height would be applied
        twice.
        """
        if self.maps is None:
            return
        h = resize_to(self.maps['height'], self.rgb.shape[:2])
        self.extern_raw = None
        self.extern_src = None
        self.extern_name = t('m.ext.taken')
        self.m_wext.set(1.0)
        # веса выведенных из яркости слоёв обнуляем: иначе высота ляжет дважды
        for sl in (self.m_wband, self.m_wmulti, self.m_wlum, self.m_wguid):
            sl.set(0.0)
        self._set_extern(h, self.extern_name)

    # ------------------------------------------------------ нейросеть -----
    def net_id(self):
        """Return the id (DN.MODEL_IDS) of the depth model chosen in the
        combobox; the first model if the displayed name matches none."""
        return next((k for k in DN.MODEL_IDS if t('nn.' + k) == self.m_netmodel.get()),
                    DN.MODEL_IDS[0])

    def net_note(self):
        """Update the line under the network button: which model, its licence,
        how much it weighs (MiB), whether it is downloaded, and what the chosen
        mode will cost.

        Better to say it in advance than to confront the user with a
        hundred-megabyte download on a click. In fine (tiled) mode the number
        of network passes is shown: tiles of tile-fraction × shorter side
        (at least 128 px) with 50 % overlap, squared. Ends with the device
        (net_device). Does nothing before the label exists.
        """
        if getattr(self, 'm_netinfo', None) is None:
            return
        k = self.net_id()
        m = DN.MODELS[k]
        state = t('m.nn.ready') if DN.have_model(k) else \
            t('m.nn.need', m['size'] / (1 << 20))
        if self.m_netfine.get() and self.rgb is not None:
            side = min(self.rgb.shape[:2])
            tile = max(128, int(side * self.m_nettile.get()))
            n = (int(np.ceil(side / max(8, tile * 0.5))) ** 2)
            state += '   ' + t('m.nn.passes', n)
        self.m_netinfo.config(text='%s, %s.   %s' % (m['license'], state,
                                                     self.net_device()))

    def net_device(self):
        """Return the translated text of what the network runs on (GPU with
        the provider's short name, or CPU).

        The provider is asked from the session itself: it may be listed in
        the build but fail to start, and onnxruntime then silently falls back
        to the CPU. Before a session exists, the first available provider is
        shown.
        """
        p = DN.provider_name(self.net_id())
        if p:
            short = DN.provider_short(p)
        else:
            # сессии ещё нет: говорим по выбранной сборке, не загружая
            # onnxruntime (загруженную сборку нельзя сменить до перезапуска)
            short = {'webgpu': 'WebGPU', 'cuda': 'CUDA',
                     'directml': 'DirectML'}.get(DN.runtime_kind(), '')
        return t('m.nn.dev.gpu', short) if short else t('m.nn.dev.cpu')

    def net_depth(self, foliage=False):
        """Compute depth with the neural network — for the material or for
        the foliage.

        Button of the Material tab (foliage=False) or of the "Foliage: light,
        AO" tab (foliage=True). Asks first before installing the runtime and
        before downloading the model (title, size in MiB, licence). Everything
        long (installing the library, downloading the model, the computation
        itself) goes to a separate thread (_net_worker), otherwise the window
        freezes and looks hung; _net_poll watches it from the main thread.
        For foliage the transparent parts are filled with flat 50 % grey, so
        the leaf reads as an object on a neutral field instead of a hole in
        depth; for the material the repaired source is used. Remembers which
        texture it was started for (self._net_tex), so a result for a texture
        that has since changed is discarded. Ignored while a run is going.
        """
        if self.rgb is None or getattr(self, '_net_busy', False):
            return
        k = self.net_id()
        m = DN.MODELS[k]
        if not DN.have_runtime():
            # сначала выбрать, где считать (процессор или видеокарта), и
            # поставить; сам счёт — следующим нажатием
            EDM.runtime_dialog(self.root, lambda ok: self.net_note())
            return
        if not DN.have_model(k):
            if not messagebox.askyesno(t('m.nn.title'),
                                       t('m.nn.ask.dl', m['title'],
                                         m['size'] / (1 << 20), m['license'])):
                return
        self._net_busy = True
        self._net_tex = id(self.rgb)          # для какой текстуры считаем
        self._net_for = 'foliage' if foliage else 'material'
        self._net_msg = t('m.nn.working')
        self._net_prog = None
        self._net_out = None
        bar = self.d_netbar if foliage else self.m_netbar
        bar.configure(value=0)
        bar.pack(fill='x', pady=(4, 0))
        self.m_netbtn.config(state='disabled')
        self.d_netbtn.config(state='disabled')
        if foliage:
            # карточке нужен фон: за альфой лежит чёрное или мусор, и сеть
            # честно считает это провалом глубины. Подставляем ровный серый —
            # тогда лист читается как предмет на нейтральном поле
            rgba = self.rgba_full
            a = rgba[..., 3:4]
            src = (rgba[..., :3] * a + 0.5 * (1.0 - a)).astype(np.float32)
        else:
            src = self.repaired(self.rgb)[..., :3].copy()
        fine = bool(self.m_netfine.get())
        tile = int(min(src.shape[:2]) * self.m_nettile.get())
        threading.Thread(target=self._net_worker,
                         args=(k, src, fine, tile), daemon=True).start()
        self._net_poll()

    # ------------------------------------------------------ ESRGAN -------
    def _esr_for(self, shape):
        """ESRGAN outputs brought to the required size (the preview is computed
        on a reduced copy).

        shape: (H, W). Returns [normal, franken], each H×W×3 float32 in 0…1 or
        None if that network has not been run. The normal is renormalised
        after resizing (resizing shortens the vectors).
        """
        raw = getattr(self, '_esr_raw', None) or {}
        out = []
        for k in ('normal', 'franken'):
            a = raw.get(k)
            if a is None:
                out.append(None)
                continue
            if a.shape[:2] != tuple(shape):
                a = resize_to3(a, shape)
            if k == 'normal':
                v = a * 2.0 - 1.0
                v = v / np.maximum(np.sqrt((v * v).sum(-1, keepdims=True)), 1e-6)
                a = (v * 0.5 + 0.5).astype(np.float32)
            out.append(a)
        return out

    def esr_note(self):
        """Update the ESRGAN info line: which models are downloaded, where the
        model folder is, and which have been computed for this texture (with
        the time taken, in seconds)."""
        raw = getattr(self, '_esr_raw', None) or {}
        have = [t('m.esr.k.' + k) for k in ES.MODEL_IDS if ES.have_model(k)]
        done = [t('m.esr.k.' + k) for k in ES.MODEL_IDS if k in raw]
        txt = t('m.esr.have', ', '.join(have) if have else t('m.esr.none'), DN.model_dir())
        if done:
            txt += '\n' + t('m.esr.done', ', '.join(done),
                             getattr(self, '_esr_secs', 0.0))
        self.m_esrinfo.config(text=txt)

    def esr_file(self):
        """Point to a manually downloaded .pth: it is copied into the model
        folder. Needed when direct downloading is blocked.

        The file is checked by loading and describing its weights; which
        network it is ('franken' or 'normal') is taken from the file name. A
        previously converted .onnx of that network is deleted so it is rebuilt
        from the new file, and loaded models are unloaded.
        """
        p = filedialog.askopenfilename(title=t('m.esr.file'),
                                       filetypes=[('PyTorch', '*.pth'), ('*', '*.*')])
        if not p:
            return
        try:
            sd = ES.load_pth(p)
            ES.describe(dict(sd))
        except Exception as e:
            messagebox.showerror(t('m.esr.title'), t('m.esr.bad', os.path.basename(p), e))
            return
        name = os.path.basename(p).lower()
        key = 'franken' if 'franken' in name else 'normal' if 'normal' in name else None
        if key is None:
            messagebox.showerror(t('m.esr.title'), t('m.esr.which', os.path.basename(p)))
            return
        import shutil
        dst = os.path.join(DN.model_dir(), ES.MODELS[key]['file'])
        shutil.copyfile(p, dst)
        try:
            os.remove(ES.onnx_path(key))     # пересобрать из нового файла
        except OSError:
            pass
        ES.unload()
        self.esr_note()

    def esr_run(self):
        """Run the selected ESRGAN networks ('normal', 'franken') over the
        texture — in a separate thread, with a progress bar (see _esr_poll).

        Asks before downloading missing models (titles, licence, author). The
        repaired source is used. Remembers the texture (self._esr_tex), so a
        result that arrives after the texture changed is discarded. Ignored
        while a run is going.
        """
        if self.rgb is None or getattr(self, '_esr_busy', False):
            return
        keys = [k for k, v in (('normal', self.m_esrn_on), ('franken', self.m_esrf_on))
                if v.get()]
        if not keys:
            return
        need = [k for k in keys if not ES.have_model(k)]
        if need and not messagebox.askyesno(
                t('m.esr.title'),
                t('m.esr.ask', ', '.join(ES.MODELS[k]['title'] for k in need),
                  ES.MODELS[need[0]]['license'], ES.MODELS[need[0]]['author'])):
            return
        self._esr_busy = True
        self._esr_tex = id(self.rgb)
        self._esr_msg, self._esr_prog, self._esr_out = t('m.nn.working'), None, None
        self.m_esrbar.configure(value=0)
        self.m_esrbar.pack(fill='x', pady=(4, 0), before=self.m_esrinfo)
        self.m_esrbtn.config(state='disabled')
        src = self.repaired(self.rgb)[..., :3].copy()
        threading.Thread(target=self._esr_worker, args=(keys, src), daemon=True).start()
        self._esr_poll()

    def _esr_worker(self, keys, src):
        """Worker thread of esr_run. Not a single access to the interface —
        only to plain attributes.

        keys: networks to run ('normal', 'franken'); src: H×W×3 float32 0…1.
        Installs onnxruntime/onnx if missing, downloads a missing .pth and
        converts it to ONNX (ES.prepare), then runs each network. Progress
        goes to self._esr_msg and self._esr_prog ((done, total) or None when
        the amount is unknown); the result to self._esr_out: ('ok', {key:
        H×W×3 array}, seconds) or ('err', message). _esr_poll reads them in
        the main thread.
        """
        import time as _t
        try:
            for pkg, mod in (('onnxruntime', 'onnxruntime'), ('onnx', 'onnx')):
                try:
                    __import__(mod)
                except Exception:
                    self._esr_msg = t('m.nn.installing')
                    self._esr_prog = None
                    ok, out = DN.install_runtime(pkg, mod)
                    if not ok:
                        raise RuntimeError(t('m.nn.nort') + '\n\n' + out)
            res, t0 = {}, _t.time()
            for j, k in enumerate(keys):
                if not os.path.isfile(ES.onnx_path(k)):
                    def dl(got, total, k=k):
                        """Download progress: got of total bytes."""
                        self._esr_msg = t('m.esr.loading', ES.MODELS[k]['title'],
                                          got / (1 << 20), total / (1 << 20))
                        self._esr_prog = (got, total)
                    try:
                        if not os.path.isfile(ES.pth_path(k)):
                            ES.download(k, dl)
                    except Exception as e:
                        raise RuntimeError(t('m.esr.nodl', ES.MODELS[k]['page'],
                                             DN.model_dir(), e))
                    self._esr_msg = t('m.esr.convert', ES.MODELS[k]['title'])
                    self._esr_prog = None
                    ES.prepare(k)

                def prog(i, n, j=j, k=k):
                    """Run progress: pass i of n of network j, counted over
                    all selected networks."""
                    self._esr_prog = (j * n + i, n * len(keys))
                    self._esr_msg = t('m.esr.pass', ES.MODELS[k]['title'], i, n)
                res[k] = ES.run(k, src, progress=prog)
            self._esr_out = ('ok', res, _t.time() - t0)
        except Exception as e:
            self._esr_out = ('err', str(e))

    def _esr_poll(self):
        """Main-thread poll (every 250 ms via after) of the ESRGAN worker.

        Shows the worker's status text and progress (a running indeterminate
        bar when the amount is unknown). When done: frees the button, shows an
        error, or — if the texture is still the one the run started for —
        merges the results into self._esr_raw, switches the view to the
        network's output at once (otherwise pressing the button changes
        nothing visible until a slider moves), passes them to the material
        job and schedules a recompute. Results for a texture that has changed
        meanwhile are discarded with a "stale" message.
        """
        self.m_status.config(text=self._esr_msg or '')
        bar = self.m_esrbar
        pr = self._esr_prog
        if pr is None:
            if str(bar.cget('mode')) != 'indeterminate':
                bar.configure(mode='indeterminate')
                bar.start(60)
        else:
            if str(bar.cget('mode')) != 'determinate':
                bar.stop()
                bar.configure(mode='determinate')
            bar.configure(value=int(1000 * pr[0] / max(pr[1], 1)))
        out = self._esr_out
        if out is None:
            self.root.after(250, self._esr_poll)
            return
        self._esr_busy = False
        bar.stop()
        bar.pack_forget()
        self.m_esrbtn.config(state='normal')
        self.m_status.config(text='')
        if out[0] == 'err':
            messagebox.showerror(t('m.esr.title'), out[1])
            self.esr_note()
            return
        if getattr(self, '_esr_tex', None) != id(self.rgb):
            self.m_status.config(text=t('m.nn.stale'))
            return
        self._esr_raw = dict(getattr(self, '_esr_raw', None) or {}, **out[1])
        self._esr_secs = out[2]
        # сразу показать то, что дала сеть: иначе нажатие ничего видимого
        # не меняет, пока не сдвинешь ползунок
        self.view_m.set('esrn' if 'normal' in out[1] else 'esrh')
        self.mat.set_esr(*self._esr_for(self.mat.src.shape[:2]))
        self.maps = None
        self.esr_note()
        self.schedule(30)

    def guess_light(self):
        """Estimate where the light comes from by the angular skew of the
        spectrum (MG.light_azimuth on the linear luminance of the repaired
        source).

        A hint for the first press: elongated relief (rows of masonry) skews
        it just like real light does, so the angle is then corrected by hand.
        Sets the azimuth slider (degrees 0…360), raises the shape-from-shading
        weight to 0.6 if it was 0, and shows the angle and confidence in %.
        """
        if self.rgb is None:
            return
        L = MG.luminance(MG.srgb_to_lin(self.repaired(self.rgb)[..., :3]))
        az, conf = MG.light_azimuth(L)
        self.m_sfsaz.set(round(math.degrees(az)) % 360)
        if self.m_wsfs.get() <= 0:
            self.m_wsfs.set(0.6)
        self.m_sfsinfo.config(text=t('m.sfs.found', math.degrees(az) % 360,
                                     100.0 * conf))
        self.schedule(30)

    def net_apply(self):
        """Split the network depth into large and fine scales with the weights
        currently set (DN.shape: scale, low, high, contrast) and set it as the
        external depth layer at full size.

        The network is not rerun: its answer is kept raw (self._net_raw), and
        the sliders work instantly. The external weight is raised to at least 1.
        """
        d = getattr(self, '_net_raw', None)
        if d is None or self.rgb is None:
            return
        m = DN.shape(d, min(d.shape[:2]), self.m_netscale.get(),
                     self.m_netlow.get(), self.m_nethigh.get(),
                     self.m_netcon.get())
        self.m_wext.set(max(self.m_wext.get(), 1.0))
        self._set_extern(resize_to(m, self.rgb.shape[:2]), self.extern_name)

    def _net_poll(self):
        """Main-thread poll of the depth-network worker.

        The thread computes, and the window collects its status line and the
        finished answer every quarter of a second (after(250)). The exchange
        goes through plain attributes: self._net_msg (text), self._net_prog
        ((done, total), or None when the wait is of unknown length — then a
        running bar is shown), self._net_out (None while working, ('ok',
        depth, model id) or ('err', message)). Touching widgets from another
        thread is forbidden: Tk does not survive it — that includes
        root.after, which gave "RuntimeError: main thread is not in main
        loop". Hands over to _net_done or _net_fail. Shows the status and bar
        of the tab the run belongs to (material or foliage).
        """
        fol = getattr(self, '_net_for', '') == 'foliage'
        lab = self.d_status if fol else self.m_status
        lab.config(text=self._net_msg or '')
        bar = self.d_netbar if fol else self.m_netbar
        pr = getattr(self, '_net_prog', None)
        if pr is None:
            # сколько ждать неизвестно (ставится библиотека, греется модель) —
            # бегущая полоска честнее, чем застывшая на нуле
            if str(bar.cget('mode')) != 'indeterminate':
                bar.configure(mode='indeterminate')
                bar.start(60)
        else:
            if str(bar.cget('mode')) != 'determinate':
                bar.stop()
                bar.configure(mode='determinate')
            done, total = pr
            bar.configure(value=int(1000 * done / max(total, 1)))
        out = self._net_out
        if out is None:
            self.root.after(250, self._net_poll)
            return
        if out[0] == 'ok':
            self._net_done(out[1], out[2])
        else:
            self._net_fail(out[1])

    def _net_worker(self, key, src, fine, tile):
        """Worker thread of net_depth. There is not a single access to the
        interface here — only to plain attributes (see _net_poll).

        key: model id; src: H×W×3 float32 0…1; fine: tiled mode; tile: tile
        side in px. Installs the runtime if missing, downloads the model if
        missing (a failed download names the URL and model folder, since the
        network may be closed or a proxy may refuse, and the file can then be
        put into the folder by hand), then runs DN.compute and stores ('ok',
        depth H×W float32, key) or ('err', message) in self._net_out.
        """
        try:
            if not DN.have_runtime():
                self._net_msg = t('m.nn.installing')
                self._net_prog = None
                ok, out = DN.install_runtime()
                if not ok:
                    raise RuntimeError(t('m.nn.nort') + '\n\n' + out)
            if not DN.have_model(key):
                def dl(got, total):
                    """Download progress: got of total bytes."""
                    self._net_msg = t('m.nn.loading', got / (1 << 20), total / (1 << 20))
                    self._net_prog = (got, total)
                try:
                    DN.download(key, dl)
                except Exception as e:
                    # сеть могла быть закрыта, прокси мог не пустить; в таком
                    # случае модель можно просто положить в папку руками
                    raise RuntimeError(t('m.nn.nodl', DN.MODELS[key]['url'],
                                         DN.model_dir(), e))
            self._net_msg = t('m.nn.working')
            self._net_prog = None

            def prog(i, n):
                """Computation progress: pass i of n (n > 1 in tiled mode)."""
                self._net_prog = (i, n)
                if n > 1:
                    self._net_msg = t('m.nn.pass', i, n)
            d = DN.compute(src, key, fine=fine, tile=tile, progress=prog)
            self._net_out = ('ok', d, key)
        except Exception as e:
            self._net_out = ('err', str(e))

    def _net_free(self):
        """Main thread, after a network run ends either way: clear the busy
        flag, re-enable both network buttons (material and foliage) and hide
        and reset both progress bars."""
        self._net_busy = False
        self.m_netbtn.config(state='normal')
        self.d_netbtn.config(state='normal')
        for bar in (self.m_netbar, self.d_netbar):
            bar.stop()
            bar.configure(mode='determinate', value=0)
            bar.pack_forget()

    def _net_done(self, d, key):
        """Main thread: take a finished network depth into use.

        d: raw depth, H×W float32; key: model id. If another texture was
        opened while the network was computing, the answer is not for it and
        is discarded ("stale" message). For foliage the leaf ORDER is what is
        needed, not relief (a card has no depth of its own; only which leaf is
        nearer matters): the normalised depth goes to self._fnet_raw, the
        foliage caches are dropped and its weight raised to 0.6 if it was 0.
        For the material the raw answer is kept for the range sliders
        (net_apply), and self.extern_src = ('net', key) records it so batch
        processing reruns the same network on every texture.
        """
        self._net_free()
        if getattr(self, '_net_tex', None) != id(self.rgb):
            # пока сеть считала, открыли другую текстуру — ответ не её
            self.m_status.config(text=t('m.nn.stale'))
            self.d_status.config(text=t('m.nn.stale'))
            return
        if getattr(self, '_net_for', 'material') == 'foliage':
            # листве нужен ПОРЯДОК листьев, а не рельеф: у карточки своей
            # глубины нет вовсе, и важно только, какой лист ближе
            self._fnet_raw = DN.normalize(d)
            self.drop_cache()
            if self.d_fnet.get() <= 0:
                self.d_fnet.set(0.6)
            self.d_netinfo.config(text='%s   %dx%d' % (t('nn.' + key),
                                                       d.shape[1], d.shape[0]))
            self.d_status.config(text=t('com.ready'))
            self.recompute()
            return
        self.extern_raw = None
        # пакетная обработка повторит сеть на каждой текстуре с тем же выбором
        self.extern_src = ('net', key)
        self._net_raw = d                    # сырой ответ, для ползунков диапазонов
        self.extern_name = '%s   %dx%d' % (t('nn.' + key), d.shape[1], d.shape[0])
        self.net_apply()
        self.m_status.config(text=t('com.ready'))
        self.net_note()

    def _net_fail(self, msg):
        """Main thread: a network run failed — free the controls, clear the
        status lines and show msg in an error box."""
        self._net_free()
        self.m_status.config(text='')
        self.d_status.config(text='')
        messagebox.showerror(t('m.nn.title'), msg)

    def save_depth(self):
        """Save the external depth layer (full size, 0…1) as a 16-bit
        greyscale PNG chosen by the user; nothing happens without a layer."""
        if getattr(self, 'extern_full', None) is None:
            return
        p = filedialog.asksaveasfilename(title=t('m.ext.title'), defaultextension='.png',
                                         initialfile='depth.png',
                                         filetypes=[('PNG 16 bit', '*.png'), ('*', '*.*')])
        p = self._with_ext(p, '.png')
        if not p:
            return
        a = (np.clip(self.extern_full, 0, 1) * 65535 + 0.5).astype(np.uint16)
        Image.fromarray(a, mode='I;16').save(p)
        self.m_status.config(text=t('m.ext.saved', os.path.basename(p)))

    def paint_depth(self, x, y, erase=False, final=False):
        """Depth brush (paint_cb of the material view): raise (left button) or
        lower (right button, erase=True) the external depth layer.

        Without a layer, a flat one at 0.5 is created (weight raised to 1).
        The stroke adds strength × (1 − d/r)² — a soft edge — within radius r
        (slider in preview px, scaled to full size), clipped to 0…1. The
        material job is updated only on release (final), since that is slow.
        """
        if self.rgb is None:
            return
        if getattr(self, 'extern_full', None) is None:
            self.extern_raw = None
            self.extern_src = None
            self.extern_name = t('m.ext.hand')
            self.extern_full = np.full(self.rgb.shape[:2], 0.5, np.float32)
            self.m_extname.config(text=self.extern_name)
            if self.m_wext.get() <= 0:
                self.m_wext.set(1.0)
        x, y = self._mfull(x, y)
        h, w = self.extern_full.shape
        if not (0 <= x < w and 0 <= y < h):
            return
        r = max(1, int(self.dbrush_r.get() * self.rgb.shape[0] /
                       max(self.mat.src.shape[0], 1)))
        s = self.dbrush_s.get() * (-1.0 if erase else 1.0)
        y0, y1 = max(0, y - r), min(h, y + r + 1)
        x0, x1 = max(0, x - r), min(w, x + r + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        d = np.sqrt((xx - x) ** 2 + (yy - y) ** 2) / max(r, 1)
        soft = np.clip(1.0 - d, 0.0, 1.0) ** 2          # мягкий край
        self.extern_full[y0:y1, x0:x1] = np.clip(
            self.extern_full[y0:y1, x0:x1] + s * soft, 0.0, 1.0)
        if final:
            self.mat.set_extern(self._extern_for(self.mat.src.shape[:2]))
            self.schedule(30)

    def load_depth(self, path=''):
        """Load an external depth image, or remove the layer.

        path: '' asks with a file dialog; None removes the external layer;
        otherwise the file is read (DDS, PNG, TGA, TIFF, JPEG, EXR) as RGBA
        into self.extern_raw, recorded in self.extern_src for batch, and the
        chosen channel is applied by apply_depth_channel.
        """
        if path is None:
            self.extern_raw = None
            self.extern_src = None
            self.extern_name = ''
            self._set_extern(None, t('m.ext.none'))
            return
        if path == '':
            path = filedialog.askopenfilename(
                title=t('m.ext.title'),
                filetypes=[('Images', '*.dds *.png *.tga *.tif *.tiff *.jpg *.exr'),
                           ('*', '*.*')])
        if not path:
            return
        try:
            arr = read_rgba(path)
        except Exception as e:
            messagebox.showerror(t('com.notopen'), '%s\n\n%s' % (os.path.basename(path), e))
            return
        self.extern_raw = arr
        self.extern_src = ('file', path)
        self.extern_name = '%s   %dx%d' % (os.path.basename(path), arr.shape[1], arr.shape[0])
        self.apply_depth_channel()

    def _extern_for(self, shape):
        """Return the external depth layer resized to shape (H, W), or None."""
        return resize_to(getattr(self, 'extern_full', None), shape)

    # ============================================================ механика ==
    def rebuild_job(self, recompute=True):
        """Recreate the material job (self.mat) from the (repaired) texture.

        For the preview it is computed on a half-size copy (LANCZOS) if that
        option is ticked and the longer side exceeds 512 px; the job gets the
        scale factor (preview / full width) so build_maps can rescale
        size-dependent parameters with MG.at_scale. The external depth, ESRGAN
        outputs and height hint are resized to the job's size. Drops the
        current maps; recompute: also schedule a recompute.
        """
        if self.rgb is None:
            return
        src = self.repaired(self.rgb)
        if getattr(self, 'm_fast', None) is not None and self.m_fast.get() \
                and max(src.shape[:2]) > 512:
            im = Image.fromarray((np.clip(src, 0, 1) * 255).astype(np.uint8))
            im = im.resize((src.shape[1] // 2, src.shape[0] // 2), Image.LANCZOS)
            src = np.asarray(im).astype(np.float32) / 255
        self.mat = MG.MaterialJob(src, src.shape[1] / float(self.rgb.shape[1]))
        self.mat.set_extern(self._extern_for(src.shape[:2]))
        self.mat.set_esr(*self._esr_for(src.shape[:2]))
        self.mat.set_guide(resize_signed(self.guide, src.shape[:2])
                           if self.guide is not None else None)
        self.maps = None
        if recompute:
            self.schedule()

    def build_maps(self, job, band):
        """Compute one or two sets of maps: with the second zone on, a second
        full set is computed and stitched with the first by the zone mask.

        job: a MaterialJob at preview or full size (job.px = its scale
        relative to full size); band: tiling blend band in full-size pixels
        (0 = no tiling), rescaled to the job like all size-dependent
        parameters (MG.at_scale). The second set uses zone_params(), the
        same external depth, ESRGAN outputs and hint; the mask is kept in
        self._zmask. Returns the maps dict of MaterialJob.all_maps (or
        MG.blend_maps of both). Reads Tk variables — main thread only.
        """
        # параметры заданы для полного размера; на уменьшенной копии превью
        # их надо пересчитать, иначе превью покажет не то, что сохранится
        px = job.px
        band = int(round(band * px)) if band else 0
        base = job.all_maps(MG.at_scale(self.mat_params(), px), band)
        if not self.z_on.get():
            return base
        mask = self.zone_mask(job.src.shape[:2], base)
        j2 = MG.MaterialJob(job.src, px)
        j2.set_extern(getattr(job, 'extern', None))
        j2.set_esr(job.esr_n, job.esr_f)
        j2.set_guide(getattr(job, 'guide', None))
        over = j2.all_maps(MG.at_scale(self.zone_params(), px), band)
        out = MG.blend_maps(base, over, mask)
        self._zmask = mask
        return out

    def full_maps(self):
        """Full-size maps — for saving.

        If the preview job already is full size, its maps are returned as they
        are; otherwise a new full-size job is built from the repaired texture
        and computed here, in the main thread (this can take a while).
        """
        band = self.m_band.get() if self.m_seam.get() else 0
        if self.mat.src.shape[:2] == self.rgb.shape[:2]:
            return self.maps
        j = MG.MaterialJob(self.repaired(self.rgb))
        j.set_extern(self._extern_for(self.rgb.shape[:2]))
        j.set_esr(*self._esr_for(self.rgb.shape[:2]))
        j.set_guide(self.guide)
        return self.build_maps(j, band)

    def views(self):
        """Return the views of all six tabs, in notebook order (the nodes tab
        was missing, so Fit and 1:1 there raised an IndexError)."""
        return (self.n_view, self.d_view, self.m_view, self.v_view, self.x_view,
                self.g_view)

    def view(self):
        """Return the 2-D ZoomView of the current tab, for the Fit and 1:1
        buttons; on the 3D tab that is the material view."""
        v = self.views()[self.nb.index('current')]
        return self.m_view if isinstance(v, Stage3D) else v

    def schedule(self, delay=110):
        """Request a deferred recompute of the current tab, delay in ms.

        Called by nearly every setting. While a slider is being turned, the
        view is shown coarser and faster (self._coarse, see prev_pack), and
        a fraction of a second after the last movement it is drawn properly:
        each call cancels the pending recompute (and any pending fine job) and
        restarts the timer, so a drag produces one recompute at the end.
        """
        if not self.classic_on:
            return               # вкладки скрыты — считать для них незачем
        self._coarse = True
        if self.pending:
            self.root.after_cancel(self.pending)
        if getattr(self, 'fine_job', None):
            self.root.after_cancel(self.fine_job)
            self.fine_job = None
        self.pending = self.root.after(delay, self.recompute)

    def recompute(self):
        """Recompute what the current tab shows, then redraw (the after()
        target of schedule).

        Clears the coarse flag and updates the section marks first. Foliage
        normals: foliage_normal and the leaf/branch status line (or the
        no-alpha note). Foliage light: _fol_light, with SSS when it is shown
        or saved. 3D: just v3_redraw. Material: pending repair, then
        build_maps (preview size) and the parallax, relief and direct-normal
        curl notes. Normal repair: x_convert. What gets recomputed inside is
        decided by the caches: the material job and its maps are reset by
        rebuild_job/load; foliage maps by drop_cache (see cached).
        """
        if self.rgb is None:
            return
        # грубый показ нужен только пока крутят ползунок; дошли сюда — значит
        # отпустили, и рисуем честно
        self._coarse = False
        self.mark_sections()
        i = self.nb.index('current')
        if i == 0:
            if self.norm is None:
                self.n_status.config(text=t('f.noalpha.short'))
                self.no_alpha_note(self.n_view)
                return
            self.n_status.config(text=t('com.computing'))
            self.root.update_idletasks()
            self.normal = None
            self.foliage_normal()
            self.n_status.config(text=t(
                'f.status', self.norm.nlab, 100.0 * self.norm.branch_share(),
                100.0 * self.branch.mean() if self.branch is not None else 0.0))
        elif i == 1:
            if self.src is None:
                self.d_status.config(text=t('f.noalpha.short'))
                self.no_alpha_note(self.d_view)
                return
            self._fol_light(self.view_d.get() in ('sss', 'backlit')
                            or self.fsave['_sss'].get())
        elif i == 3:
            self.v3_redraw(False)
            return
        elif i == 2:
            self.m_status.config(text=t('com.computing'))
            self.root.update_idletasks()
            if getattr(self, '_need_rep', False):
                self.apply_repair_now()
            self.root.update_idletasks()
            band = self.m_band.get() if self.m_seam.get() else 0
            self.maps = self.build_maps(self.mat, band)
            self.m_status.config(text=t('com.ready'))
            self.plx_note()
            self.rd_note()
            c = getattr(self.mat, 'direct_curl', None)
            if c is None:
                self.m_ndirnote.config(text='')
            elif self.m_ndirfix.get() >= 0.999:
                self.m_ndirnote.config(text=t('m.ndir.curl.ok', c))
            else:
                self.m_ndirnote.config(text=t('m.ndir.curl', c))
        else:
            self.x_convert()
        self.redraw()

    def rd_note(self):
        """Show how much mixing the relief into the diffuse really changes it —
        in steps of 1/255 (mean and 95th percentile of the largest channel
        change per pixel). "Barely visible" is checked by a number this way,
        not by eye. Adds a warning when the external depth is chosen as the
        source but none is loaded; empty when the relief amount is 0.
        """
        m = self.maps or {}
        if self.m_rdamt.get() <= 0 or 'diffuse_base' not in m:
            self.m_rdnote.config(text='')
            return
        d = np.abs(m['diffuse'] - m['diffuse_base']).max(-1) * 255.0
        txt = t('m.rd.note', float(d.mean()), float(np.percentile(d, 95)))
        if self.m_rdsrc.get() == t('m.rd.s.ext') and getattr(self.mat, 'extern', None) is None:
            txt += '\n' + t('m.rd.noext')
        self.m_rdnote.config(text=txt)

    def plx_note(self):
        """Numbers under the parallax settings: what it will look like in the
        game.

        From maps['plx_info']: texture repeat and available depth in cm, step
        in mm, how far up/down the surface goes, and how many of the alpha
        levels are used; warnings when the wanted depth exceeds what is
        available, when part of the height is clipped, or when fewer than 32
        levels are used (visible stair-stepping). Also enables the manual level
        slider only for the manual reference, and greys out the normal
        strength sliders when the normal is made consistent with the parallax.
        """
        manual = self.m_plxref.get() == t('m.plx.r.manual')
        self.m_plxlev.enable(manual)
        cons = self.m_plxnorm.get()
        for sl in (self.m_nstr, self.m_nfine, self.m_nmid, self.m_nbig, self.m_nscale):
            sl.enable(not cons)
        inf = (self.maps or {}).get('plx_info')
        if not inf:
            self.m_plxnote.config(text='')
            return
        txt = t('m.plx.note', inf['repeat_cm'], inf['avail_cm'], inf['avail_cm'] / 2,
                inf['step_mm'], inf['up_cm'], inf['down_cm'], inf['used'])
        if inf['want_cm'] > inf['avail_cm'] + 1e-3:
            txt += '\n' + t('m.plx.over', inf['want_cm'], inf['avail_cm'])
        if inf['clipped'] > 0.001:
            txt += '\n' + t('m.plx.clip', 100.0 * inf['clipped'])
        if inf['used'] < 32:
            txt += '\n' + t('m.plx.few')
        self.m_plxnote.config(text=txt)

    def x_is_height(self):
        """True when the repair tab's source is set to "height map"."""
        return self.x_src.get() == t('x.src.height')

    def x_mode_changed(self):
        """View-mode radio buttons of the repair tab: refit and redraw."""
        # у видов разный размер (до и после — вдвое шире), поэтому вписать
        self.x_view._fit_pending = True
        self.redraw()

    def x_src_changed(self):
        """Source-kind combobox of the repair tab: switch to the converted view,
        refit and recompute."""
        # при смене того, чем считать картинку, сразу показываем результат
        # преобразования: иначе смена ничего видимого не делает
        self.x_mode.set('conv')
        self.x_view._fit_pending = True
        self.schedule(30)

    def x_convert(self):
        """Normal-repair tab: analysis, repair and normal ↔ height conversion.

        Everything is visible: the result and the way back as separate views,
        the numbers under the sliders.
        Height source: grey = RGB mean, centred and multiplied by the amplitude
        (px) into a normal (self.fixed, 0…1), and back to height for the check;
        the note gives the mean and 99.9th-percentile tilt in degrees.
        Normal source: NX.analyze before and after NX.fix go into the report;
        the height is taken from the already fixed normal (fix first, then
        convert), with the low-frequency cut; then converted back to a normal
        and compared — height range, curl (slope_curl, in the OpenGL
        convention) and mean angle to the fixed normal go into the note.
        Sets self.fixed, self.x_conv, self.x_back and self.x_h (raw height, or
        None for a height source).
        """
        dx = self.x_dx.get()
        self.x_amp.enable(self.x_is_height())
        self.x_cut.enable(not self.x_is_height())
        self.x_minz.enable(not self.x_is_height())
        for w in self.x_fixboxes:
            for c in ([w] if isinstance(w, ttk.Checkbutton) else w.winfo_children()):
                if isinstance(c, ttk.Checkbutton):
                    c.state(['disabled'] if self.x_is_height() else ['!disabled'])
        if self.x_is_height():
            g = self.rgb.mean(-1)
            h_px = (g - float(g.mean())) * float(self.x_amp.get())
            n = MG.normal_from_heightmap(h_px, dx)
            self.fixed = n * 0.5 + 0.5
            self.x_conv = self.fixed
            h2 = MG.height_from_normal(n, dx)
            self.x_back = (h2 - h2.min()) / max(float(np.ptp(h2)), 1e-6)
            self.x_h = None
            tilt = np.degrees(np.arccos(np.clip(n[..., 2], -1, 1)))
            self.x_convnote.config(text=t('x.conv.h.note', float(tilt.mean()),
                                          float(np.percentile(tilt, 99.9))))
            self.x_text.delete('1.0', 'end')
            self.x_text.insert('1.0', t('x.conv.h.rep', g.shape[1], g.shape[0],
                                        float(self.x_amp.get())))
            return
        rep, probs = NX.analyze(self.rgb)
        self.fixed = NX.fix(self.rgb, **self.fix_params())
        rep2, _ = NX.analyze(self.fixed)
        txt = t('x.was') + '\n'
        for k, v in rep:
            txt += '  %s: %s\n' % (t('x.r.' + k), v)
        txt += '\n' + '\n'.join(
            '  ! ' + (t('x.' + k, a) if a is not None else t('x.' + k))
            for k, a in probs)
        txt += '\n\n' + t('x.now') + '\n'
        for k, v in rep2[1:5]:
            txt += '  %s: %s\n' % (t('x.r.' + k), v)
        # высота из уже починенной нормали: сначала чиним, потом переводим
        n = self.fixed * 2.0 - 1.0
        n = n / np.maximum(np.sqrt((n * n).sum(-1, keepdims=True)), 1e-6)
        h = MG.height_from_normal(n, dx, self.x_cut.get())
        back = MG.normal_from_heightmap(h, dx)
        ndx = n if dx else n * np.array([1.0, -1.0, 1.0], np.float32)
        curl = MG.slope_curl(ndx)
        ang = MG.angle_between(n, back)
        self.x_h = h
        self.x_conv = (h - h.min()) / max(float(np.ptp(h)), 1e-6)
        self.x_back = back * 0.5 + 0.5
        note = t('x.conv.n.note', float(np.ptp(h)), curl, ang)
        self.x_convnote.config(text=note)
        txt += '\n' + note + '\n'
        self.x_text.delete('1.0', 'end')
        self.x_text.insert('1.0', txt)

    def save_conv(self):
        """Save the conversion result: the height (16-bit PNG or DDS,
        default <name>_h.png) or, for a height source, the normal (<name>_n)."""
        if getattr(self, 'x_conv', None) is None:
            return
        if self.x_is_height():
            p = self.ask_path('_n')
            if p:
                save_rgba(rgb_rgba(self.x_conv), p, self.x_compress.get(), normal=True)
                self.x_status.config(text=t('com.saved', os.path.basename(p)))
            return
        base, _ = os.path.splitext(self.path)
        p = filedialog.asksaveasfilename(
            title=t('com.save'), initialfile=os.path.basename(base) + '_h.png',
            initialdir=os.path.dirname(base), defaultextension='.png',
            filetypes=[('PNG 16 bit', '*.png'), ('DirectDraw Surface', '*.dds'),
                       ('*', '*.*')])
        if p:
            save_gray16(self.x_conv, p, self.x_compress.get())
            self.x_status.config(text=t('com.saved', os.path.basename(p)))

    def redraw(self):
        """Put the current tab's picture into its view, without recomputing
        maps (view-mode and light changes come here directly).

        Foliage normals: the chosen view, optionally side by side with a
        second one. Foliage light: AO, SSS (colour × amount), backlit,
        height, after, or before/after pair on grey. Material: the chosen
        picture (material_picture), optionally side by side. 3D: v3_redraw.
        Normal repair: converted, back-converted, after, or before/after pair.
        The reference size passed to the view is the full (or source) size, so
        brushes and zoom stay consistent when the picture is a reduced copy.
        """
        if self.rgb is None:
            return
        i = self.nb.index('current')
        if i == 0 and self.normal is not None:
            arr = self.fol_pic(self.view_n.get())
            h0, w0 = self.src.rgba.shape[:2]
            if self.split_n.get():
                arr = self.side_by_side(arr, self.fol_pic(self.view_n2.get()))
                w0 *= 2
            self.push(self.n_view, arr, (h0, w0))
        elif i == 1 and self.diffuse is not None:
            v = self.view_d.get()
            if v == 'ao':
                arr = self.fol_ao
            elif v == 'sss':
                arr = (self.fol_sss[0] * self.fol_sss[1][..., None]
                       if self.fol_sss else self.fol_ao)
            elif v == 'backlit':
                arr = self.backlit_picture()
            elif v == 'height':
                arr = FO.height_map(self.norm) if self.foliage_normal() is not None else self.fol_ao
            elif v == 'after':
                arr = self.on_grey(self.diffuse)
            else:
                arr = np.concatenate([self.on_grey(self.src.rgba),
                                      self.on_grey(self.diffuse)], axis=1)
            h0, w0 = self.src.rgba.shape[:2]
            self.push(self.d_view, arr, (h0, w0 * 2) if v == 'pair' else (h0, w0))
        elif i == 2 and self.maps is not None:
            arr = self.material_picture(self.view_m.get())
            h0, w0 = self.mat.src.shape[:2]
            if self.split_m.get():
                arr = self.side_by_side(arr, self.material_picture(self.view_m2.get()))
                w0 *= 2
            self.push(self.m_view, arr, (h0, w0))
        elif i == 3:
            self.v3_redraw(False)
        elif i == 4 and self.fixed is not None:
            mode = self.x_mode.get()
            if mode == 'conv' and getattr(self, 'x_conv', None) is not None:
                arr = self.x_conv
            elif mode == 'back' and getattr(self, 'x_back', None) is not None:
                arr = self.x_back
            elif mode == 'after' or self.x_is_height():
                arr = self.fixed
            else:
                arr = np.concatenate([self.rgb, self.fixed], axis=1)
            if arr.ndim == 2:
                arr = np.repeat(arr[..., None], 3, -1)
            self.push(self.x_view, arr, arr.shape[:2])

    PREV_MAX = 640          # во сколько пикселей считать превью под светом

    @staticmethod
    def _shrink(a, k):
        """Return every k-th row and column of a (a itself for k <= 1)."""
        # прореживание, а не усреднение: усреднение само по себе стоит целого
        # прохода по картинке и съедает весь выигрыш
        return a if k <= 1 else a[::k, ::k]

    def prev_pack(self, tag, arrays, scale):
        """Return reduced copies of arrays for the lit preview.

        While the picture is fitted into the window there is no point in
        computing the light at full size: the eye gets a reduced copy anyway,
        and the recompute runs on every mouse movement over the light ring.
        tag: cache name ('fol', 'mat'); arrays: same-size maps, the first one
        identifies the cache; scale: current view zoom. At zoom ≥ 0.9 nothing
        is reduced; otherwise every k-th pixel is taken so the longer side is
        about PREV_MAX (640) px, and k is doubled while the ring is dragged or
        a slider is moving (self._coarse). The result is cached until tag, k
        or the first array change.
        """
        big = max(arrays[0].shape[:2])
        k = 1 if scale >= 0.9 else max(1, int(big // self.PREV_MAX))
        if LightRing.DRAG or getattr(self, '_coarse', False):
            k *= 2
        key = (tag, k, id(arrays[0]))
        if getattr(self, '_pk_key', None) != key:
            self._pk = [self._shrink(a, k) for a in arrays]
            self._pk_key = key
        return self._pk

    def push(self, view, arr, ref=None):
        """Show arr in view (ZoomView), flipped vertically if the top-bar Flip
        box is on; ref: reference (full) size (H, W) for brush coordinates."""
        view.set_image(arr[::-1] if self.flip.get() else arr, ref=ref)

    @staticmethod
    def on_grey(rgba):
        """Composite an RGBA image (0…1) over 45 % grey; returns H×W×3 in 0…1."""
        a = rgba[..., 3:4]
        return np.clip(rgba[..., :3] * a + 0.45 * (1 - a), 0, 1)

    def side_by_side(self, a, b):
        """Two panels side by side.

        They are computed at different resolutions, so the larger one is
        brought to the same height by decimation (then both are cropped to the
        smaller height). Grey panels are repeated into three channels.
        Returns H×(Wa+Wb)×3.
        """
        to3 = lambda x: x if x.ndim == 3 else np.stack([x] * 3, -1)
        a, b = to3(np.asarray(a)), to3(np.asarray(b))
        ka = max(1, a.shape[0] // max(b.shape[0], 1))
        kb = max(1, b.shape[0] // max(a.shape[0], 1))
        a, b = self._shrink(a, ka), self._shrink(b, kb)
        n = min(a.shape[0], b.shape[0])
        return np.concatenate([a[:n], b[:n]], axis=1)

    def fol_pic(self, v):
        """Picture for the "Foliage: normals" view.

        v: 'orig' (source on grey), 'map' (normal map, 0…1), 'seg' (leaf
        segments), 'branch' (branch mask), anything else: the lit preview
        (relight_foliage). With the mask display on, the hand-painted branch
        mask is tinted orange at 45 % over it. Returns H×W×3 float32 0…1.
        """
        arr = (self.on_grey(self.src.rgba) if v == 'orig' else
               self.normal * 0.5 + 0.5 if v == 'map' else
               self.norm.segments_picture() if v == 'seg' else
               self.norm.branch_picture() if v == 'branch' else
               self.relight_foliage())
        if self.mask_vis.get() and self.branch is not None and self.branch.any():
            k = max(1, self.branch.shape[0] // max(arr.shape[0], 1))
            m = self._shrink(self.branch.astype(np.float32), k)[..., None] * 0.45
            if m.shape[:2] == arr.shape[:2]:
                arr = arr * (1 - m) + np.array([1.0, 0.35, 0.2], np.float32) * m
        return arr

    def backlit_picture(self):
        """The crown against the light: the light source is moved behind the
        card (its z made negative) and FO.backlit shows the translucency.

        Uses the SSS colour and amount from _fol_light and the SSS power
        slider; without them it just shows the source on grey.
        """
        n = self.foliage_normal()
        if n is None or not self.fol_sss:
            return self.on_grey(self.src.rgba)
        rgb, a = self.fol_sss
        L = self.ring_d.vector().copy()
        L[2] = -abs(L[2])                       # свет уходит за карточку
        base = self.diffuse if self.diffuse is not None else self.src.rgba
        return FO.backlit(base, n, rgb, a, L, power=self.d_spow.get())

    def relight_foliage(self):
        """Lit preview of the foliage normals under the light ring's direction.

        Simple Lambert: colour × (0.30 + 1.30·max(n·l, 0)) on the (de-lit, if
        available) diffuse normalised to a 0.95 peak, dark background where
        alpha is 0. Computed on reduced copies (prev_pack). Returns H×W×3.
        """
        base = self.diffuse if self.diffuse is not None else self.src.rgba
        n, C, A = self.prev_pack('fol', (self.normal, base[..., :3], self.src.alpha),
                                 self.n_view.scale)
        ld = self.ring.vector()
        d = np.clip((n * ld).sum(-1, keepdims=True), 0, 1)
        C = C / max(float(C.max()), 1e-6) * 0.95
        A = A[..., None]
        return (C * (0.30 + 1.30 * d)) * A + np.array([0.09, 0.10, 0.09], np.float32) * (1 - A)

    def material_picture(self, v=None):
        """Picture for the Material view, at the job (preview) size.

        v: view id (default: the view combobox). 'ext' external depth;
        'esrn'/'esrh'/'esrr' the ESRGAN answer as is, unmixed, so it is
        visible what exactly the network gave (normal; blue/green channel of
        the franken output as height/roughness); 'guide' height hint (-1…1
        shown as 0…1); 'zone' zone mask; 'damage' damage mask in red over the
        source; 'orig' the source before repair; 'fixed' the repaired source;
        'normal'; 'nh' the alpha exactly as it will go into _nh, with clipped
        values (0 or 255) in red to show where the depth did not fit;
        'dbase' diffuse before relief mixing; 'height', 'ao', 'rough', 'spec',
        'diffuse' the maps themselves. Anything else: the lit preview under
        the light ring — Lambert plus a specular term spec·(n·l)^(2 + 200 ×
        shininess), on reduced copies (prev_pack). Ambient: with the
        hemispherical switch on, as in engines that use AO and bent
        normals — 1.6·(0.10 + 0.24·n_z)·AO (hemisphere
        around the normal, times AO; direct light is not touched by AO).
        Without it, as in OpenMW: one constant ambient 0.30 for the whole
        surface, no AO map — shading is visible only if baked into the
        diffuse. A constant ambient would hide the bent normal completely.
        Returns H×W or H×W×3 float32 0…1.
        """
        m = self.maps
        v = v or self.view_m.get()
        shape = self.mat.src.shape[:2]
        if v == 'ext':
            e = self._extern_for(shape)
            return np.full(shape, 0.5, np.float32) if e is None else e
        if v in ('esrn', 'esrh', 'esrr'):
            # ответ сети как есть, без смешивания: видно, что именно она дала
            en, ef = self._esr_for(shape)
            if v == 'esrn':
                return en if en is not None else np.full(shape + (3,), 0.5, np.float32)
            if ef is None:
                return np.full(shape, 0.5, np.float32)
            return ef[..., 2] if v == 'esrh' else ef[..., 1]
        if v == 'guide':
            g = resize_signed(self.guide, shape) if self.guide is not None else None
            return np.full(shape, 0.5, np.float32) if g is None else \
                np.clip(g * 0.5 + 0.5, 0, 1)
        if v == 'zone':
            return self.zone_mask(shape, m)
        if v == 'damage':
            d = self.damage_mask()
            base = self.mat.src
            if d is None:
                return base
            a = resize_to(d, shape)[..., None]
            return np.clip(base * (1 - a) + np.array([1.0, 0.25, 0.2], np.float32) * a, 0, 1)
        if v == 'orig':
            # именно исходная картинка, до починки: иначе сравнивать не с чем
            return resize_to(self.rgb, shape) if self.rgb is not None else self.mat.src
        if v == 'fixed':
            return self.mat.src
        if v == 'normal':
            return m['normal'] * 0.5 + 0.5
        if v == 'nh':
            # альфа ровно так, как уйдёт в _nh; обрезанное — красным, чтобы
            # было видно, где глубина не влезла
            a = m.get('nh_alpha', m['height'])
            g = np.repeat(a[..., None], 3, -1)
            cut = ((a <= 0.5 / 255) | (a >= 254.5 / 255))[..., None]
            return np.where(cut, np.array([0.9, 0.2, 0.15], np.float32), g)
        if v == 'dbase':
            return m.get('diffuse_base', m['diffuse'])
        if v in ('height', 'ao', 'rough', 'spec', 'diffuse'):
            return m[v]
        n, dif, ao, sp, sh = self.prev_pack(
            'mat', (m['normal'], m['diffuse'], m['ao'], m['spec'], m['shininess']),
            self.m_view.scale)
        ld = self.ring_m.vector()
        dot = np.clip((n * ld).sum(-1), 0, 1)
        d = dot[..., None]
        spec = (dot ** np.maximum(2.0 + 200.0 * sh, 1.0))[..., None] * sp
        # рассеянный свет считается по полусфере, а не постоянным числом:
        # иначе bent normal в превью не виден вовсе — он ведь и нужен затем,
        # чтобы в щели меньше приходило именно рассеянного
        if self.amb_hemi.get():
            # как в движках, где AO и bent normal применяются: рассеянный свет
            # полусферой по нормали и умножен на AO; прямой свет AO не трогает
            amb = (1.6 * (0.10 + 0.24 * np.clip(n[..., 2], 0, 1)) * ao)[..., None]
        else:
            # OpenMW: рассеянная составляющая одна на всю поверхность, карты
            # AO нет — затенение видно, только если запечено в диффуз
            amb = np.full(n.shape[:2] + (1,), 0.30, np.float32)
        return np.clip(dif * (amb + 1.1 * d) + spec, 0, 1)

    # ------------------------------------------------------------- действия
    def on_drop(self, event):
        """tkinterdnd2 <<Drop>> on the window: load the first dropped file
        (Tcl wraps paths with spaces in braces, several files are
        space-separated)."""
        p = event.data.strip()
        if p.startswith('{') and p.endswith('}'):
            p = p[1:-1]
        p = p.split('} {')[0].strip('{}')
        if self.classic_on:
            self.load(p)
        else:
            self.editor.on_drop(p)

    def open_file(self):
        """Top-bar Open button: choose a texture and load it."""
        p = filedialog.askopenfilename(
            title=t('top.open'),
            filetypes=[('Textures', '*.dds *.png *.tga *.tif *.tiff *.jpg'), ('*', '*.*')])
        if p:
            self.load(p)

    def set_name_line(self):
        """Show the file name, size and alpha state (coverage in %, or why the
        alpha is not used) in the top bar."""
        if self.src is not None:
            has = t('f.hasalpha', 100.0 * self.src.cover)
        else:
            has = t('f.noalpha.tag', getattr(self, '_alpha_why', ''))
        self.name.config(text='%s   %dx%d   %s' % (os.path.basename(self.path),
                                                   self.rgb.shape[1], self.rgb.shape[0], has))

    def load(self, p):
        """Load texture p and reset all per-texture state.

        Everything tied to the previous texture is dropped: height hint, zone
        and damage strokes, network and ESRGAN answers, and the external depth
        (it used to be silently mixed into the new texture). The material job
        is rebuilt. For foliage, the image (halved if the fast option is on and
        the longer side exceeds 1024 px) is parsed into FO.Source/FO.Normals:
        a ValueError means there is no usable alpha, which is not a failure
        (the reason is shown in the name line); any other exception is a real
        failure and is reported in an error box. Finally every view is refit
        and the current tab recomputed. An unreadable file only shows an
        error box.
        """
        try:
            rgba = read_rgba(p)
        except Exception as e:
            messagebox.showerror(t('com.notopen'), '%s\n\n%s' % (os.path.basename(p), e))
            return
        self.path = p
        self.rgb = rgba[..., :3]
        self.alpha = rgba[..., 3]
        self.rgba_full = rgba
        self.g_pos = self.g_neg = self.guide = None
        self.z_hand = self.z_kill = None
        self.dmg_hand = self.dmg_kill = self.dmg = None
        self._dmg_key = None
        self._fnet_raw = None               # ответы сети — от прошлой текстуры
        self._net_raw = None                # ответ сети относится к прошлой текстуре
        self._esr_raw = {}                  # и ответы ESRGAN тоже
        self.extern_src = None
        # внешняя глубина относится к прошлой текстуре — раньше она молча
        # подмешивалась к новой
        self.extern_full = self.extern_raw = None
        self.extern_name = ''
        if getattr(self, 'm_extname', None) is not None:
            self.m_extname.config(text=t('m.ext.none'))
        self._rep_cache = {}
        self.drop_cache()
        self.rebuild_job(recompute=False)
        self.normal = self.diffuse = self.maps = self.fixed = None
        self.fol_ao = self.fol_sss = None
        self._sss_ckey = None
        try:
            small = rgba
            if getattr(self, 'f_fast', None) is not None and self.f_fast.get() \
                    and max(rgba.shape[:2]) > 1024:
                im2 = Image.fromarray((np.clip(rgba, 0, 1) * 255).astype(np.uint8))
                im2 = im2.resize((rgba.shape[1] // 2, rgba.shape[0] // 2), Image.LANCZOS)
                small = np.asarray(im2).astype(np.float32) / 255
            self.src = FO.Source(p, small)
            self.norm = FO.Normals(self.src)
            self.branch = np.zeros(self.src.mask.shape, bool)
        except ValueError as e:                 # альфы нет — это не поломка
            self.src = self.norm = None
            self.branch = None
            self._alpha_why = str(e)
        except Exception as e:                  # а вот это уже поломка, и молчать нельзя
            self.src = self.norm = None
            self.branch = None
            self._alpha_why = t('f.parse.err')
            messagebox.showerror(t('f.err.title'),
                                 '%s\n\n%s: %s' % (os.path.basename(p), type(e).__name__, e))
        self.set_name_line()
        self.esr_note()
        for v in self.views():
            v.scale = 0
            v._fit_pending = True
        self.recompute()

    @staticmethod
    def _with_ext(p, ext):
        """Append ext to path p if it does not already end with it (p '' or
        None is returned unchanged).

        Windows does not add the extension itself when no list of file types
        is set in the dialog, so it is added by hand.
        """
        if p and not p.lower().endswith(ext):
            p += ext
        return p

    def ask_path(self, suffix):
        """Ask where to save a DDS, suggesting <texture name><suffix>.dds next
        to the texture. Returns the path with .dds ensured, or '' if cancelled."""
        base, _ = os.path.splitext(self.path)
        p = filedialog.asksaveasfilename(
            title=t('com.save'),
            initialfile=os.path.basename(base) + suffix + '.dds',
            initialdir=os.path.dirname(base), defaultextension='.dds',
            filetypes=[('DirectDraw Surface', '*.dds'), ('*', '*.*')])
        return self._with_ext(p, '.dds')

    def fix_params(self):
        """Keyword arguments for NX.fix from the repair tab (min Z 0 when
        clamping is off)."""
        return dict(rebuild_z=self.x_rebuild.get(),
                    renormalize=self.x_renorm.get(),
                    min_z=self.x_minz.get() if self.x_clamp.get() else 0.0,
                    flip_green=self.x_flip.get())

    def save_fixed_normal(self):
        """Save the repaired normal as a normal-map DDS: over the source file
        (after confirmation) when overwrite is on, else to <name>_fix.dds;
        the source alpha is kept if that option is on."""
        if self.fixed is None:
            return
        if self.x_over.get():
            if not messagebox.askyesno(t('x.over.title'),
                                       t('x.over.q', os.path.basename(self.path))):
                return
            p = self.path
        else:
            p = self.ask_path('_fix')
        if p:
            a = self.alpha if (self.x_keepalpha.get() and self.alpha is not None) else None
            save_rgba(rgb_rgba(self.fixed, a), p, self.x_compress.get(), normal=True)
            self.x_status.config(text=t('com.saved', os.path.basename(p)))

    def save_normal(self):
        """Save the foliage normal map (<name>_n.dds).

        The preview may be computed on a half-size copy, so the normal is
        recomputed here at full size (full_foliage, then bend), in the main
        thread, before writing.
        """
        if self.normal is None:
            return
        p = self.ask_path('_n')
        if p:
            self.n_status.config(text=t('com.computing'))
            self.root.update_idletasks()
            src, nj, br = self.full_foliage()
            n = self.bend(nj.run(*self.fol_args(br)), src, nj)
            FO.save_normal_dds(n, p, self.n_compress.get())
            self.n_status.config(text=t('com.saved', os.path.basename(p)))

    def material_outputs(self, m):
        """Turn a maps dict into the images to write, by file suffix.

        Returns {suffix: H×W×4 uint8 RGBA}: '_n' normal; '_nh' normal with the
        parallax height in alpha (nh_alpha from MG.parallax_alpha: laid around
        128 = no offset, as OpenMW reads it; the plain height if parallax is
        not set up); '_spec'
        specular colour with shininess in alpha; '_ao', '_h', '_rough',
        '_metal' grey; '_fixed' the repaired, de-lit diffuse.
        """
        n01 = m['normal'] * 0.5 + 0.5
        return {
            '_n': rgb_rgba(n01),
            '_nh': rgb_rgba(n01, m.get('nh_alpha', m['height'])),
            '_spec': rgb_rgba(m['spec'], m['shininess']),
            '_ao': gray_rgba(m['ao']),
            '_h': gray_rgba(m['height']),
            '_rough': gray_rgba(m['rough']),
            '_metal': gray_rgba(m['metal']),
            '_fixed': rgb_rgba(m['diffuse']),
        }

    def save_material(self):
        """Material Save button: write the ticked maps as <name><suffix>.dds
        into a chosen folder. Maps are computed at full size (full_maps),
        not taken from the preview."""
        if self.maps is None:
            return
        want = [k for k, v in self.saveflags.items() if v.get()]
        if not want:
            messagebox.showinfo(t('com.nothing'), t('com.nothing.b'))
            return
        d = filedialog.askdirectory(title=t('com.where'),
                                    initialdir=os.path.dirname(self.path))
        if not d:
            return
        self.m_status.config(text=t('com.computing'))
        self.root.update_idletasks()
        outs = self.material_outputs(self.full_maps())
        stem = os.path.splitext(os.path.basename(self.path))[0]
        for k in want:
            save_rgba(outs[k], os.path.join(d, stem + k + '.dds'), self.m_compress.get(),
                      normal=k in ('_n', '_nh'))
        self.m_status.config(text=t('com.saved.n', len(want)))

    # ------------------------------------------------------ кисть по веткам
    def set_brush(self):
        """Branch-brush check box: turn painting in the foliage view on or off."""
        on = self.brush_on.get()
        self.n_view.paint_on = on
        self.n_view.paint_cb = self.paint_branch if on else None

    def paint_branch(self, x, y, erase=False, final=False):
        """Branch brush (paint_cb of the foliage view): mark (left button) or
        unmark (right button) pixels of the branch mask in a disc of the brush
        radius (px). While painting only the view is redrawn; on release the
        normal is dropped and a recompute scheduled."""
        if self.branch is None:
            return
        if not self._disc(self.branch, x, y, int(self.brush_r.get()), not erase):
            return
        if final:
            self.normal = None
            self.schedule(30)
        else:
            self.redraw()

    def clear_brush(self):
        """Clear the branch mask and recompute."""
        if self.branch is not None:
            self.branch[:] = False
            self.normal = None
            self.schedule(30)

    def save_brush(self):
        """Save the branch mask as an 8-bit PNG (255 = branch)."""
        if self.branch is None:
            return
        p = filedialog.asksaveasfilename(title=t('f.mask.save'), defaultextension='.png',
                                         initialfile='branch_mask.png',
                                         filetypes=[('PNG', '*.png'), ('*', '*.*')])
        p = self._with_ext(p, '.png')
        if p:
            Image.fromarray((self.branch * 255).astype(np.uint8)).save(p)
            self.n_status.config(text=t('f.mask.saved'))

    def load_brush(self):
        """Load a branch mask from an image (grey > 127 = branch), resized with
        nearest neighbour if its size differs, and recompute."""
        p = filedialog.askopenfilename(title=t('f.mask.load'),
                                       filetypes=[('Images', '*.png *.tga *.tif *.bmp'),
                                                  ('*', '*.*')])
        if not p or self.branch is None:
            return
        try:
            im = Image.open(p)
            im.load()
            a = np.asarray(im.convert('L'))
            if a.shape != self.branch.shape:
                im = im.convert('L').resize((self.branch.shape[1], self.branch.shape[0]),
                                            Image.NEAREST)
                a = np.asarray(im)
            self.branch = a > 127
        except Exception as e:
            messagebox.showerror(t('com.notopen'), str(e))
            return
        self.normal = None
        self.schedule(30)

    def no_alpha_note(self, view):
        """Write "no alpha" and what to do about it right in the view canvas.

        An empty preview without an explanation looks like a breakage, so the
        note is written directly into it.
        """
        view.img = None
        view.delete('all')
        w, h = max(view.winfo_width(), 1), max(view.winfo_height(), 1)
        view.create_text(w // 2, h // 2 - 58, fill=FG, justify='center',
                         font=('TkDefaultFont', 11), text=t('f.na.head'))
        view.create_text(w // 2, h // 2 + 10, fill=DIM, justify='center', width=470,
                         font=('TkDefaultFont', 9), text=t('f.na.body'))

    def make_alpha(self):
        """Build an alpha from the background and re-read the texture as
        foliage.

        FO.alpha_from_background with the tolerance slider; on success the new
        RGBA replaces the loaded one and the foliage state is reset; on
        failure an error box is shown and nothing changes.
        """
        if getattr(self, 'rgba_full', None) is None:
            return
        rgba = FO.alpha_from_background(self.rgba_full, self.f_tol.get())
        try:
            self.src = FO.Source(self.path, rgba)
            self.norm = FO.Normals(self.src)
            self.branch = np.zeros(self.src.mask.shape, bool)
        except Exception as e:
            messagebox.showerror(APP, t('f.alphafail', e))
            return
        self.rgba_full = rgba
        self.alpha = rgba[..., 3]
        self.normal = self.diffuse = self.fol_ao = self.fol_sss = None
        self._sss_ckey = None
        self.name.config(text=t('f.alphadone', os.path.basename(self.path)))
        self.n_view._fit_pending = True
        self.schedule(30)

    def reload_foliage(self):
        """Reload the current file (e.g. after the fast/half-size option changed)."""
        if getattr(self, 'path', None):
            self.load(self.path)

    def full_foliage(self):
        """Source and normals at full size — for saving.

        Returns (FO.Source, FO.Normals, branch mask) — the current ones if the
        preview already is full size, otherwise new ones built from the full
        RGBA, with the branch mask resized by nearest neighbour.
        """
        if self.src.rgba.shape[:2] == self.rgba_full.shape[:2]:
            return self.src, self.norm, self.branch
        src = FO.Source(self.path, self.rgba_full)
        nj = FO.Normals(src)
        br = self.branch
        if br is not None and br.shape != src.mask.shape:
            br = resize_to(br, src.mask.shape, nearest=True)
        return src, nj, br

    def bent_pair(self, src, nrm):
        """Horizon scan without the tilt correction — the bent normal is taken
        from it.

        Returns FO.crown_occlusion(..., want_bent=True) for src/nrm with the
        current AO spread, bulge, radius, seed and network depth: (occlusion,
        bent normal). Cached (see cached): the normal has to be bent before
        the AO is computed, otherwise it would be chicken and egg.
        """
        net, nw = self.fol_net(src)
        key = (id(src), getattr(nrm, 'nlab', 0), int(self.d_aospread.get()),
               int(self.d_aobulge.get()), int(self.d_aorad.get()), self.s_seed.get(),
               id(net), round(nw, 3))
        return self.cached('bent', key, lambda: FO.crown_occlusion(
            src, nrm, None, float(self.d_aospread.get()),
            float(self.d_aobulge.get()), float(self.d_aorad.get()),
            self.s_seed.get(), 0.0, want_bent=True, net=net, net_w=nw))

    def bend(self, n, src, nrm):
        """Mix the bent normal into normal map n (H×W×3, -1…1) with the bent
        slider's amount; returns n unchanged if the amount is 0 or there are no
        leaf normals."""
        if self.s_bent.get() <= 0 or nrm is None:
            return n
        return FO.bend_normal(n, self.bent_pair(src, nrm)[1], self.s_bent.get())

    def foliage_normal(self):
        """Return the foliage normal map (H×W×3 float32, -1…1) at preview size,
        computing and bending it if self.normal was dropped; None without
        alpha."""
        if self.norm is None:
            return None
        if self.normal is None:
            self.normal = self.bend(self.norm.run(*self.fol_args(self.branch)),
                                    self.src, self.norm)
        return self.normal

    def cached(self, name, key, make):
        """A small single-value cache: return the value stored under name if
        its key equals key, else call make(), store and return it.

        The light tab computes several expensive things, and moving any slider
        used to recompute them all — hence the slowness. Each value lives in
        attribute '_c_<name>'; key must include everything the value depends
        on (source id, number of leaves, the relevant sliders, seed, network
        depth id). drop_cache removes all of them at once — on load and when
        switching between preview and full size.
        """
        st = getattr(self, '_c_' + name, None)
        if st is not None and st[0] == key:
            return st[1]
        v = make()
        setattr(self, '_c_' + name, (key, v))
        return v

    def drop_cache(self):
        """Forget every value stored by cached() (all '_c_*' attributes).

        Needed when a key could not tell the difference: a new texture or new
        network depth, and the switch to and from full size in _fol_full.
        """
        for k in [k for k in list(self.__dict__) if k.startswith('_c_')]:
            delattr(self, k)

    def _veins(self, src):
        """Vein and edge map of src (FO.vein_map), cached per source."""
        return self.cached('veins', id(src), lambda: FO.vein_map(src))

    def fol_net(self, src):
        """Network depth resized to the current source, or None.

        Returns (depth H×W or None, weight): (None, 0.0) when no network depth
        is loaded or its weight slider is 0.
        """
        d = getattr(self, '_fnet_raw', None)
        if d is None or self.d_fnet.get() <= 0:
            return None, 0.0
        return resize_to(d, src.mask.shape), float(self.d_fnet.get())

    def _depth(self, src, nrm):
        """Plausible crown depth (FO.leaf_depth) from the leaf segmentation,
        spread, bulge, seed and optional network depth; cached."""
        net, nw = self.fol_net(src)
        key = (id(src), getattr(nrm, 'nlab', 0), int(self.d_aospread.get()),
               int(self.d_aobulge.get()), self.s_seed.get(),
               id(net), round(nw, 3))
        return self.cached('depth', key, lambda: FO.leaf_depth(
            nrm, src.mask, float(self.d_aospread.get()),
            float(self.d_aobulge.get()), self.s_seed.get(),
            net=net, net_w=nw))

    def _behind(self, src, nrm):
        """Silhouettes of leaves lying behind each leaf (FO.behind_leaves) for
        the translucency, with the reach slider; cached. None without leaf
        labels."""
        if nrm is None or getattr(nrm, 'lab', None) is None:
            return None
        key = (id(src), getattr(nrm, 'nlab', 0), int(self.d_sreach.get()),
               int(self.d_aospread.get()), int(self.d_aobulge.get()), self.s_seed.get(),
               round(self.d_fnet.get(), 3), id(getattr(self, '_fnet_raw', None)))
        return self.cached('behind', key, lambda: FO.behind_leaves(
            self._depth(src, nrm), src.mask, float(self.d_sreach.get()),
            max(float(self.d_aospread.get()), 1.0)))

    def occlusion(self, src, nrm, normal):
        """Raw crown occlusion (FO.crown_occlusion, 0 open sky … 1 enclosed)
        with the tilt correction by the given normal map; cached. Used for
        both the AO map and the translucency."""
        net, nw = self.fol_net(src)
        key = (id(src), getattr(nrm, 'nlab', 0), int(self.d_aospread.get()),
               int(self.d_aobulge.get()), int(self.d_aorad.get()),
               self.s_seed.get(), round(self.d_aotilt.get(), 3), id(normal),
               id(net), round(nw, 3))
        return self.cached('occ', key, lambda: FO.crown_occlusion(
            src, nrm, normal, float(self.d_aospread.get()),
            float(self.d_aobulge.get()), float(self.d_aorad.get()),
            self.s_seed.get(), self.d_aotilt.get(), net=net, net_w=nw))

    def translucency(self, src=None, nrm=None):
        """Translucency map: how much light passes through the leaves and of
        what colour.

        src, nrm: default the current source and leaf normals. The filter
        colour depends on only one control (tint) and costs more than
        everything else, so it is kept separately and recomputed only when
        that control is moved. Occlusion is reused from _fol_light if it
        matches the source size. Returns FO.translucency's (colour H×W×3,
        amount H×W), or None without a source.
        """
        src = src or self.src
        if src is None:
            return None
        nrm = nrm or self.norm
        if nrm is not None:
            self.foliage_normal()
        key = (id(src), round(self.d_stint.get(), 3))
        if getattr(self, '_sss_ckey', None) != key:
            self._sss_color = FO.trans_color(src, self.d_stint.get())
            self._sss_ckey = key
        occ = getattr(self, '_occ', None)
        if occ is None or occ.shape != src.mask.shape:
            occ = self.occlusion(src, nrm, self.normal)
        return FO.translucency(src, nrm, self.d_sstr.get(), self.d_sbase.get(),
                               self.d_sveins.get(), self.d_slay.get(),
                               self.d_spig.get(), self.d_sedge.get(),
                               self.d_sbr.get(), self.d_stint.get(),
                               self.d_sgam.get(), self.d_scon.get(),
                               self._sss_color, self._veins(src), occ,
                               self.d_sbehind.get(), self._behind(src, nrm))

    def foliage_outputs(self, want):
        """Build the foliage images to write, for the suffixes in want.

        Returns {suffix: H×W×4 uint8}: '_fixed' de-lit diffuse; '_ao' AO with
        the leaf alpha; '_sss' translucency colour with amount in alpha; '_n'
        normal; '_nh' normal with the foliage height (FO.height_map, 0…1) in
        alpha; '_h' height; '_spec'/'_rough' from MG.material_maps with fixed
        foliage values. Uses the current state, so for saving it is run inside
        _fol_full at full size.
        """
        out = {}
        A = self.src.alpha
        if '_fixed' in want:
            out['_fixed'] = (np.clip(self.diffuse, 0, 1) * 255 + 0.5).astype(np.uint8)
        if '_ao' in want:
            out['_ao'] = gray_rgba(self.fol_ao, A)
        if '_sss' in want:
            rgb, a = self.fol_sss if self.fol_sss else (None, None)
            if rgb is not None:
                out['_sss'] = rgb_rgba(rgb, a)
        if {'_n', '_nh', '_h'} & set(want):
            n = self.foliage_normal()
            h = FO.height_map(self.norm)
            n01 = n * 0.5 + 0.5
            if '_n' in want:
                out['_n'] = rgb_rgba(n01)
            if '_nh' in want:
                out['_nh'] = rgb_rgba(n01, h)
            if '_h' in want:
                out['_h'] = gray_rgba(h, A)
        if {'_spec', '_rough'} & set(want):
            base = np.clip(self.diffuse[..., :3], 0, 1)
            spec, shin, rough, _ = MG.material_maps(base, self.fol_ao, self.fol_ao,
                                                    0.040, 0.25, 0.35, 0.0, 0.2)
            if '_spec' in want:
                out['_spec'] = rgb_rgba(spec, shin)
            if '_rough' in want:
                out['_rough'] = gray_rgba(rough, A)
        return out

    def _fol_light(self, want_sss):
        """De-lit diffuse, occlusion and (when needed) translucency of the
        foliage on the current self.src/self.norm.

        Sets self.diffuse (FO.delight, optionally keeping saturation; cached by
        the de-light sliders), self._occ, self.fol_ao and, if want_sss (the
        translucency map is not always needed, and it is costly),
        self.fol_sss. The leaf segmentation is computed once and shared by the
        occlusion and the translucency.
        """
        dkey = tuple(round(v, 3) for v in (self.d_un.get(), self.d_unr.get(),
                                           self.d_ao.get(), self.d_aor.get(),
                                           self.d_lift.get(), self.d_sat.get()))

        def _delight():
            """Compute the de-lit RGBA diffuse (value for cached)."""
            out = FO.delight(self.src, self.d_un.get(), self.d_unr.get(),
                             self.d_ao.get(), self.d_aor.get(), self.d_lift.get())
            if self.d_sat.get() > 0:
                rgb = MG.keep_saturation(self.src.rgba[..., :3], out[..., :3],
                                         self.d_sat.get())
                out = np.concatenate([rgb, out[..., 3:]], -1)
            return out

        self.diffuse = self.cached('delight', (id(self.src), dkey), _delight)
        # разбиение на листья нужно и затенению (собственный объём листа),
        # и просвету (край, ветки), поэтому считаем его один раз
        self.foliage_normal()
        self._occ = self.occlusion(self.src, self.norm, self.normal)
        self.fol_ao = FO.ao_map(self.src, strength=self.d_aostr.get(),
                                gamma=self.d_aogam.get(), contrast=self.d_aocon.get(),
                                nrm=self.norm, occ=self._occ)
        # карта просвета нужна не всегда, а стоит заметно
        if want_sss:
            self.fol_sss = self.translucency()

    def _fol_full(self, fn, want):
        """Run fn at the full foliage size and return its result.

        The preview may be computed on a half-size copy, but the full size
        must be saved — previously the preview copy went into the file. The
        source, normals and branch mask are swapped for full-size ones, the
        caches dropped and _fol_light rerun (with SSS if '_sss' is in want);
        afterwards (also on error) everything is put back as it was and the
        caches dropped again. Runs in the main thread.
        """
        half = self.src.rgba.shape[:2] != self.rgba_full.shape[:2]
        keep = (self.src, self.norm, self.branch, self.normal, self.diffuse,
                self.fol_ao, self.fol_sss, getattr(self, '_occ', None))
        try:
            if half:
                self.src, self.norm, self.branch = self.full_foliage()
                self.drop_cache()
                self.normal = None
            self._fol_light('_sss' in want)
            return fn()
        finally:
            if half:
                (self.src, self.norm, self.branch, self.normal, self.diffuse,
                 self.fol_ao, self.fol_sss, self._occ) = keep
                self.drop_cache()

    def save_foliage_maps(self):
        """Foliage light tab Save button: write the ticked maps as
        <name><suffix>.dds into a chosen folder, recomputed at full size via
        _fol_full. The diffuse is written by FO.save_diffuse_dds (optionally
        keeping foliage coverage constant across mips), normals as normal
        maps."""
        if self.diffuse is None:
            return
        want = [k for k, v in self.fsave.items() if v.get()]
        if not want:
            messagebox.showinfo(t('com.nothing'), t('com.nothing.b'))
            return
        d = filedialog.askdirectory(title=t('com.where'),
                                    initialdir=os.path.dirname(self.path))
        if not d:
            return
        self.d_status.config(text=t('com.computing'))
        self.root.update_idletasks()
        outs = self._fol_full(lambda: self.foliage_outputs(want), want)
        stem = os.path.splitext(os.path.basename(self.path))[0]
        for k, v in outs.items():
            p = os.path.join(d, stem + k + '.dds')
            if k == '_fixed':
                FO.save_diffuse_dds(v, p, self.d_compress.get(), self.d_cov.get())
            else:
                save_rgba(v, p, self.d_compress.get(), normal=k in ('_n', '_nh'))
        self.d_status.config(text=t('com.saved.n', len(outs)))

    # ----------------------------------------------------------- пачкой ----
    def batch(self, what):
        """Batch buttons: process every texture in a chosen folder with the
        settings currently on screen.

        what: 'normal' (foliage normals), 'diffuse' (foliage light tab maps),
        'material', or 'fixnormal' (normal repair). All Tk variables are read
        here, in the main thread, into plain tuples (args, opts) for the
        worker _batch; the bent normal is applied as in manual saving.
        Material: neural networks are recomputed for each texture with the
        same settings as on screen (the depth network only if the external
        layer came from it; ESRGAN if its weights are non-zero); automatic
        damage detection is repeated per texture. What cannot be reproduced
        is announced before starting, not silently: external depth from a file
        or brush (with a note that the result will be flat if all other height
        layers are 0), the second zone, and brush hints/damage — the user may
        cancel. Missing ESRGAN models need consent to download. Fixnormal with
        overwrite asks for confirmation.
        """
        d = filedialog.askdirectory(title=t('com.folder'))
        if not d:
            return
        if what == 'normal':
            args = self.fol_args(None)
            opts = (self.n_compress.get(), self.s_bent.get(),
                    float(self.d_aospread.get()), float(self.d_aobulge.get()),
                    float(self.d_aorad.get()), self.s_seed.get())
        elif what == 'diffuse':
            args = ((self.d_un.get(), self.d_unr.get(), self.d_ao.get(),
                     self.d_aor.get(), self.d_lift.get()),
                    (self.d_aostr.get(), self.d_aogam.get(), self.d_aocon.get(),
                     float(self.d_aospread.get()), float(self.d_aobulge.get()),
                     float(self.d_aorad.get()), self.d_aotilt.get(),
                     self.s_seed.get(), self.d_sat.get()),
                    (self.d_sstr.get(), self.d_sbase.get(), self.d_sveins.get(),
                     self.d_slay.get(), self.d_spig.get(), self.d_sedge.get(),
                     self.d_sbr.get(), self.d_stint.get(), self.d_sgam.get(),
                     self.d_scon.get(), self.d_sbehind.get(),
                     float(self.d_sreach.get())),
                    self.fol_args(None),
                    [k for k, v in self.fsave.items() if v.get()],
                    # bent normal — как в пакете нормалей и как при ручном сохранении
                    (self.s_bent.get(), float(self.d_aospread.get()),
                     float(self.d_aobulge.get()), float(self.d_aorad.get()),
                     self.s_seed.get()))
            opts = (self.d_compress.get(), self.d_cov.get())
        elif what == 'material':
            p = self.mat_params()
            # нейросети в пакете считаются заново на каждой текстуре — с теми
            # же настройками, что сейчас на экране
            src = getattr(self, 'extern_src', None) or ()
            depth = None
            if src[:1] == ('net',) and p.get('w_ext', 0) > 0:
                depth = (src[1], bool(self.m_netfine.get()), self.m_nettile.get(),
                         self.m_netscale.get(), self.m_netlow.get(), self.m_nethigh.get(),
                         self.m_netcon.get())
            esr = [k for k, on in (('normal', p.get('esr_nmix', 0) > 0),
                                   ('franken', p.get('w_esr', 0) > 0
                                    or p.get('esr_rough', 0) > 0)) if on]
            # о том, что в пакет не попадёт, сказать до начала, а не молча
            warn = []
            if (getattr(self, 'extern_full', None) is not None and depth is None
                    and p.get('w_ext', 0) > 0):
                others = sum(p.get(k, 0) for k in ('w_band', 'w_multi', 'w_lum',
                                                   'w_guided', 'w_sfs', 'w_esr'))
                warn.append(t('b.w.ext') + ('\n' + t('b.w.flat') if others <= 0 else ''))
            if self.z_on.get():
                warn.append(t('b.w.zone'))
            if self.guide is not None or self.dmg_hand is not None:
                warn.append(t('b.w.hand'))
            if warn and not messagebox.askyesno(t('com.batch.title'),
                                                '\n\n'.join(warn) + '\n\n' + t('b.w.go')):
                return
            need = [k for k in esr if not ES.have_model(k)]
            if need and not messagebox.askyesno(
                    t('m.esr.title'),
                    t('m.esr.ask', ', '.join(ES.MODELS[k]['title'] for k in need),
                      ES.MODELS[need[0]]['license'], ES.MODELS[need[0]]['author'])):
                return
            # починка: автоматический поиск битых мест повторяется на каждой
            # текстуре; то, что нарисовано кистью, относится только к открытой
            rep = None
            if self.r_on.get() and self.r_auto.get():
                rep = (self.r_hi.get(), self.r_lo.get(), self.r_flat.get(),
                       float(self.r_grow.get()), self.r_method.get(), self.r_mix.get())
            args = (p, self.m_band.get() if self.m_seam.get() else 0,
                    [k for k, v in self.saveflags.items() if v.get()],
                    depth, esr, rep)
            opts = (self.m_compress.get(),)
        else:
            if self.x_over.get() and not messagebox.askyesno(t('x.over.title'),
                                                             t('x.over.batch')):
                return
            args = (self.fix_params(), self.x_keepalpha.get(), self.x_over.get())
            opts = (self.x_compress.get(),)
        status = {'normal': self.n_status, 'diffuse': self.d_status,
                  'material': self.m_status, 'fixnormal': self.x_status}[what]
        self._bat_start(status)
        threading.Thread(target=self._batch, args=(d, what, args, opts), daemon=True).start()

    # ------------------------------------------------ пакет: связь с окном --
    def _bat_start(self, status):
        """Prepare the main-thread side of a batch and start polling it.

        The batch runs in a separate thread; the window collects its status
        line and the fraction done every quarter of a second. Widgets must not
        be touched from the worker thread — the batch used to do exactly that.
        status: the tab's status label; a progress bar (0…1000) is put under
        it. Resets self._bat_msg, self._bat_prog (0…1) and self._bat_out
        (None until the worker stores (done, errors)).
        """
        self._bat_msg, self._bat_prog, self._bat_out = '', 0.0, None
        bar = getattr(self, '_bat_bar', None)
        if bar is not None:
            bar.destroy()
        self._bat_bar = ttk.Progressbar(status.master, mode='determinate', maximum=1000)
        self._bat_bar.pack(fill='x', after=status)
        self._bat_poll(status)

    def _bat_poll(self, status):
        """Main-thread poll (every 250 ms via after) of a batch worker: show
        its text and progress in status and the bar; when self._bat_out
        arrives, remove the bar and report the count done and failed (with up
        to 12 error lines) in a message box."""
        status.config(text=self._bat_msg or '')
        self._bat_bar.configure(value=int(1000 * min(max(self._bat_prog, 0.0), 1.0)))
        out = self._bat_out
        if out is None:
            self.root.after(250, lambda: self._bat_poll(status))
            return
        self._bat_bar.destroy()
        self._bat_bar = None
        done, bad = out
        status.config(text=t('com.batch.done', done, len(bad)))
        msg = t('com.batch.done', done, len(bad))
        if bad:
            msg += '\n\n' + '\n'.join(bad[:12])
        messagebox.showinfo(t('com.batch.title'), msg)

    def _batch(self, d, what, args, opts):
        """Batch worker thread: not a single access to widgets, only the
        attributes _bat_msg, _bat_prog and _bat_out (see _bat_poll).

        d: folder; what, args, opts: as prepared by batch. Sources are
        .dds/.png/.tga files; finished maps are skipped (their suffixes in
        MAP_SUFFIXES), except for normal repair, which needs exactly the
        finished _n and _nh and skips only its own _fix output. Each file is
        processed from scratch at full size and written as <stem><suffix>.dds
        next to it; per-file errors are collected, not raised. Stores (number
        done, list of 'file — error') in self._bat_out at the end.
        """
        # правке нормалей нужны как раз готовые _n и _nh, остальным — наоборот
        skip = (('_fix',) if what == 'fixnormal' else MAP_SUFFIXES)
        files = [f for f in sorted(os.listdir(d))
                 if f.lower().endswith(('.dds', '.png', '.tga'))
                 and not os.path.splitext(f)[0].lower().endswith(skip)]
        done, bad = 0, []
        nfile = max(len(files), 1)
        for i, f in enumerate(files):
            self._bat_msg = '%d/%d  %s' % (i + 1, len(files), f)
            self._bat_prog = i / nfile

            def sub(frac, what_now='', i=i, f=f):
                """Progress within file i: frac 0…1, what_now names the step."""
                # доля внутри одной текстуры: сети считают долго, и без этого
                # полоска стояла бы на месте минутами
                self._bat_prog = (i + min(max(frac, 0.0), 1.0)) / nfile
                self._bat_msg = '%d/%d  %s  %s' % (i + 1, len(files), f, what_now)
            path = os.path.join(d, f)
            stem = os.path.splitext(f)[0]
            try:
                if what == 'normal':
                    src = FO.Source(path)
                    nj = FO.Normals(src)
                    nn = nj.run(*args)
                    if opts[1] > 0:
                        _, bt = FO.crown_occlusion(src, nj, None, opts[2], opts[3],
                                                   opts[4], opts[5], 0.0, want_bent=True)
                        nn = FO.bend_normal(nn, bt, opts[1])
                    FO.save_normal_dds(nn, os.path.join(d, stem + '_n.dds'), opts[0])
                elif what == 'diffuse':
                    dl, aop, ssp, nrm, want, bent = args
                    src = FO.Source(path)
                    diff = FO.delight(src, *dl)
                    if aop[8] > 0:
                        rgb = MG.keep_saturation(src.rgba[..., :3], diff[..., :3], aop[8])
                        diff = np.concatenate([rgb, diff[..., 3:]], -1)
                    nj = FO.Normals(src)
                    nn = nj.run(*nrm)
                    occ = FO.crown_occlusion(src, nj, nn, aop[3], aop[4],
                                             aop[5], aop[7], aop[6])
                    ao = FO.ao_map(src, strength=aop[0], gamma=aop[1],
                                   contrast=aop[2], nrm=nj, occ=occ)
                    A = src.alpha
                    if '_fixed' in want:
                        FO.save_diffuse_dds((np.clip(diff, 0, 1) * 255 + 0.5).astype(np.uint8),
                                            os.path.join(d, stem + '_fixed.dds'), *opts)
                    if '_ao' in want:
                        save_rgba(gray_rgba(ao, A), os.path.join(d, stem + '_ao.dds'), opts[0])
                    if '_sss' in want:
                        zz = FO.leaf_depth(src and nj, src.mask, aop[3], aop[4], aop[7])
                        bk = FO.behind_leaves(zz, src.mask, ssp[11], max(aop[3], 1.0))
                        sr, sa = FO.translucency(src, nj, *ssp[:10], None, None, occ,
                                                 ssp[10], bk)
                        save_rgba(rgb_rgba(sr, sa),
                                  os.path.join(d, stem + '_sss.dds'), opts[0])
                    if {'_n', '_nh', '_h'} & set(want):
                        if bent[0] > 0:
                            _, bt = FO.crown_occlusion(src, nj, None, bent[1], bent[2],
                                                       bent[3], bent[4], 0.0, want_bent=True)
                            nn = FO.bend_normal(nn, bt, bent[0])
                        n01 = nn * 0.5 + 0.5
                        hh = FO.height_map(nj)
                        if '_n' in want:
                            save_rgba(rgb_rgba(n01), os.path.join(d, stem + '_n.dds'), opts[0],
                                      normal=True)
                        if '_nh' in want:
                            save_rgba(rgb_rgba(n01, hh),
                                      os.path.join(d, stem + '_nh.dds'), opts[0], normal=True)
                        if '_h' in want:
                            save_rgba(gray_rgba(hh, A),
                                      os.path.join(d, stem + '_h.dds'), opts[0])
                    if {'_spec', '_rough'} & set(want):
                        sp, sh, rg, _ = MG.material_maps(np.clip(diff[..., :3], 0, 1), ao, ao,
                                                         0.040, 0.25, 0.35, 0.0, 0.2)
                        if '_spec' in want:
                            save_rgba(rgb_rgba(sp, sh),
                                      os.path.join(d, stem + '_spec.dds'), opts[0])
                        if '_rough' in want:
                            save_rgba(gray_rgba(rg, A),
                                      os.path.join(d, stem + '_rough.dds'), opts[0])
                elif what == 'fixnormal':
                    im = Image.open(path)
                    im.load()
                    rgba = np.array(im.convert('RGBA')).astype(np.float32) / 255
                    fx = NX.fix(rgba[..., :3], **args[0])
                    al = rgba[..., 3] if args[1] else None
                    dst = path if args[2] else os.path.join(d, stem + '_fix.dds')
                    if args[2] and not dst.lower().endswith('.dds'):
                        dst = os.path.splitext(dst)[0] + '.dds'
                    save_rgba(rgb_rgba(fx, al), dst, opts[0], normal=True)
                else:
                    p, band, want, depth, esr, rep = args
                    im = Image.open(path)
                    im.load()
                    rgb = np.array(im.convert('RGB')).astype(np.float32) / 255
                    if rep is not None:
                        b, c = RP.exposure_masks(rgb, hi=rep[0], lo=rep[1], flat=rep[2],
                                                 grow=rep[3])
                        mk = np.clip(b + c, 0, 1)
                        if (mk > 0.5).any():
                            fx = RP.repair(rgb, mk, rep[4])
                            rgb = np.clip(rgb * (1.0 - rep[5]) + fx * rep[5],
                                          0, 1).astype(np.float32)
                    mj = MG.MaterialJob(rgb)
                    # внешняя глубина из файла относится к открытой текстуре, а
                    # не к каждой из пачки; из нейросети — считается заново
                    ext = None
                    steps = (1 if depth else 0) + len(esr)
                    st = 0
                    if depth:
                        key, fine, tfrac, sc, lo, hi, cu = depth
                        dd = DN.compute(rgb, key, fine=fine,
                                        tile=int(min(rgb.shape[:2]) * tfrac),
                                        progress=lambda a, b_, st=st: sub(
                                            (st + a / max(b_, 1)) / (steps + 1),
                                            t('m.nn.title')))
                        ext = resize_to(DN.shape(dd, min(dd.shape[:2]), sc, lo, hi, cu),
                                        rgb.shape[:2])
                        st += 1
                    mj.set_extern(ext)
                    got = {}
                    for k in esr:
                        got[k] = ES.run(k, rgb, progress=lambda a, b_, st=st, k=k: sub(
                            (st + a / max(b_, 1)) / (steps + 1), ES.MODELS[k]['title']))
                        st += 1
                    if got:
                        nn_ = got.get('normal')
                        if nn_ is not None:
                            v = nn_ * 2.0 - 1.0
                            nn_ = (v / np.maximum(np.sqrt((v * v).sum(-1, keepdims=True)),
                                                  1e-6)) * 0.5 + 0.5
                        mj.set_esr(nn_, got.get('franken'))
                    sub(steps / (steps + 1), '')
                    maps = mj.all_maps(p, band)
                    outs = self.material_outputs(maps)
                    for k in want:
                        save_rgba(outs[k], os.path.join(d, stem + k + '.dds'), opts[0],
                                  normal=k in ('_n', '_nh'))
                done += 1
            except Exception as e:
                bad.append('%s — %s' % (f, e))
        self._bat_prog = 1.0
        self._bat_out = (done, bad)


import imgfilters as IMF     # noqa: E402 — только для сверки версий
import materials as MATS     # noqa: E402


def check_build():
    """Check that every module has the same BUILD as this file.

    Files from different releases are the most common cause of strange
    behaviour: an old module does not understand a new call, the error is
    swallowed, and the texture silently ends up in the wrong tab. Better to
    say so directly. Called at start-up before the main window exists; on a
    mismatch shows an error box listing the mismatching files (from a
    temporary hidden Tk root) and returns False, otherwise True.
    """
    mods = (('foliage.py', FO), ('matgen.py', MG), ('normalfix.py', NX),
            ('repair.py', RP), ('view3d.py', V3), ('i18n.py', i18n),
            ('depthnet.py', DN), ('nodes.py', ND), ('esrgan.py', ES),
            ('deepbump.py', DB), ('supermat_bridge.py', SB), ('project.py', PR),
            ('editor.py', EDM), ('texio.py', TXIO), ('xpu.py', XPU),
            ('imgfilters.py', IMF), ('materials.py', MATS))
    bad = [n for n, m in mods if getattr(m, 'BUILD', 0) != BUILD]
    if not bad:
        return True
    r = tk.Tk()
    r.withdraw()
    messagebox.showerror(t('sys.mismatch'), t('sys.mismatch.b', ',  '.join(bad)))
    r.destroy()
    return False


if __name__ == '__main__':
    i18n.load()
    if not check_build():
        sys.exit()
    taskbar_id()
    EDM.init_gpu()          # узлы и 3D на видеокарте, если стоит CuPy
    root = TkinterDnD.Tk() if HAVE_DND else tk.Tk()
    App(root)
    # смена сборки onnxruntime, отложенная до перезапуска
    root.after(200, lambda: EDM.run_pending_runtime(root))
    root.mainloop()
