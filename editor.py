# -*- coding: utf-8 -*-
"""
Texelith — the node workspace: the main window of the program since build 22.

Laid out the way Substance Designer and Blender's node editors are:

    ┌ 2D view ─────┬──────────── node graph ─────────────┬ inspector ─┐
    │ picture of   │  every node shows its own result as │ parameters │
    │ the selected │  a thumbnail; wires coloured by the │ of the     │
    │ node         │  kind of value they carry           │ selected   │
    ├ 3D view ─────┤                                     │ node, its  │
    │ the outputs  │                                     │ readouts   │
    │ on a plane   │                                     │ (numbers)  │
    └──────────────┴─────────────────────────────────────┴────────────┘

Only the selected node's settings are on screen; everything else is in the
graph. Each node's effect is visible at all times (its thumbnail), the 2D view
shows the selected node, and the numbers each node reports (fitted scales,
curl, clipped share…) are in the inspector.

Mouse: left — select, drag nodes, drag from a port to connect (from an
occupied input — to pick its wire up), drag on empty space — box select;
middle drag or Alt+left — pan; wheel — zoom; right — menu. Dropping a wire on
empty space offers the nodes that can take it.
Keys: Shift+A or Tab — add node; X / Delete — delete; Ctrl+D — duplicate;
F — frame all; H — thumbnails on/off; Ctrl+Z / Ctrl+Y — undo / redo;
Ctrl+S / Ctrl+O / Ctrl+N — save / open / new project; Ctrl+T — open texture.

The whole graph is evaluated in a worker thread in dependency order; the
thread never touches Tk (results go into plain dicts that a timer in the main
thread picks up). Textures larger than PREVIEW_MAX are previewed on a reduced
copy, with every radius rescaled (nodes._px), so the preview agrees with the
saved file.

Texelith — узловое рабочее место, главное окно программы с 22-й сборки.
Устроено как Substance Designer и узловые редакторы Blender: слева 2D и 3D,
в середине граф с миниатюрами результата каждого узла, справа параметры
выбранного узла и его показания. Горячие клавиши — выше.
"""

import copy
import os
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

import foliage as FO
import nodes as ND
import project as PR
import texio as TX
import view3d as V3
import xpu as X
from i18n import t
import i18n

BUILD = 25          # версия набора файлов, проверяется при запуске

# Виджеты и цвета живут в Texelith.py; он передаёт их сюда через bind_ui,
# чтобы не импортировать главный модуль второй раз.
ZoomView = Stage3D = LightRing = Slider = Info = ScrollPanel = sec = None
BG = PANEL = EDGE = FG = DIM = ACC = APP = None


def bind_ui(**kw):
    """Receive the shared widgets and colours from Texelith.py."""
    globals().update(kw)


PREVIEW_MAX = 1024          # больше — превью на уменьшенной копии
GAME_TILES = 6              # сколько повторов текстуры на стене в режиме «как в игре»
GAME_FOV = 60.0             # угол обзора OpenMW по умолчанию (field of view = 60)
PREVIEW_FAST = 512          # пока двигают ползунок — считаем на копии не больше этой

# Образцы текстур рядом с программой (samples/, CC0, ambientCG): открываются
# при первом запуске, чтобы граф не был пустым
DEFAULT_SAMPLE = 'Rock064.jpg'


def sample_dir():
    """The samples/ folder next to the program files."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'samples')


def samples():
    """Sample texture files, sorted by name (empty if the folder is missing)."""
    d = sample_dir()
    try:
        return sorted(os.path.join(d, f) for f in os.listdir(d)
                      if f.lower().endswith(TX.TEX_EXT))
    except OSError:
        return []
UNDO_MAX = 100

CAT_COLOR = {'input': '#4f6f45', 'prepare': '#625283', 'height': '#76643f',
             'normal': '#3d668a', 'neural': '#84496a', 'maps': '#437270',
             'openmw': '#86633a', 'output': '#50565c', 'other': '#50565c'}
ERR_COLOR = '#9a3d3d'

# геометрия узла в единицах холста (при масштабе 1 — пиксели)
NW, HEAD, ROW, THUMB, PAD = 176, 22, 17, 132, 6


# Параметры в игровых юнитах: под ними показывается перевод в см, м и дюймы.
UNIT_PARAMS = {'repeat'}
CM_PER_UNIT = 30.48 / 22.1          # 22,1 юнита в футе (MWSE) — как в matgen


def units_text(u):
    """Game units as centimetres or metres and inches, e.g.
    '128 юн. = 1,77 м = 69,6″'."""
    cm = float(u) * CM_PER_UNIT
    length = t('u.cm', cm) if cm < 100 else t('u.m', cm / 100.0)
    return t('u.line', float(u), length, cm / 2.54)


def fmt_value(key, v):
    """Readout value as text: units by the kind of quantity."""
    if isinstance(v, str):
        return v
    if isinstance(v, bool):
        return t('ni.yes' if v else 'ni.no')
    v = float(v)
    if key in ('depth_px', 'crossover'):
        return '%.1f px' % v
    if key.startswith('angle') or key.startswith('tilt'):
        return '%.1f°' % v
    if key in ('corr',):
        return '%+.2f' % v
    if key.startswith('curl'):
        return '%.3f' % v
    if key == 'steps':
        return '%.1f / 255' % v
    return '%.3g' % v


def display(v, channel='rgb', lit=False, directx=True, light=None, base=None, alpha=None):
    """Turn a node value into an H×W×3 picture 0…1 for the 2D view.

    v: grey H×W; colour H×W×3; normal H×W×3 in −1…1; RGBA H×W×4 (_nh, _spec).
    channel: 'rgb', 'r', 'g', 'b' or 'a'. lit: show a normal (or the RGB of
    an RGBA normal map) under light, as the "lit" view of the classic tabs:
    colour × (0.30 + 1.30·max(n·L, 0)), the colour being ``base`` (H×W×3,
    the graph's diffuse or the texture) brought to a 0.95 peak, and where
    ``alpha`` (H×W) is 0 a dark background. light: direction (x, y, z) of
    the light ring, z out of the texture; directx: the normal's layout (an
    OpenGL one is turned to DirectX first, as the ring expects).
    """
    v = np.asarray(v, np.float32)
    if v.ndim == 2:
        return np.dstack([np.clip(v, 0, 1)] * 3)
    signed = v.shape[2] == 3 and float(v.min()) < -0.01
    rgb = v[..., :3]
    if channel == 'a':
        a = v[..., 3] if v.shape[2] == 4 else np.ones(v.shape[:2], np.float32)
        return np.dstack([np.clip(a, 0, 1)] * 3)
    if lit:
        n = rgb if signed else rgb * 2.0 - 1.0
        if abs(float(np.sqrt((n * n).sum(-1)).mean()) - 1.0) < 0.15:
            H, W = n.shape[:2]
            if not directx:
                n = n * np.array([1.0, -1.0, 1.0], np.float32)
            L = np.asarray(light if light is not None else (-0.45, -0.5, 0.74), np.float32)
            L = L / max(float(np.linalg.norm(L)), 1e-6)
            d = np.clip((n * L).sum(-1) / np.maximum(np.sqrt((n * n).sum(-1)), 1e-6), 0, 1)
            if base is None:
                C = np.full((H, W, 3), 0.6, np.float32)
            else:
                C = np.asarray(base, np.float32)
                C = (np.dstack([C] * 3) if C.ndim == 2 else C[..., :3])
                if C.shape[:2] != (H, W):
                    C = TX.resize_wrap(C, H, W)
                C = np.clip(C, 0, 1) / max(float(C.max()), 1e-6) * 0.95
            out = C * (0.30 + 1.30 * d)[..., None]
            if alpha is not None:
                A = np.asarray(alpha, np.float32)
                if A.shape[:2] != (H, W):
                    A = TX.resize_wrap(A, H, W)
                A = np.clip(A, 0, 1)[..., None]
                out = out * A + np.array([0.09, 0.10, 0.09], np.float32) * (1 - A)
            return np.clip(out, 0, 1)
    if signed:
        rgb = rgb * 0.5 + 0.5
    rgb = np.clip(rgb, 0, 1)
    if channel in ('r', 'g', 'b'):
        return np.dstack([rgb[..., 'rgb'.index(channel)]] * 3)
    return rgb


def thumb_of(v, size=THUMB):
    """Small PIL image of a value for the node thumbnail (square, letterboxed)."""
    a = display(v)
    im = Image.fromarray((a * 255 + 0.5).astype(np.uint8))
    im.thumbnail((size, size), Image.BILINEAR)
    return im


def _inside(p, poly):
    """Is point p = (x, y) inside polygon poly (list of (x, y))? Even-odd rule."""
    x, y = p
    c = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi:
            c = not c
        j = i
    return c


def _area(poly):
    """Area of a polygon in square pixels (shoelace formula)."""
    s = 0.0
    for i in range(len(poly)):
        x0, y0 = poly[i - 1]
        x1, y1 = poly[i]
        s += x0 * y1 - x1 * y0
    return abs(s) / 2.0


class Editor:
    """The node workspace inside ``parent``. See the module docstring.

    State that must survive a rebuild of the window (language change) is kept
    in ``app.ed``, a plain dict: the graph, the project path, the texture and
    its alpha, batch settings, view position.
    """

    def __init__(self, parent, app):
        """Build the workspace into parent; app is the Texelith App."""
        self.app = app
        self.root = app.root
        st = app.ed
        if 'graph' not in st:
            st.update(graph=PR.template('classic'), path='', src=None, alpha=None,
                      tex='', batch=dict(PR.BATCH_DEFAULTS), zoom=0.8, ox=30.0, oy=30.0,
                      thumbs=True, dirty=False, sel=[], view=None, port=None)
        self.st = st
        self.results, self.errors, self.thumbs = {}, {}, {}
        self._err_key = {}       # узел → вид графа, при котором записана ошибка
        self._tk_thumbs = {}
        self.undo_stack, self.redo_stack = [], []
        self._last_param = (None, 0.0)
        self._gen = 0
        self._busy = False
        self._again = False
        self._pending = None
        self._running = None          # узел, который сейчас считается
        self._drag = None
        self._menu_xy = (0, 0)
        self.dead = False              # окно пересобрано — таймеры этого экземпляра молчат
        self.frame = ttk.Frame(parent)
        self.frame.pack(fill='both', expand=True)
        self.build()
        self.sync_dxt()
        self.root.after(60, self.redraw)
        self.root.after(120, self.fill_inspector)
        if self.st['src'] is not None:
            self.root.after(150, self.schedule)
        elif not st.get('sample_tried'):
            # первый запуск: открыть образец, чтобы граф сразу что-то показывал
            st['sample_tried'] = True
            p = os.path.join(sample_dir(), DEFAULT_SAMPLE)
            if os.path.isfile(p):
                self.root.after(150, lambda: self.open_texture(p, mark=False))
            else:
                self.status(t('ed.hello'))
        else:
            self.status(t('ed.hello'))

    # ------------------------------------------------------------ сборка --
    @property
    def graph(self):
        """The current graph (lives in app.ed so it survives a UI rebuild)."""
        return self.st['graph']

    def build(self):
        """Toolbar, the three panes, the status bar."""
        tb = ttk.Frame(self.frame, padding=(8, 6, 8, 4))
        tb.pack(fill='x')
        for key, cmd in (('ed.new', self.new_project), ('ed.open', self.open_project),
                         ('ed.save', self.save_project)):
            ttk.Button(tb, text=t(key), command=cmd, style='Small.TButton').pack(side='left', padx=1)
        ttk.Separator(tb, orient='vertical').pack(side='left', fill='y', padx=8)
        ttk.Button(tb, text=t('ed.tex'), command=self.open_texture,
                   style='Small.TButton').pack(side='left', padx=1)
        self.tex_label = ttk.Label(tb, style='Hint.TLabel', text='')
        self.tex_label.pack(side='left', padx=8)
        ttk.Separator(tb, orient='vertical').pack(side='left', fill='y', padx=8)
        ttk.Button(tb, text=t('ed.savemaps'), command=self.save_maps,
                   style='Small.TButton').pack(side='left', padx=1)
        ttk.Button(tb, text=t('ed.batch'), command=self.batch_dialog,
                   style='Small.TButton').pack(side='left', padx=1)
        # раскладка нормалей сразу во всех узлах
        mb = ttk.Menubutton(tb, text=t('ed.dxall'), style='Small.TButton')
        mm = tk.Menu(mb, tearoff=0, background=PANEL, foreground=FG,
                     activebackground=EDGE, activeforeground=FG)
        mm.add_command(label=t('ed.dxall.dx'), command=lambda: self.set_all_dx(True))
        mm.add_command(label=t('ed.dxall.gl'), command=lambda: self.set_all_dx(False))
        mb['menu'] = mm
        mb.pack(side='left', padx=(8, 1))
        # справа: язык, справка, классический режим
        self.app.lang_widgets(tb)
        ttk.Button(tb, text=t('ed.classic'), command=lambda: self.app.show_classic(True),
                   style='Small.TButton').pack(side='right', padx=6)

        pw = ttk.PanedWindow(self.frame, orient='horizontal')
        pw.pack(fill='both', expand=True)
        self.pw = pw
        left = ttk.PanedWindow(pw, orient='vertical')
        mid = ttk.Frame(pw)
        right = ttk.Frame(pw)
        pw.add(left, weight=2)
        pw.add(mid, weight=5)
        pw.add(right, weight=0)
        self.left_pw = left
        self.root.after(80, self._init_sashes)

        # --- 2D
        f2 = ttk.Frame(left)
        left.add(f2, weight=1)
        row = ttk.Frame(f2)
        row.pack(fill='x', pady=(2, 2))
        self.v2_title = ttk.Label(row, text='2D', style='Head.TLabel')
        self.v2_title.pack(side='left', padx=(4, 8))
        self.port_var = tk.StringVar(value='')
        # какой выход узла смотреть; то же — щелчок по гнезду выхода на узле
        ttk.Label(row, text=t('ed.port'), style='Hint.TLabel').pack(side='left', padx=(0, 3))
        self.port_cb = ttk.Combobox(row, textvariable=self.port_var, width=16,
                                    state='readonly')
        self.port_cb.pack(side='left')
        self.port_cb.bind('<<ComboboxSelected>>', lambda e: self._port_chosen())
        row = ttk.Frame(f2)                 # каналы — второй строкой, чтобы влезало
        row.pack(fill='x', pady=(0, 2))
        self.chan = tk.StringVar(value='rgb')
        for c, lab in (('rgb', 'RGB'), ('r', 'R'), ('g', 'G'), ('b', 'B'), ('a', 'A')):
            ttk.Radiobutton(row, text=lab, value=c, variable=self.chan,
                            command=self.show_2d).pack(side='left', padx=1)
        self.lit = tk.BooleanVar(value=False)
        ttk.Checkbutton(row, text=t('ed.lit'), variable=self.lit,
                        command=self.show_2d).pack(side='left', padx=(6, 0))
        # закрепить: 2D продолжает показывать этот узел, пока в инспекторе другой
        self.pin_var = tk.BooleanVar(value=self.st.get('pin') is not None)
        ttk.Checkbutton(row, text=t('ed.pin'), variable=self.pin_var,
                        command=self.toggle_pin).pack(side='left', padx=(6, 0))
        self.v2 = ZoomView(f2)
        self.v2.pack(fill='both', expand=True)
        self._pt = {'mode': None, 'pts': [], 'sel': None, 'start': None, 'cur': None}
        self._mk = {'v': None, 'pts': []}          # мазок рисованной маски, пока ведётся
        self.v2.bind('<Motion>', self._brush_cursor, add='+')
        self.v2.bind('<Leave>', lambda e: self.v2.delete('mcursor'), add='+')
        self.v2_info = ttk.Label(f2, style='Hint.TLabel', text='')
        self.v2_info.pack(fill='x', padx=4)

        # --- 3D
        f3 = ttk.Frame(left)
        left.add(f3, weight=1)
        row = ttk.Frame(f3)
        row.pack(fill='x', pady=(2, 2))
        # 3D — плоскость в перспективе; 2D — карточка прямо, под тем же светом
        # (как вид «со светом» на вкладках листвы); колесо — масштаб
        self.v3_flat = tk.BooleanVar(value=False)
        for val, lab in ((False, '3D'), (True, '2D')):
            ttk.Radiobutton(row, text=lab, value=val, variable=self.v3_flat,
                            style='Toolbutton', command=self.show_3d).pack(side='left',
                                                                           padx=(4 if not val else 0, 0))
        ttk.Label(row, text='').pack(side='left', padx=4)
        self.v3_mode = tk.StringVar(value=t('v3.m.openmw'))
        self._v3_modes = {t('v3.m.' + m): m for m in ('openmw', 'occlusion', 'offset', 'flat')}
        cb = ttk.Combobox(row, values=list(self._v3_modes), textvariable=self.v3_mode,
                          state='readonly', width=16)
        cb.pack(side='left')
        cb.bind('<<ComboboxSelected>>', lambda e: self.show_3d())
        ttk.Label(row, text=t('v3.tiles')).pack(side='left', padx=(8, 2))
        self.v3_tiles = tk.IntVar(value=2)
        ttk.Spinbox(row, from_=1, to=8, width=3, textvariable=self.v3_tiles,
                    command=self.show_3d).pack(side='left')
        self.v3_hemi = tk.BooleanVar(value=False)
        row = ttk.Frame(f3)
        row.pack(fill='x', pady=(0, 2))
        ttk.Checkbutton(row, text=t('ed.hemi'), variable=self.v3_hemi,
                        command=self.show_3d).pack(side='left', padx=4)
        # глубина для режимов «с перекрытием» и «смещение»; в режиме OpenMW
        # её задаёт альфа _nh, как в игре
        self.v3_depth = tk.DoubleVar(value=4.0)
        ttk.Label(row, text=t('ed.v3depth')).pack(side='left', padx=(10, 2))
        self.v3_depth_scale = ttk.Scale(row, from_=0.0, to=15.0, variable=self.v3_depth,
                                        length=110, command=lambda v: self.show_3d(True))
        self.v3_depth_scale.pack(side='left')
        self.v3_depth_scale.bind('<ButtonRelease-1>', lambda e: self.show_3d(False))
        # просвет листвы: виден, когда свет за карточкой (внешнее кольцо у
        # кольца света); узость — как на вкладке «Листва: свет»
        row = ttk.Frame(f3)
        row.pack(fill='x', pady=(0, 2))
        self.v3_spow = tk.DoubleVar(value=2.0)
        ttk.Label(row, text=t('ed.v3spow')).pack(side='left', padx=(4, 2))
        sc = ttk.Scale(row, from_=0.5, to=8.0, variable=self.v3_spow, length=110,
                       command=lambda v: self.show_3d(True))
        sc.pack(side='left')
        sc.bind('<ButtonRelease-1>', lambda e: self.show_3d(False))
        Info(row, t('ed.v3spow.info')).pack(side='left', padx=4)
        # «как в игре»: стена того размера, каким её видно в OpenMW с заданного
        # расстояния — повтор текстуры в юнитах берётся из узла параллакса, угол
        # обзора как у движка по умолчанию. Иначе смещение ±2 % плитки на
        # мелкой плитке превью занимает пару пикселей и параллакса не видно
        row = ttk.Frame(f3)
        row.pack(fill='x', pady=(0, 2))
        self.v3_game = tk.BooleanVar(value=True)
        ttk.Checkbutton(row, text=t('ed.v3game'), variable=self.v3_game,
                        command=self.show_3d).pack(side='left', padx=4)
        self.v3_gdist = tk.DoubleVar(value=96.0)
        self.v3_gdist_lab = ttk.Label(row, text='', style='Hint.TLabel', width=30)
        self.v3_gdist_scale = ttk.Scale(row, from_=24.0, to=512.0, variable=self.v3_gdist,
                                        length=130, command=lambda v: self.show_3d(True))
        self.v3_gdist_scale.pack(side='left', padx=(4, 2))
        self.v3_gdist_scale.bind('<ButtonRelease-1>', lambda e: self.show_3d(False))
        self.v3_gdist_lab.pack(side='left')
        ttk.Button(row, text='⛶', width=3, style='Small.TButton',
                   command=self.toggle_big_3d).pack(side='right', padx=4)
        body = ttk.Frame(f3)
        body.pack(fill='both', expand=True)
        self.v3 = Stage3D(body, lambda live: self.show_3d(live))
        self.v3.pack(fill='both', expand=True)
        # кольцо света лежит поверх угла 3D-вида, а не отнимает у него ширину
        self.ring = LightRing(body, self._light_moved, back=True)
        self.ring.place(relx=1.0, rely=1.0, x=-4, y=-4, anchor='se')
        self.spin = tk.BooleanVar(value=False)
        ttk.Checkbutton(row, text=t('com.spin'), variable=self.spin,
                        command=lambda: self.ring.toggle_spin(self.spin.get())
                        ).pack(side='left', padx=(8, 2))
        # ось вращения света: вокруг нормали или через верх / бок назад
        axes = ('z', 'x', 'y')
        ax_labels = [t('ed.spin.' + a) for a in axes]
        self.spin_axis = tk.StringVar(value=ax_labels[0])
        cb = ttk.Combobox(row, values=ax_labels, textvariable=self.spin_axis, width=18,
                          state='readonly')
        cb.pack(side='left')
        cb.bind('<<ComboboxSelected>>', lambda e: setattr(
            self.ring, 'axis', axes[ax_labels.index(self.spin_axis.get())]))

        # --- граф
        self.cv = tk.Canvas(mid, background='#1c1f22', highlightthickness=0)
        self.cv.pack(fill='both', expand=True)
        self.bind_canvas()

        # --- инспектор
        sp = ScrollPanel(right, width=330)
        sp.pack(fill='both', expand=True)
        self.insp = sp.inner

        # --- строка состояния
        sb = ttk.Frame(self.frame, padding=(8, 2, 8, 4))
        sb.pack(fill='x')
        self.msg = ttk.Label(sb, style='Hint.TLabel', text='')
        self.msg.pack(side='left')
        self.bar = ttk.Progressbar(sb, length=180, mode='determinate', maximum=1.0)
        self.bar.pack(side='right')
        self.res_label = ttk.Label(sb, style='Hint.TLabel', text='')
        self.res_label.pack(side='right', padx=10)
        # черновой проход: сначала на уменьшенной копии, потом подробно. Его
        # картинка у некоторых узлов (починка, пятна, сети) не та, что у
        # подробного, и при настройке это сбивает — тогда его выключают
        self.draft = tk.BooleanVar(value=bool(self.st.get('draft', True)))
        ttk.Checkbutton(sb, text=t('ed.draft'), variable=self.draft,
                        command=lambda: (self.st.__setitem__('draft', self.draft.get()),
                                         self.schedule(10))).pack(side='right', padx=6)
        self.dev_label = ttk.Label(sb, style='Hint.TLabel', text='', cursor='hand2')
        self.dev_label.pack(side='right', padx=10)
        self.dev_label.bind('<Button-1>', lambda e: runtime_dialog(self.root, self._after_install))
        self.update_dev_label()
        self.update_title()
        self.update_tex_label()

    def _init_sashes(self, tries=0):
        """Initial pane sizes: views ~27 %, inspector 345 px, graph the rest."""
        if self.dead:
            return
        try:
            W = self.pw.winfo_width()
            H = self.left_pw.winfo_height()
            if (W < 300 or H < 200) and tries < 20:
                self.root.after(100, lambda: self._init_sashes(tries + 1))
                return
            self.pw.sashpos(0, int(W * 0.27))
            self.pw.sashpos(1, W - 345)
            self.left_pw.sashpos(0, int(H * 0.52))
        except Exception:
            pass
        self.root.after(50, self.frame_all)

    def update_dev_label(self):
        """Status bar: where the nodes compute (click — to choose)."""
        self.dev_label.config(text=(t('gpu.short', X.device_name()) if X.active()
                                    else t('gpu.cpu')))

    def status(self, text):
        """Show a message in the status bar."""
        self.msg.config(text=text)

    def update_title(self):
        """Window title: program, project name, * if unsaved."""
        name = os.path.basename(self.st['path']) if self.st['path'] else t('ed.untitled')
        self.root.title('%s — %s%s' % (APP, name, ' *' if self.st['dirty'] else ''))

    def update_tex_label(self):
        """Texture name and size next to the toolbar button."""
        if self.st['src'] is None:
            self.tex_label.config(text=t('ed.notex'))
            return
        h, w = self.st['src'].shape[:2]
        self.tex_label.config(text='%s  %d×%d%s' % (
            os.path.basename(self.st['tex']), w, h,
            '  α' if self.st['alpha'] is not None else ''))

    def touch(self):
        """Mark the project as changed."""
        if not self.st['dirty']:
            self.st['dirty'] = True
            self.update_title()

    def close(self):
        """Stop this instance's timers before its widgets are destroyed."""
        self.dead = True
        self._gen += 1
        for a in ('_pending', '_v3_after'):
            if getattr(self, a, None):
                try:
                    self.root.after_cancel(getattr(self, a))
                except Exception:
                    pass

    # ------------------------------------------------------ преобразование --
    def w2s(self, x, y):
        """Graph (world) coordinates → canvas pixels."""
        z = self.st['zoom']
        return x * z + self.st['ox'], y * z + self.st['oy']

    def s2w(self, x, y):
        """Canvas pixels → graph coordinates."""
        z = self.st['zoom']
        return (x - self.st['ox']) / z, (y - self.st['oy']) / z

    def node_h(self, nid):
        """Height of a node box in graph units."""
        sp = ND.REG[self.graph.nodes[nid]['type']]
        rows = max(len(sp.ins), len(sp.outs), 1)
        return HEAD + rows * ROW + PAD + ((THUMB + PAD) if self.st['thumbs'] else 0)

    def port_w(self, nid, port, is_in):
        """Graph position of a port."""
        n = self.graph.nodes[nid]
        sp = ND.REG[n['type']]
        lst = sp.ins if is_in else sp.outs
        idx = [p for p, _ in lst].index(port)
        return (n['x'] if is_in else n['x'] + NW), n['y'] + HEAD + ROW * idx + ROW / 2 + 2

    def hit_port(self, sx, sy):
        """(node, port, is_input, kind) under canvas point, or None."""
        z = self.st['zoom']
        r = max(7.0, 8.0 * z)
        for nid in reversed(list(self.graph.nodes)):
            sp = ND.REG[self.graph.nodes[nid]['type']]
            for is_in, lst in ((True, sp.ins), (False, sp.outs)):
                for port, kind in lst:
                    px, py = self.w2s(*self.port_w(nid, port, is_in))
                    if abs(px - sx) <= r and abs(py - sy) <= r:
                        return nid, port, is_in, kind
        return None

    def hit_node(self, sx, sy):
        """Topmost node under canvas point, or None."""
        x, y = self.s2w(sx, sy)
        for nid in reversed(list(self.graph.nodes)):
            n = self.graph.nodes[nid]
            if n['x'] <= x <= n['x'] + NW and n['y'] <= y <= n['y'] + self.node_h(nid):
                return nid
        return None

    # ------------------------------------------------------------ рисование --
    def redraw(self):
        """Redraw the whole graph canvas."""
        if self.dead:
            return
        c = self.cv
        c.delete('all')
        z = self.st['zoom']
        W, H = max(c.winfo_width(), 50), max(c.winfo_height(), 50)
        # сетка
        step = 40 * z
        if step >= 12:
            x0 = self.st['ox'] % step
            y0 = self.st['oy'] % step
            for i in range(int(W / step) + 2):
                c.create_line(x0 + i * step, 0, x0 + i * step, H, fill='#212427')
            for i in range(int(H / step) + 2):
                c.create_line(0, y0 + i * step, W, y0 + i * step, fill='#212427')
        g = self.graph
        lw = max(1.0, 2.2 * z)
        for a, ap, b, bp in g.links:
            if a not in g.nodes or b not in g.nodes:
                continue
            x1, y1 = self.w2s(*self.port_w(a, ap, False))
            x2, y2 = self.w2s(*self.port_w(b, bp, True))
            kind = dict(ND.REG[g.nodes[a]['type']].outs)[ap]
            dx = max(30.0, abs(x2 - x1) * 0.45)
            c.create_line(x1, y1, x1 + dx, y1, x2 - dx, y2, x2, y2, smooth=True,
                          splinesteps=24, width=lw, fill=ND.KIND_COLOR.get(kind, DIM))
        sel = set(self.st['sel'])
        fs = max(6, int(round(8 * z)))
        for nid, n in g.nodes.items():
            self.draw_node(nid, n, nid in sel, fs)
        d = self._drag
        if d and d[0] == 'wire':
            c.create_line(d[3], d[4], d[5], d[6], fill=ACC, width=lw, dash=(5, 3))
        elif d and d[0] == 'box':
            c.create_rectangle(d[1], d[2], d[3], d[4], outline=ACC, dash=(3, 3))
        if not g.nodes:
            c.create_text(W / 2, H / 2, fill=DIM, text=t('ed.empty'), justify='center')

    def fit_text(self, text, size, width):
        """Cut text with '…' so that it fits into width pixels in one line."""
        import tkinter.font as tkfont
        fonts = self.__dict__.setdefault('_fonts', {})
        f = fonts.get(size)
        if f is None:
            f = fonts[size] = tkfont.Font(family='TkDefaultFont', size=size, weight='bold')
        if f.measure(text) <= width:
            return text
        while text and f.measure(text + '…') > width:
            text = text[:-1]
        return text.rstrip() + '…'

    def draw_node(self, nid, n, selected, fs):
        """Draw one node: body, coloured header, ports, thumbnail, state."""
        c = self.cv
        z = self.st['zoom']
        sp = ND.REG[n['type']]
        x0, y0 = self.w2s(n['x'], n['y'])
        x1, y1 = self.w2s(n['x'] + NW, n['y'] + self.node_h(nid))
        err = nid in self.errors
        head = ERR_COLOR if err else CAT_COLOR.get(sp.cat, CAT_COLOR['other'])
        c.create_rectangle(x0 + 3, y0 + 3, x1 + 3, y1 + 3, fill='#141618', outline='')
        c.create_rectangle(x0, y0, x1, y1, fill=PANEL,
                           outline=ACC if selected else EDGE, width=2 if selected else 1)
        c.create_rectangle(x0 + 1, y0 + 1, x1 - (1 if not selected else 2), y0 + HEAD * z,
                           fill=head, outline='')
        if z >= 0.4:
            c.create_text(x0 + 7 * z, y0 + HEAD * z / 2, anchor='w', fill=FG,
                          text=self.fit_text(t('nd.' + n['type']), fs, (NW - 26) * z),
                          font=('TkDefaultFont', fs, 'bold'))
        mark = ''
        if self._running == nid:
            mark = '…'
        elif nid == self.shown():
            mark = '◉'
        if mark:
            c.create_text(x1 - 7 * z, y0 + HEAD * z / 2, text=mark, anchor='e', fill=FG,
                          font=('TkDefaultFont', fs))
        if (self.results.get(nid) or {}).get('_warn'):
            # предупреждение узла (например, альфа исходника заменена входом)
            c.create_text(x1 - (20 if mark else 7) * z, y0 + HEAD * z / 2, text='⚠',
                          anchor='e', fill='#f0c060', font=('TkDefaultFont', fs, 'bold'))
        r = max(3.0, 5.0 * z)
        wired = self.graph.inputs_of(nid)
        for is_in, lst in ((True, sp.ins), (False, sp.outs)):
            other = len(sp.outs if is_in else sp.ins)
            for row_i, (port, kind) in enumerate(lst):
                px, py = self.w2s(*self.port_w(nid, port, is_in))
                col = ND.KIND_COLOR.get(kind, DIM)
                # пустой кружок — необязательный вход, к которому ничего не подключено
                opt = is_in and port in sp.opt and port not in wired
                c.create_oval(px - r, py - r, px + r, py + r, fill=PANEL if opt else col,
                              outline=col, width=max(1, 1.5 * z))
                if z >= 0.55:
                    shown = (not is_in and len(sp.outs) > 1
                             and port == (self.vport().get(nid) or sp.outs[0][0]))
                    # подпись — в свою половину узла: длинные вход и выход в одной
                    # строке иначе налезают друг на друга
                    c.create_text(px + (9 * z if is_in else -9 * z), py,
                                  text=self.fit_text(ND.label(n['type'], port), max(6, fs - 1),
                                                     ((NW / 2 - 12) if row_i < other
                                                      else (NW - 24)) * z),
                                  anchor='w' if is_in else 'e',
                                  fill=FG if shown else DIM,
                                  font=('TkDefaultFont', max(6, fs - 1),
                                        'bold underline' if shown else 'normal'))
        if self.st['thumbs']:
            rows = max(len(sp.ins), len(sp.outs), 1)
            tx, ty = self.w2s(n['x'] + (NW - THUMB) / 2, n['y'] + HEAD + rows * ROW + PAD)
            size = int(THUMB * z)
            im = self.thumbs.get(nid)
            if im is not None and size >= 16:
                key = (nid, size, id(im))
                ph = self._tk_thumbs.get(key)
                if ph is None:
                    w, h = im.size
                    k = size / float(max(w, h))
                    ph = ImageTk.PhotoImage(im.resize((max(1, int(w * k)), max(1, int(h * k))),
                                                      Image.BILINEAR))
                    for old in [kk for kk in self._tk_thumbs if kk[0] == nid]:
                        del self._tk_thumbs[old]
                    self._tk_thumbs[key] = ph
                c.create_image(tx + size / 2, ty + size / 2, image=ph)
            else:
                c.create_rectangle(tx, ty, tx + size, ty + size, fill='#1a1d20', outline=EDGE)
                if err and z >= 0.5:
                    c.create_text(tx + size / 2, ty + size / 2, text='⚠', fill='#e08080',
                                  font=('TkDefaultFont', max(8, int(20 * z))))

    # ------------------------------------------------------------- мышь ---
    def bind_canvas(self):
        """Mouse and keyboard bindings of the graph canvas."""
        c = self.cv
        c.bind('<Button-1>', self.on_press)
        c.bind('<B1-Motion>', self.on_motion)
        c.bind('<ButtonRelease-1>', self.on_release)
        c.bind('<Alt-Button-1>', self.pan_start)
        c.bind('<Alt-B1-Motion>', self.pan_move)
        c.bind('<Button-2>', self.pan_start)
        c.bind('<B2-Motion>', self.pan_move)
        c.bind('<Button-3>', self.on_menu)
        c.bind('<Double-Button-1>', self.on_double)
        c.bind('<MouseWheel>', lambda e: self.zoom_at(e.x, e.y, 1.12 if e.delta > 0 else 1 / 1.12))
        c.bind('<Button-4>', lambda e: self.zoom_at(e.x, e.y, 1.12))
        c.bind('<Button-5>', lambda e: self.zoom_at(e.x, e.y, 1 / 1.12))
        c.bind('<Configure>', lambda e: self.redraw())
        c.bind('<Motion>', lambda e: setattr(self, '_mouse', (e.x, e.y)))
        for seq, fn in (('<Shift-A>', self.menu_at_mouse), ('<Tab>', self.menu_at_mouse),
                        ('<x>', self.delete_sel), ('<Delete>', self.delete_sel),
                        ('<BackSpace>', self.delete_sel), ('<Control-d>', self.duplicate),
                        ('<f>', self.frame_all), ('<Home>', self.frame_all),
                        ('<h>', self.toggle_thumbs), ('<Control-a>', self.select_all)):
            c.bind(seq, lambda e, f=fn: (f(), 'break')[1])

    def pan_start(self, e):
        """Middle button (or Alt+left): start panning."""
        self._pan = (e.x, e.y, self.st['ox'], self.st['oy'])

    def pan_move(self, e):
        """Pan by the mouse travel."""
        x, y, ox, oy = self._pan
        self.st['ox'], self.st['oy'] = ox + e.x - x, oy + e.y - y
        self.redraw()

    def zoom_at(self, sx, sy, k):
        """Zoom by k keeping the point under the cursor in place (0.25…2)."""
        z0 = self.st['zoom']
        z = min(2.0, max(0.25, z0 * k))
        wx, wy = self.s2w(sx, sy)
        self.st['zoom'] = z
        self.st['ox'], self.st['oy'] = sx - wx * z, sy - wy * z
        self.redraw()

    def on_press(self, e):
        """Left button: port → wire; node → select and move; empty → box select."""
        self.cv.focus_set()
        hit = self.hit_port(e.x, e.y)
        if hit:
            nid, port, is_in, kind = hit
            if is_in:
                src = self.graph.inputs_of(nid).get(port)
                if src:
                    # подхватить провод: отсоединить и тянуть от его начала
                    self.push_undo()
                    self.graph.disconnect(nid, port)
                    sx, sy = self.w2s(*self.port_w(src[0], src[1], False))
                    k = dict(ND.REG[self.graph.nodes[src[0]]['type']].outs)[src[1]]
                    self._drag = ['wire', (src[0], src[1], False, k), True, sx, sy, e.x, e.y]
                    self.schedule()
                    self.redraw()
                    return
            sx, sy = self.w2s(*self.port_w(nid, port, is_in))
            self._drag = ['wire', (nid, port, is_in, kind), False, sx, sy, e.x, e.y]
            return
        nid = self.hit_node(e.x, e.y)
        shift = bool(e.state & 0x0001)
        if nid:
            sel = self.st['sel']
            if shift:
                if nid in sel:
                    sel.remove(nid)
                else:
                    sel.append(nid)
            elif nid not in sel:
                self.st['sel'] = [nid]
            self.set_active(nid)
            wx, wy = self.s2w(e.x, e.y)
            offs = {k: (self.graph.nodes[k]['x'] - wx, self.graph.nodes[k]['y'] - wy)
                    for k in self.st['sel'] if k in self.graph.nodes}
            self._drag = ['move', offs, False]
            # перемещённый узел рисуется поверх остальных
            n = self.graph.nodes.pop(nid)
            self.graph.nodes[nid] = n
        else:
            if not shift:
                self.st['sel'] = []
                self.set_active(None)
            self._drag = ['box', e.x, e.y, e.x, e.y]
        self.redraw()

    def on_motion(self, e):
        """Drag: extend the wire, move the selection or grow the box."""
        d = self._drag
        if not d:
            return
        if d[0] == 'wire':
            d[5], d[6] = e.x, e.y
        elif d[0] == 'move':
            if not d[2]:
                self.push_undo()
                d[2] = True
            wx, wy = self.s2w(e.x, e.y)
            for k, (dx, dy) in d[1].items():
                if k in self.graph.nodes:
                    self.graph.nodes[k]['x'] = round(wx + dx)
                    self.graph.nodes[k]['y'] = round(wy + dy)
        elif d[0] == 'box':
            d[3], d[4] = e.x, e.y
        self.redraw()

    def on_release(self, e):
        """Finish a drag: connect a wire, offer nodes for a loose wire, box-select."""
        d, self._drag = self._drag, None
        if not d:
            return
        if d[0] == 'wire':
            (nid, port, is_in, kind), picked = d[1], d[2]
            hit = self.hit_port(e.x, e.y)
            if (hit and not is_in and not picked and hit[0] == nid and hit[1] == port
                    and abs(e.x - d[3]) < 6 and abs(e.y - d[4]) < 6):
                # щелчок по выходу без протяжки — показать этот выход
                self.choose_port(nid, port)
                return
            if hit and hit[2] != is_in and hit[0] != nid:
                a, ap, b, bp = ((nid, port, hit[0], hit[1]) if not is_in
                                else (hit[0], hit[1], nid, port))
                if not picked:
                    self.push_undo()
                if self.graph.connect(a, ap, b, bp):
                    self.touch()
                    self.schedule()
                else:
                    self.status(t('g.badlink'))
            elif not hit and self.hit_node(e.x, e.y) is None and not picked:
                self.add_menu(e.x_root, e.y_root, e.x, e.y, wire=(nid, port, is_in, kind))
            elif picked:
                self.touch()
        elif d[0] == 'move' and d[2]:
            self.touch()
        elif d[0] == 'box' and (abs(d[3] - d[1]) > 4 or abs(d[4] - d[2]) > 4):
            x0, y0 = self.s2w(min(d[1], d[3]), min(d[2], d[4]))
            x1, y1 = self.s2w(max(d[1], d[3]), max(d[2], d[4]))
            got = [k for k, n in self.graph.nodes.items()
                   if n['x'] < x1 and n['x'] + NW > x0 and n['y'] < y1
                   and n['y'] + self.node_h(k) > y0]
            sel = self.st['sel']
            for k in got:
                if k not in sel:
                    sel.append(k)
            if got:
                self.set_active(got[-1])
        self.redraw()

    def on_double(self, e):
        """Double click on empty space: the add menu."""
        if self.hit_node(e.x, e.y) is None:
            self.add_menu(e.x_root, e.y_root, e.x, e.y)

    def on_menu(self, e):
        """Right click: node menu on a node, add menu on empty space."""
        nid = self.hit_node(e.x, e.y)
        if not nid:
            self.add_menu(e.x_root, e.y_root, e.x, e.y)
            return
        if nid not in self.st['sel']:
            self.st['sel'] = [nid]
        self.set_active(nid)
        self.redraw()
        m = tk.Menu(self.root, tearoff=0, background=PANEL, foreground=FG,
                    activebackground=EDGE, activeforeground=FG)
        m.add_command(label=t('ed.m.dup'), accelerator='Ctrl+D', command=self.duplicate)
        m.add_command(label=t('ed.m.del'), accelerator='X', command=self.delete_sel)
        m.add_separator()
        m.add_command(label=t('ed.m.unlink'), command=lambda: self.unlink(nid))
        m.tk_popup(e.x_root, e.y_root)

    def menu_at_mouse(self):
        """Shift+A / Tab: add menu at the mouse pointer."""
        x, y = getattr(self, '_mouse', (100, 100))
        self.add_menu(self.cv.winfo_rootx() + x, self.cv.winfo_rooty() + y, x, y)

    def add_menu(self, rx, ry, sx, sy, wire=None):
        """Popup menu of node types by category.

        wire: (node, port, is_input, kind) of a wire dropped on empty space —
        then only nodes that can take it are listed, and the new node is
        connected to it.
        """
        m = tk.Menu(self.root, tearoff=0, background=PANEL, foreground=FG,
                    activebackground=EDGE, activeforeground=FG)
        for cat in ND.CATEGORIES:
            keys = [k for k, sp in ND.REG.items() if sp.cat == cat and not sp.hidden]
            if wire:
                kind, want_in = wire[3], not wire[2]
                keys = [k for k in keys if any(
                    (ND.kinds_fit(kind, kk) if want_in else ND.kinds_fit(kk, kind))
                    for _, kk in (ND.REG[k].ins if want_in else ND.REG[k].outs))]
            if not keys:
                continue
            sub = tk.Menu(m, tearoff=0, background=PANEL, foreground=FG,
                          activebackground=EDGE, activeforeground=FG)
            for k in sorted(keys, key=lambda k: t('nd.' + k)):
                sub.add_command(label=t('nd.' + k),
                                command=lambda k=k: self.add_node(k, sx, sy, wire))
            m.add_cascade(label=t('cat.' + cat), menu=sub)
        m.tk_popup(rx, ry)

    # ------------------------------------------------------------ правка ---
    # ------------------------------------------------------- заплатка ----
    # Узел «Заплатка»: пока он активен, в 2D левой кнопкой обводится место,
    # которое надо исправить, а потом обводка перетаскивается туда, откуда
    # взять замену — как инструмент «Заплатка» в Photoshop. Правая кнопка —
    # снять обводку; вид двигается средней кнопкой, колесо — масштаб.
    def patch_mode(self, nid):
        """Turn the 2D view's patch drawing on for patch node ``nid``, or
        off (nid None)."""
        v = self.v2
        if nid is None:
            if v.paint_cb == self._patch_ev:
                v.paint_on, v.paint_cb, v.overlay_cb = False, None, None
                self._pt.update(mode=None, pts=[], sel=None)
            return
        v.paint_on, v.paint_cb, v.overlay_cb = True, self._patch_ev, self._patch_overlay

    def _patch_ev(self, x, y, erase, final):
        """Mouse in the 2D view while a patch node is active (image pixels of
        the full-size texture): outline, then drag the outline to the source."""
        pt = self._pt
        nid = self.st.get('view')
        if nid not in self.graph.nodes or self.st['src'] is None:
            return
        if erase:
            pt.update(mode=None, pts=[], sel=None)
            self.v2.render()
            return
        if pt['mode'] is None:                      # нажатие
            if pt['sel'] and _inside((x, y), pt['sel']):
                pt.update(mode='drag', start=(x, y), cur=(x, y))
            else:
                pt.update(mode='lasso', pts=[(x, y)], sel=None)
        elif pt['mode'] == 'lasso':
            if (x, y) != pt['pts'][-1]:
                pt['pts'].append((x, y))
        else:
            pt['cur'] = (x, y)
        if final:
            H, W = self.st['src'].shape[:2]
            if pt['mode'] == 'lasso':
                pts = pt['pts']
                pt['sel'] = pts if len(pts) >= 3 and _area(pts) >= 4 else None
                if pt['sel'] and self.graph.nodes[nid]['params'].get('auto'):
                    # автоподбор: откуда взять — ищется по совпадению окрестности
                    poly = [[px / W, py / H] for px, py in pt['sel']]
                    off = self._auto_source(nid, poly)
                    if off is not None:
                        self._add_patch(nid, pt['sel'], off[0], off[1])
            elif pt['mode'] == 'drag':
                (x0, y0), (x1, y1) = pt['start'], pt['cur']
                if abs(x1 - x0) + abs(y1 - y0) >= 2:
                    self._add_patch(nid, pt['sel'], (x1 - x0) / W, (y1 - y0) / H)
            pt['mode'] = None
        self.v2.render()

    def _add_patch(self, nid, sel, dx, dy):
        """Store one patch: outline ``sel`` (full-size pixels), shift (dx, dy)
        in fractions of the size; one undo step, recompute."""
        H, W = self.st['src'].shape[:2]
        new = {'poly': [[round(px / W, 5), round(py / H, 5)] for px, py in sel],
               'dx': round(float(dx), 5), 'dy': round(float(dy), 5)}
        self.push_undo()
        prm = self.graph.nodes[nid]['params']
        prm['patches'] = list(prm.get('patches') or []) + [new]
        self.touch()
        self.schedule()
        self.fill_inspector()
        self._pt['sel'] = None

    def _auto_source(self, nid, poly):
        """Shift to take a patch from, found on what the node shows now
        (earlier patches included), or on its input; None (with a status
        message) if nothing is computed yet or nothing fits."""
        import imgfilters as F
        v = (self.results.get(nid) or {}).get('h')
        if not isinstance(v, np.ndarray):
            src = self.graph.inputs_of(nid).get('h')
            v = (self.results.get(src[0]) or {}).get(src[1]) if src else None
        if not isinstance(v, np.ndarray):
            self.status(t('ed.patch.wait'))
            return None
        off = F.find_source(v, poly)
        if off is None:
            self.status(t('ed.patch.nofit'))
        return off

    def _patch_overlay(self, v):
        """Draw the patches of the active node and the outline being made."""
        nid = self.st.get('view')
        if nid not in self.graph.nodes or self.st['src'] is None:
            return
        H, W = self.st['src'].shape[:2]

        def scr(pts, dx=0.0, dy=0.0):
            """Image pixels → canvas coordinates, flattened."""
            out = []
            for x, y in pts:
                out += [(x + dx) * v.scale + v.ox, (y + dy) * v.scale + v.oy]
            return out
        for p in self.graph.nodes[nid]['params'].get('patches') or []:
            pts = [(x * W, y * H) for x, y in p['poly']]
            if len(pts) >= 3:
                v.create_polygon(scr(pts), outline='#e0c060', fill='', width=1)
                # источник показываем внутри текстуры: сдвиг идёт через край
                cx = sum(x for x, _ in pts) / len(pts) + p['dx'] * W
                cy = sum(y for _, y in pts) / len(pts) + p['dy'] * H
                sx = p['dx'] * W - (cx // W) * W
                sy = p['dy'] * H - (cy // H) * H
                v.create_polygon(scr(pts, sx, sy), outline='#6ec0e0',
                                 fill='', width=1, dash=(3, 3))
        pt = self._pt
        if pt['mode'] == 'lasso' and len(pt['pts']) >= 2:
            v.create_line(scr(pt['pts']), fill='white', width=1)
        if pt['sel']:
            v.create_polygon(scr(pt['sel']), outline='white', fill='', width=1, dash=(4, 3))
            if pt['mode'] == 'drag':
                (x0, y0), (x1, y1) = pt['start'], pt['cur']
                v.create_polygon(scr(pt['sel'], x1 - x0, y1 - y0), outline='#6ec0e0',
                                 fill='', width=2)

    def patch_block(self, nid):
        """Inspector part of the patch node: how to use it and two buttons."""
        body = sec(self.insp, 'ed.patch')
        ttk.Label(body, text=t('ed.patch.how'), style='Hint.TLabel', wraplength=300,
                  justify='left').pack(anchor='w', pady=(2, 4))
        n = len(self.graph.nodes[nid]['params'].get('patches') or [])
        ttk.Label(body, text=t('ed.patch.n', n)).pack(anchor='w')

        def drop(last):
            """Remove the last patch, or all of them (one undo step)."""
            prm = self.graph.nodes[nid]['params']
            cur = list(prm.get('patches') or [])
            if not cur:
                return
            self.push_undo()
            prm['patches'] = cur[:-1] if last else []
            self.touch()
            self.schedule()
            self.fill_inspector()
            self.v2.render()
        row = ttk.Frame(body)
        row.pack(fill='x', pady=(4, 0))
        ttk.Button(row, text=t('ed.patch.undo'), command=lambda: drop(True),
                   style='Small.TButton').pack(side='left')
        ttk.Button(row, text=t('ed.patch.clear'), command=lambda: drop(False),
                   style='Small.TButton').pack(side='left', padx=4)

    # ------------------------------------------------- рисованная маска ---
    # Узел «Нарисовать маску»: пока он открыт в инспекторе и показан в 2D,
    # левая кнопка в 2D рисует белым, правая — чёрным. Размер, жёсткость и
    # непрозрачность кисти — в инспекторе; они общие для всех таких узлов и
    # записываются в каждый мазок. Вид двигается средней кнопкой.
    def brush(self):
        """Brush settings: {'size': px of the full-size texture (diameter),
        'hard': 0…1, 'op': 0…1}, kept with the editor state."""
        return self.st.setdefault('brush', {'size': 40, 'hard': 0.7, 'op': 1.0})

    def mask_mode(self, nid):
        """Turn mask painting in the 2D view on for paint-mask node nid, or
        off (nid None)."""
        v = self.v2
        if nid is None:
            if v.paint_cb == self._mask_ev:
                v.paint_on, v.paint_cb, v.overlay_cb = False, None, None
                self._mk.update(v=None, pts=[])
                v.delete('mcursor')
            return
        v.paint_on, v.paint_cb, v.overlay_cb = True, self._mask_ev, None

    def _brush_cursor(self, e=None, x=None, y=None):
        """Circle of the brush size round the mouse while painting a mask."""
        v = self.v2
        v.delete('mcursor')
        if v.paint_cb != self._mask_ev:
            return
        if e is not None:
            x, y = e.x, e.y
        r = max(1.0, self.brush()['size'] / 2.0 * v.scale)
        v.create_oval(x - r, y - r, x + r, y + r, outline=FG, tags='mcursor')
        v.create_oval(x - r - 1, y - r - 1, x + r + 1, y + r + 1, outline='#000000',
                      tags='mcursor')

    def _mask_ev(self, x, y, erase, final):
        """Mouse in the 2D view while a paint-mask node is active (pixels of
        the full-size texture): collect the stroke, draw it at once on the
        view, store it on release (one undo step) and recompute."""
        nid = self.st.get('view')
        if nid not in self.graph.nodes or self.st['src'] is None:
            return
        H, W = self.st['src'].shape[:2]
        mk, b, v = self._mk, self.brush(), self.v2
        if mk['v'] is None:                          # начало мазка
            mk.update(v=0.0 if erase else 1.0, pts=[])
        r = b['size'] / 2.0
        pts = mk['pts']
        if not pts or (x - pts[-1][0]) ** 2 + (y - pts[-1][1]) ** 2 >= max(1.0, r / 4) ** 2:
            # мазок сразу виден на 2D, не дожидаясь пересчёта
            s, sx, sy = v.scale, v.ox + x * v.scale, v.oy + y * v.scale
            col = '#ffffff' if mk['v'] else '#000000'
            if pts:
                px, py = v.ox + pts[-1][0] * s, v.oy + pts[-1][1] * s
                v.create_line(px, py, sx, sy, fill=col, width=max(1.0, 2 * r * s),
                              capstyle='round', tags='mstroke')
            else:
                v.create_oval(sx - r * s, sy - r * s, sx + r * s, sy + r * s, fill=col,
                              outline='', tags='mstroke')
            pts.append((x, y))
        self._brush_cursor(x=v.ox + x * v.scale, y=v.oy + y * v.scale)
        if not final:
            return
        new = {'pts': [[round(px / W, 5), round(py / H, 5)] for px, py in pts],
               'r': round(r / float(W), 6), 'hard': round(float(b['hard']), 3),
               'op': round(float(b['op']), 3), 'v': mk['v']}
        mk.update(v=None, pts=[])
        self.push_undo()
        prm = self.graph.nodes[nid]['params']
        prm['strokes'] = list(prm.get('strokes') or []) + [new]
        self.touch()
        self.schedule(60)
        lab = getattr(self, '_mask_n', None)
        if lab is not None and lab.winfo_exists():
            lab.config(text=t('ed.mask.n', len(prm['strokes'])))

    def mask_block(self, nid):
        """Inspector part of the paint-mask node: the brush and two buttons."""
        body = sec(self.insp, 'ed.mask')
        ttk.Label(body, text=t('ed.mask.how'), style='Hint.TLabel', wraplength=300,
                  justify='left').pack(anchor='w', pady=(2, 4))
        b = self.brush()
        for key, lo, hi, integer in (('size', 1, 1000, True), ('hard', 0.0, 1.0, False),
                                     ('op', 0.0, 1.0, False)):
            sl = Slider(body, t('ed.mask.' + key), lo, hi, b[key], '', lambda: None,
                        integer=integer)
            sl.var.trace_add('write', lambda *a, k=key, s=sl: self.brush().__setitem__(k, s.get()))
        n = len(self.graph.nodes[nid]['params'].get('strokes') or [])
        self._mask_n = ttk.Label(body, text=t('ed.mask.n', n))
        self._mask_n.pack(anchor='w', pady=(6, 0))

        def drop(last):
            """Remove the last stroke, or all of them (one undo step)."""
            prm = self.graph.nodes[nid]['params']
            cur = list(prm.get('strokes') or [])
            if not cur:
                return
            self.push_undo()
            prm['strokes'] = cur[:-1] if last else []
            self.touch()
            self.schedule()
            self._mask_n.config(text=t('ed.mask.n', len(prm['strokes'])))
        row = ttk.Frame(body)
        row.pack(fill='x', pady=(4, 0))
        ttk.Button(row, text=t('ed.mask.undo'), command=lambda: drop(True),
                   style='Small.TButton').pack(side='left')
        ttk.Button(row, text=t('ed.mask.clear'), command=lambda: drop(False),
                   style='Small.TButton').pack(side='left', padx=4)

    def sync_dxt(self):
        """Make the DXT choice of the batch settings the default for every
        save, the classic tabs included (foliage.DXT)."""
        b = self.st['batch']
        FO.DXT.update(algo=b.get('dxt_algo', 'quicktex'), level=int(b.get('dxt_level', 14)))

    def set_all_dx(self, dx):
        """Put every node that has a 'directx' setting into one normal-map
        layout: True — DirectX (green down), False — OpenGL (green up). One
        undo step; new nodes added afterwards get the same layout."""
        nodes = [n for n in self.graph.nodes.values() if 'directx' in n['params']]
        self.st['dx'] = bool(dx)
        if nodes:
            self.push_undo()
            for n in nodes:
                n['params']['directx'] = bool(dx)
            self.touch()
            self.schedule()
            self.fill_inspector()
        self.status(t('ed.dxall.done', 'DirectX' if dx else 'OpenGL', len(nodes)))

    def add_node(self, key, sx, sy, wire=None):
        """Add a node of type key at canvas point (sx, sy); connect a dropped wire."""
        self.push_undo()
        wx, wy = self.s2w(sx, sy)
        nid = self.graph.add(key, round(wx - (NW / 2 if not wire or not wire[2] else NW)),
                             round(wy - HEAD / 2))
        if 'dx' in self.st and 'directx' in self.graph.nodes[nid]['params']:
            # раскладка, выбранная кнопкой «Нормали» для всех узлов
            self.graph.nodes[nid]['params']['directx'] = self.st['dx']
        if wire:
            src, port, is_in, kind = wire
            sp = ND.REG[key]
            lst = sp.ins if not is_in else sp.outs
            p2 = next((p for p, kk in lst if kk == kind), None) or \
                next((p for p, kk in lst if ND.ANY in (kk, kind)), None) or \
                next((p for p, kk in lst if ND.kinds_fit(kk, kind)), None)
            if p2:
                if is_in:
                    self.graph.connect(nid, p2, src, port)
                else:
                    self.graph.connect(src, port, nid, p2)
        self.st['sel'] = [nid]
        self.set_active(nid)
        self.touch()
        self.redraw()
        self.schedule()

    def delete_sel(self):
        """Delete the selected nodes."""
        if not self.st['sel']:
            return
        self.push_undo()
        for nid in self.st['sel']:
            self.graph.remove(nid)
            self.results.pop(nid, None)
            self.errors.pop(nid, None)
            self.thumbs.pop(nid, None)
        self.st['sel'] = []
        self.set_active(None)
        self.touch()
        self.redraw()
        self.schedule()

    def duplicate(self):
        """Copy the selected nodes (with the wires between them) 30 px aside."""
        sel = [k for k in self.st['sel'] if k in self.graph.nodes]
        if not sel:
            return
        self.push_undo()
        g = self.graph
        new = {}
        for k in sel:
            n = g.nodes[k]
            nk = g.add(n['type'], n['x'] + 30, n['y'] + 30)
            g.nodes[nk]['params'] = dict(n['params'])
            new[k] = nk
        for a, ap, b, bp in list(g.links):
            if b in new:
                g.connect(new.get(a, a), ap, new[b], bp)
        self.st['sel'] = list(new.values())
        self.set_active(self.st['sel'][-1])
        self.touch()
        self.redraw()
        self.schedule()

    def unlink(self, nid):
        """Remove all wires into and out of a node."""
        self.push_undo()
        g = self.graph
        g.links = [l for l in g.links if l[0] != nid and l[2] != nid]
        self.touch()
        self.redraw()
        self.schedule()

    def select_all(self):
        """Ctrl+A."""
        self.st['sel'] = list(self.graph.nodes)
        self.redraw()

    def toggle_thumbs(self):
        """H: node thumbnails on/off (compact graph)."""
        self.st['thumbs'] = not self.st['thumbs']
        self.redraw()

    def frame_all(self):
        """F / Home: fit all nodes into the canvas."""
        if self.dead:
            return
        g = self.graph
        if not g.nodes:
            return
        xs = [n['x'] for n in g.nodes.values()]
        ys = [n['y'] for n in g.nodes.values()]
        x0, y0 = min(xs) - 30, min(ys) - 30
        x1 = max(xs) + NW + 30
        y1 = max(n['y'] + self.node_h(k) for k, n in g.nodes.items()) + 30
        W, H = max(self.cv.winfo_width(), 100), max(self.cv.winfo_height(), 100)
        z = min(1.2, max(0.25, min(W / (x1 - x0), H / (y1 - y0))))
        self.st['zoom'] = z
        self.st['ox'] = (W - (x1 - x0) * z) / 2 - x0 * z
        self.st['oy'] = (H - (y1 - y0) * z) / 2 - y0 * z
        self.redraw()

    def push_undo(self):
        """Remember the graph before a change."""
        self.undo_stack.append(self.graph.to_json())
        del self.undo_stack[:-UNDO_MAX]
        self.redo_stack.clear()

    def _restore(self, text):
        """Replace the graph content from JSON, keeping the evaluation cache.

        The cache is keyed by content, so entries of nodes that did not change
        are still right after an undo."""
        cache = self.graph._cache
        self.graph.from_json(text)
        self.graph._cache = cache
        self.st['sel'] = [k for k in self.st['sel'] if k in self.graph.nodes]
        if self.st.get('view') not in self.graph.nodes:
            self.st['view'] = None
        self.touch()
        self.redraw()
        self.fill_inspector()
        self.schedule()

    def undo(self):
        """Ctrl+Z."""
        if self.undo_stack:
            self.redo_stack.append(self.graph.to_json())
            self._restore(self.undo_stack.pop())

    def redo(self):
        """Ctrl+Y / Ctrl+Shift+Z."""
        if self.redo_stack:
            self.undo_stack.append(self.graph.to_json())
            self._restore(self.redo_stack.pop())

    def set_active(self, nid):
        """Make nid the node shown in the inspector and the 2D view."""
        if nid != self.st.get('view'):
            self.st['view'] = nid
        self.fill_inspector()
        self.show_2d()

    # ------------------------------------------------------------- счёт ----
    def preview_src(self, limit=PREVIEW_MAX):
        """(src, alpha, px) for the preview: the texture, or a copy reduced so
        that its longer side is at most ``limit`` (px = the reduction).
        Remembered per texture and limit."""
        src, alpha = self.st['src'], self.st['alpha']
        memo = self.__dict__.setdefault('_prev_memo', {})
        key = (id(src), limit)
        if key in memo:
            return memo[key]
        memo.clear() if len(memo) > 4 else None
        h, w = src.shape[:2]
        if max(h, w) <= limit:
            memo[key] = (src, alpha, 1.0)
            return memo[key]
        k = limit / float(max(h, w))
        hh, ww = max(1, int(round(h * k))), max(1, int(round(w * k)))
        s2 = np.clip(TX.resize_wrap(src, hh, ww), 0, 1)
        a2 = None if alpha is None else np.clip(TX.resize_wrap(alpha, hh, ww), 0, 1)
        memo[key] = (s2, a2, ww / float(w))
        return memo[key]

    def schedule(self, delay=160):
        """Ask for a re-evaluation of the graph after ``delay`` ms.

        Two passes: first a fast one on a copy of at most PREVIEW_FAST
        pixels, then — if nothing changed meanwhile — a fine one at up to
        PREVIEW_MAX. While a slider is being moved only fast passes run."""
        if self._pending:
            self.root.after_cancel(self._pending)
        self._pending = self.root.after(delay, lambda: self.evaluate(False))

    def evaluate(self, fine=False):
        """Evaluate every node in a worker thread (see module docstring).

        fine: False — the fast pass (copy ≤ PREVIEW_FAST), True — the fine
        pass (≤ PREVIEW_MAX); poll() starts the fine one after a fast one."""
        if self.dead:
            return
        self._pending = None
        if self.st['src'] is None:
            return
        if InstallWindow.active:
            # пока ставится onnxruntime, сети не трогаем: загруженную сборку
            # нельзя заменить
            self.schedule(1000)
            return
        if self._busy:
            self._again = True
            self._gen += 1                 # текущий проход пусть бросит работу
            return
        if not self.st.get('draft', True):
            fine = True                    # черновой проход выключен
        limit = PREVIEW_MAX if fine else PREVIEW_FAST
        self._stage_fine = fine or max(self.st['src'].shape[:2]) <= PREVIEW_FAST
        self._prev = self.preview_src(limit)
        src, alpha, px = self._prev
        self.res_label.config(text=t('ed.prevres', src.shape[1], src.shape[0]) +
                              ('' if self._stage_fine else '  ' + t('ed.refining')))
        g2 = ND.Graph()
        g2.from_json(self.graph.to_json())
        g2._cache = self.graph._cache            # кэш общий: ключи по содержимому
        # ошибка узла, записанная для прежнего вида графа (до новой связи или
        # другого параметра выше), уже неправда: убираем её, пока узел не
        # пересчитан — иначе при долгом проходе висит старое сообщение
        ek = self._err_key
        for nid in list(self.errors):
            try:
                same = nid in g2.nodes and ek.get(nid) == g2.key_of(nid, '')
            except Exception:                    # noqa: BLE001
                same = False
            if not same:
                self.errors.pop(nid, None)
                ek.pop(nid, None)
        self._gen += 1
        gen = self._gen
        self._busy, self._again = True, False
        self._done = False
        self._w_prog = 0.0
        # исходник кладём на видеокарту один раз (если она включена)
        ctx = {'src': X.to_dev(np.ascontiguousarray(src)), 'alpha': X.to_dev(alpha), 'px': px,
               'key': 'prev:%d:%.4f' % (id(self.st['src']), px),
               'name': os.path.basename(self.st['tex'] or '')}
        order = g2.order()

        def sub(i, n):
            """Progress of a long node (network windows)."""
            self._w_sub = (i, n)

        ctx['sub'] = sub
        res, errs, th = self.results, self.errors, self.thumbs

        def work():
            """Worker: evaluate nodes in order, fill results/errors/thumbnails."""
            try:
                for k, nid in enumerate(order):
                    if self._gen != gen:
                        return
                    self._running = nid
                    self._w_sub = None
                    self._w_prog = k / max(len(order), 1)
                    try:
                        # в графе результаты остаются на видеокарте; окну
                        # (миниатюры, 2D, 3D, показания) — копия на процессоре
                        out = X.host_tree(g2.eval(nid, ctx))
                        res[nid] = out
                        errs.pop(nid, None)
                        v = out.get(self.vport().get(nid))
                        if not isinstance(v, np.ndarray):
                            v = next((x for kk, x in out.items() if not kk.startswith('_')
                                      and isinstance(x, np.ndarray)), None)
                        if v is not None:
                            th[nid] = thumb_of(v)
                        else:
                            th.pop(nid, None)
                    except Exception as e:           # noqa: BLE001 — показать на узле
                        res.pop(nid, None)
                        th.pop(nid, None)
                        errs[nid] = str(e) or e.__class__.__name__
                        try:
                            self._err_key[nid] = g2.key_of(nid, '')
                        except Exception:            # noqa: BLE001
                            self._err_key[nid] = None
            finally:
                self._running = None
                self._done = True
        self._t0 = time.time()
        threading.Thread(target=work, daemon=True).start()
        self.poll()

    def poll(self):
        """Main-thread timer: show progress and pick up finished results."""
        if self.dead:
            return
        if not self._done:
            sub = getattr(self, '_w_sub', None)
            run = self._running
            txt = t('g.node', t('nd.' + self.graph.nodes[run]['type'])) \
                if run in self.graph.nodes else t('com.computing')
            if sub:
                txt += '  %d/%d' % sub
            self.status(txt)
            self.bar['value'] = self._w_prog
            self.redraw()
            self.root.after(150, self.poll)
            return
        self._busy = False
        self.bar['value'] = 0
        # от удалённых узлов ничего не храним
        for d in (self.results, self.errors, self.thumbs):
            for k in [k for k in d if k not in self.graph.nodes]:
                d.pop(k, None)
        n_err = len(self.errors)
        self.status(t('ed.done', time.time() - self._t0) +
                    ('  ·  ' + t('ed.errors', n_err) if n_err else ''))
        self.redraw()
        self.show_2d()
        self.show_readouts()
        self.show_3d()
        if self._again:
            self._again = False
            self.evaluate(False)
        elif not self._stage_fine:
            # быстрый проход готов и ничего не менялось — уточняем
            self._pending = self.root.after(250, lambda: self.evaluate(True))

    # ------------------------------------------------------------- 2D -------
    def shown(self):
        """Node the 2D view shows: the pinned one ("pin" check box above
        the 2D view) while it exists, otherwise the active node."""
        p = self.st.get('pin')
        if p is not None and p not in self.graph.nodes:
            # закреплённый узел удалён — закрепление снимается
            self.st['pin'] = p = None
            if hasattr(self, 'pin_var'):
                self.pin_var.set(False)
        return self.st.get('view') if p is None else p

    def toggle_pin(self):
        """The pin check box: on — the 2D view keeps showing the node shown
        now, while clicks on other nodes change only the inspector; off — the
        2D view follows the active node again."""
        nid = self.st.get('view')
        if self.pin_var.get() and nid in self.graph.nodes:
            self.st['pin'] = nid
        else:
            self.st['pin'] = None
            self.pin_var.set(False)
        self.show_2d()
        self.redraw()

    def view_value(self):
        """(value, node id, port) shown in the 2D view, or (None, …)."""
        nid = self.shown()
        if nid is None:
            outs = self.graph.outputs()
            nid = outs[0] if outs else None
        if nid is None or nid not in self.results:
            return None, nid, None
        out = self.results[nid]
        ports = [k for k, v in out.items() if not k.startswith('_')
                 and isinstance(v, np.ndarray)]
        if not ports:
            return None, nid, None
        want = self.vport().get(nid)
        port = want if want in ports else ports[0]
        return out[port], nid, port

    def vport(self):
        """Which output each node shows (2D view and its thumbnail):
        dict node id → port. Kept with the project state, so it survives a
        rebuild of the window; a node not listed shows its first output."""
        return self.st.setdefault('vport', {})

    def choose_port(self, nid, port):
        """Show output ``port`` of node ``nid``: in the 2D view and on the
        node's thumbnail. Used by the list above the 2D view and by a click on
        an output socket."""
        self.vport()[nid] = port
        out = self.results.get(nid) or {}
        v = out.get(port)
        if isinstance(v, np.ndarray):
            self.thumbs[nid] = thumb_of(v)
        if nid == self.shown():
            self.show_2d()
            self.show_readouts()
        elif self.st.get('pin') is not None:
            # при закреплении щелчок по выходу переносит закрепление на этот
            # узел, инспектор остаётся прежним
            self.st['pin'] = nid
            self.show_2d()
        else:
            self.set_active(nid)
        self.redraw()

    def _port_chosen(self):
        """Output port picked in the 2D header."""
        i = self.port_cb.current()
        nid = self.shown()
        if nid in self.graph.nodes and i >= 0:
            outs = ND.REG[self.graph.nodes[nid]['type']].outs
            if i < len(outs):
                self.choose_port(nid, outs[i][0])
                return
        self.show_2d()

    def show_2d(self):
        """Put the viewed node's value into the 2D view with its numbers."""
        if self.dead:
            return
        v, nid, port = self.view_value()
        if nid in self.graph.nodes:
            sp = ND.REG[self.graph.nodes[nid]['type']]
            pinned = nid == self.st.get('pin') and nid != self.st.get('view')
            self.v2_title.config(text=t('nd.' + sp.key) + ('  ' + t('ed.pinned') if pinned else ''))
            self.port_cb.config(values=[ND.label(sp.key, p) for p, _ in sp.outs])
            self.port_var.set(ND.label(sp.key, port) if port else '')
        else:
            self.v2_title.config(text='2D')
            self.port_cb.config(values=[])
            self.port_var.set('')
        if v is None:
            self.v2.delete('all')
            self.v2.img = None
            txt = self.errors.get(nid, '') if nid else ''
            self.v2_info.config(text=txt[:200])
            return
        dx = True
        if nid in self.graph.nodes:
            dx = bool(self.graph.nodes[nid]['params'].get('directx', True))
        base = alpha = None
        if self.lit.get():
            # цвет — диффуз графа (выход «diffuse» обычного набора: диффуз
            # набора PBR — без AO, он для другого шейдера), иначе сама текстура
            for o in self.graph.outputs():
                pp = self.graph.nodes[o]['params']
                if pp.get('name') == 'diffuse' and pp.get('set') != 'pbr':
                    r = (self.results.get(o) or {}).get('any')
                    if isinstance(r, np.ndarray) and r.ndim == 3:
                        base = r
            prev = getattr(self, '_prev', None)
            if base is None:
                base = prev[0] if prev else self.st['src']
            alpha = prev[1] if prev else self.st['alpha']
        img = display(v, self.chan.get(), self.lit.get(), dx, tuple(self.ring.vector()),
                      base, alpha)
        ref = self.st['src'].shape[:2] if self.st['src'] is not None else None
        typ = (self.graph.nodes[nid]['type'] if nid in self.graph.nodes
               and nid == self.st.get('view') else None)
        self.patch_mode(nid if typ == 'patch' else None)
        self.mask_mode(nid if typ == 'paint_mask' else None)
        self.v2.set_image(img, keep_view=True, ref=ref)
        a = np.asarray(v, np.float32)
        kind = ('%d×%d' % (a.shape[1], a.shape[0]) +
                ('  ×%d' % a.shape[2] if a.ndim == 3 else ''))
        self.v2_info.config(text='%s   min %.3f  max %.3f  mean %.3f' % (
            kind, float(a.min()), float(a.max()), float(a.mean())))

    # ------------------------------------------------------------- 3D -------
    def maps_3d(self):
        """Maps for view3d.render from the graph's outputs (preview size).

        _nh gives the normal and the height (alpha, used as is by the OpenMW
        mode); _n a normal; 'diffuse' the colour (the source otherwise);
        _spec colour and exponent; _ao the occlusion for hemispherical
        ambient. Missing maps are neutral.
        """
        if self.st['src'] is None:
            return None
        # одни и те же массивы — одни и те же карты: иначе view3d каждый раз
        # заново строит пирамиды мипов (это половина времени кадра)
        key = (id(getattr(self, '_prev', (None,))[0]),) + tuple(
            (self.graph.nodes[n]['params']['name'], id((self.results.get(n) or {}).get('any')))
            for n in self.graph.outputs()) + tuple(
            id((self.results.get(n) or {}).get('sss')) for n in self.graph.nodes
            if self.graph.nodes[n]['type'] == 'foliage')
        memo = getattr(self, '_m3', None)
        if memo is not None and memo[0] == key:
            return memo[1]
        m = X.dev_tree(self._maps_3d())       # 3D считается там же, где узлы
        self._m3 = (key, m)
        return m

    def _maps_3d(self):
        """Build the maps for maps_3d (not memoised)."""
        outs = {}
        for nid in self.graph.outputs():
            if self.graph.nodes[nid]['params'].get('set') == 'pbr':
                continue        # набор PBR — карты для другого шейдера, 3D их не читает
            r = self.results.get(nid)
            if r is not None and r.get('any') is not None:
                outs[self.graph.nodes[nid]['params']['name']] = np.asarray(r['any'], np.float32)
        src = self._prev[0] if getattr(self, '_prev', None) else self.st['src']
        H, W = src.shape[:2]
        m = {'diffuse': src[..., :3], 'normal': None, 'height': np.full((H, W), 0.5, np.float32),
             'spec': np.zeros((H, W, 3), np.float32), 'shininess': np.zeros((H, W), np.float32),
             'ao': np.ones((H, W), np.float32)}
        nh = outs.get('_nh')
        if nh is not None and nh.ndim == 3 and nh.shape[2] == 4:
            n = nh[..., :3] * 2.0 - 1.0
            m['normal'] = n
            m['height'] = nh[..., 3]
            m['nh_alpha'] = nh[..., 3]
        elif '_n' in outs:
            n = outs['_n']
            m['normal'] = n if float(n.min()) < -0.01 else n[..., :3] * 2.0 - 1.0
        if '_h' in outs and outs['_h'].ndim == 2:
            m['height'] = outs['_h']
        if m['normal'] is None:
            m['normal'] = np.dstack([np.zeros((H, W)), np.zeros((H, W)), np.ones((H, W))]
                                    ).astype(np.float32)
        d = outs.get('diffuse')
        if d is not None and d.ndim == 3:
            m['diffuse'] = np.clip(d[..., :3], 0, 1)
        sp = outs.get('_spec')
        if sp is not None and sp.ndim == 3:
            m['spec'] = np.clip(sp[..., :3], 0, 1)
            if sp.shape[2] == 4:
                # альфа _spec — показатель Блинна-Фонга /255; у превью 2 + 400·s
                m['shininess'] = np.clip((sp[..., 3] * 255.0 - 2.0) / 400.0, 0, 1)
        if '_ao' in outs and outs['_ao'].ndim == 2:
            m['ao'] = outs['_ao']
        # просвет: выход «_sss» (цвет + сила в альфе, как файл вкладки), иначе
        # прямо из узла «Листва»
        ss = outs.get('_sss')
        if ss is not None and ss.ndim == 3 and ss.shape[2] == 4:
            m['sss'], m['sss_a'] = np.clip(ss[..., :3], 0, 1), np.clip(ss[..., 3], 0, 1)
        else:
            for nid, n in self.graph.nodes.items():
                r = self.results.get(nid) or {}
                if n['type'] == 'foliage' and r.get('sss') is not None:
                    m['sss'] = np.asarray(r['sss'], np.float32)
                    m['sss_a'] = np.asarray(r['sss_a'], np.float32)
                    break
        prev = getattr(self, '_prev', None)
        a = prev[1] if prev else self.st['alpha']
        if a is not None and float(np.min(a)) < 0.5:
            m['alpha'] = np.asarray(a, np.float32)   # листва и решётки: дырки видны
        for k in list(m):
            if m[k] is not None and m[k].shape[:2] != (H, W):
                m[k] = TX.resize_wrap(m[k], H, W)
        n = m['normal']
        m['normal'] = (n / np.maximum(np.sqrt((n * n).sum(-1, keepdims=True)), 1e-6)
                       ).astype(np.float32)
        return m

    def show_3d(self, live=False):
        """Render the 3D view.

        While the camera or the light is dragged: a small frame right here.
        Otherwise the full frame is rendered in a worker thread (numpy only)
        and put on the canvas when ready, so the window does not freeze for
        the half-second a large frame takes; a request that arrives while a
        frame is being rendered is served right after it.
        """
        if self.dead:
            return
        if getattr(self, '_v3_after', None):
            self.root.after_cancel(self._v3_after)
            self._v3_after = None
        m = self.maps_3d()
        if m is None:
            self.v3.note(t('v3.nomaps'))
            return
        coarse = bool(live or self.v3.dragging or LightRing.DRAG)
        cw, ch = max(self.v3.winfo_width(), 120), max(self.v3.winfo_height(), 90)
        if self.v3_flat.get():
            self.v3.show(self.flat_lit(m, cw, ch))
            return
        w = min(cw, 900)
        h = max(90, int(round(w * ch / float(cw))))
        if coarse:
            w, h = max(160, w // 2), max(100, h // 2)
        try:
            tiles = max(1, min(8, int(self.v3_tiles.get())))
        except Exception:
            tiles = 2
        dist, fov = self.v3.dist, 42.0
        last = getattr(self, '_v3_lastdist', None)
        if self.v3_game.get() and last and abs(self.v3.dist - last) > 1e-6:
            # колесо мыши в режиме «как в игре» двигает расстояние в юнитах
            self.v3_gdist.set(min(512.0, max(24.0, float(self.v3_gdist.get()) *
                                             self.v3.dist / last)))
        self._v3_lastdist = self.v3.dist
        if self.v3_game.get():
            # плоскость шириной 2 единицы превью = «Повторов текстуры» повторам;
            # расстояние в юнитах игры переводится в единицы превью. Раньше
            # здесь всегда было 6 повторов — листва мостилась, даже когда не надо
            rep = self.game_repeat()
            gd = float(self.v3_gdist.get())
            dist = gd / (tiles * rep / 2.0)
            fov = GAME_FOV
            self.v3_gdist_lab.config(text=units_text(gd))
        else:
            self.v3_gdist_lab.config(text='')
        self.v3_gdist_scale.state(['!disabled'] if self.v3_game.get() else ['disabled'])
        mode = self._v3_modes.get(self.v3_mode.get(), 'openmw')
        self.v3_depth_scale.state(['disabled'] if mode in ('openmw', 'flat') else ['!disabled'])
        kw = dict(yaw=self.v3.yaw, pitch=self.v3.pitch, dist=dist, fov=fov,
                  tiles=float(tiles), depth=float(self.v3_depth.get()) / 100.0,
                  steps=12 if coarse else 32, mode=mode,
                  shadow=not coarse and mode == 'occlusion',
                  light=tuple(self.ring.vector()), hemi=self.v3_hemi.get(),
                  sss_pow=float(self.v3_spow.get()))
        if coarse:
            self._v3_gen = getattr(self, '_v3_gen', 0) + 1       # старый кадр не нужен
            self.v3.show(X.to_host(V3.render(m, w, h, **kw)))
            if not self.v3.dragging:
                self._v3_after = self.root.after(250, lambda: self.show_3d(False))
            return
        if getattr(self, '_v3_busy', False):
            self._v3_again = True
            return
        self._v3_busy, self._v3_again, self._v3_img = True, False, None
        self._v3_gen = gen = getattr(self, '_v3_gen', 0) + 1

        def work():
            """Worker: render the frame (numpy only, no Tk)."""
            try:
                self._v3_img = (gen, X.to_host(V3.render(m, w, h, **kw)))
            except Exception:                            # noqa: BLE001
                self._v3_img = (gen, None)

        threading.Thread(target=work, daemon=True).start()
        self._v3_poll()

    def _light_moved(self):
        """The light ring moved: the 3D view, and the 2D view when it is lit."""
        self.show_3d(True)
        # окно 2D со «Светом» считается в полном размере превью: пока кольцо
        # тянут, его не трогаем, обновляем, когда отпустили
        if self.lit.get() and not LightRing.DRAG:
            self.show_2d()

    def flat_lit(self, m, cw, ch):
        """The 2D mode of the 3D view: the texture straight on, under the
        light ring. Light in front — colour × (0.30 + 1.30·max(n·L, 0)), as
        the "lit" view of the foliage tab; light behind — foliage.backlit,
        as its "against the light" view. A dark background where the alpha
        is cut. Fitted into the canvas (cw × ch); the wheel zooms.

        Only what is visible is computed, at the size it is shown: the maps
        are cropped and brought to that size once (_flat_prep, remembered
        until the maps, the canvas or the zoom change), and a move of the
        light costs a few operations on a picture the size of the canvas.
        Returns ch × cw × 3."""
        import foliage as FOL
        zoom = min(8.0, max(0.25, 2.6 / max(float(self.v3.dist), 1e-3)))
        key = (id(m), cw, ch, round(zoom, 4))
        memo = getattr(self, '_flat_memo', None)
        if memo is None or memo[0] != key:
            memo = (key, self._flat_prep(m, cw, ch, zoom))
            self._flat_memo = memo
        g, (dx0, dy0) = memo[1]
        bg = np.array([0.09, 0.10, 0.09], np.float32)
        out = np.tile(bg, (ch, cw, 1))
        if g is None:
            return out
        n, C, A = g['normal'], g['diffuse'], g['alpha'][..., None]
        L = np.asarray(self.ring.vector(), np.float32)
        if float(L[2]) < 0 and g.get('sss') is not None:
            # свет за карточкой — ровно «против света» вкладки листвы
            img = FOL.backlit(np.concatenate([C, A], -1), n, g['sss'], g['sss_a'], L,
                              power=float(self.v3_spow.get()))
        else:
            d = np.clip((n * L).sum(-1), 0, 1)
            img = (C * g['cpeak']) * (0.30 + 1.30 * d)[..., None]
            img = img * A + bg * (1 - A)
        hh, ww = img.shape[:2]
        out[dy0:dy0 + hh, dx0:dx0 + ww] = np.clip(img, 0, 1)
        return out

    def _flat_prep(self, m, cw, ch, zoom):
        """For flat_lit: the visible part of the maps at its on-screen size
        (dict, or None if nothing is visible) and where it goes on the
        canvas. Colour, normal, alpha, translucency."""
        def host(v):
            return None if v is None else np.asarray(X.to_host(v), np.float32)
        n = host(m['normal'])
        H, W = n.shape[:2]
        k = min(cw / float(W), ch / float(H)) * 0.96 * zoom
        x0, y0 = (cw - W * k) / 2.0, (ch - H * k) / 2.0
        # видимый кусок текстуры, в её пикселях
        cx0, cy0 = max(0, int(np.floor(-x0 / k))), max(0, int(np.floor(-y0 / k)))
        cx1 = min(W, int(np.ceil((cw - x0) / k)))
        cy1 = min(H, int(np.ceil((ch - y0) / k)))
        if cx1 <= cx0 or cy1 <= cy0:
            return None, (0, 0)
        dx0, dy0 = int(round(x0 + cx0 * k)), int(round(y0 + cy0 * k))
        dw = max(1, int(round((cx1 - cx0) * k)))
        dh = max(1, int(round((cy1 - cy0) * k)))
        # то, что вылезает за холст, обрезаем уже в экранных пикселях
        sx, sy = max(0, -dx0), max(0, -dy0)
        dx0, dy0 = max(0, dx0), max(0, dy0)
        dw2, dh2 = min(dw - sx, cw - dx0), min(dh - sy, ch - dy0)
        if dw2 <= 0 or dh2 <= 0:
            return None, (0, 0)
        filt = Image.BOX if k < 1 else Image.BILINEAR

        def fit(a):
            """Crop a map to the visible part and bring it to screen size."""
            if a is None:
                return None
            a = a[cy0:cy1, cx0:cx1]
            chans = [a] if a.ndim == 2 else [a[..., c] for c in range(a.shape[2])]
            res = [np.asarray(Image.fromarray(np.ascontiguousarray(c, np.float32), 'F')
                              .resize((dw, dh), filt), np.float32)[sy:sy + dh2, sx:sx + dw2]
                   for c in chans]
            return res[0] if a.ndim == 2 else np.stack(res, -1)
        C = np.clip(host(m['diffuse'])[..., :3], 0, 1)
        g = {'normal': fit(n), 'diffuse': fit(C),
             'cpeak': 0.95 / max(float(C.max()), 1e-6)}
        a = host(m.get('alpha'))
        g['alpha'] = ((fit(a) >= 0.5).astype(np.float32) if a is not None
                      else np.ones(g['normal'].shape[:2], np.float32))
        nn = g['normal']
        g['normal'] = nn / np.maximum(np.sqrt((nn * nn).sum(-1, keepdims=True)), 1e-6)
        if m.get('sss') is not None and m.get('sss_a') is not None:
            g['sss'], g['sss_a'] = fit(host(m['sss'])), fit(host(m['sss_a']))
        return g, (dx0, dy0)

    def repeat_switch(self, nid, mode):
        """Parameters for switching node nid's repeat to ``mode`` without
        changing the repeat itself: «in units» and «in proportion to the
        resolution» are two ways of writing the same size. Going to
        'per_px' the units per pixel become repeat / width, going to 'units'
        the repeat becomes units per pixel × width (the full-size width of
        the current texture). The value is kept inside its slider's range;
        if it had to be cut, the status bar says so. Returns the dict to
        store (always with 'repeat_mode')."""
        n = self.graph.nodes[nid]
        prm = n['params']
        out = {'repeat_mode': mode}
        if 'repeat' not in prm or 'units_px' not in prm or prm.get('repeat_mode') == mode:
            return out
        w = self.st['src'].shape[1] if self.st.get('src') is not None else 256
        rng = {pr[0]: (pr[3], pr[4]) for pr in ND.REG[n['type']].params}
        cur = ND.repeat_units(prm, w)          # сколько юнитов сейчас
        if mode == 'per_px':
            lo, hi = rng['units_px']
            val = min(max(round(cur / w, 6), lo), hi)
            out['units_px'] = val
            got = val * w
        else:
            lo, hi = rng['repeat']
            val = int(min(max(round(cur), lo), hi))
            out['repeat'] = val
            got = float(val)
        if abs(got - cur) > max(0.01 * cur, 0.5):
            self.status(t('ed.repeat_cut', cur, got))
        return out

    def game_repeat(self):
        """Game units per texture repeat: from the first 'Parallax for
        OpenMW' node of the graph (its repeat, or units per pixel × the
        texture width), else 128."""
        for n in self.graph.nodes.values():
            if n['type'] == 'parallax_nh':
                w = self.st['src'].shape[1] if self.st.get('src') is not None else 256
                return ND.repeat_units(n['params'], w)
        return 128.0

    def toggle_big_3d(self):
        """⛶: make the 3D view take most of the window, and back."""
        try:
            W = self.pw.winfo_width()
            H = self.left_pw.winfo_height()
            if not getattr(self, '_big3d', False):
                self._sash = (self.pw.sashpos(0), self.pw.sashpos(1), self.left_pw.sashpos(0))
                self.pw.sashpos(1, W - 4)
                self.pw.sashpos(0, W - 8)
                self.left_pw.sashpos(0, int(H * 0.08))
                self._big3d = True
            else:
                a, b, c = self._sash
                self.pw.sashpos(0, a)
                self.pw.sashpos(1, b)
                self.left_pw.sashpos(0, c)
                self._big3d = False
        except Exception:
            pass
        self.root.after(80, lambda: self.show_3d(False))

    def _v3_poll(self):
        """Main thread: show the frame rendered by show_3d's worker."""
        if self.dead:
            return
        if self._v3_img is None:
            self.root.after(40, self._v3_poll)
            return
        gen, img = self._v3_img
        self._v3_busy = False
        if img is not None and gen == self._v3_gen:
            self.v3.show(img)
        if self._v3_again:
            self._v3_again = False
            self.show_3d(False)

    # --------------------------------------------------------- инспектор ---
    def clear_inspector(self):
        """Remove everything from the inspector column."""
        for w in self.insp.winfo_children():
            w.destroy()
        self._readouts = None

    def fill_inspector(self):
        """Inspector for the active node, or the project summary."""
        if self.dead:
            return
        if (getattr(self, 'v2', None) is not None and self.v2.paint_cb == self._matpick_ev
                and self.st.get('view') != getattr(self, '_matpick_nid', None)):
            self.matpick_mode(False)        # пипетка живёт, пока открыт её узел
        self.clear_inspector()
        nid = self.st.get('view')
        if nid not in self.graph.nodes:
            self.fill_project_info()
            return
        n = self.graph.nodes[nid]
        sp = ND.REG[n['type']]
        head = tk.Frame(self.insp, background=CAT_COLOR.get(sp.cat, CAT_COLOR['other']),
                        height=4)
        head.pack(fill='x', pady=(4, 0))
        ttk.Label(self.insp, text=t('nd.' + sp.key), style='Head.TLabel',
                  font=('TkDefaultFont', 11, 'bold')).pack(anchor='w', pady=(4, 0))
        ttk.Label(self.insp, text=t('cat.' + sp.cat) + '   ·   ' + nid,
                  style='Hint.TLabel').pack(anchor='w')
        info = 'nd.' + sp.key + '.info'
        if info in i18n.S:
            ttk.Label(self.insp, text=t(info), style='Hint.TLabel', wraplength=300,
                      justify='left').pack(anchor='w', pady=(6, 2))
        self.model_block(sp.key, n)
        if sp.params:
            body = sec(self.insp, 'ed.params')
            if sp.groups:
                self.param_tabs(body, nid, sp)
            else:
                for pr in sp.params:
                    if sp.show_if and not sp.show_if(n['params'], pr[0]):
                        continue        # ползунок не для выбранного режима
                    self.param_widget(body, nid, pr)
        if sp.key == 'patch':
            self.patch_block(nid)
        if sp.key == 'paint_mask':
            self.mask_block(nid)
        body = sec(self.insp, 'ed.readouts')
        self._readouts = body
        self.show_readouts()

    def param_tabs(self, parent, nid, sp):
        """Parameters of a node with many of them, on tabs (sp.groups): a
        block of buttons, three in a row (all titles fit the inspector), and
        under it the parameters of the chosen one. The open tab is remembered
        per node type."""
        by = {pr[0]: pr for pr in sp.params}
        groups, used = [], set()
        for key, names in sp.groups:
            prs = [by[nm] for nm in names if nm in by]
            used.update(nm for nm in names if nm in by)
            groups.append((key, prs))
        rest = [pr for pr in sp.params if pr[0] not in used and pr[1] != 'd']
        if rest:
            groups.append(('ed.tab.other', rest))
        bar = ttk.Frame(parent)
        bar.pack(fill='x', pady=(4, 2))
        for c in range(3):
            bar.columnconfigure(c, weight=1, uniform='tab')
        body = ttk.Frame(parent)
        body.pack(fill='x')
        tabs = self.st.setdefault('tabs', {})
        cur = tk.IntVar(value=min(int(tabs.get(sp.key, 0)), len(groups) - 1))
        frames = []
        for k, (key, prs) in enumerate(groups):
            f = ttk.Frame(body)
            for pr in prs:
                if sp.show_if and not sp.show_if(self.graph.nodes[nid]['params'], pr[0]):
                    continue
                self.param_widget(f, nid, pr)
            frames.append(f)

        def show():
            k = cur.get()
            tabs[sp.key] = k
            for j, f in enumerate(frames):
                if j == k:
                    f.pack(fill='x')
                else:
                    f.pack_forget()
        for k, (key, _) in enumerate(groups):
            ttk.Radiobutton(bar, text=t(key), value=k, variable=cur, command=show,
                            style='Toolbutton').grid(row=k // 3, column=k % 3,
                                                     sticky='ew', padx=1, pady=1)
        show()

    def param_widget(self, parent, nid, pr):
        """One parameter control: slider, check box or list."""
        name, kind = pr[0], pr[1]
        if kind == 'd':
            if self.graph.nodes[nid]['type'] == 'curves':
                self.curve_block(parent, nid, name)
            elif name == 'mats':
                self.material_block(parent, nid)
            return              # прочие данные (заплатки) правятся на холсте 2D
        cur = self.graph.nodes[nid]['params'][name]
        key = self.graph.nodes[nid]['type']
        hint = ND.hint(key, name)
        if kind in ('f', 'i'):
            s = Slider(parent, ND.label(key, name), pr[3], pr[4], cur, hint, lambda: None,
                       integer=(kind == 'i'), fmt='%.2f' if pr[4] - pr[3] <= 20 else '%.1f')
            s.var.trace_add('write', lambda *a, sl=s: self.set_param(nid, name, sl.get()))
            if name in UNIT_PARAMS:
                # юниты OpenMW — сразу в сантиметрах, метрах и дюймах
                lab = ttk.Label(parent, style='Hint.TLabel', text=units_text(cur))
                lab.pack(anchor='w')
                s.var.trace_add('write', lambda *a, sl=s, lb=lab:
                                lb.config(text=units_text(sl.get())))
        elif kind == 'b':
            v = tk.BooleanVar(value=bool(cur))
            row = ttk.Frame(parent)
            row.pack(fill='x', pady=(4, 0))
            ttk.Checkbutton(row, text=ND.label(key, name), variable=v,
                            command=lambda: self.set_param(nid, name, bool(v.get()))
                            ).pack(side='left')
            if hint:
                Info(row, hint).pack(side='right')
        else:
            vals = list(pr[3])
            # подпись значения: сперва своя у узла (nv.<узел>.<значение>), потом общая
            labels = [t('nv.%s.%s' % (key, x)) if ('nv.%s.%s' % (key, x)) in i18n.S
                      else t('nv.' + str(x)) if ('nv.' + str(x)) in i18n.S else str(x)
                      for x in vals]
            row = ttk.Frame(parent)
            row.pack(fill='x', pady=(6, 0))
            ttk.Label(row, text=ND.label(key, name)).pack(side='left')
            if hint:
                Info(row, hint).pack(side='right')
            v = tk.StringVar(value=labels[vals.index(cur)] if cur in vals else str(cur))
            cb = ttk.Combobox(row, values=labels, textvariable=v, state='readonly', width=14)
            cb.pack(side='right', padx=4)
            def picked(e=None):
                val = vals[labels.index(v.get())]
                sp = ND.REG[self.graph.nodes[nid]['type']]
                upd = sp.preset_fn(self.graph.nodes[nid]['params'], name, val) \
                    if sp.preset_fn else None
                if name == 'repeat_mode':
                    # смена единиц повтора — тот же размер, другим числом
                    upd = dict(upd or {}, **self.repeat_switch(nid, val))
                if upd:
                    # выбор материала сразу ставит его ползунки, одним шагом отмены;
                    # только те, что у этого узла есть
                    prm = self.graph.nodes[nid]['params']
                    upd = {k: v for k, v in upd.items() if k in prm}
                    self.push_undo()
                    prm.update(upd)
                    self.graph.nodes[nid]['params'].setdefault(name, val)
                    self.sync_mat(nid, list(upd) + [name])
                    self.sync_relief(nid, list(upd) + [name])
                    self._last_param = ((nid, name), time.time())
                    self.touch()
                    self.schedule()
                    self.root.after(10, self.fill_inspector)
                    return
                presets = sp.presets
                if name == 'preset' and val in presets:
                    # пресет материала: переставить сразу все его ползунки,
                    # одним шагом отмены
                    self.push_undo()
                    self.graph.nodes[nid]['params'].update(presets[val])
                    self.graph.nodes[nid]['params'][name] = val
                    self._last_param = ((nid, name), time.time())
                    self.touch()
                    self.schedule()
                    self.root.after(10, self.fill_inspector)
                    return
                self.set_param(nid, name, val)
                # от выбора модели зависит, что показывает раздел «Нейросеть»
                if name == 'model':
                    self.root.after(10, self.fill_inspector)
            cb.bind('<<ComboboxSelected>>', picked)

    def curve_block(self, parent, nid, name):
        """The Curves node's editor: the curve over the input's histogram."""
        def changed(pts):
            self.set_param(nid, name, pts)
        self._curve = CurveBox(parent, self.graph.nodes[nid]['params'][name], changed,
                               self._curve_hist(nid))
        self._curve.nid = nid

    def material_block(self, parent, nid):
        """The PBR nodes' materials: palette of spheres and the shininess scale."""
        def changed(upd):
            self.set_params(nid, upd)
        self._matbox = MaterialBox(parent, self.graph.nodes[nid]['params'], changed,
                                   (self.results.get(nid) or {}).get('_mat_hist'))
        self._matbox.nid = nid
        self._matbox.on_pick_mode = self.matpick_mode
        if getattr(self, '_matpick_nid', None) == nid:
            self._matbox.pick_var.set(True)         # пипетка остаётся при перестройке
            self.matpick_mode(True)
        else:
            self.matpick_mode(False)

    def matpick_mode(self, on):
        """Material pipette in the 2D view on or off (a helper of MaterialBox)."""
        v = self.v2
        mb = getattr(self, '_matbox', None)
        self._matpick_nid = mb.nid if (on and mb is not None) else None
        if on:
            v.paint_on, v.paint_cb, v.overlay_cb = True, self._matpick_ev, None
        elif v.paint_cb == self._matpick_ev:
            v.paint_on, v.paint_cb = False, None

    def _matpick_ev(self, x, y, erase, final):
        """Click in the 2D view with the pipette on: shininess of the incoming
        roughness map at that pixel → the material box."""
        mb = getattr(self, '_matbox', None)
        if not final or mb is None or not mb.alive() or self.st['src'] is None:
            return
        src = self.graph.inputs_of(mb.nid).get('rough')
        r = (self.results.get(src[0]) or {}).get(src[1]) if src else None
        if not isinstance(r, np.ndarray):
            mb.set_pick(None, t('ed.mat.pick_norough'))
            return
        r = np.asarray(r, np.float32)
        if r.ndim == 3:
            r = r[..., :3].mean(-1)
        H, W = self.st['src'].shape[:2]
        h, w = r.shape[:2]
        yy = int(min(max(y * h / float(H), 0), h - 1))
        xx = int(min(max(x * w / float(W), 0), w - 1))
        mb.set_pick(1.0 - float(np.clip(r[yy, xx], 0, 1)))

    def set_params(self, nid, upd):
        """Store several parameters at once: one undo step, then the same
        as set_param (shared material copied to the other nodes)."""
        n = self.graph.nodes.get(nid)
        if n is None:
            return
        upd = {k: v for k, v in upd.items() if n['params'].get(k) != v}
        if not upd:
            return
        self.push_undo()
        self._last_param = ((nid, tuple(upd)), time.time())
        n['params'].update(copy.deepcopy(upd))
        self.sync_mat(nid, list(upd))
        self.sync_relief(nid, list(upd))
        self.touch()
        self.schedule(220)

    def _curve_hist(self, nid):
        """Histogram (64 bins, 0…1 heights) of what comes into node nid, or
        None while it is not computed."""
        src = self.graph.inputs_of(nid).get('h')
        v = (self.results.get(src[0]) or {}).get(src[1]) if src else None
        if not isinstance(v, np.ndarray):
            return None
        v = np.asarray(v, np.float32)
        if v.ndim == 3:
            v = v[..., :3].mean(axis=-1)
        c = np.histogram(np.clip(v, 0, 1), 64, (0.0, 1.0))[0].astype(np.float32)
        top = float(np.percentile(c, 98)) or float(c.max()) or 1.0
        return np.minimum(c / top, 1.0)

    def set_param(self, nid, key, val):
        """Store a parameter; one undo step per burst of changes of one setting."""
        n = self.graph.nodes.get(nid)
        if n is None or n['params'].get(key) == val:
            return
        now = time.time()
        if self._last_param[0] != (nid, key) or now - self._last_param[1] > 1.5:
            self.push_undo()
        self._last_param = ((nid, key), now)
        n['params'][key] = val
        if key == 'mat_sync' and val:
            if self.sync_mat(nid, pull=True):
                self.root.after(10, self.fill_inspector)    # показать взятые значения
        elif key == 'relief_sync' and val:
            if self.sync_relief(nid, pull=True):
                self.root.after(10, self.fill_inspector)
        else:
            self.sync_mat(nid, [key])
            self.sync_relief(nid, [key])
        sp = ND.REG.get(n['type'])
        if sp is not None and sp.show_if and any(pr[0] == key and pr[1] in ('b', 'c')
                                                 for pr in sp.params):
            self.root.after(10, self.fill_inspector)    # набор ползунков мог смениться
        self.touch()
        self.schedule(220)

    def sync_relief(self, nid, keys=(), pull=False):
        """Relief settings shared by «Parallax for OpenMW», «Relief for OpenMW»
        and AO nodes that have «relief_sync» on (nodes.RELIEF_SYNC maps each node's parameter
        names to the shared ones: parallax 'depth' is AO 'depth_cm').

        keys: parameters just changed in node nid — copied to the other ticked
        nodes. pull=True: nid has just been ticked — it takes the values of
        the first other ticked node. Returns True if anything was copied."""
        n = self.graph.nodes.get(nid)
        if n is None or not n['params'].get('relief_sync'):
            return False
        mine = ND.RELIEF_SYNC.get(n['type'], {})
        others = [v for k, v in self.graph.nodes.items() if k != nid
                  and v['type'] in ND.RELIEF_SYNC and v['params'].get('relief_sync')]
        if not others:
            return False
        if pull:
            src = others[0]
            theirs = {c: nm for nm, c in ND.RELIEF_SYNC[src['type']].items()}
            for nm, c in mine.items():
                if c in theirs:
                    n['params'][nm] = src['params'][theirs[c]]
            return True
        changed = False
        for key in keys:
            c = mine.get(key)
            if c is None:
                continue
            for o in others:
                for nm, cc in ND.RELIEF_SYNC[o['type']].items():
                    if cc == c:
                        o['params'][nm] = n['params'][key]
                        changed = True
        return changed

    def sync_mat(self, nid, keys=(), pull=False):
        """Material settings shared by the PBR nodes that have «mat_sync» on.

        keys: material parameters just changed in node nid — copied to every
        other node with the box ticked (only those it has: «Отражение» lacks the
        defaults). pull=True: nid has just been ticked — it takes all the
        material parameters from the first other ticked node instead.
        Returns True if anything was copied."""
        n = self.graph.nodes.get(nid)
        if n is None or not n['params'].get('mat_sync'):
            return False
        names = {pr[0] for pr in ND.MAT_PARAMS} - {'mat_sync'}
        others = [v for k, v in self.graph.nodes.items()
                  if k != nid and v['params'].get('mat_sync')]
        if pull:
            if not others:
                return False
            src = others[0]['params']
            for k in names:
                if k in src and k in n['params']:
                    n['params'][k] = copy.deepcopy(src[k])
            return True
        done = False
        for v in others:
            for k in keys:
                if k in names and k in v['params']:
                    v['params'][k] = copy.deepcopy(n['params'][k])
                    done = True
        return done

    def show_readouts(self):
        """Fill the readouts section of the active node (numbers it reports)."""
        body = getattr(self, '_readouts', None)
        if body is None or not body.winfo_exists():
            return
        for w in body.winfo_children():
            w.destroy()
        nid = self.st.get('view')
        cb = getattr(self, '_curve', None)
        if cb is not None and cb.alive() and cb.nid == nid:
            cb.hist = self._curve_hist(nid)
            cb.draw()
        mb = getattr(self, '_matbox', None)
        if mb is not None and mb.alive() and mb.nid == nid and mb.drag is None:
            mb.hist = (self.results.get(nid) or {}).get('_mat_hist')
            mb.params = copy.deepcopy(self.graph.nodes[nid]['params'])
            mb.draw()
        if nid in self.errors:
            ttk.Label(body, text=self.errors[nid], foreground='#e08a8a', wraplength=300,
                      justify='left').pack(anchor='w', pady=2)
            return
        out = self.results.get(nid)
        if out is None:
            ttk.Label(body, text=t('ed.noresult'), style='Hint.TLabel').pack(anchor='w')
            return
        if out.get('_warn'):
            ttk.Label(body, text='⚠ ' + t(out['_warn']), foreground='#e0b060', wraplength=300,
                      justify='left').pack(anchor='w', pady=2)
        rows = list(out.get('_info') or [])
        for k, v in out.items():
            if isinstance(v, np.ndarray) and not k.startswith('_'):
                rows.append(('port:' + k, '%s  %.3f…%.3f' % (
                    'x'.join(str(s) for s in v.shape), float(v.min()), float(v.max()))))
        for k, v in rows:
            row = ttk.Frame(body)
            row.pack(fill='x')
            lab = (ND.label(self.graph.nodes[nid]['type'], k[5:])
                   if k.startswith('port:') and nid in self.graph.nodes
                   else t('np.' + k[5:]) if k.startswith('port:') else t('ni.' + k))
            ttk.Label(row, text=lab, style='Hint.TLabel').pack(side='left')
            ttk.Label(row, text=fmt_value(k, v)).pack(side='right')
            if not k.startswith('port:') and ('ni.' + k + '.info') in i18n.S:
                Info(row, t('ni.' + k + '.info')).pack(side='right', padx=4)

    def fill_project_info(self):
        """Inspector with nothing selected: project, texture, outputs, keys."""
        p = self.insp
        ttk.Label(p, text=t('ed.project'), style='Head.TLabel',
                  font=('TkDefaultFont', 11, 'bold')).pack(anchor='w', pady=(6, 2))
        ttk.Label(p, text=self.st['path'] or t('ed.untitled'), style='Hint.TLabel',
                  wraplength=300).pack(anchor='w')
        ttk.Label(p, text=t('ed.outputs'), style='Head.TLabel').pack(anchor='w', pady=(10, 2))
        outs = self.graph.outputs()
        if not outs:
            ttk.Label(p, text=t('g.noout'), style='Hint.TLabel', wraplength=300).pack(anchor='w')
        for o in outs:
            nm = self.graph.nodes[o]['params']['name']
            pbr = self.graph.nodes[o]['params'].get('set') == 'pbr'
            ttk.Label(p, text='•  ' + (t('ed.samename') if nm == 'diffuse' else nm)
                      + ('   [_PBR]' if pbr else '')).pack(anchor='w')
        ttk.Label(p, text=t('ed.keys'), style='Head.TLabel').pack(anchor='w', pady=(12, 2))
        ttk.Label(p, text=t('ed.keys.list'), style='Hint.TLabel', justify='left',
                  wraplength=310).pack(anchor='w')

    # ---------------------------------------------------- модели сетей -----
    def model_block(self, key, n):
        """Network status and one button that installs everything missing.

        Rows: the onnxruntime build and where it computes (processor or
        graphics card, with a button to choose), and each file the node needs.
        The install itself runs in InstallWindow, which shows pip's output and
        the download progress live, so it is always visible what is going on.
        """
        import depthnet as DN
        steps = []
        if key not in ('depth_net', 'deepbump', 'esrgan_normal', 'esrgan_franken', 'nh_fuse'):
            if key == 'supermat':
                self.supermat_block()
            return
        body = sec(self.insp, 'ed.model')
        row = ttk.Frame(body)
        row.pack(fill='x', pady=2)
        have_rt = DN.have_runtime()
        kind = DN.runtime_kind() if have_rt else ''
        if have_rt:
            txt = '✓  onnxruntime %s — %s' % (DN.runtime_version(), t('rt.' + (kind or 'cpu')))
        else:
            txt = '✗  ' + t('ed.m.rt')
            steps.append(runtime_step(DN.load_settings().get('runtime') or 'cpu'))
        ttk.Label(row, text=txt, wraplength=230,
                  foreground=FG if have_rt else '#e0a060').pack(side='left')
        ttk.Button(row, text=t('rt.choose'), style='Small.TButton',
                   command=lambda: runtime_dialog(self.root, self._after_install)
                   ).pack(side='right')

        def need(ok, label, step, manual=None):
            r = ttk.Frame(body)
            r.pack(fill='x', pady=2)
            ttk.Label(r, text=('✓  ' if ok else '✗  ') + label, wraplength=250,
                      foreground=FG if ok else '#e0a060').pack(side='left')
            if not ok:
                steps.append(step)
                if manual:
                    ttk.Button(r, text=t('ed.m.file'), style='Small.TButton',
                               command=lambda: self.model_from_file(*manual)
                               ).pack(side='right')

        def dl(title, url, fn):
            # при ошибке сети — адрес и папка, чтобы скачать браузером
            def run(line, prog):
                try:
                    fn(line, prog)
                    return True, ''
                except KeyboardInterrupt:
                    raise
                except Exception as e:                   # noqa: BLE001
                    return False, t('ed.m.nodl', e, url, DN.model_dir())
            return (title, run)
        if key in ('depth_net', 'nh_fuse'):
            mk = n['params']['model']
            mm = DN.MODELS[mk]
            need(DN.have_model(mk), '%s  (%d MB, %s)' % (mm['title'], mm['size'] >> 20,
                                                          mm['license']),
                 dl(mm['title'], mm['url'], lambda line, prog: DN.download(mk, prog)),
                 (mm['file'], mm['size'], None))
        if key in ('deepbump', 'nh_fuse'):
            import deepbump as DB
            need(DB.have_model(), '%s  (%d MB, %s)' % (DB.MODEL['title'], DB.MODEL['size'] >> 20,
                                                      DB.MODEL['license']),
                 dl(DB.MODEL['title'], DB.MODEL['url'], lambda line, prog: DB.download(prog)),
                 (DB.MODEL['file'], DB.MODEL['size'], None))
        if key in ('esrgan_normal', 'esrgan_franken'):
            import esrgan as ES
            mk = 'normal' if key == 'esrgan_normal' else 'franken'
            mm = ES.MODELS[mk]
            need(ES.have_onnx_lib(), t('ed.m.onnx'),
                 ('onnx', lambda line, prog: DN.install_runtime('onnx', 'onnx', line)))
            need(ES.have_model(mk), '%s  (%s, %s)' % (mm['title'], mm['author'], mm['license']),
                 dl(mm['title'], mm['url'], lambda line, prog: ES.prepare(mk, prog)),
                 (mm['file'], 0, ES.load_pth))
        if steps:
            def go():
                # onnxruntime ещё не выбран — сначала спросить, где считать
                if not have_rt and not DN.load_settings().get('runtime'):
                    runtime_dialog(self.root, self._after_install)
                    return
                InstallWindow(self.root, steps, self._after_install)
            ttk.Button(body, text=t('ed.m.all'), command=go).pack(fill='x', pady=(6, 0))

    def model_from_file(self, name, size, check):
        """Put a model file downloaded by hand (e.g. in a browser) in place.

        name: the file name the program expects in the model folder; size:
        exact size in bytes (0 = not known — then ``check(path)`` must accept
        the file, e.g. esrgan.load_pth)."""
        import shutil
        import depthnet as DN
        p = filedialog.askopenfilename(title=name,
                                       filetypes=[(name, '*' + os.path.splitext(name)[1]),
                                                  ('*', '*.*')])
        if not p:
            return
        try:
            if size and os.path.getsize(p) != size:
                raise IOError(t('ed.m.badsize', os.path.getsize(p), size))
            if check:
                check(p)
            shutil.copyfile(p, os.path.join(DN.model_dir(), name))
        except Exception as e:                           # noqa: BLE001
            messagebox.showerror(APP, str(e))
            return
        self._after_install(True)

    def _after_install(self, ok):
        """Refresh the inspector and recompute after an install window closes."""
        if self.dead:
            return
        # видеокарту могли включить или выключить: старые ответы лежат не
        # там, где теперь считаем, — всё заново
        self.graph._cache.clear()
        self.results.clear()
        self._m3 = None
        self.update_dev_label()
        self.status(t('ed.m.ok') if ok else t('ed.m.fail'))
        self.fill_inspector()
        self.schedule()

    def supermat_block(self):
        """SuperMat setup: its folder, its Python, optional weights, a test start."""
        import supermat_bridge as SB
        body = sec(self.insp, 'ed.supermat')
        ttk.Label(body, text=t('ed.supermat.info'), style='Hint.TLabel', wraplength=300,
                  justify='left').pack(anchor='w')
        cfg = SB.load_config()
        vars_ = {}
        for k, lab, isdir in (('repo', 'ed.supermat.repo', True), ('python', 'ed.supermat.py', False),
                              ('weights', 'ed.supermat.w', False)):
            ttk.Label(body, text=t(lab)).pack(anchor='w', pady=(6, 0))
            row = ttk.Frame(body)
            row.pack(fill='x')
            v = tk.StringVar(value=cfg[k])
            vars_[k] = v
            ttk.Entry(row, textvariable=v).pack(side='left', fill='x', expand=True)

            def browse(v=v, isdir=isdir):
                """Pick a folder or file for this field."""
                p = (filedialog.askdirectory() if isdir else
                     filedialog.askopenfilename())
                if p:
                    v.set(p)
            ttk.Button(row, text='…', width=3, command=browse).pack(side='left', padx=2)
        state = ttk.Label(body, style='Hint.TLabel', wraplength=300, justify='left')
        state.pack(anchor='w', pady=(6, 0))
        bad = SB.problem(cfg)
        state.config(text=t(bad) if bad else (t('ed.supermat.run', SB.device()) if SB.device()
                                             else t('ed.supermat.ready')))

        def apply_and_test():
            """Store the SuperMat paths and start the worker once to test them."""
            SB.save_config({k: v.get().strip() for k, v in vars_.items()})
            bad = SB.problem()
            if bad:
                state.config(text=t(bad))
                return
            state.config(text=t('ed.supermat.starting'))
            self._f_out, self._fetching = None, True

            def work():
                """Worker thread body; writes only plain attributes, never Tk."""
                try:
                    self._f_out = (True, SB.start())
                except Exception as e:               # noqa: BLE001
                    self._f_out = (False, str(e))
            threading.Thread(target=work, daemon=True).start()

            def poll():
                """Main-thread timer: wait for the worker and show its result."""
                if self._f_out is None:
                    self.root.after(300, poll)
                    return
                self._fetching = False
                ok, txt = self._f_out
                if state.winfo_exists():
                    state.config(text=t('ed.supermat.run', txt) if ok else txt[-600:])
                if ok:
                    self.schedule()
            poll()
        ttk.Button(body, text=t('ed.supermat.test'), command=apply_and_test).pack(fill='x', pady=4)

    # ------------------------------------------------------ файлы ----------
    def open_texture(self, p=None, mark=True):
        """Open a texture as the preview source (Ctrl+T).

        mark: count it as a change of the project (not for the sample opened
        at start-up, so closing an untouched window asks nothing)."""
        if not p:
            p = filedialog.askopenfilename(
                title=t('ed.tex'), initialdir=None if self.st['tex'] else sample_dir(),
                filetypes=[('Textures', '*.dds *.png *.tga *.tif *.tiff *.jpg *.bmp'),
                           ('*', '*.*')])
        if not p:
            return
        try:
            rgba = TX.read_rgba(p)
        except Exception as e:                       # noqa: BLE001
            messagebox.showerror(t('com.notopen'), str(e))
            return
        self.st['src'] = np.ascontiguousarray(rgba[..., :3])
        a = rgba[..., 3]
        self.st['alpha'] = a if float(a.min()) < 0.999 else None
        self.st['tex'] = p
        self.results.clear()
        self.errors.clear()
        self.thumbs.clear()
        self._prev_key = None
        self.update_tex_label()
        self.v2.scale = 0
        self.v2._fit_pending = True
        if mark:
            self.touch()
        self.redraw()
        self.schedule(10)

    def on_drop(self, path):
        """A file dropped on the window: a project or a texture."""
        if path.lower().endswith(PR.EXTS):
            self.open_project(path)
        else:
            self.open_texture(path)

    def ask_discard(self):
        """True if it is fine to replace the current project (saving if asked)."""
        if not self.st['dirty']:
            return True
        r = messagebox.askyesnocancel(APP, t('ed.ask.save'))
        if r is None:
            return False
        if r:
            return self.save_project()
        return True

    def new_project(self):
        """Ctrl+N: choose a template and start a new project."""
        if not self.ask_discard():
            return
        dlg = tk.Toplevel(self.root)
        dlg.title(t('ed.new.title'))
        dlg.configure(background=BG)
        dlg.transient(self.root)
        ttk.Label(dlg, text=t('ed.new.head'), style='Head.TLabel').pack(anchor='w', padx=14,
                                                                         pady=(12, 6))
        choice = tk.StringVar(value=PR.TEMPLATES[0][0])
        for k, _ in PR.TEMPLATES:
            f = ttk.Frame(dlg)
            f.pack(fill='x', padx=14, pady=3)
            ttk.Radiobutton(f, text=t('tpl.' + k), value=k, variable=choice).pack(anchor='w')
            ttk.Label(f, text=t('tpl.' + k + '.info'), style='Hint.TLabel', wraplength=460,
                      justify='left').pack(anchor='w', padx=(22, 0))
        keep = tk.BooleanVar(value=True)
        ttk.Checkbutton(dlg, text=t('ed.new.keeptex'), variable=keep).pack(anchor='w',
                                                                            padx=14, pady=6)

        def go():
            """Create the project from the chosen template and close the dialog."""
            self.st['graph'] = PR.template(choice.get())
            self.st.update(path='', dirty=False, sel=[], view=None, port=None, pin=None)
            self.pin_var.set(False)
            if not keep.get():
                self.st.update(src=None, alpha=None, tex='')
            self.results.clear()
            self.errors.clear()
            self.thumbs.clear()
            self.undo_stack.clear()
            self.redo_stack.clear()
            dlg.destroy()
            self.update_title()
            self.update_tex_label()
            self.fill_inspector()
            self.root.after(50, self.frame_all)
            self.schedule(10)
        ttk.Button(dlg, text=t('ed.new.go'), command=go).pack(pady=(4, 12))
        dlg.grab_set()

    def open_project(self, p=None):
        """Ctrl+O: open a .texelith project (and its preview texture)."""
        if not p:
            if not self.ask_discard():
                return
            p = filedialog.askopenfilename(title=t('ed.open'),
                                           filetypes=[('Texelith', '*' + PR.EXT + ' *' + PR.OLD_EXT), ('*', '*.*')])
        if not p:
            return
        try:
            pr = PR.load(p)
        except Exception as e:                       # noqa: BLE001
            messagebox.showerror(t('com.notopen'), str(e))
            return
        self.st['graph'] = pr['graph']
        v = pr['view']
        self.pin_var.set(False)
        self.st.update(path=p, batch=pr['batch'], sel=[], view=v.get('view'), port=None, pin=None,
                       zoom=float(v.get('zoom', 0.8)), ox=float(v.get('ox', 30)),
                       oy=float(v.get('oy', 30)), thumbs=bool(v.get('thumbs', True)))
        if self.st['view'] not in self.graph.nodes:
            self.st['view'] = None
        self.results.clear()
        self.errors.clear()
        self.thumbs.clear()
        self.undo_stack.clear()
        self.redo_stack.clear()
        if pr['source'] and os.path.isfile(pr['source']):
            self.open_texture(pr['source'])
        elif pr['source']:
            self.status(t('ed.texmissing', pr['source']))
        self.st['dirty'] = False
        self.update_title()
        self.fill_inspector()
        self.redraw()
        self.schedule(10)

    def save_project(self, as_new=False):
        """Ctrl+S (Ctrl+Shift+S — under a new name). Returns True if saved."""
        p = self.st['path']
        if as_new or not p:
            base = os.path.splitext(os.path.basename(self.st['tex']))[0] if self.st['tex'] else 'project'
            p = filedialog.asksaveasfilename(
                title=t('ed.save'), defaultextension=PR.EXT, initialfile=base + PR.EXT,
                filetypes=[('Texelith', '*' + PR.EXT), ('*', '*.*')])
            if not p:
                return False
        view = {k: self.st[k] for k in ('zoom', 'ox', 'oy', 'thumbs', 'view')}
        try:
            PR.save(p, self.graph, self.st['tex'], self.st['batch'], view)
        except Exception as e:                       # noqa: BLE001
            messagebox.showerror(APP, str(e))
            return False
        self.st['path'] = p
        self.st['dirty'] = False
        self.update_title()
        self.status(t('m.ext.saved', os.path.basename(p)))
        return True

    # ------------------------------------------------ запись одной текстуры --
    def save_maps(self):
        """Evaluate at full size and write every output for the open texture."""
        if self.st['src'] is None or not self.graph.outputs():
            messagebox.showinfo(APP, t('g.noout') if self.st['src'] is not None else t('ed.notex'))
            return
        tex_dir = os.path.dirname(self.st['tex']) or '.'
        d = filedialog.askdirectory(title=t('ed.savemaps'), initialdir=tex_dir)
        if not d:
            return
        rel = os.path.basename(self.st['tex'])
        fmt = self.st['batch'].get('format', '.dds')
        tg = PR.targets(self.graph, rel, tex_dir, d, fmt, self.st['batch'].get('layout', 'flat'))
        if len(tg) < len(self.graph.outputs()):
            self.status(t('ed.nodiffuse'))
        g2 = ND.Graph()
        g2.from_json(self.graph.to_json())
        comp = TX.dxt(self.st['batch'])
        ctx = {'src': self.st['src'], 'alpha': self.st['alpha'], 'px': 1.0,
               'key': 'full:%d' % id(self.st['src']),
               'name': os.path.basename(self.st['tex'] or '')}
        self._s_out = None

        def work():
            """Worker thread body; writes only plain attributes, never Tk."""
            done, bad = [], []
            for nid, path in tg:
                try:
                    v = X.to_host(g2.eval(nid, ctx).get('any'))
                    if v is None:
                        raise ValueError(t('g.novalue'))
                    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
                    nm = g2.nodes[nid]['params']['name']
                    TX.write_value(v, path, comp, PR.diffuse_alpha(v, ctx['alpha'], nm),
                                   as_normal=PR.normal_flag(nm))
                    done.append(path)
                except Exception as e:               # noqa: BLE001
                    bad.append('%s — %s' % (os.path.basename(path), e))
            self._s_out = (done, bad)
        threading.Thread(target=work, daemon=True).start()
        self.status(t('ed.saving'))
        self.bar.config(mode='indeterminate')
        self.bar.start(12)

        def poll():
            """Main-thread timer: wait for the worker and show its result."""
            if self.dead:
                return
            if self._s_out is None:
                self.root.after(200, poll)
                return
            self.bar.stop()
            self.bar.config(mode='determinate')
            done, bad = self._s_out
            self.status(t('com.saved.n', len(done)) + '  →  ' + d)
            if bad:
                messagebox.showerror(t('ed.savemaps'), '\n'.join(bad))
        poll()

    # ------------------------------------------------------------ пакет ----
    def batch_dialog(self):
        """Batch window: folders, options, start/stop, progress and log."""
        if not self.graph.outputs():
            messagebox.showinfo(APP, t('g.noout'))
            return
        b = self.st['batch']
        if not b.get('in') and self.st['tex']:
            b['in'] = os.path.dirname(self.st['tex'])
        dlg = tk.Toplevel(self.root)
        dlg.title(t('ed.batch.title'))
        dlg.configure(background=BG)
        dlg.geometry('720x560')
        frm = ttk.Frame(dlg, padding=12)
        frm.pack(fill='both', expand=True)
        vs = {}
        for k, lab in (('in', 'ed.b.in'), ('out', 'ed.b.out')):
            ttk.Label(frm, text=t(lab)).pack(anchor='w', pady=(4, 0))
            row = ttk.Frame(frm)
            row.pack(fill='x')
            v = tk.StringVar(value=b.get(k, ''))
            vs[k] = v
            ttk.Entry(row, textvariable=v).pack(side='left', fill='x', expand=True)
            ttk.Button(row, text='…', width=3,
                       command=lambda v=v: v.set(filedialog.askdirectory() or v.get())
                       ).pack(side='left', padx=2)
        ttk.Label(frm, text=t('ed.b.out.info'), style='Hint.TLabel', wraplength=680,
                  justify='left').pack(anchor='w')
        row = ttk.Frame(frm)
        row.pack(fill='x', pady=(8, 0))
        rec = tk.BooleanVar(value=b.get('recursive', True))
        skd = tk.BooleanVar(value=b.get('skip_done', True))
        cmp_ = tk.BooleanVar(value=b.get('compress', True))
        ttk.Checkbutton(row, text=t('ed.b.rec'), variable=rec).pack(side='left')
        ttk.Checkbutton(row, text=t('ed.b.skipdone'), variable=skd).pack(side='left', padx=10)
        ttk.Checkbutton(row, text=t('ed.b.compress'), variable=cmp_).pack(side='left')
        fmt = tk.StringVar(value=b.get('format', '.dds'))
        ttk.Combobox(row, values=('.dds', '.png', '.tga'), textvariable=fmt, width=6,
                     state='readonly').pack(side='right')
        ttk.Label(row, text=t('ed.b.fmt')).pack(side='right', padx=4)
        # чем сжимать DXT и насколько тщательно (действует и на «Сохранить карты»)
        row = ttk.Frame(frm)
        row.pack(fill='x', pady=(6, 0))
        ttk.Label(row, text=t('ed.b.dxt')).pack(side='left')
        algos = ('quicktex', 'pillow')
        algo_labels = [t('ed.b.dxt.' + a) for a in algos]
        cur = b.get('dxt_algo', 'quicktex')
        algo = tk.StringVar(value=algo_labels[algos.index(cur)] if cur in algos
                            else algo_labels[0])
        ttk.Combobox(row, values=algo_labels, textvariable=algo, width=30,
                     state='readonly').pack(side='left', padx=6)
        ttk.Label(row, text=t('ed.b.dxt.level')).pack(side='left', padx=(8, 3))
        lvl = tk.IntVar(value=int(b.get('dxt_level', 14)))
        lvl_w = ttk.Spinbox(row, from_=0, to=18, width=4, textvariable=lvl)
        lvl_w.pack(side='left')
        if not FO.have_quicktex():
            ttk.Label(row, text=t('ed.b.dxt.noqt'), style='Hint.TLabel').pack(side='left',
                                                                              padx=8)

        def algo_changed(*a):
            """The level applies to quicktex only."""
            on = algo.get() == algo_labels[0]
            lvl_w.state(['!disabled'] if on else ['disabled'])
        algo.trace_add('write', algo_changed)
        algo_changed()
        ttk.Label(frm, text=t('ed.b.dxt.info'), style='Hint.TLabel', wraplength=680,
                  justify='left').pack(anchor='w')
        # раскладка по папкам: всё вместе / по виду карты / по текстуре
        row = ttk.Frame(frm)
        row.pack(fill='x', pady=(8, 0))
        ttk.Label(row, text=t('ed.b.layout')).pack(side='left')
        lay_labels = [t('ed.b.layout.' + k) for k in PR.LAYOUTS]
        cur = b.get('layout', 'flat')
        lay = tk.StringVar(value=lay_labels[PR.LAYOUTS.index(cur)] if cur in PR.LAYOUTS
                           else lay_labels[0])
        ttk.Combobox(row, values=lay_labels, textvariable=lay, width=34,
                     state='readonly').pack(side='left', padx=6)
        ttk.Label(frm, text=t('ed.b.layout.info'), style='Hint.TLabel', wraplength=680,
                  justify='left').pack(anchor='w')
        ttk.Label(frm, text=t('ed.b.skip')).pack(anchor='w', pady=(8, 0))
        skip = tk.StringVar(value=b.get('skip', PR.DEFAULT_SKIP))
        ttk.Entry(frm, textvariable=skip).pack(fill='x')
        info = ttk.Label(frm, style='Hint.TLabel', text='')
        info.pack(anchor='w', pady=(6, 0))
        bar = ttk.Progressbar(frm, mode='determinate', maximum=1.0)
        bar.pack(fill='x', pady=(6, 0))
        now = ttk.Label(frm, style='Hint.TLabel', text='')
        now.pack(anchor='w')
        log = tk.Text(frm, height=12, background=PANEL, foreground=FG, relief='flat',
                      font=('TkFixedFont', 8))
        log.pack(fill='both', expand=True, pady=(6, 0))
        btns = ttk.Frame(frm)
        btns.pack(fill='x', pady=(8, 0))
        state = {'run': False, 'stop': False, 'lines': [], 'prog': (0.0, ''), 'out': None}

        def opts():
            """Read the dialog fields into a batch-settings dict (also kept in the project)."""
            o = {'in': vs['in'].get().strip(), 'out': vs['out'].get().strip(),
                 'recursive': rec.get(), 'skip_done': skd.get(), 'compress': cmp_.get(),
                 'format': fmt.get(), 'skip': skip.get(),
                 'layout': PR.LAYOUTS[lay_labels.index(lay.get())]
                 if lay.get() in lay_labels else 'flat',
                 'dxt_algo': algos[algo_labels.index(algo.get())]
                 if algo.get() in algo_labels else 'quicktex'}
            try:
                o['dxt_level'] = min(max(int(lvl.get()), 0), 18)
            except (tk.TclError, ValueError):
                o['dxt_level'] = 14
            self.st['batch'].update(o)
            self.sync_dxt()
            return o

        def count():
            """Count the textures the batch would process with the current settings."""
            o = opts()
            if not os.path.isdir(o['in']):
                info.config(text=t('ed.b.noin'))
                return
            names = [self.graph.nodes[n]['params']['name'] for n in self.graph.outputs()]
            fl = PR.collect(o['in'], o['recursive'], o['skip'], names, o['layout'],
                            o['out'] or o['in'])
            info.config(text=t('ed.b.count', len(fl)))

        def start():
            """Start the batch in a worker thread."""
            if state['run']:
                return
            o = opts()
            if not os.path.isdir(o['in']):
                info.config(text=t('ed.b.noin'))
                return
            if not o['out']:
                o['out'] = o['in']
            if self.errors:
                if not messagebox.askyesno(APP, t('ed.b.haserr', len(self.errors)), parent=dlg):
                    return
            self.touch()
            state.update(run=True, stop=False, out=None)
            g = ND.Graph()
            g.from_json(self.graph.to_json())

            def work():
                """Worker thread body; writes only plain attributes, never Tk."""
                try:
                    state['out'] = PR.run_batch(
                        g, o, lambda fr, tx: state.__setitem__('prog', (fr, tx)),
                        lambda: state['stop'], state['lines'].append)
                except Exception as e:               # noqa: BLE001
                    state['out'] = {'error': str(e)}
            threading.Thread(target=work, daemon=True).start()
            poll()

        def poll():
            """Main-thread timer: wait for the worker and show its result."""
            while state['lines']:
                log.insert('end', state['lines'].pop(0) + '\n')
                log.see('end')
            fr, tx = state['prog']
            bar['value'] = fr
            now.config(text=tx)
            if state['out'] is None:
                dlg.after(250, poll)
                return
            state['run'] = False
            r = state['out']
            if 'error' in r:
                info.config(text=r['error'])
            else:
                info.config(text=t('ed.b.result', r['done'], r['skipped'], len(r['failed']),
                                   r['seconds']))

        def stop():
            """Ask the running batch to stop after the current file."""
            state['stop'] = True
        ttk.Button(btns, text=t('ed.b.count.btn'), command=count).pack(side='left')
        ttk.Button(btns, text=t('ed.b.start'), command=start).pack(side='right')
        ttk.Button(btns, text=t('ed.b.stop'), command=stop).pack(side='right', padx=6)
        count()


# ------------------------------------------------ установка и сборки сетей --
def runtime_step(kind):
    """Install step for onnxruntime build ``kind`` (see depthnet.RUNTIMES)."""
    import depthnet as DN
    return ('onnxruntime: ' + t('rt.' + kind),
            lambda line, prog: DN.install_runtime(line=line, kind=kind))


class CurveBox:
    """Curves editor in the inspector, as the Curves dialogue of Photoshop
    or GIMP: the curve drawn over the histogram of the node's input.

    Mouse: click on the plot — a new point there; drag a point — move it;
    right click on a point, or drag it out of the plot — remove it (at least
    two stay). The selected point's input and output can be typed in levels
    0…255 under the plot. on_change(points) gets the new list of [input,
    output] pairs 0…1 on every change."""
    S = 256            # сторона графика, px
    M = 7              # поля
    GAP = 0.01         # точки не ближе 1 % по входу (как в «Кривых»)

    def __init__(self, parent, pts, on_change, hist=None):
        """Build the plot and the level entries in parent; pts — list of
        [input, output] pairs 0…1, hist — optional histogram (values 0…1) drawn
        under the curve."""
        self.pts = sorted([float(a), float(b)] for a, b in pts) or [[0.0, 0.0], [1.0, 1.0]]
        self.on_change, self.hist = on_change, hist
        self.sel, self.drag = None, False
        side = self.S + 2 * self.M
        self.cv = tk.Canvas(parent, width=side, height=side, background='#1a1d20',
                            highlightthickness=1, highlightbackground=EDGE)
        self.cv.pack(anchor='w', pady=(6, 2))
        self.cv.bind('<Button-1>', self._press)
        self.cv.bind('<B1-Motion>', self._motion)
        self.cv.bind('<ButtonRelease-1>', lambda e: setattr(self, 'drag', False))
        self.cv.bind('<Button-3>', self._remove_at)
        row = ttk.Frame(parent)
        row.pack(fill='x', pady=(2, 0))
        self.vin, self.vout = tk.StringVar(), tk.StringVar()
        for key, var in (('ed.curve.in', self.vin), ('ed.curve.out', self.vout)):
            ttk.Label(row, text=t(key)).pack(side='left', padx=(0, 3))
            e = ttk.Entry(row, textvariable=var, width=5, justify='right')
            e.pack(side='left', padx=(0, 10))
            for ev in ('<Return>', '<KP_Enter>', '<FocusOut>'):
                e.bind(ev, self._typed)
        ttk.Button(row, text=t('ed.curve.reset'), style='Small.TButton',
                   command=self.reset).pack(side='right')
        ttk.Label(parent, text=t('ed.curve.hint'), style='Hint.TLabel', wraplength=300,
                  justify='left').pack(anchor='w', pady=(2, 0))
        self.select(None)
        self.draw()

    def alive(self):
        """False once the inspector that held the plot has been destroyed."""
        return bool(self.cv.winfo_exists())

    # координаты: вход 0…1 слева направо, выход 0…1 снизу вверх
    def _xy(self, e):
        """Mouse event → (input, output) 0…1, clamped to the plot."""
        x = (e.x - self.M) / float(self.S)
        y = 1.0 - (e.y - self.M) / float(self.S)
        return min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)

    def _scr(self, x, y):
        """(input, output) 0…1 → canvas pixel coordinates."""
        return self.M + x * self.S, self.M + (1.0 - y) * self.S

    def _near(self, e, r=8):
        """Index of the point within r px of the mouse, or None."""
        best, bd = None, r * r
        for k, (x, y) in enumerate(self.pts):
            sx, sy = self._scr(x, y)
            d = (sx - e.x) ** 2 + (sy - e.y) ** 2
            if d <= bd:
                best, bd = k, d
        return best

    def _limits(self, k):
        """Range of input a point may take: between its neighbours."""
        lo = self.pts[k - 1][0] + self.GAP if k > 0 else 0.0
        hi = self.pts[k + 1][0] - self.GAP if k < len(self.pts) - 1 else 1.0
        return lo, hi

    def _put(self, k, x, y):
        """Move point k to (x, y), keeping it between its neighbours."""
        lo, hi = self._limits(k)
        self.pts[k] = [round(min(max(x, lo), hi), 4), round(min(max(y, 0.0), 1.0), 4)]

    def _changed(self):
        """Redraw and report the new points through on_change."""
        self.select(self.sel)
        self.draw()
        self.on_change([list(p) for p in self.pts])

    def _press(self, e):
        """Left click: grab the point under the mouse, or add a new one there."""
        self.cv.focus_set()
        k = self._near(e)
        if k is None:
            x, y = self._xy(e)
            if any(abs(x - p[0]) < self.GAP for p in self.pts):
                # слишком близко к имеющейся точке — берём её
                k = min(range(len(self.pts)), key=lambda j: abs(self.pts[j][0] - x))
            else:
                self.pts.append([x, y])
                self.pts.sort()
                k = next(j for j, p in enumerate(self.pts) if p == [x, y])
                self._put(k, x, y)
                self.sel, self.drag = k, True
                self._changed()
                return
        self.sel, self.drag = k, True
        self.select(k)
        self.draw()

    def _motion(self, e):
        """Drag the grabbed point; dragged far outside the plot, it is removed."""
        if not self.drag or self.sel is None:
            return
        side = self.S + 2 * self.M
        if (e.x < -24 or e.y < -24 or e.x > side + 24 or e.y > side + 24) and len(self.pts) > 2:
            # утащили за край — точка убирается, как в «Кривых»
            del self.pts[self.sel]
            self.sel, self.drag = None, False
            self._changed()
            return
        self._put(self.sel, *self._xy(e))
        self._changed()

    def _remove_at(self, e):
        """Right click: remove the point under the mouse (two always stay)."""
        k = self._near(e)
        if k is not None and len(self.pts) > 2:
            del self.pts[k]
            self.sel = None
            self._changed()

    def _typed(self, e=None):
        """Input/output of the selected point typed in levels 0…255."""
        if self.sel is None:
            self.select(None)
            return
        try:
            x = float(self.vin.get().strip().replace(',', '.')) / 255.0
            y = float(self.vout.get().strip().replace(',', '.')) / 255.0
        except ValueError:
            self.select(self.sel)
            return
        old = list(self.pts[self.sel])
        self._put(self.sel, x, y)
        if self.pts[self.sel] != old:
            self._changed()
        else:
            self.select(self.sel)

    def select(self, k):
        """Make point k the selected one and show its levels (blank for none)."""
        self.sel = k
        if k is None or k >= len(self.pts):
            self.sel = None
            self.vin.set('')
            self.vout.set('')
            return
        x, y = self.pts[k]
        self.vin.set(str(int(round(x * 255))))
        self.vout.set(str(int(round(y * 255))))

    def reset(self):
        """Back to the straight line (0,0)–(1,1)."""
        self.pts = [[0.0, 0.0], [1.0, 1.0]]
        self.sel = None
        self._changed()

    def draw(self):
        """Redraw histogram, grid, the curve through curve_lut and the points."""
        import imgfilters as F
        c, S, M = self.cv, self.S, self.M
        if not self.alive():
            return
        c.delete('all')
        if self.hist is not None:
            w = S / float(len(self.hist))
            for k, hv in enumerate(self.hist):
                if hv > 0:
                    c.create_rectangle(M + k * w, M + S * (1 - 0.95 * hv), M + (k + 1) * w,
                                       M + S, fill='#30353b', outline='')
        for q in (0.25, 0.5, 0.75):
            c.create_line(M + q * S, M, M + q * S, M + S, fill='#2a2e33')
            c.create_line(M, M + q * S, M + S, M + q * S, fill='#2a2e33')
        c.create_rectangle(M, M, M + S, M + S, outline='#3a3f45')
        c.create_line(M, M + S, M + S, M, fill='#454a50', dash=(3, 3))
        lut = F.curve_lut([tuple(p) for p in self.pts])
        xy = []
        for k in range(0, 256, 2):
            xy += list(self._scr(k / 255.0, float(lut[k])))
        xy += list(self._scr(1.0, float(lut[255])))
        c.create_line(*xy, fill=ACC, width=2)
        for k, (x, y) in enumerate(self.pts):
            sx, sy = self._scr(x, y)
            on = k == self.sel
            c.create_rectangle(sx - 4, sy - 4, sx + 4, sy + 4, outline=FG,
                               fill=ACC if on else '#1a1d20', width=1)


class MaterialBox:
    """The materials of a PBR node in the inspector: a palette of material
    spheres (click — add or remove) and under it the shininess scale.

    The scale shows the histogram of the shininess (1 − roughness) of the
    incoming roughness map, the two cut marks — the dullest and the
    shiniest clip_dull / clip_shiny % of pixels are not counted when the
    range is found (drag them) — and between the marks one band per chosen
    material, dullest on the left, the width being its share of the range
    (drag the borders). One material — the whole texture is of it.
    on_change(dict of new parameters) is called on every change."""
    W, HH, BAND = 280, 46, 24        # ширина шкалы, высота гистограммы, полосы
    M = 8                            # поля по краям
    SPH = 34                         # шар в палитре, px
    PER_ROW = 7

    def __init__(self, parent, params, on_change, hist=None):
        """Build the palette, the scale, the buttons and the pipette in parent.

        params — the node's parameters (copied: dragging edits the copy and the
        node gets only what on_change sends); hist — the node's '_mat_hist'
        readout (histogram 'h', range 'lo'/'hi', cut positions
        'cut_dull'/'cut_shiny' in shininess) or None before the first run."""
        import materials as MAT
        self.MAT = MAT
        # своя копия: перетаскивание меняет её, а в узел уходит через on_change
        self.params, self.on_change, self.hist = copy.deepcopy(params), on_change, hist
        self.drag = None
        self.imgs = {}
        ttk.Label(parent, text=t('ed.mat.hint'), style='Hint.TLabel', wraplength=290,
                  justify='left').pack(anchor='w', pady=(4, 2))
        self.pal = []                 # (canvas, {item: key})
        for gkey, keys in MAT.GROUPS:
            ttk.Label(parent, text=t(gkey), style='Hint.TLabel').pack(anchor='w', pady=(2, 0))
            rows = (len(keys) + self.PER_ROW - 1) // self.PER_ROW
            step = self.SPH + 6
            cv = tk.Canvas(parent, width=self.PER_ROW * step, height=rows * step,
                           background=PANEL, highlightthickness=0)
            cv.pack(anchor='w')
            items = {}
            bg = tuple(int(PANEL[i:i + 2], 16) / 255.0 for i in (1, 3, 5)) \
                if isinstance(PANEL, str) and PANEL.startswith('#') else (0.11, 0.12, 0.13)
            for j, k in enumerate(keys):
                x = (j % self.PER_ROW) * step + 3
                y = (j // self.PER_ROW) * step + 3
                im = ImageTk.PhotoImage(Image.fromarray(MAT.sphere_preview(k, self.SPH, bg)))
                self.imgs[k] = im
                ring = cv.create_oval(x - 2, y - 2, x + self.SPH + 2, y + self.SPH + 2,
                                      outline='', width=2)
                it = cv.create_image(x, y, image=im, anchor='nw')
                items[it] = (k, ring)
                cv.tag_bind(it, '<Button-1>', lambda e, k=k: self.toggle(k))
                cv.tag_bind(it, '<Enter>', lambda e, k=k: self.show_info(k))
                cv.tag_bind(it, '<Leave>', lambda e: self.show_info(None))
            self.pal.append((cv, items))
        self.info = ttk.Label(parent, text='', style='Hint.TLabel', wraplength=290,
                              justify='left')
        self.info.pack(anchor='w', pady=(2, 4))
        h = self.HH + self.BAND + 40
        self.cv = tk.Canvas(parent, width=self.W + 2 * self.M, height=h, background='#1a1d20',
                            highlightthickness=1, highlightbackground=EDGE)
        self.cv.pack(anchor='w', pady=(2, 2))
        self.cv.bind('<Button-1>', self._press)
        self.cv.bind('<B1-Motion>', self._motion)
        self.cv.bind('<ButtonRelease-1>', self._release)
        row = ttk.Frame(parent)
        row.pack(fill='x', pady=(2, 0))
        ttk.Button(row, text=t('ed.mat.equal'), style='Small.TButton',
                   command=self.equal).pack(side='left')
        self.gloss_btn = ttk.Button(row, text=t('ed.mat.by_gloss'), style='Small.TButton',
                                    command=self.by_gloss)
        self.gloss_btn.pack(side='left', padx=(4, 0))
        ttk.Button(row, text=t('ed.mat.clear'), style='Small.TButton',
                   command=self.clear).pack(side='right')
        self.list = ttk.Label(parent, text='', wraplength=290, justify='left')
        self.list.pack(anchor='w', pady=(4, 0))
        self.note = ttk.Label(parent, text='', style='Hint.TLabel', wraplength=290,
                              justify='left')
        self.note.pack(anchor='w', pady=(2, 4))
        # пипетка — вспомогательный инструмент: щелчок по текстуре в 2D
        # показывает место пикселя на шкале и позволяет поставить туда срез
        # или границу; основной выбор — палитра и шкала выше
        self.pick_s = None
        self.on_pick_mode = None
        self.pick_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(parent, text=t('ed.mat.pick'), variable=self.pick_var,
                        command=lambda: self.on_pick_mode and self.on_pick_mode(
                            bool(self.pick_var.get()))).pack(anchor='w', pady=(4, 0))
        self.pick_info = ttk.Label(parent, text='', style='Hint.TLabel', wraplength=290,
                                   justify='left')
        self.pick_info.pack(anchor='w')
        self.pick_row = ttk.Frame(parent)
        for key, cmd in (('ed.mat.pick_cd', lambda: self.pick_to('cd')),
                         ('ed.mat.pick_b', lambda: self.pick_to('b')),
                         ('ed.mat.pick_cs', lambda: self.pick_to('cs'))):
            ttk.Button(self.pick_row, text=t(key), style='Small.TButton',
                       command=cmd).pack(side='left', padx=(0, 3))
        self.draw()

    def set_pick(self, s, text=None):
        """A picked pixel: its shininess s (1 − roughness), or None with a
        message (no roughness map and the like)."""
        self.pick_s = s
        if s is None:
            self.pick_info.config(text=text or '')
            self.pick_row.pack_forget()
        else:
            self.pick_info.config(text=self._pick_text(s))
            self.pick_row.pack(anchor='w', pady=(2, 4))
        self.draw()

    def _pick_text(self, s):
        """Text for the picked shininess s: the value and where it falls —
        below the dull cut, above the shiny cut, or in which material's band."""
        rg = self._range()
        if not rg:
            return t('ed.mat.pick_val', s, '')
        cd, cs = rg[2], rg[3]
        if s <= cd:
            where = t('ed.mat.pick_cutd')
        elif s >= cs:
            where = t('ed.mat.pick_cuts')
        else:
            lys = self.layers()
            if not lys:
                where = t('ed.mat.sliders')
            else:
                tt = (s - cd) / max(cs - cd, 1e-6)
                acc, k = 0.0, lys[-1][0]
                for key, w in lys:
                    acc += w
                    if tt <= acc:
                        k = key
                        break
                where = '«%s»' % t('nv.' + k).split(',')[0]
        return t('ed.mat.pick_val', s, where)

    def pick_to(self, what):
        """Put the dull cut, the shiny cut or the nearest band border at the
        picked pixel's shininess."""
        if self.pick_s is None or not self._range():
            return
        x = self._x(self.pick_s)
        if what in ('cd', 'cs'):
            key = 'clip_dull' if what == 'cd' else 'clip_shiny'
            val = round(min(max(self._pct(x, what), 0.0), 40.0), 1)
            self.params[key] = val
            self.hist = dict(self.hist, **{'cut_dull' if what == 'cd' else 'cut_shiny':
                                           self.pick_s})
            self.on_change({key: val})
        else:
            lys = self.layers()
            if len(lys) < 2:
                return
            xd, xs = self._cuts_x()
            bx = self._bounds_x()
            j = min(range(1, len(bx) - 1), key=lambda i: abs(bx[i] - x))
            fr = [(b - xd) / max(xs - xd, 1e-6) for b in bx]
            f = min(max((x - xd) / max(xs - xd, 1e-6), fr[j - 1] + 0.02), fr[j + 1] - 0.02)
            fr[j] = f
            mats = [[k, round(fr[i + 1] - fr[i], 4)] for i, (k, _) in enumerate(lys)]
            self.params['mats'] = mats
            self.on_change({'mats': mats})
        self.draw()
        self.pick_info.config(text=self._pick_text(self.pick_s))

    def alive(self):
        """False once the inspector that held the scale has been destroyed."""
        return bool(self.cv.winfo_exists())

    # --- данные
    def layers(self):
        """Chosen materials as [(key, share)], shares summing to 1
        (materials.layers)."""
        return self.MAT.layers(self.params.get('mats'), bool(self.params.get('mats_manual')))

    def _set(self, lys, **extra):
        """New list of (key, share) → parameters, sent to on_change."""
        upd = {'mats': [[k, round(float(w), 4)] for k, w in lys]}
        if lys and not self.layers():
            # первый выбранный материал: карты укладываются в него полностью,
            # ручной множитель блика — снова 1 (как было при выборе из списка)
            upd['fit'] = 1.0
            if 'spec_gain' in self.params:
                upd['spec_gain'] = 1.0
        upd.update(extra)
        self.on_change(upd)
        self.params.update(upd)
        self.draw()

    def toggle(self, k):
        """Palette click: remove material k, or add it with an equal share (the
        others shrink proportionally)."""
        lys = self.layers()
        keys = [a for a, _ in lys]
        if k in keys:
            lys = [(a, w) for a, w in lys if a != k]
        else:
            n = len(lys) + 1
            lys = [(a, w * (n - 1) / n) for a, w in lys]
            # при своём порядке новый встаёт туда, куда его ставит его блеск
            # среди уже выбранных; иначе порядок всё равно наведёт layers
            r = self.MAT.gloss_rank(k)
            j = next((i for i, (a, _) in enumerate(lys) if self.MAT.gloss_rank(a) > r), len(lys))
            lys.insert(j, (k, 1.0 / n))
        self._set(self.MAT.layers([[a, w] for a, w in lys],
                                  bool(self.params.get('mats_manual'))))

    def by_gloss(self):
        """Put the bands back in the order of the materials' own shininess
        (undo the swaps)."""
        lys = self.MAT.layers(self.params.get('mats'), False)
        self._set(lys, mats_manual=False)

    def swap(self, a, b):
        """Swap bands a and b (indices from the dull end), each keeping its
        width in place: the material moves, the border stays."""
        lys = self.layers()
        if a == b or not (0 <= a < len(lys) and 0 <= b < len(lys)):
            return
        keys = [k for k, _ in lys]
        keys[a], keys[b] = keys[b], keys[a]
        self._set([(k, w) for k, (_, w) in zip(keys, lys)], mats_manual=True)

    def equal(self):
        """Give every chosen material the same share of the range."""
        lys = self.layers()
        if lys:
            self._set([(k, 1.0 / len(lys)) for k, _ in lys])

    def clear(self):
        """Remove all materials: the node falls back to its sliders."""
        self._set([])

    def show_info(self, k):
        """Under the palette: roughness, metalness and f0 of material k (None
        clears)."""
        if k is None:
            self.info.config(text='')
            return
        m = self.MAT.mat(k)
        txt = t('ed.mat.info', t('nv.' + k), m['rough_lo'], m['rough_hi'], m['metal_lo'],
                m['metal_hi'], m['f0'])
        self.info.config(text=txt)

    # --- шкала: x ← блеск
    def _range(self):
        """(lo, hi, cut_dull, cut_shiny) in shininess, or None without a histogram."""
        h = self.hist
        if not h:
            return None
        return float(h['lo']), float(h['hi']), float(h['cut_dull']), float(h['cut_shiny'])

    def _x(self, s):
        """Shininess s → x on the scale."""
        rg = self._range()
        lo, hi = (rg[0], rg[1]) if rg else (0.0, 1.0)
        return self.M + (s - lo) / max(hi - lo, 1e-6) * self.W

    def _cuts_x(self):
        """x of the dull and the shiny cut marks."""
        rg = self._range()
        if rg:
            return self._x(rg[2]), self._x(rg[3])
        cd = float(self.params.get('clip_dull', 2.0)) / 100.0
        cs = float(self.params.get('clip_shiny', 2.0)) / 100.0
        return self.M + cd * self.W, self.M + (1.0 - cs) * self.W

    def _bounds_x(self):
        """x of every band border, from the dull cut to the shiny cut."""
        xd, xs = self._cuts_x()
        out, acc = [xd], 0.0
        for _, w in self.layers():
            acc += w
            out.append(xd + acc * (xs - xd))
        return out

    def draw(self):
        """Redraw palette rings, histogram, cut marks, material bands, the
        pipette marker and the text under the scale."""
        if not self.alive():
            return
        lys = self.layers()
        keys = [k for k, _ in lys]
        for cv, items in self.pal:
            for it, (k, ring) in items.items():
                cv.itemconfigure(ring, outline=ACC if k in keys else '')
        c = self.cv
        c.delete('all')
        top, bot = 4, 4 + self.HH
        if self.hist:
            hh = self.hist['h']
            mx = max(max(hh), 1e-9)
            n = len(hh)
            for j, v in enumerate(hh):
                x0 = self.M + j * self.W / n
                x1 = self.M + (j + 1) * self.W / n
                c.create_rectangle(x0, bot - min(v / mx, 1.0) * self.HH, x1, bot,
                                   fill='#5c636b', outline='')
        else:
            c.create_text(self.M + self.W / 2, (top + bot) / 2, text=t('ed.mat.nohist'),
                          fill='#8a9298', width=self.W - 10)
        xd, xs = self._cuts_x()
        c.create_rectangle(self.M, top, xd, bot, fill='#000000', stipple='gray50', outline='')
        c.create_rectangle(xs, top, self.M + self.W, bot, fill='#000000', stipple='gray50',
                           outline='')
        y0, y1 = bot + 4, bot + 4 + self.BAND
        bx = self._bounds_x()
        if lys:
            for j, (k, w) in enumerate(lys):
                a, b = bx[j], bx[j + 1]
                col = '#%02x%02x%02x' % tuple(int(v * 255) for v in
                                              self.MAT._LOOK.get(k, (0.6, 0.6, 0.6)))
                if k in self.MAT._METAL_F0:
                    col = '#%02x%02x%02x' % tuple(int((v ** (1 / 2.2)) * 255)
                                                  for v in self.MAT._METAL_F0[k])
                c.create_rectangle(a, y0, b, y1, fill=col, outline='#1a1d20')
                name = t('nv.' + k).split(',')[0]
                if b - a > 18:
                    lum = sum(int(col[i:i + 2], 16) for i in (1, 3, 5)) / 3
                    c.create_text((a + b) / 2, (y0 + y1) / 2, text=name[:max(1, int((b - a) / 7))],
                                  fill='#000000' if lum > 140 else '#ffffff')
            for x in bx[1:-1]:
                c.create_line(x, y0 - 3, x, y1 + 3, fill='#ffffff', width=2)
            if isinstance(self.drag, tuple) and self.drag[0] == 'm':
                j, to = self.drag[1], getattr(self, 'drop', None)
                c.create_rectangle(bx[j] + 1, y0 + 1, bx[j + 1] - 1, y1 - 1, outline=ACC,
                                   width=2, dash=(3, 2))
                if to is not None and to != j:
                    c.create_rectangle(bx[to] + 1, y0 + 1, bx[to + 1] - 1, y1 - 1,
                                       outline=ACC, width=3)
        else:
            c.create_rectangle(xd, y0, xs, y1, fill='#2a2f34', outline='')
            c.create_text((xd + xs) / 2, (y0 + y1) / 2, text=t('ed.mat.sliders'),
                          fill='#8a9298')
        for x in (xd, xs):
            c.create_line(x, top, x, y1 + 3, fill=ACC, width=2)
            c.create_polygon(x - 5, y1 + 4, x + 5, y1 + 4, x, y1 - 2, fill=ACC, outline='')
        if self.pick_s is not None and self._range():
            px = self._x(self.pick_s)
            c.create_line(px, top, px, y1, fill='#e05ad0', width=2, dash=(3, 2))
            c.create_oval(px - 4, top - 1, px + 4, top + 7, fill='#e05ad0', outline='')
        c.create_text(self.M, y1 + 14, text=t('ed.mat.dull'), anchor='w', fill='#8a9298')
        c.create_text(self.M + self.W, y1 + 14, text=t('ed.mat.shiny'), anchor='e',
                      fill='#8a9298')
        c.create_text(self.M + self.W / 2, y1 + 28, fill='#8a9298',
                      text=t('ed.mat.cuts', float(self.params.get('clip_dull', 2.0)),
                             float(self.params.get('clip_shiny', 2.0))))
        if lys:
            self.list.config(text='  ·  '.join('%s %d %%' % (t('nv.' + k).split(',')[0],
                                                          round(w * 100)) for k, w in lys))
            self.note.config(text=t('ed.mat.note') + (
                '\n' + t('ed.mat.manual') if self.params.get('mats_manual') else ''))
        else:
            self.list.config(text=t('ed.mat.none'))
            self.note.config(text='')

    # --- мышь
    def _press(self, e):
        """Grab the cut mark or band border within 7 px of the click."""
        xd, xs = self._cuts_x()
        cand = [('cd', xd), ('cs', xs)] + [(('b', j), x)
                                            for j, x in enumerate(self._bounds_x()[1:-1], 1)]
        best = min(cand, key=lambda c: abs(c[1] - e.x)) if cand else None
        self.drag = best[0] if best and abs(best[1] - e.x) <= 7 else None
        if self.drag is None:
            # щелчок внутри полосы — её можно перетащить на место другой
            j = self._band_at(e.x, e.y)
            if j is not None:
                self.drag = ('m', j)
                self.drop = j

    def _band_at(self, x, y):
        """Index of the material band under (x, y), or None."""
        y0 = 4 + self.HH + 4
        if not (y0 - 2 <= y <= y0 + self.BAND + 2):
            return None
        bx = self._bounds_x()
        for j in range(len(bx) - 1):
            if bx[j] <= x < bx[j + 1]:
                return j
        return None

    def _pct(self, x, side):
        """Share of pixels (%) duller than x ('cd') or shinier than x ('cs')."""
        rg = self._range()
        f = min(max((x - self.M) / self.W, 0.0), 1.0)
        if not rg:
            return 100.0 * (f if side == 'cd' else 1.0 - f)
        h = self.hist['h']
        n = len(h)
        pos = f * n
        j = int(min(pos, n - 1))
        below = sum(h[:j]) + h[j] * (pos - j)
        tot = sum(h) or 1.0
        return 100.0 * (below / tot if side == 'cd' else 1.0 - below / tot)

    def _motion(self, e):
        """Drag the grabbed mark (cut share in % of pixels, 0…40) or border
        (shares of the neighbouring materials); only the local copy changes."""
        if self.drag is None:
            return
        if self.drag[0] == 'm':
            # полоса тянется: подсвечиваем, на чьё место она встанет
            bx = self._bounds_x()
            x = min(max(e.x, bx[0]), bx[-1] - 1)
            self.drop = next((j for j in range(len(bx) - 1) if bx[j] <= x < bx[j + 1]),
                             self.drag[1])
            self.draw()
            return
        xd, xs = self._cuts_x()
        if self.drag in ('cd', 'cs'):
            x = min(e.x, xs - 4) if self.drag == 'cd' else max(e.x, xd + 4)
            key = 'clip_dull' if self.drag == 'cd' else 'clip_shiny'
            self.params[key] = round(min(max(self._pct(x, self.drag), 0.0), 40.0), 1)
            if self.hist:
                # метки на шкале двигаем сразу, по самой мыши
                s = self.hist['lo'] + (x - self.M) / self.W * (self.hist['hi'] - self.hist['lo'])
                self.hist = dict(self.hist, **{'cut_dull' if self.drag == 'cd'
                                               else 'cut_shiny': s})
        else:
            j = self.drag[1]
            bx = self._bounds_x()
            lo, hi = bx[j - 1] + 4, bx[j + 1] - 4
            if hi <= lo:
                return
            x = min(max(e.x, lo), hi)
            fr = [(b - xd) / max(xs - xd, 1e-6) for b in bx]
            fr[j] = (x - xd) / max(xs - xd, 1e-6)
            lys = self.layers()
            self.params['mats'] = [[k, round(fr[i + 1] - fr[i], 4)] for i, (k, _) in enumerate(lys)]
        self.draw()

    def _release(self, e):
        """Drag finished: send the changed parameter to the node."""
        if self.drag is None:
            return
        d, self.drag = self.drag, None
        if d[0] == 'm':
            to, self.drop = getattr(self, 'drop', d[1]), None
            if to != d[1]:
                self.swap(d[1], to)
            else:
                self.draw()
            return
        if d in ('cd', 'cs'):
            key = 'clip_dull' if d == 'cd' else 'clip_shiny'
            self.on_change({key: self.params[key]})
        else:
            self.on_change({'mats': self.params['mats']})


class InstallWindow:
    """A window that runs install/download steps one after another.

    Shows every line pip prints and the download progress while it happens —
    before, an install ran silently and the only sign of it was a word in
    the status bar. The steps run in a worker thread; the thread only appends
    to a list and sets numbers, the window polls them.

    steps: list of (title, fn(line, prog) → (ok, text)); line(text) receives
    output lines, prog(done, total) download progress. done_cb(ok) is called
    when the window is closed.
    """

    active = 0          # сколько окон установки сейчас работает

    def __init__(self, root, steps, done_cb=None):
        """Open the window and start the first step."""
        self.root, self.steps, self.done_cb = root, list(steps), done_cb
        self.w = tk.Toplevel(root)
        self.w.title(t('ed.m.title'))
        self.w.configure(background=BG)
        self.w.geometry('760x460')
        self.w.transient(root)
        self.head = ttk.Label(self.w, text='', style='Head.TLabel')
        self.head.pack(anchor='w', padx=12, pady=(10, 4))
        self.bar = ttk.Progressbar(self.w, mode='determinate', maximum=1.0)
        self.bar.pack(fill='x', padx=12)
        self.log = tk.Text(self.w, background=PANEL, foreground=FG, relief='flat',
                           font=('TkFixedFont', 8), wrap='word')
        self.log.pack(fill='both', expand=True, padx=12, pady=8)
        self.btn = ttk.Button(self.w, text=t('ed.m.close'), command=self.close,
                              state='disabled')
        self.btn.pack(pady=(0, 10))
        self.w.protocol('WM_DELETE_WINDOW', self.close)
        self.lines, self.prog, self.cur, self.ok = [], None, None, True
        self.running = False
        self.w.grab_set()
        self.next()

    def next(self):
        """Start the next step, or finish."""
        if not self.steps:
            self.head.config(text=t('ed.m.alldone') if self.ok else t('ed.m.fail'))
            self.bar.stop()
            self.bar.config(mode='determinate', value=1.0 if self.ok else 0.0)
            self.btn.config(state='normal')
            if self.running:
                InstallWindow.active -= 1
            self.running = False
            return
        title, fn = self.steps.pop(0)
        self.cur = None
        self.prog = None
        self.__dict__.pop('_speed0', None)
        self.head.config(text=title)
        self.lines.append('── ' + title)
        self.bar.config(mode='indeterminate')
        self.bar.start(12)
        if not self.running:
            InstallWindow.active += 1
        self.running = True

        def prog(done, total):
            """Download progress (worker thread)."""
            self.prog = (done, total)

        def work():
            """Worker thread: run the step; errors become text."""
            try:
                self.cur = fn(self.lines.append, prog)
            except Exception as e:                   # noqa: BLE001
                self.cur = (False, '%s: %s' % (e.__class__.__name__, e))
        threading.Thread(target=work, daemon=True).start()
        self.poll()

    def poll(self):
        """Main thread: move new lines into the log and show progress."""
        if not self.w.winfo_exists():
            return
        if self.lines:
            new, self.lines[:] = self.lines[:], []
            self.log.insert('end', '\n'.join(new) + '\n')
            self.log.see('end')
        if self.prog:
            d, tot = self.prog
            if str(self.bar.cget('mode')) != 'determinate':
                self.bar.stop()
                self.bar.config(mode='determinate')
            self.bar['value'] = d / float(max(tot, 1))
            now = time.time()
            t0, d0 = getattr(self, '_speed0', (now, d))
            if now - t0 > 8:                       # скорость по последним ~8 с
                self._speed0 = (now, d)
            elif not hasattr(self, '_speed0'):
                self._speed0 = (now, d)
            v = (d - d0) / max(now - t0, 1e-3)
            eta = (tot - d) / v if v > 1 and tot else 0
            self.head.config(text='%s   %.1f / %.1f MB   %.0f KB/s%s' % (
                self.head.cget('text').split('   ')[0], d / 1048576.0, tot / 1048576.0,
                v / 1024.0, ('   ~%d min' % max(1, eta // 60)) if eta > 60 else ''))
        if self.cur is None:
            self.root.after(200, self.poll)
            return
        ok, text = self.cur
        if ok:
            self.log.insert('end', '✓  ' + (text or t('ed.m.ok')) + '\n\n')
        else:
            self.ok = False
            msg = t('rt.restart') if text == 'RESTART' else text
            self.log.insert('end', '✗  ' + t('ed.m.fail') + '\n' + msg + '\n\n')
            self.steps.clear()          # дальше смысла нет — ошибка на экране
        self.log.see('end')
        self.next()

    def close(self):
        """Close when finished (the steps cannot be stopped half-way)."""
        if self.running:
            return
        try:
            self.w.grab_release()
        except Exception:
            pass
        self.w.destroy()
        if self.done_cb:
            self.done_cb(self.ok)


def runtime_dialog(root, done_cb=None):
    """Choose where the networks compute: processor or graphics card.

    Lists the onnxruntime builds of depthnet.RUNTIMES with what each needs
    and its licence; installing one replaces the other.
    """
    import depthnet as DN
    cur = DN.runtime_kind() if DN.have_runtime() else ''
    dlg = tk.Toplevel(root)
    dlg.title(t('rt.title'))
    dlg.configure(background=BG)
    dlg.transient(root)

    # --- узлы и 3D-вид: CuPy на видеокарте NVIDIA
    ttk.Label(dlg, text=t('gpu.head'), style='Head.TLabel').pack(anchor='w', padx=14,
                                                                pady=(12, 2))
    st = ttk.Label(dlg, text=gpu_status_text(), style='Hint.TLabel', wraplength=540,
                   justify='left')
    st.pack(anchor='w', padx=14)
    gf = ttk.Frame(dlg)
    gf.pack(fill='x', padx=14, pady=(2, 6))
    if X.have_cupy():
        gv = tk.BooleanVar(value=X.active())

        def toggle():
            on, info = X.use(gv.get())
            DN.save_settings(gpu=bool(gv.get()))
            gv.set(on)
            st.config(text=gpu_status_text())
            if done_cb:
                done_cb(True)
        ttk.Checkbutton(gf, text=t('gpu.use'), variable=gv, command=toggle).pack(anchor='w')
    else:
        ttk.Label(gf, text=t('gpu.info'), style='Hint.TLabel', wraplength=540,
                  justify='left').pack(anchor='w')

        def inst():
            dlg.destroy()
            InstallWindow(root, [gpu_step()], done_cb)
        ttk.Button(gf, text=t('gpu.install'), command=inst).pack(anchor='w', pady=(4, 0))
    ttk.Separator(dlg).pack(fill='x', padx=14, pady=6)

    ttk.Label(dlg, text=t('rt.head'), style='Head.TLabel').pack(anchor='w', padx=14,
                                                               pady=(6, 4))
    ttk.Label(dlg, text=t('rt.cur', t('rt.' + cur) if cur else t('rt.none')),
              style='Hint.TLabel').pack(anchor='w', padx=14)
    # с CuPy на NVIDIA сетям лучше та же CUDA: библиотеки общие
    choice = tk.StringVar(value=cur or DN.load_settings().get('runtime') or
                          ('cuda' if X.active() else 'webgpu'))
    for k in DN.RUNTIME_IDS:
        f = ttk.Frame(dlg)
        f.pack(fill='x', padx=14, pady=4)
        ttk.Radiobutton(f, text=t('rt.' + k), value=k, variable=choice).pack(anchor='w')
        ttk.Label(f, text=t('rt.' + k + '.info'), style='Hint.TLabel', wraplength=520,
                  justify='left').pack(anchor='w', padx=(22, 0))

    def go():
        k = choice.get()
        dlg.destroy()
        DN.save_settings(runtime=k)
        InstallWindow(root, [runtime_step(k)], done_cb)
    ttk.Button(dlg, text=t('rt.go'), command=go).pack(pady=(6, 12))
    dlg.grab_set()


def gpu_status_text():
    """Where the nodes and the 3D view compute now, as a sentence."""
    if X.active():
        return t('gpu.on', X.device_name())
    if not X.have_cupy():
        return t('gpu.nocupy')
    return t('gpu.off', X.last_error() or '—')


def gpu_step():
    """Install step: CuPy for CUDA 13 with its CUDA libraries, then switch on."""
    import depthnet as DN

    def run(line, prog):
        ok, text = DN.install_runtime(GPU_PIP, 'cupy', line)
        if not ok:
            return False, text
        on, info = X.use(True)
        DN.save_settings(gpu=True)
        return (True, t('gpu.on', info)) if on else (False, t('gpu.off', info))
    return ('CuPy', run)


GPU_PIP = 'cupy-cuda13x[ctk]'


def init_gpu():
    """At start-up: switch the card on if CuPy is installed and not turned off."""
    import depthnet as DN
    if X.have_cupy() and DN.load_settings().get('gpu', True):
        X.use(True)


def run_pending_runtime(root):
    """At start-up, before anything loads onnxruntime: finish a build switch
    that had to wait for a restart (see depthnet.install_runtime)."""
    import depthnet as DN
    k = DN.load_settings().get('pending')
    if not k:
        return
    DN.save_settings(pending='')
    InstallWindow(root, [runtime_step(k)])


# ---------------------------------------------------- меню и клавиши --------
def install_menu(app):
    """Menu bar and window-wide shortcuts (bound once per window).

    The handlers look up ``app.editor`` at call time, so they keep working
    after the window is rebuilt for another language.
    """
    root = app.root
    ed = lambda: app.editor                                    # noqa: E731

    def node_mode():
        """True while the node workspace (not the classic tabs) is shown."""
        return not app.classic_on

    kw = dict(tearoff=0, background=PANEL, foreground=FG, activebackground=EDGE,
              activeforeground=FG)
    mb = tk.Menu(root, **kw)
    m = tk.Menu(mb, **kw)
    m.add_command(label=t('ed.new'), accelerator='Ctrl+N', command=lambda: ed().new_project())
    m.add_command(label=t('ed.open'), accelerator='Ctrl+O', command=lambda: ed().open_project())
    m.add_command(label=t('ed.save'), accelerator='Ctrl+S', command=lambda: ed().save_project())
    m.add_command(label=t('ed.saveas'), accelerator='Ctrl+Shift+S',
                  command=lambda: ed().save_project(True))
    m.add_separator()
    m.add_command(label=t('ed.tex'), accelerator='Ctrl+T', command=lambda: ed().open_texture())
    sm = tk.Menu(m, **kw)
    for p in samples():
        sm.add_command(label=os.path.splitext(os.path.basename(p))[0],
                       command=lambda p=p: ed().open_texture(p))
    m.add_cascade(label=t('ed.samples'), menu=sm)
    m.add_command(label=t('ed.savemaps'), command=lambda: ed().save_maps())
    m.add_command(label=t('ed.batch'), command=lambda: ed().batch_dialog())
    m.add_separator()
    m.add_command(label=t('ed.quit'), command=app.quit)
    mb.add_cascade(label=t('ed.menu.project'), menu=m)
    m = tk.Menu(mb, **kw)
    m.add_command(label=t('ed.undo'), accelerator='Ctrl+Z', command=lambda: ed().undo())
    m.add_command(label=t('ed.redo'), accelerator='Ctrl+Y', command=lambda: ed().redo())
    m.add_separator()
    m.add_command(label=t('ed.addnode'), accelerator='Shift+A',
                  command=lambda: ed().menu_at_mouse())
    m.add_command(label=t('ed.m.dup'), accelerator='Ctrl+D', command=lambda: ed().duplicate())
    m.add_command(label=t('ed.m.del'), accelerator='X', command=lambda: ed().delete_sel())
    m.add_command(label=t('ed.selall'), accelerator='Ctrl+A', command=lambda: ed().select_all())
    mb.add_cascade(label=t('ed.menu.edit'), menu=m)
    m = tk.Menu(mb, **kw)
    m.add_command(label=t('ed.frame'), accelerator='F', command=lambda: ed().frame_all())
    m.add_command(label=t('ed.thumbs'), accelerator='H', command=lambda: ed().toggle_thumbs())
    m.add_separator()
    m.add_command(label=t('rt.title') + '…',
                  command=lambda: runtime_dialog(root, ed()._after_install))
    m.add_command(label=t('ed.classic'), command=lambda: app.show_classic(not app.classic_on))
    mb.add_cascade(label=t('ed.menu.view'), menu=m)
    m = tk.Menu(mb, **kw)
    m.add_command(label=t('top.guide'), command=app.open_guide)
    m.add_command(label=t('ed.keys'), command=lambda: messagebox.showinfo(
        t('ed.keys'), t('ed.keys.list')))
    mb.add_cascade(label=t('ed.menu.help'), menu=m)
    root.config(menu=mb)
    if getattr(root, '_texelith_keys', False):
        return
    root._texelith_keys = True

    def key(seq, fn):
        """Bind a window-wide shortcut that acts only in the node workspace."""
        def h(e):
            """Key handler: run fn unless typing in a text field; stop further handling."""
            if not node_mode():
                return None
            # в поле ввода Ctrl+Z и Ctrl+A — его собственные
            if isinstance(e.widget, (tk.Entry, ttk.Entry, tk.Text)) and seq in (
                    '<Control-z>', '<Control-a>', '<Control-y>'):
                return None
            fn()
            return 'break'
        root.bind(seq, h)
    key('<Control-n>', lambda: ed().new_project())
    key('<Control-o>', lambda: ed().open_project())
    key('<Control-s>', lambda: ed().save_project())
    key('<Control-S>', lambda: ed().save_project(True))
    key('<Control-t>', lambda: ed().open_texture())
    key('<Control-z>', lambda: ed().undo())
    key('<Control-y>', lambda: ed().redo())
    key('<Control-Z>', lambda: ed().redo())
