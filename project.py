# -*- coding: utf-8 -*-
"""
Texelith — projects: templates, saving, and batch processing of a folder.

A project is one node graph plus everything around it that a person would not
want to set up twice: the texture used for the preview, the batch folders and
options, and where the view was. It is stored as JSON in a ``.texelith`` file;
paths inside are kept relative to the project file when they can be, so a
project folder can be moved or given to someone else.

Batch processing lives here rather than in the window, so it can also run
without one:

    python project.py wall.texelith  <input folder>  <output folder>

goes through the input folder (with subfolders), evaluates the graph for every
texture and writes the outputs into the same relative place under the output
folder — ready to be dropped into a data folder of OpenMW. Textures whose
outputs already exist are skipped, so an interrupted run of a few thousand
files continues where it stopped. A log is written next to the outputs.

Texelith — проекты: шаблоны, сохранение и пакетная обработка папки. Проект —
это граф плюс то, что не хочется настраивать дважды: текстура для превью,
папки и настройки пакета, положение вида. Пакет можно запустить и без окна
(см. выше); уже сделанные текстуры пропускаются, так что прерванный прогон
продолжается с места остановки.
"""

import fnmatch
import json
import os
import sys
import time

import numpy as np

import nodes as ND
import texio as TX
import xpu as X

BUILD = 25          # версия набора файлов, проверяется при запуске

EXT = '.texelith'
# проекты, сохранённые до переименования (сборка 22 и раньше), тоже открываются
OLD_EXT = '.tessera'
EXTS = (EXT, OLD_EXT)
FORMAT = 1          # версия формата файла проекта

# Что пакет по умолчанию не трогает: интерфейс и картинки, которым рельеф не
# нужен. Шаблоны fnmatch по относительному пути в нижнем регистре, через «/».
DEFAULT_SKIP = 'icons/*; bookart/*; levelup/*; splash/*; distantland/*; menu_*; cursor*; tx_sky*'

def diffuse_alpha(v, src_alpha, name):
    """The alpha to write a graph output with: for a 'diffuse' output its
    own fourth channel if it has one (the output node puts there the 'alpha'
    input or the source alpha), else the source alpha; for other outputs
    None (their fourth channel is part of the map itself)."""
    if name != 'diffuse':
        return None
    if v is not None and v.ndim == 3 and v.shape[2] == 4:
        return v[..., 3]
    return src_alpha


BATCH_DEFAULTS = {'in': '', 'out': '', 'recursive': True, 'format': '.dds',
                  'compress': True, 'skip_done': True, 'skip': DEFAULT_SKIP,
                  'layout': 'flat', 'dxt_algo': 'quicktex', 'dxt_level': 14}

# Как раскладывать карты в папке вывода: 'flat' — рядом друг с другом, как
# лежат исходники; 'by_map' — в подпапку на каждый вид карты (n, nh, spec…);
# 'by_texture' — в подпапку на каждую текстуру.
LAYOUTS = ('flat', 'by_map', 'by_texture')


def map_folder(name):
    """Subfolder of an output in the 'by_map' layout: the suffix without
    its underscore ('_nh' → 'nh'), 'diffuse' for the replacement diffuse."""
    return name.lstrip('_') or 'diffuse'


# ------------------------------------------------------------ шаблоны ------
def _grid(g, key, col, row, **params):
    """Add a node at grid cell (col, row) with parameter overrides; return its id."""
    nid = g.add(key, 40 + col * 230, 40 + row * 200)
    g.nodes[nid]['params'].update(params)
    return nid


def _out(g, col, row, src, port, name):
    """Add an output node named ``name`` fed from src.port."""
    o = _grid(g, 'out', col, row, name=name)
    g.connect(src, port, o, 'any')
    return o


def t_classic():
    """No networks: de-light, height from luminance and lighting,
    normal, OpenMW _nh, diffuse with AO, _spec."""
    g = ND.Graph()
    # починки битых мест в шаблоне нет: на кладке она принимает тёмные швы
    # за провалы и заливает их; её добавляют руками, когда она нужна
    r = s = _grid(g, 'source', 0, 1)
    d = _grid(g, 'delight', 2, 1)
    hl = _grid(g, 'height_lum', 3, 0)
    hs = _grid(g, 'height_light', 3, 1)
    mx = _grid(g, 'mix', 4, 0, wa=1.0, wb=0.6)
    n = _grid(g, 'normal', 5, 0)
    px = _grid(g, 'parallax_nh', 6, 0)
    ao = _grid(g, 'ao', 5, 1)
    rl = _grid(g, 'relief', 6, 2, mode='multiply', occl=True, amount=1.0)
    sp = _grid(g, 'spec', 5, 2)
    pb = _grid(g, 'pbr_openmw', 7, 2, metal_dark=0.0, ao_diffuse=0.0)
    for a, ap, b, bp in ((r, 'rgb', d, 'rgb'),
                         (d, 'rgb', hl, 'rgb'), (d, 'rgb', hs, 'rgb'),
                         (hl, 'h', mx, 'a'), (hs, 'h', mx, 'b'),
                         (mx, 'h', n, 'h'), (n, 'n', px, 'n'), (mx, 'h', px, 'h'),
                         (mx, 'h', ao, 'h'), (d, 'rgb', rl, 'rgb'), (ao, 'ao', rl, 's'),
                         (d, 'rgb', sp, 'rgb'), (ao, 'ao', sp, 'ao'),
                         (rl, 'rgb', pb, 'base'), (sp, 'rough', pb, 'rough'),
                         (ao, 'ao', pb, 'ao')):
        g.connect(a, ap, b, bp)
    _out(g, 7, 0, px, 'nh', '_nh')
    _out(g, 8, 1, rl, 'rgb', 'diffuse')
    _out(g, 8, 2, pb, 'spec', '_spec')
    return g


def t_neural():
    """DeepBump normal balanced with a depth network, OpenMW _nh, diffuse with
    AO, _spec."""
    g = ND.Graph()
    r = s = _grid(g, 'source', 0, 1)
    db = _grid(g, 'deepbump', 2, 0)
    dn = _grid(g, 'depth_net', 2, 2)
    b = _grid(g, 'nh_balance', 3, 1, balance=0.55)
    px = _grid(g, 'parallax_nh', 4, 0)
    ao = _grid(g, 'ao', 4, 2)
    rl = _grid(g, 'relief', 5, 2, mode='multiply', occl=True, amount=1.0)
    d = _grid(g, 'delight', 2, 3)
    sp = _grid(g, 'spec', 5, 3)
    pb = _grid(g, 'pbr_openmw', 6, 3, metal_dark=0.0, ao_diffuse=0.0)
    for a, ap, bb, bp in ((r, 'rgb', db, 'rgb'), (r, 'rgb', dn, 'rgb'),
                          (db, 'n', b, 'n'), (dn, 'h', b, 'h'),
                          (b, 'n', px, 'n'), (b, 'h', px, 'h'), (b, 'h', ao, 'h'),
                          (d, 'rgb', rl, 'rgb'), (ao, 'ao', rl, 's'),
                          (r, 'rgb', d, 'rgb'), (d, 'rgb', sp, 'rgb'),
                          (ao, 'ao', sp, 'ao'), (rl, 'rgb', pb, 'base'),
                          (sp, 'rough', pb, 'rough'), (ao, 'ao', pb, 'ao')):
        g.connect(a, ap, bb, bp)
    _out(g, 5, 0, px, 'nh', '_nh')
    _out(g, 6, 2, rl, 'rgb', 'diffuse')
    _out(g, 7, 3, pb, 'spec', '_spec')
    return g


def t_supermat():
    """SuperMat PBR maps converted for OpenMW; normal and height from DeepBump,
    balanced with a depth network (SuperMat itself gives neither)."""
    g = ND.Graph()
    s = _grid(g, 'source', 0, 1)
    c = _grid(g, 'supermat', 1, 1)
    db = _grid(g, 'deepbump', 1, 2)
    dn = _grid(g, 'depth_net', 1, 3)
    b = _grid(g, 'nh_balance', 2, 2, balance=0.55)
    px = _grid(g, 'parallax_nh', 3, 0)
    ao = _grid(g, 'ao', 3, 2)
    pb = _grid(g, 'pbr_openmw', 4, 1)
    for a, ap, bb, bp in ((s, 'rgb', c, 'rgb'), (s, 'rgb', db, 'rgb'), (s, 'rgb', dn, 'rgb'),
                          (db, 'n', b, 'n'), (dn, 'h', b, 'h'),
                          (b, 'n', px, 'n'), (b, 'h', px, 'h'), (b, 'h', ao, 'h'),
                          (c, 'base', pb, 'base'), (c, 'metal', pb, 'metal'),
                          (c, 'rough', pb, 'rough'), (ao, 'ao', pb, 'ao')):
        g.connect(a, ap, bb, bp)
    _out(g, 4, 0, px, 'nh', '_nh')
    _out(g, 5, 1, pb, 'diffuse', 'diffuse')
    _out(g, 5, 2, pb, 'spec', '_spec')
    return g


def t_normal():
    """The quickest: height from luminance, a plain _n map."""
    g = ND.Graph()
    s = _grid(g, 'source', 0, 0)
    d = _grid(g, 'delight', 1, 0)
    h = _grid(g, 'height_lum', 2, 0)
    n = _grid(g, 'normal', 3, 0)
    g.connect(s, 'rgb', d, 'rgb')
    g.connect(d, 'rgb', h, 'rgb')
    g.connect(h, 'h', n, 'h')
    _out(g, 4, 0, n, 'n', '_n')
    return g


def t_fixnormal():
    """A finished normal map as the source: layout detected, strength and
    tilt limit, height integrated from it, packed as _nh."""
    g = ND.Graph()
    s = _grid(g, 'source', 0, 0)
    a = _grid(g, 'as_normal', 1, 0, green='auto')
    f = _grid(g, 'normal_fix', 2, 0)
    h = _grid(g, 'normal_to_height', 3, 1, cutoff=0.25)
    px = _grid(g, 'parallax_nh', 4, 0)
    for x, xp, y, yp in ((s, 'rgb', a, 'rgb'), (a, 'n', f, 'n'), (f, 'n', h, 'n'),
                         (f, 'n', px, 'n'), (h, 'h', px, 'h')):
        g.connect(x, xp, y, yp)
    _out(g, 5, 0, px, 'nh', '_nh')
    return g


def t_foliage():
    """Foliage card with alpha: the «Foliage» node gives the per-leaf normal
    with the foliage height (_nh) and the crown AO; the diffuse is de-lit by
    the «Remove light» node."""
    g = ND.Graph()
    s = _grid(g, 'source', 0, 1)
    f = _grid(g, 'foliage', 1, 1)
    px = _grid(g, 'parallax_nh', 2, 0)
    dl = _grid(g, 'delight', 1, 2)
    g.connect(s, 'rgb', dl, 'rgb')
    # ветки вручную, как кистью на вкладке: белым по веткам в окне 2D
    bm = _grid(g, 'paint_mask', 0, 0)
    g.connect(s, 'rgb', bm, 'rgb')
    g.connect(bm, 'mask', f, 'hand')
    g.connect(s, 'rgb', f, 'rgb')
    g.connect(s, 'a', f, 'a')
    g.connect(f, 'n', px, 'n')
    g.connect(f, 'h', px, 'h')
    _out(g, 3, 0, px, 'nh', '_nh')
    _out(g, 3, 1, dl, 'rgb', 'diffuse')
    _out(g, 3, 2, f, 'ao', '_ao')
    return g


def t_empty():
    """Only the source node."""
    g = ND.Graph()
    _grid(g, 'source', 0, 0)
    return g


TEMPLATES = (('neural', t_neural), ('classic', t_classic), ('supermat', t_supermat),
             ('normal', t_normal), ('fixnormal', t_fixnormal), ('foliage', t_foliage),
             ('empty', t_empty))


def template(key):
    """New graph from template ``key`` (a key of TEMPLATES)."""
    return dict(TEMPLATES)[key]()


# ------------------------------------------------------ файл проекта -------
def _rel(path, base):
    """Path relative to the folder ``base`` when on the same drive, else absolute."""
    if not path:
        return ''
    try:
        return os.path.relpath(path, base)
    except ValueError:
        return os.path.abspath(path)


def _abs(path, base):
    """Inverse of _rel."""
    if not path:
        return ''
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(base, path))


def save(path, graph, source='', batch=None, view=None):
    """Write a project file.

    Args:
        path: output .texelith path.
        graph: nodes.Graph.
        source: path of the preview texture ('' if none).
        batch: dict of batch settings (BATCH_DEFAULTS keys).
        view: free-form dict of window state (selection, pan, zoom…).

    The file is written to ``path + '.part'`` and renamed, so a crash while
    saving never leaves half a project.
    """
    base = os.path.dirname(os.path.abspath(path))
    b = dict(BATCH_DEFAULTS)
    b.update(batch or {})
    b['in'], b['out'] = _rel(b['in'], base), _rel(b['out'], base)
    d = {'texelith': FORMAT, 'build': BUILD,
         'graph': json.loads(graph.to_json()),
         'source': _rel(source, base), 'batch': b, 'view': view or {}}
    tmp = path + '.part'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(d, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def load(path):
    """Read a project file.

    Returns:
        dict with 'graph' (nodes.Graph), 'source' (absolute path or ''),
        'batch' (dict, absolute folders), 'view' (dict).

    Raises:
        ValueError: not a Texelith project.
    """
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    if 'graph' not in d:
        raise ValueError('not a Texelith project')
    base = os.path.dirname(os.path.abspath(path))
    g = ND.Graph()
    g.from_json(json.dumps(d['graph']))
    b = dict(BATCH_DEFAULTS)
    b.update(d.get('batch') or {})
    b['in'], b['out'] = _abs(b['in'], base), _abs(b['out'], base)
    return {'graph': g, 'source': _abs(d.get('source', ''), base), 'batch': b,
            'view': d.get('view') or {}}


# ------------------------------------------------------------- пакет -------
def pbr_root(out_dir):
    """Folder of the 'pbr' output set: next to the output folder, with
    '_PBR' added to its name (C:/mods/out → C:/mods/out_PBR)."""
    return os.path.normpath(out_dir).rstrip('\\/') + '_PBR'


def normal_flag(name):
    """as_normal for TX.write_value by output name: None (tell by the
    values) for the normal maps, False for the rest — a packed four-channel
    map (_spec, _ms…) must not be taken for a normal."""
    return None if name in ('_nh', '_n', '_bent') else False


def _skip_patterns(text):
    """Split the skip field ('a/*; b*') into fnmatch patterns."""
    return [x.strip().lower().replace('\\', '/') for x in (text or '').replace(',', ';').split(';')
            if x.strip()]


def collect(in_dir, recursive=True, skip='', out_names=(), layout='flat', out_dir=None):
    """Source textures of a batch, as paths relative to ``in_dir``.

    Finished maps are never sources: names ending with a known map suffix or
    with one of ``out_names`` are left out, otherwise a second run over the
    same folder would process its own output. Paths matching a skip pattern
    are left out too, and so are the subfolders the sorted layouts make
    ('by_map': a folder named after a map kind; 'by_texture': name/name.ext),
    counted from the output folder when it lies inside the input folder —
    a replacement diffuse there has no suffix to tell it by.
    """
    pats = _skip_patterns(skip)
    ends = tuple(x.lower() for x in set(TX.MAP_SUFFIXES) | set(out_names) if x.startswith('_'))
    folders = {map_folder(n).lower() for n in out_names}
    # папки, которые делают раскладки по папкам (этого или прошлого прогона,
    # с любой раскладкой), источниками не считаем: если вывод внутри входной
    # папки — считая от него, иначе — от самой входной папки
    sub = ''
    psub = ''
    if out_dir:
        r = os.path.relpath(os.path.abspath(out_dir), os.path.abspath(in_dir))
        if not r.startswith('..') and r != '.':
            sub = r.replace('\\', '/').lower() + '/'
        # папка набора «PBR» (вывод_PBR) внутри входной — тоже не источник
        r = os.path.relpath(os.path.abspath(pbr_root(out_dir)), os.path.abspath(in_dir))
        if not r.startswith('..') and r != '.':
            psub = r.replace('\\', '/').lower() + '/'
    res = []
    for root, dirs, files in os.walk(in_dir):
        dirs.sort()
        for f in sorted(files):
            stem, ext = os.path.splitext(f)
            if ext.lower() not in TX.TEX_EXT or stem.lower().endswith(ends):
                continue
            rel = os.path.relpath(os.path.join(root, f), in_dir).replace('\\', '/')
            if any(fnmatch.fnmatch(rel.lower(), p) for p in pats):
                continue
            if sub and rel.lower().startswith(sub):
                continue                # вывод внутри входной папки — не источник
            if psub and rel.lower().startswith(psub):
                continue                # и папка набора «PBR»
            if rel.lower().startswith(sub):
                inner = rel[len(sub):].split('/')
                if len(inner) > 1 and inner[0].lower() in folders:
                    continue            # подпапка вида карты (раскладка by_map)
                if len(inner) > 1 and \
                        inner[-2].lower() == os.path.splitext(inner[-1])[0].lower():
                    continue            # папка текстуры (раскладка by_texture)
            res.append(rel)
        if not recursive:
            break
    return res


def targets(graph, rel, in_dir, out_dir, fmt='.dds', layout='flat'):
    """Output files for one source: list of (output node id, path).

    layout (LAYOUTS): 'flat' — suffix outputs go next to where the source
    lies relative to the input folder, under the output folder; 'by_map' —
    the same, but under a subfolder per kind of map (map_folder: n, nh,
    spec…, diffuse); 'by_texture' — under a subfolder named after the
    texture. A 'diffuse' output keeps the source's own name; it is skipped
    when that would be the source file itself — a batch never overwrites
    its sources. Outputs of the 'pbr' set go the same way under
    pbr_root(out_dir) instead of out_dir.
    """
    stem = os.path.splitext(rel)[0]
    sub, base = os.path.split(stem)
    src = os.path.normcase(os.path.abspath(os.path.join(in_dir, rel)))
    res = []
    for nid in graph.outputs():
        name = graph.nodes[nid]['params']['name']
        suffix = '' if name == 'diffuse' else name
        root = pbr_root(out_dir) if graph.nodes[nid]['params'].get('set') == 'pbr' else out_dir
        if layout == 'by_map':
            path = os.path.join(root, map_folder(name), sub, base + suffix + fmt)
        elif layout == 'by_texture':
            path = os.path.join(root, sub, base, base + suffix + fmt)
        else:
            path = os.path.join(root, stem + suffix + fmt)
        if name == 'diffuse' and os.path.normcase(os.path.abspath(path)) == src:
            continue
        res.append((nid, path))
    return res


def run_batch(graph, opts, progress=None, cancel=None, log=None):
    """Evaluate the graph for every texture of a folder and write the outputs.

    Args:
        graph: nodes.Graph; a copy is used, so the caller may keep editing.
        opts: dict with BATCH_DEFAULTS keys ('in', 'out', 'recursive',
            'format', 'compress', 'skip_done', 'skip', 'layout' — see
            targets).
        progress: optional callable(done_fraction, text) — called from the
            calling thread; it must not touch a GUI (the window polls).
        cancel: optional callable() → True to stop after the current file.
        log: optional callable(line) for the log; a file
            ``texelith_batch.log`` is written into the output folder anyway.

    Returns:
        dict: 'done', 'skipped', 'failed' (list of 'file — error'),
        'total', 'seconds', 'log' (path of the log file).
    """
    g = ND.Graph()
    g.from_json(graph.to_json())
    o = dict(BATCH_DEFAULTS)
    o.update(opts or {})
    in_dir, out_dir = o['in'], o['out'] or o['in']
    fmt = o['format'] if o['format'].startswith('.') else '.' + o['format']
    names = [g.nodes[n]['params']['name'] for n in g.outputs()]
    layout = o.get('layout', 'flat')
    files = collect(in_dir, o['recursive'], o['skip'], names, layout, out_dir)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, 'texelith_batch.log')
    t0 = time.time()
    done, skipped, failed = 0, 0, []
    nf = max(len(files), 1)
    with open(log_path, 'a', encoding='utf-8') as lf:
        def say(line):
            """Write a line to the log file and pass it to the caller."""
            lf.write(line + '\n')
            lf.flush()
            if log:
                log(line)
        say('=== %s  %d files  %s → %s' % (time.strftime('%Y-%m-%d %H:%M:%S'),
                                           len(files), in_dir, out_dir))
        for i, rel in enumerate(files):
            if cancel and cancel():
                say('stopped')
                break
            tg = targets(g, rel, in_dir, out_dir, fmt, layout)
            if not tg:
                continue
            if o['skip_done'] and all(os.path.exists(p) for _, p in tg):
                skipped += 1
                continue
            if progress:
                progress(i / nf, rel)
            try:
                rgba = TX.read_rgba(os.path.join(in_dir, rel))
                has_a = float(rgba[..., 3].min()) < 0.999

                def sub(a, b, i=i, rel=rel):
                    """Progress inside one file (network windows)."""
                    if progress:
                        progress((i + a / max(b, 1) * 0.9) / nf, rel)
                ctx = {'src': X.to_dev(np.ascontiguousarray(rgba[..., :3])),
                       'alpha': rgba[..., 3] if has_a else None,
                       'key': 'batch:' + rel, 'sub': sub, 'name': os.path.basename(rel)}
                info = []
                for nid, path in tg:
                    r = g.eval(nid, ctx)
                    v = X.to_host(r.get('any'))
                    if v is None:
                        raise ValueError(nid + ': no value')
                    if r.get('_warn'):
                        import i18n
                        say('warn  %s  %s' % (rel, i18n.t(r['_warn'])))
                    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
                    nm = g.nodes[nid]['params']['name']
                    TX.write_value(v, path, TX.dxt(o), diffuse_alpha(v, ctx['alpha'], nm),
                                   as_normal=normal_flag(nm))
                    info.append(os.path.basename(path))
                done += 1
                say('ok    %s  → %s' % (rel, ', '.join(info)))
            except Exception as e:                    # noqa: BLE001 — в лог и дальше
                failed.append('%s — %s' % (rel, e))
                say('FAIL  %s  %s' % (rel, e))
            g._cache.clear()        # память: 2048² × десяток узлов на файл
        secs = time.time() - t0
        say('=== done %d, skipped %d, failed %d, %.0f s' % (done, skipped, len(failed), secs))
    if progress:
        progress(1.0, '')
    return {'done': done, 'skipped': skipped, 'failed': failed, 'total': len(files),
            'seconds': secs, 'log': log_path}


def main(argv):
    """Command line: ``python project.py project.texelith [in_dir] [out_dir]``.

    Folders not given are taken from the project's batch settings.
    """
    if len(argv) < 2:
        print(__doc__)
        return 2
    pr = load(argv[1])
    opts = dict(pr['batch'])
    if len(argv) > 2:
        opts['in'] = argv[2]
    if len(argv) > 3:
        opts['out'] = argv[3]
    if not opts['in'] or not os.path.isdir(opts['in']):
        print('input folder not found: %r' % opts['in'])
        return 2
    last = [0.0]

    def prog(fr, text):
        """Print progress at most twice a second."""
        if time.time() - last[0] > 0.5:
            last[0] = time.time()
            sys.stdout.write('\r%5.1f%%  %-60s' % (fr * 100, text[-60:]))
            sys.stdout.flush()
    r = run_batch(pr['graph'], opts, prog)
    print('\ndone %d, skipped %d, failed %d, %.0f s; log: %s'
          % (r['done'], r['skipped'], len(r['failed']), r['seconds'], r['log']))
    return 0 if not r['failed'] else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv))
