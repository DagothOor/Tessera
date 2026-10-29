# -*- coding: utf-8 -*-
"""
Texelith — neural monocular depth (MiDaS, Depth Anything).

Why this is here. Everything else in the program derives height from
brightness: darker means lower. That works while the picture is honest and
lies as soon as the artist painted a light patch of dirt or a dark stone next to
a light one. Brightness does not know that a brick protrudes and the joint
between bricks is recessed — it only knows that the joint is darker.

Single-image depth networks know exactly that: they were trained on pairs
"photograph — measured depth" and answer the question "what is closer to the
viewer here", not "what is lighter here". For a texture this is precisely the
missing hint: the network provides the large-scale shape (the brick is convex,
the joint is recessed, the plank is bowed), while fine relief is still taken
from brightness. That is why the result is put into the "external depth map"
and mixed there with the other height layers using the ordinary sliders.

What is here:

  * Depth Anything V2 Small — newer and noticeably more accurate on textures:
    it keeps the edges of bricks and ornament instead of smearing them.
    Licence Apache-2.0. Yang et al., "Depth Anything V2", 2024;
  * MiDaS v2.1 Small — the classic Intel ISL network, twice as fast and half
    the size, but on textures it mostly sees the overall tilt of the wall.
    Licence MIT. Ranftl et al., "Towards Robust Monocular Depth Estimation:
    Mixing Datasets for Zero-shot Cross-dataset Transfer".

Both are open and both run entirely on the user's own machine through
onnxruntime: nothing is sent anywhere, the model file is downloaded once and
kept next to the program settings (~/.texelith/models). The models are
downloaded rather than shipped with the program because of their size (about
99 MB and 67 MB) and because each carries its own licence (Apache-2.0 for
Depth Anything V2 Small, MIT for MiDaS) that the user should see separately
from the program's own. Computation runs on the CPU by default; if an
onnxruntime build with GPU support (CUDA, DirectML, ROCm) is installed it is
picked up automatically, and if it fails to start there is a silent fallback
to the CPU so that the button always works.

Two computation modes:

  * fast — one pass over the whole texture. The network sees it as a whole,
    is responsible for the overall shape, takes a fraction of a second;
  * fine — the texture is cut into overlapping windows, each is sent to the
    network separately and is therefore looked at "up close". Small details
    appear, but the price is dozens of passes.

Seamlessness. The network knows nothing about the texture wrapping around and
honestly draws the edge of a scene at the frame border, so the tile stops
matching its neighbours. This is cured differently in the two modes:

  * in fast mode the picture is first wrapped around itself with a margin,
    the network runs on the enlarged frame, and from the result not the
    values but the slopes are taken — and those are assembled back into a map
    by periodic integration (Poisson via FFT). Periodicity is not fitted here
    but obtained by construction: a solution on a torus simply has no other
    option;
  * in fine mode all sampling is done modulo the image size, i.e. a window
    that reaches the right edge continues from the left one. A seam has
    nowhere to come from.

Texelith — глубина нейросетью (MiDaS, Depth Anything).

Зачем это здесь. Всё остальное в программе выводит высоту из яркости: где
темнее — там ниже. Работает, пока картинка честная, и врёт, как только
художник нарисовал светлое пятно грязи или тёмный камень рядом со светлым.
Яркость не знает, что кирпич выступает, а шов между кирпичами утоплен, —
она знает только, что шов темнее.

Сети предсказания глубины по одному снимку знают именно это: их учили на
парах «фотография — измеренная глубина», и они отвечают на вопрос «что тут
ближе к наблюдателю», а не «что тут светлее». Для текстуры это ровно та
подсказка, которой не хватает: сеть даёт крупную форму (кирпич выпуклый,
шов утоплен, доска выгнута), а мелкий рельеф по-прежнему берётся из
яркости. Поэтому результат кладётся во «внешнюю карту глубины» и там
смешивается с остальными слоями высоты обычными ползунками.

Что тут есть:

  * Depth Anything V2 Small — новее и заметно точнее на текстурах: границы
    кирпича и орнамента она держит, а не размазывает. Лицензия Apache-2.0;
  * MiDaS v2.1 Small — та самая классическая сеть Intel ISL, вдвое быстрее
    и вдвое меньше, но на текстурах видит в основном общий наклон стены.
    Лицензия MIT.

Обе — открытые, обе считаются через onnxruntime целиком на своей машине:
никуда ничего не отправляется, файл модели скачивается один раз и лежит
рядом с настройками программы. Считается по умолчанию на процессоре; если
в системе стоит сборка onnxruntime с поддержкой видеокарты (CUDA, DirectML,
ROCm), она подхватывается сама, а если не заводится — тихий откат на
процессор, чтобы кнопка работала всегда.

Два режима счёта:

  * быстрый — один проход по всей текстуре. Сеть видит её целиком, отвечает
    за общую форму, считает доли секунды;
  * подробный — текстура режется на перекрывающиеся окна, каждое уходит в
    сеть отдельно и потому рассматривается «вблизи». Мелкие детали
    появляются, но платить приходится десятками проходов.

Бесшовность. Сеть про заворот текстуры не знает и честно рисует по краям
кадра край сцены, из-за чего плитка перестаёт стыковаться. Лечится это
по-разному в двух режимах:

  * в быстром картинка сначала обёртывается сама вокруг себя с запасом,
    сеть считает по расширенному кадру, а из результата берутся не значения,
    а наклоны — и уже они собираются обратно в карту периодическим
    интегрированием (Пуассон через БПФ). Периодичность тут не подгоняется,
    а получается по построению: у решения на окружности другого варианта
    просто нет;
  * в подробном все выборки идут по модулю размера, то есть окно, дойдя до
    правого края, продолжается с левого. Шва там неоткуда взяться.
"""

import os
import threading

import numpy as np
from PIL import Image

BUILD = 25          # версия набора файлов, проверяется при запуске

# Нормировка входа: обе сети обучены на ImageNet и ждут именно её.
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

MODELS = {
    'da2s': {
        'title': 'Depth Anything V2 Small',
        'file': 'depth_anything_v2_vits.onnx',
        'url': ('https://github.com/fabio-sim/Depth-Anything-ONNX/releases/'
                'download/v2.0.0/depth_anything_v2_vits.onnx'),
        'size': 99373606,
        'license': 'Apache-2.0',
    },
    'midas': {
        'title': 'MiDaS v2.1 Small',
        'file': 'model-small.onnx',
        'url': ('https://github.com/isl-org/MiDaS/releases/'
                'download/v2_1/model-small.onnx'),
        'size': 66764249,
        'license': 'MIT',
    },
}
MODEL_IDS = ('da2s', 'midas')

_sessions = {}
_lock = threading.Lock()


# --------------------------------------------------------------- хозяйство
def model_dir():
    """Return the directory for downloaded models, creating it if needed.

    The directory is ``~/.texelith/models``, next to the settings rather than in
    the program folder: the program may be placed anywhere, including a place
    where the user has no write permission. Also used by esrgan.py.

    Returns:
        Absolute path (str) of an existing directory.
    """
    import i18n
    d = os.path.join(i18n.APP_DIR, 'models')
    os.makedirs(d, exist_ok=True)
    return d


def model_path(key):
    """Return the full path of the ONNX file for model ``key`` ('da2s' or 'midas')."""
    return os.path.join(model_dir(), MODELS[key]['file'])


def have_model(key):
    """Tell whether model ``key`` is downloaded completely.

    The file size is compared with the exact expected size in bytes
    (``MODELS[key]['size']``), so that a partially downloaded file does not pass
    itself off as a ready model.

    Returns:
        True if the file exists and has exactly the expected size, else False.
    """
    p = model_path(key)
    try:
        # размер сверяем, чтобы недокачанный файл не выдавал себя за готовый
        return os.path.getsize(p) == MODELS[key]['size']
    except OSError:
        return False


# Сборки onnxruntime. Все дают один и тот же модуль onnxruntime, поэтому
# стоять может только одна: перед установкой другой прежняя удаляется.
RUNTIMES = {
    'cpu': {'pip': 'onnxruntime', 'dist': 'onnxruntime'},
    'webgpu': {'pip': 'onnxruntime-webgpu', 'dist': 'onnxruntime-webgpu'},
    'cuda': {'pip': 'onnxruntime-gpu[cuda,cudnn]', 'dist': 'onnxruntime-gpu'},
    'directml': {'pip': 'onnxruntime-directml', 'dist': 'onnxruntime-directml'},
}
RUNTIME_IDS = ('cpu', 'webgpu', 'cuda', 'directml')


def have_runtime():
    """Return True if the ``onnxruntime`` module is installed.

    Looks the module up without importing it: importing loads the native
    libraries, and on Windows a loaded library cannot be replaced — the
    runtime could then not be switched to another build until a restart.
    """
    import importlib.util
    try:
        return importlib.util.find_spec('onnxruntime') is not None
    except Exception:
        return False


def runtime_kind():
    """Which onnxruntime build is installed: a key of RUNTIMES, or ''."""
    from importlib import metadata
    for k in RUNTIME_IDS:
        try:
            metadata.version(RUNTIMES[k]['dist'])
            return k
        except Exception:
            continue
    return ''


def runtime_version():
    """Return the installed onnxruntime version string, or '' if it is unavailable."""
    from importlib import metadata
    k = runtime_kind()
    try:
        return metadata.version(RUNTIMES[k]['dist']) if k else ''
    except Exception:
        return ''


def _pip_stream(args, line=None):
    """Run ``python -m pip <args>``; feed every output line to line(text).

    Returns (returncode, full output). No console window on Windows.
    """
    import subprocess
    import sys
    flags = 0x08000000 if sys.platform == 'win32' else 0
    try:
        p = subprocess.Popen([sys.executable, '-m', 'pip'] + args + ['--disable-pip-version-check'],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             errors='replace', creationflags=flags)
    except Exception as e:                                   # noqa: BLE001
        return 1, str(e)
    buf = []
    for ln in p.stdout:
        buf.append(ln)
        if line:
            line(ln.rstrip())
    p.wait()
    return p.returncode, ''.join(buf)


def check_runtime():
    """Import onnxruntime in a FRESH Python process and list its providers.

    A fresh process is the honest check right after an install: the running
    program may have stale import caches, and a separate process also shows
    whether the native libraries load at all. For the CUDA build the NVIDIA
    libraries installed by pip are preloaded first, as the program does.

    Returns (ok, providers list, error text).
    """
    import subprocess
    import sys
    code = ('import onnxruntime as o\n'
            'f = getattr(o, "preload_dlls", None)\n'
            'if f:\n'
            '    try: f()\n'
            '    except Exception: pass\n'
            'print("@@" + ",".join(o.get_available_providers()))\n')
    flags = 0x08000000 if sys.platform == 'win32' else 0
    try:
        r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                           timeout=300, creationflags=flags)
    except Exception as e:                                   # noqa: BLE001
        return False, [], str(e)
    for ln in (r.stdout or '').splitlines():
        if ln.startswith('@@'):
            return True, [p for p in ln[2:].split(',') if p], ''
    return False, [], ((r.stderr or '') + (r.stdout or ''))[-2000:]


def install_runtime(pkg='onnxruntime', module=None, line=None, kind=None):
    """Install a package with pip into the Python that runs the program.

    For onnxruntime the build is chosen by ``kind`` (a key of RUNTIMES; by
    default the one saved in the settings, else 'cpu'). Other onnxruntime
    builds are uninstalled first — they all provide the same module and must
    not be mixed. That is impossible while onnxruntime is already loaded in
    this process (Windows keeps its libraries locked); then it asks for a
    restart instead of breaking the installation.

    Three ways are tried in a row: a plain install, ``--user`` and
    ``--break-system-packages``; none needs administrator rights. The user
    site directory is appended to ``sys.path`` afterwards and import caches
    are invalidated. The result is checked in a fresh process (check_runtime).

    Args:
        pkg: pip package name ('onnxruntime' means "the runtime", see kind).
        module: import name to verify with, if it differs from ``pkg``.
        line: optional callable(text) receiving pip's output line by line.
        kind: runtime build for pkg == 'onnxruntime'.

    Returns:
        (ok, text): text is the last 2000 characters of pip's output, or the
        reason of failure.
    """
    import sys
    is_rt = pkg == 'onnxruntime'
    if is_rt:
        kind = kind or load_settings().get('runtime') or 'cpu'
        cur = runtime_kind()
        if cur and cur != kind:
            if 'onnxruntime' in sys.modules:
                save_settings(pending=kind)      # сменится при следующем запуске
                return False, 'RESTART'
            if line:
                line('pip uninstall ' + RUNTIMES[cur]['dist'])
            _pip_stream(['uninstall', '-y'] + [RUNTIMES[k]['dist'] for k in RUNTIME_IDS], line)
        target = RUNTIMES[kind]['pip']
    else:
        target = pkg
    out = ''
    for extra in ([], ['--user'], ['--break-system-packages']):
        if line:
            line('pip install %s %s' % (target, ' '.join(extra)))
        rc, out = _pip_stream(['install'] + extra + [target], line)
        if rc == 0:
            break
    if rc != 0 and 'from versions: none' in out:
        # pip не видит ни одной сборки: почти всегда это неподходящий Python
        # (32-битный или версия, под которую пакет не собран), а не сеть
        note = why_no_wheel()
        out += '\n' + note
        if line:
            line(note)
    try:
        import site
        import importlib
        d = site.getusersitepackages()
        if d and d not in sys.path:
            sys.path.append(d)
        importlib.invalidate_caches()
    except Exception:
        pass
    if is_rt:
        ok, prov, err = check_runtime()
        if ok:
            save_settings(runtime=kind)
            return True, 'providers: ' + ', '.join(prov)
        return False, (out + '\n' + err)[-2000:]
    try:
        __import__(module or pkg)
        return True, out[-2000:]
    except Exception:
        return False, out[-2000:]


def why_no_wheel():
    """Explain "from versions: none": which Python runs the program.

    The onnxruntime packages are built for 64-bit Python 3.11–3.14 on
    Windows; for any other Python pip sees no version at all.
    """
    import platform
    import sys
    import struct
    bits = struct.calcsize('P') * 8
    v = sys.version_info
    ok = bits == 64 and (3, 11) <= (v.major, v.minor) <= (3, 14)
    head = 'Python %d.%d.%d, %d-bit (%s).' % (v.major, v.minor, v.micro, bits, sys.executable)
    if ok:
        return head + ' This Python is supported; pip may be unable to reach PyPI ' \
                      '(network, proxy or pip settings).'
    return head + ' onnxruntime is built only for 64-bit Python 3.11-3.14: install one ' \
                  'from python.org and start the program with it. ' + platform.platform()


def settings_path():
    """``~/.texelith/runtime.json``: which onnxruntime build the person chose."""
    import i18n
    d = i18n.APP_DIR
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, 'runtime.json')


def load_settings():
    """Read runtime.json ({} if missing or broken)."""
    import json
    try:
        with open(settings_path(), encoding='utf-8') as f:
            return dict(json.load(f))
    except Exception:
        return {}


def save_settings(**kw):
    """Merge keys into runtime.json."""
    import json
    d = load_settings()
    d.update(kw)
    try:
        with open(settings_path(), 'w', encoding='utf-8') as f:
            json.dump(d, f)
    except Exception:
        pass


def download(key, progress=None, chunk=1 << 18):
    """Download model ``key`` into model_dir().

    The data is written to ``<file>.part`` first (by fetch(), which resumes
    after a stall) and renamed to the final name only after the size in bytes
    has been checked against the exact expected ``MODELS[key]['size']``. Why:
    an interrupted download would otherwise look like a working model
    (have_model() only compares sizes). On an error the ``.part`` file is kept
    so that the next attempt continues from it; a file of the wrong size is
    removed.

    Models are downloaded on demand rather than shipped with the program: they
    are large (about 99 MB and 67 MB) and each has its own licence (Depth
    Anything V2 Small — Apache-2.0, MiDaS — MIT).

    Args:
        key: model id from MODEL_IDS.
        progress: optional callback ``progress(done, total)`` in bytes, called as
            data arrives; ``total`` comes from Content-Length or, failing that,
            the expected size. If it returns False, the download is aborted and
            the file is deleted (the abort surfaces to the caller as
            KeyboardInterrupt).
        chunk: read block size in bytes (256 KiB by default).

    Returns:
        Path of the downloaded model file.

    Raises:
        IOError('bad size') if the downloaded size is not the expected one;
        network and file errors are re-raised after cleanup.
    """
    m = MODELS[key]
    dst = model_path(key)
    tmp = dst + '.part'
    fetch(m['url'], tmp, m['size'], progress, chunk=chunk)
    if os.path.getsize(tmp) != m['size']:
        os.remove(tmp)
        raise IOError('bad size')
    if os.path.exists(dst):
        os.remove(dst)
    os.replace(tmp, dst)
    return dst


def fetch(url, tmp, size=0, progress=None, line=None, tries=8, chunk=1 << 18):
    """Download ``url`` into the file ``tmp``, surviving a slow or broken link.

    A model is 20–100 MB, and on a slow or unstable connection (a VPN, a
    throttled CDN) a single pass often stalls: before, one stall of 60 s
    aborted the whole download and it started from zero. Now a stalled or
    dropped connection is reopened up to ``tries`` times and the download
    CONTINUES from the bytes already on disk (an HTTP Range request); servers
    that do not support Range send the file again from the start, and that is
    handled too. The partial file is kept between attempts and between runs,
    so pressing the button again continues as well.

    Args:
        url: address; redirects are followed.
        tmp: destination (the partial file); the caller renames it.
        size: expected size in bytes (0 = unknown), used as the total when
            the server sends none and to decide that the file is complete.
        progress: optional callable(done, total); returning False aborts
            (KeyboardInterrupt).
        line: optional callable(text) for messages about retries.
        tries: attempts in a row without any progress before giving up.

    Raises:
        The last network error when all attempts fail.
    """
    import time
    import urllib.request
    fails, last = 0, None
    while True:
        have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        if size and have >= size:
            return tmp
        hdr = {'User-Agent': 'Texelith'}
        if have:
            hdr['Range'] = 'bytes=%d-' % have
        before = have
        try:
            req = urllib.request.Request(url, headers=hdr)
            with urllib.request.urlopen(req, timeout=30) as r:
                if have and r.status != 206:
                    have = 0                 # докачка не поддержана — сначала
                length = int(r.headers.get('Content-Length') or 0)
                total = (have + length) if length else (size or 0)
                with open(tmp, 'ab' if have else 'wb') as f:
                    while True:
                        b = r.read(chunk)
                        if not b:
                            break
                        f.write(b)
                        have += len(b)
                        if progress is not None and progress(have, total or size) is False:
                            raise KeyboardInterrupt
            if not size or have >= size or (length and have >= total):
                return tmp
        except KeyboardInterrupt:
            raise
        except Exception as e:                       # noqa: BLE001 — повторим
            last = e
        fails = 0 if have > before else fails + 1
        if fails >= tries:
            raise last or IOError('download stalled')
        if line:
            line('%s — %s, %.1f MB' % ('retry', last.__class__.__name__ if last else 'short',
                                        have / 1048576.0))
        time.sleep(min(2 * fails, 10))


# Порядок, в котором пробуем считать. CUDA и DirectML появляются только
# если поставлен соответствующий пакет onnxruntime, поэтому список всегда
# фильтруется по тому, что среда реально предлагает.
PROVIDERS = ('CUDAExecutionProvider',        # NVIDIA, нужен CUDA + cuDNN
             'WebGpuExecutionProvider',      # любая видеокарта через Dawn (D3D12/Vulkan)
             'DmlExecutionProvider',         # Windows, любая видеокарта
             'ROCMExecutionProvider',        # AMD под Linux
             'CoreMLExecutionProvider',
             'CPUExecutionProvider')

_used = {}


def preload():
    """Load the NVIDIA libraries pip installed with the CUDA build (once).

    onnxruntime-gpu[cuda,cudnn] brings CUDA and cuDNN as pip packages; they
    are not on the system path, and onnxruntime finds them only after
    ``onnxruntime.preload_dlls()``. Harmless for the other builds."""
    global _preloaded
    if _preloaded:
        return
    _preloaded = True
    try:
        import onnxruntime as ort
        f = getattr(ort, 'preload_dlls', None)
        if f and runtime_kind() == 'cuda':
            f()
    except Exception:
        pass


_preloaded = False


def available_providers():
    """Return the entries of PROVIDERS offered by the installed onnxruntime.

    The list keeps the preference order of PROVIDERS; it is empty if
    onnxruntime cannot be imported.
    """
    try:
        import onnxruntime as ort
        have = set(ort.get_available_providers())
    except Exception:
        return []
    return [p for p in PROVIDERS if p in have]


SHORT = {'CUDAExecutionProvider': 'CUDA', 'DmlExecutionProvider': 'DirectML',
         'WebGpuExecutionProvider': 'WebGPU',
         'ROCMExecutionProvider': 'ROCm', 'CoreMLExecutionProvider': 'CoreML',
         'CPUExecutionProvider': ''}


def provider_short(p):
    """Return a short display name for provider ``p`` ('' for the CPU provider)."""
    return SHORT.get(p, p.replace('ExecutionProvider', ''))


def provider_name(key=None):
    """Return the execution provider that was actually used.

    The value is taken from the session itself (recorded in session()), not
    from the list of available providers: a provider may be listed in the build
    and still fail to start — no driver, no cuDNN, wrong CUDA version. In that
    case onnxruntime silently falls back to the CPU, and it is useful to know.

    Args:
        key: model id; if None, the provider of any session created so far.

    Returns:
        Provider name such as 'CPUExecutionProvider', or '' if no session has
        been created yet.
    """
    if key is not None:
        return _used.get(key, '')
    return next(iter(_used.values()), '')


def session(key):
    """Return the onnxruntime session for model ``key``, creating it once.

    The session is created once and cached in ``_sessions`` (under a lock, so
    worker threads may call this concurrently): parsing the model takes
    noticeably longer than a pass itself. Graph optimisation is set to
    ORT_ENABLE_ALL.

    Provider choice: all available providers in PROVIDERS order (GPU first). If
    creating the session with them raises — the GPU is advertised but does not
    start — it is silently retried on the CPU only: slower, but it always works.
    onnxruntime may also fall back to the CPU on its own without raising; the
    provider really used is stored for provider_name(). The GPU path has not
    been tested (there is no GPU in the development environment).

    If even the CPU-only attempt fails, the model file is deleted before the
    exception is re-raised: a broken file is better removed at once, otherwise
    it keeps pretending to be a ready model (the size check in have_model()
    passes) and every next press hits the same error. Caveat: any failure at
    that point is treated this way, not only a corrupt file.

    Args:
        key: model id from MODEL_IDS; the file must already be downloaded.

    Returns:
        onnxruntime.InferenceSession.
    """
    with _lock:
        s = _sessions.get(key)
        if s is not None:
            return s
        import onnxruntime as ort
        preload()
        opt = ort.SessionOptions()
        opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        prov = available_providers() or ['CPUExecutionProvider']
        try:
            try:
                s = ort.InferenceSession(model_path(key), opt, providers=prov)
            except Exception:
                # видеокарта заявлена, но не завелась — считаем на процессоре,
                # это медленнее, но работает всегда
                s = ort.InferenceSession(model_path(key), opt,
                                         providers=['CPUExecutionProvider'])
            _used[key] = (s.get_providers() or ['?'])[0]
        except Exception:
            # битый файл лучше убрать сразу: иначе он так и будет притворяться
            # готовой моделью, и следующее нажатие упрётся в то же самое
            try:
                os.remove(model_path(key))
            except OSError:
                pass
            raise
        _sessions[key] = s
        return s


def unload():
    """Drop all cached sessions so their memory can be freed."""
    with _lock:
        _sessions.clear()


# ----------------------------------------------------------------- счётное
def _side(s):
    """Return the model's input side in pixels (both models have it fixed).

    Read from the input shape of session ``s``; if that dimension is symbolic,
    518 (the Depth Anything V2 input size) is assumed.
    """
    n = s.get_inputs()[0].shape[2]
    return int(n) if isinstance(n, int) else 518


def _run(s, rgb):
    """Run one network pass.

    The input piece is converted to 8 bits, resized with Lanczos to the model's
    square input (see _side; the aspect ratio is not preserved) and normalised
    with the ImageNet MEAN/STD both networks were trained with.

    Args:
        s: onnxruntime session.
        rgb: H×W×3 float image of any size, sRGB 0…1 (clipped).

    Returns:
        n×n float32 array (n = model input side): relative inverse depth in
        arbitrary units — a larger value means closer to the viewer.
    """
    im = Image.fromarray((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8))
    n = _side(s)
    im = im.resize((n, n), Image.LANCZOS)
    x = (np.asarray(im, np.float32) / 255.0 - MEAN) / STD
    x = np.ascontiguousarray(np.transpose(x, (2, 0, 1))[None])
    y = s.run(None, {s.get_inputs()[0].name: x})[0]
    return np.array(y[0], np.float32)


def _resize(d, h, w):
    """Resize a 2-D float map to h×w with bicubic interpolation; float32 out."""
    return np.array(Image.fromarray(d).resize((w, h), Image.BICUBIC), np.float32)


def _poisson(gy, gx):
    """Assemble a map back from its slopes so that it is periodic.

    An ordinary Poisson equation is solved, only on a torus: the picture is
    treated as closed both horizontally and vertically. With the FFT this is
    one line — the spectrum of the divergence divided by the spectrum of the
    discrete Laplacian. The divergence uses backward differences, matching the
    forward-difference slopes produced by depth_fast. The constant component
    (zero frequency) is undefined and is simply set to zero: the map is
    normalised later anyway.

    Args:
        gy, gx: H×W forward-difference slopes along rows and columns.

    Returns:
        H×W float32 map with zero mean, periodic by construction.
    """
    H, W = gy.shape
    div = (gx - np.roll(gx, 1, 1)) + (gy - np.roll(gy, 1, 0))
    u = np.fft.fftfreq(W)[None, :]
    v = np.fft.fftfreq(H)[:, None]
    den = (2 * np.cos(2 * np.pi * u) - 2) + (2 * np.cos(2 * np.pi * v) - 2)
    den[0, 0] = 1.0
    F = np.fft.fft2(div) / den
    F[0, 0] = 0.0
    return np.real(np.fft.ifft2(F)).astype(np.float32)


def depth_fast(s, rgb, pad=0.25):
    """Whole-texture passes on the torus, joined through their slopes.

    A depth network treats any picture as a scene: towards the frame border
    it invents a "far away" or "close up" that has nothing to do with the
    texture. One pass therefore leaves a wrong slope along the edges, and
    after making it periodic that shows as a ridge (or a valley) along the
    tile seams with a peak at the corners.

    So the network runs four times, on the texture rolled by 0 or half its
    size along each axis, each time on a wrap-padded frame (``pad`` of each
    side on every edge). From every pass only the slopes are taken (after
    bringing the pass to unit spread), and they are blended with the weight
    sin²(π·y/H)·sin²(π·x/W) in that pass's own coordinates: full weight at
    the middle of its frame, zero at the frame edges. The four weights add up
    to exactly 1 everywhere, so the seams of one pass are covered by the
    middles of the others. The blended slopes are assembled into a map with
    _poisson, periodic by construction.

    Args:
        s: onnxruntime session.
        rgb: H×W×3 float32, sRGB 0…1.
        pad: wrap margin as a fraction of each side (0.25 = a quarter).

    Returns:
        H×W float32 depth, zero mean, arbitrary scale (larger = closer).
    """
    H, W = rgb.shape[:2]
    py, px = max(1, int(H * pad)), max(1, int(W * pad))
    wy = np.sin(np.pi * (np.arange(H) + 0.5) / H)[:, None] ** 2
    wx = np.sin(np.pi * (np.arange(W) + 0.5) / W)[None, :] ** 2
    gy_acc = np.zeros((H, W), np.float64)
    gx_acc = np.zeros((H, W), np.float64)
    for sy in (0, H // 2):
        for sx in (0, W // 2):
            img = np.roll(rgb, (-sy, -sx), (0, 1))
            big = np.pad(img, ((py, py), (px, px), (0, 0)), mode='wrap')
            d = _resize(_run(s, big), H + 2 * py, W + 2 * px)
            d = (d - d.mean()) / max(float(d.std()), 1e-6)
            gy = (np.roll(d, -1, 0) - d)[py:py + H, px:px + W]
            gx = (np.roll(d, -1, 1) - d)[py:py + H, px:px + W]
            w = wy * wx
            # обратно в координаты текстуры: вес и наклоны едут вместе
            gy_acc += np.roll(gy * w, (sy, sx), (0, 1))
            gx_acc += np.roll(gx * w, (sy, sx), (0, 1))
    # сумма четырёх весов sin²·sin² со сдвигами на полпериода равна 1
    return _poisson(np.ascontiguousarray(gy_acc, np.float32),
                    np.ascontiguousarray(gx_acc, np.float32))


def tile_fit(d, g, tile, sigma):
    """Gain and level that put a window onto the scale of the common pass.

    The network measures depth relatively: in one window "closer" is 3, in the
    neighbouring one 12, and they cannot be added as they are. The common
    (whole-texture) pass works as a ruler here — there is one for the whole
    texture, so all windows brought to it become comparable with each other.

    The fit is done on the LARGE scale, on blurred copies of both. If detail
    were compared with the common pass, the pass knows nothing about it, the
    detail would count as noise, the gain would come out too low, and exactly
    the details would sink — while they are the whole point of the fine mode.
    The gain is the least-squares slope of the blurred base on the blurred
    window, ``cov(blur d, blur g) / var(blur d)``. The blur here uses
    ``mode='reflect'``; it only feeds these two statistics, never the map, so
    it does not affect seamlessness.

    The gain is clamped so that the window's contribution is between a quarter
    and three spreads (standard deviations) of the common pass: otherwise a flat
    window, where the network returned almost nothing, would stretch its own
    noise over the whole range. ``g`` is expected to be normalised to unit
    spread (depth_fine does that), so the clamp is ``[0.25/sd, 3/sd]`` with
    ``sd`` the spread of the unblurred window.

    Args:
        d: tile×tile raw network answer for the window.
        g: tile×tile crop of the normalised common pass at the same pixels.
        tile: window side in pixels (not used by the computation).
        sigma: blur radius in pixels for the large-scale fit (depth_fine uses
            max(2, tile/6)).

    Returns:
        (a, dm, gm) as floats: gain, mean of ``d``, mean of ``g``; the window is
        mapped as ``a * (d - dm) + gm``.
    """
    from matgen import fast_blur
    dm, gm = float(d.mean()), float(g.mean())
    dl = fast_blur(d - dm, sigma, mode='reflect')
    gl = fast_blur(g - gm, sigma, mode='reflect')
    v = float((dl * dl).mean())
    sd = float(np.sqrt(max(float(((d - dm) ** 2).mean()), 1e-12)))
    a = float((dl * gl).mean()) / max(v, 1e-12)
    return float(np.clip(a, 0.25 / sd, 3.0 / sd)), dm, gm


def depth_fine(s, rgb, tile=0, overlap=0.5, progress=None, base=None):
    """Pass over the texture with overlapping windows.

    ``base`` is the map of the common pass. If it is given, each window is
    brought to its scale (see tile_fit) and the two approaches add up: large
    shape from the common pass, details from the windows. If it is absent, each
    window just has its best-fit plane (tilt) removed and its spread equalised
    to 1 — then there is no large-scale shape at all.

    Windows are taken modulo the image size, so the rightmost window continues
    with the left edge of the texture and no seam appears. Windows are blended
    with a Hann weight (plus 1e-3) that goes to zero towards the window edges,
    so the joins between windows are not visible.

    Args:
        s: onnxruntime session.
        rgb: H×W×3 float32, sRGB 0…1.
        tile: window side in pixels; 0 means max(128, min(H, W)//2). Clamped to
            min(H, W).
        overlap: overlap fraction of neighbouring windows, 0…<1 (0.5 = half);
            the step is max(8, tile·(1 − overlap)) pixels.
        progress: optional ``progress(done, total)`` in windows; returning False
            stops early and returns the partial blend.
        base: optional H×W common-pass map (any scale; normalised inside).

    Returns:
        H×W float32 depth: on the base's normalised scale if ``base`` is
        given, otherwise roughly zero-mean and unit-spread per window.
    """
    H, W = rgb.shape[:2]
    if tile <= 0:
        tile = max(128, min(H, W) // 2)
    tile = int(min(tile, min(H, W)))
    step = max(8, int(round(tile * (1.0 - overlap))))
    ys = list(range(0, H, step))
    xs = list(range(0, W, step))
    acc = np.zeros((H, W), np.float32)
    wsum = np.zeros((H, W), np.float32)
    # окно веса: к краям сходит в ноль, поэтому стык окон не виден
    win = (np.hanning(tile)[:, None] * np.hanning(tile)[None, :]).astype(np.float32) + 1e-3
    yy, xx = np.mgrid[0:tile, 0:tile].astype(np.float32)
    A = np.stack([yy.ravel(), xx.ravel(), np.ones(tile * tile, np.float32)], 1)
    Ainv = np.linalg.pinv(A)
    g = None
    if base is not None:
        g = (base - base.mean()) / max(float(base.std()), 1e-6)
    sigma = max(2.0, tile / 6.0)
    total = len(ys) * len(xs)
    k = 0
    for y0 in ys:
        for x0 in xs:
            iy = np.mod(np.arange(y0, y0 + tile), H)
            ix = np.mod(np.arange(x0, x0 + tile), W)
            d = _resize(_run(s, rgb[iy][:, ix]), tile, tile)
            if g is None:
                d = d - (A @ (Ainv @ d.ravel())).reshape(tile, tile)
                sd = float(d.std())
                if sd > 1e-6:
                    d /= sd
            else:
                a, dm, gm = tile_fit(d, g[np.ix_(iy, ix)], tile, sigma)
                d = a * (d - dm) + gm
            acc[np.ix_(iy, ix)] += d * win
            wsum[np.ix_(iy, ix)] += win
            k += 1
            if progress is not None and progress(k, total) is False:
                return acc / np.maximum(wsum, 1e-6)
    return acc / np.maximum(wsum, 1e-6)


def macro_sigma(side, amount):
    """Convert the "scale boundary" slider into a blur radius.

    The radius falls from a quarter of the texture to one sixty-fourth
    (matgen.band_radius: ``side / (4 + 60·amount)``): on the left almost
    everything counts as large, on the right only things noticeably larger than
    a brick. The useful middle is about the side divided by sixteen: there the
    large band holds the level of each brick, and the fine band its own shape
    and texture.

    Args:
        side: texture side in pixels (usually the smaller one).
        amount: slider value 0…1.

    Returns:
        Radius in pixels (float).
    """
    from matgen import band_radius
    return band_radius(side, amount)


def bands(d, radius):
    """Split a map into a large-scale and a fine band.

    Large is a blurred copy, fine is what remains of the map after removing it.
    The blur wraps around, so each band on its own stays seamless, and so does
    any mix of them.

    The boundary between them is set by the radius. On a brick wall the large
    band holds "this brick is higher than its neighbour", the fine band "there
    is a chip inside the brick". These are different statements about the
    surface and they work differently in the engine: large drives the parallax
    and reads from three steps away, fine lives in the normal map and is seen up
    close. That is why it is useful to tune them separately.

    Args:
        d: H×W float map.
        radius: blur radius in pixels (at least 1), see macro_sigma.

    Returns:
        (base, detail): two H×W float32 arrays with base + detail == d.
    """
    from matgen import fast_blur
    base = fast_blur(d, max(1.0, float(radius)), mode='wrap')
    return base.astype(np.float32), (d - base).astype(np.float32)


def shape(d, side, scale=0.3, low=0.4, high=1.0, curve=1.0):
    """Rebuild the map from its large and fine bands with given weights.

    ``curve`` is contrast around the middle applied on top of the finished map.
    At large values the brick faces reach a flat plateau while dark furrows
    remain at the joints: the result is no longer so much depth as a crevice
    map, and it works well for AO.

    Band splitting and tone are the only things happening here, so the network
    does not need to be re-run while the sliders are being moved.

    Args:
        d: H×W raw depth as returned by compute().
        side: texture side in pixels, for macro_sigma.
        scale: "scale boundary" slider 0…1 (see macro_sigma).
        low: weight of the large band (typical 0…1.5).
        high: weight of the fine band (typical 0…2).
        curve: contrast around 0.5 (matgen.tone); 1 means unchanged.

    Returns:
        H×W float32 in 0…1.
    """
    from matgen import tone
    base, det = bands(d, macro_sigma(side, scale))
    out = normalize(float(low) * base + float(high) * det)
    return out if abs(curve - 1.0) < 1e-3 else tone(out, 1.0, float(curve))


def normalize(d, clip=1.0):
    """Map to 0…1, outliers bent in softly.

    Without that a single stray spot eats the whole range and the rest of
    the map turns grey; a hard clip instead would flatten the tops and
    bottoms, so the tails go through matgen.soft01.

    Args:
        d: float map of any shape and scale.
        clip: percent cut at each end (1.0 maps the 1st…99th percentile to 0…1).

    Returns:
        float32 array of the same shape, in 0…1.
    """
    import matgen as MG
    lo = np.percentile(d, clip)
    hi = np.percentile(d, 100.0 - clip)
    # хвосты не срезаем, а мягко подгибаем (matgen.soft01): иначе вершины и
    # дно рельефа становились плато, а нормаль из них — плоскими пятнами
    t = (np.asarray(d, np.float32) - lo) / max(float(hi - lo), 1e-6)
    return np.asarray(MG.soft01(t), np.float32)


def compute(rgb, model='da2s', fine=False, tile=0, overlap=0.5, progress=None):
    """Compute a depth map for the picture ``rgb`` (H×W×3, 0…1).

    The raw network answer is returned, brought to zero mean and unit spread
    (standard deviation): splitting it into bands and normalising to 0…1 is
    done by shape(), and that can be repeated any number of times without a new
    pass.

    In fine mode a common pass is still done first: it costs one frame out of
    dozens but provides the ruler by which the windows are put on one scale.
    The large shape then comes from it, the details from the windows.

    Args:
        rgb: H×W×3 (or ×4; alpha ignored) float image, sRGB 0…1.
        model: 'da2s' (Depth Anything V2 Small) or 'midas' (MiDaS v2.1 Small);
            it must be downloaded (see download()).
        fine: False — fast single pass (depth_fast); True — windows
            (depth_fine) on top of a common pass.
        tile: window side in pixels for fine mode (0 = automatic).
        overlap: window overlap fraction for fine mode.
        progress: optional ``progress(done, total)`` callback; in fine mode
            returning False stops the window loop early.

    Returns:
        H×W float32, zero mean, unit standard deviation; larger = closer.
        Seamless (periodic) in both modes.
    """
    rgb = np.ascontiguousarray(np.clip(np.asarray(rgb, np.float32)[..., :3], 0, 1))
    s = session(model)
    if fine:
        if progress is not None:
            progress(0, 2)
        base = depth_fast(s, rgb)

        def prog(i, total):
            """Report window progress offset by one step for the common pass."""
            return None if progress is None else progress(i + 1, total + 1)
        d = depth_fine(s, rgb, tile, overlap, prog, base=base)
    else:
        if progress is not None:
            progress(0, 1)
        d = depth_fast(s, rgb)
        if progress is not None:
            progress(1, 1)
    return ((d - d.mean()) / max(float(d.std()), 1e-6)).astype(np.float32)
