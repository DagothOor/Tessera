# -*- coding: utf-8 -*-
"""
Texelith — bridge to SuperMat running in its own environment.

SuperMat (https://github.com/hyj542682306/SuperMat, weights
https://huggingface.co/oyiya/SuperMat, MIT) is not part of Texelith: it needs
PyTorch, diffusers and a 5 GB Stable Diffusion 2.1 base model (see
supermat_worker.py). install_supermat_portable.bat sets up a portable Python
with all of it next to Texelith. Texelith needs paths kept in
``~/.texelith/supermat.json``:

  repo    — the folder of the SuperMat code (the unpacked repository);
  python  — the python executable of that environment;
  weights — optional path of supermat.pth; empty means
            <repo>/checkpoints/supermat.pth or a download from Hugging Face.

One worker process is started on first use and kept for all following textures,
because loading the model takes far longer than one pass. Calls are serialised
with a lock.

Texelith — мост к SuperMat, установленному отдельно. Нужны пути: папка кода
SuperMat и python его окружения; хранятся в ~/.texelith/supermat.json.
Процесс запускается один раз и обслуживает все текстуры подряд.
"""

import json
import os
import subprocess
import sys
import tempfile
import threading

import numpy as np

from texio import resize_wrap

BUILD = 25          # версия набора файлов, проверяется при запуске

LICENSE = 'MIT'

_proc = None
_proc_cfg = None
_device = ''
_lock = threading.Lock()


def config_path():
    """``~/.texelith/supermat.json``."""
    import i18n
    d = i18n.APP_DIR
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, 'supermat.json')


def load_config():
    """Return {'repo', 'python', 'weights'} (missing keys as '')."""
    try:
        with open(config_path(), encoding='utf-8') as f:
            d = json.load(f)
    except Exception:
        d = {}
    return {k: str(d.get(k, '') or '') for k in ('repo', 'python', 'weights')}


def save_config(cfg):
    """Store the three paths; stops a running worker if they changed."""
    old = load_config()
    with open(config_path(), 'w', encoding='utf-8') as f:
        json.dump({k: cfg.get(k, '') for k in ('repo', 'python', 'weights')}, f,
                  ensure_ascii=False, indent=1)
    if old != load_config():
        stop()


def problem(cfg=None):
    """What is missing in the setup, as an i18n key, or '' if it looks complete.

    Only checks the files that must exist; whether PyTorch and the weights
    work is found out by starting the worker (start()).
    """
    c = cfg or load_config()
    if not c['repo'] or not os.path.isdir(c['repo']):
        return 'supermat.norepo'
    if not c['python'] or not os.path.isfile(c['python']):
        return 'supermat.nopython'
    if c['weights'] and not os.path.isfile(c['weights']):
        return 'supermat.noweights'
    if not os.path.isfile(os.path.join(c['repo'], 'src', 'adapters.py')):
        return 'supermat.notrepo'
    return ''


def worker_script():
    """Path of supermat_worker.py, which lies next to this module."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'supermat_worker.py')


_tail = []          # последние строки вывода worker'а — показать, если он упал


def _read(proc):
    """Next protocol line ('@@' + JSON) from the worker, skipping the chatter.

    The chatter (library output, tracebacks) is kept in _tail, the last 40
    lines, so a failure can be shown with its cause. Returns the decoded
    dict, or None when the process has ended."""
    while True:
        line = proc.stdout.readline()
        if not line:
            return None
        line = line.rstrip()
        if line.startswith('@@'):
            msg = json.loads(line[2:])
            if 'note' in msg and len(msg) == 1:
                _tail.append(msg['note'])       # сообщение по ходу, не ответ
                continue
            return msg
        _tail.append(line)
        del _tail[:-40]


def last_output():
    """The last lines the worker printed besides the protocol (for errors)."""
    return '\n'.join(_tail[-15:])


def start():
    """Start the worker if it is not running; return the device name.

    Raises:
        RuntimeError: setup incomplete (message is the i18n key from
            problem()) or the worker failed to load the model (its message).
    """
    global _proc, _proc_cfg, _device
    cfg = load_config()
    if _proc is not None and _proc.poll() is None and _proc_cfg == cfg:
        return _device
    stop()
    bad = problem(cfg)
    if bad:
        raise RuntimeError(bad)
    flags = 0x08000000 if sys.platform == 'win32' else 0      # без окна консоли
    env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
    p = subprocess.Popen([cfg['python'], worker_script(), cfg['repo'], cfg['weights']],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                         errors='replace', bufsize=1, env=env, creationflags=flags)
    msg = _read(p)
    if not msg or not msg.get('ready'):
        try:
            p.kill()
        except Exception:
            pass
        raise RuntimeError((msg or {}).get('error') or
                           'SuperMat worker exited:\n' + last_output())
    _proc, _proc_cfg, _device = p, cfg, msg.get('device', '')
    return _device


def stop():
    """Stop the worker (closing its stdin ends its loop)."""
    global _proc
    p, _proc = _proc, None
    if p is not None:
        try:
            p.stdin.close()
            p.wait(timeout=5)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass


def device():
    """Device the running worker uses ('cuda', 'cpu'), '' if not running."""
    return _device if _proc is not None and _proc.poll() is None else ''


def run(rgb, res=512, back=True):
    """Run SuperMat on one texture.

    Args:
        rgb: H×W×3 sRGB 0…1.
        res: the square size the model works at (it was trained at 512).

    Returns:
        dict: 'basecolor' H×W×3 sRGB 0…1, 'roughness' and 'metalness' H×W
        0…1 — resized back to H×W on the torus (back=False: left at
        res × res). Seamless: the worker blends four passes rolled by half
        a period (see supermat_worker.py).
    """
    with _lock:
        start()
        H, W = rgb.shape[:2]
        tmp = tempfile.mkdtemp(prefix='texelith_supermat_')
        fin, fout = os.path.join(tmp, 'in.npy'), os.path.join(tmp, 'out.npz')
        try:
            # к размеру модели приводим здесь, на торе: у интерполяции в
            # PyTorch край берётся «прижатым», и шов появился бы уже на входе
            x = resize_wrap(np.clip(rgb[..., :3], 0, 1), int(res), int(res))
            np.save(fin, np.ascontiguousarray(x, np.float32))
            _proc.stdin.write(json.dumps({'in': fin, 'out': fout}) + '\n')
            _proc.stdin.flush()
            msg = _read(_proc)
            if msg is None:
                stop()
                raise RuntimeError('SuperMat worker exited:\n' + last_output())
            if not msg.get('ok'):
                raise RuntimeError(msg.get('error', 'SuperMat error'))
            with np.load(fout) as z:
                maps = {k: z[k] for k in z.files}
        finally:
            for f in (fin, fout):
                try:
                    os.remove(f)
                except OSError:
                    pass
            try:
                os.rmdir(tmp)
            except OSError:
                pass
    out = {}
    for k, v in maps.items():
        if back:
            v = resize_wrap(v, H, W)
        if v.ndim == 3 and v.shape[2] == 1:
            v = v[..., 0]
        out[k] = np.clip(v, 0, 1).astype(np.float32)
    return out
