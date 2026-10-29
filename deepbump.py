# -*- coding: utf-8 -*-
"""
Texelith — DeepBump: a normal map from a colour texture by a small network.

DeepBump (Hugo Tini, https://github.com/HugoTini/DeepBump, GPL-3.0) is a U-Net
with a MobileNetV2 encoder trained on photogrammetry and procedural materials,
i.e. on flat tiling textures — the same kind of picture Texelith works with.
It takes a grey 256×256 window and returns a normal map for it.

What this module does differently from the original add-on: the windows are
laid out on the torus. The texture is treated as a tile, window positions are
spaced evenly over the whole period, and overlapping results are blended with
a smooth weight and summed back modulo the size. Every pixel is thus covered
by the same number of windows wherever it is, including the texture edge, and
the result tiles. The original pads the picture by wrapping too, but blends
over a padded rectangle, so the edge is handled differently from the middle.

The model is not shipped with Texelith. It is downloaded on request from the
DeepBump repository into ``~/.texelith/models`` (27 MB) under its own licence.

Texelith — DeepBump: карта нормалей из цветной текстуры маленькой сетью.

Сеть обучена на плоских тайлящихся материалах (фотограмметрия и
процедурные), вход — серое окно 256×256, выход — нормаль для него. Окна
раскладываются по тору: позиции равномерно на весь период, перекрытия
смешиваются гладким весом и складываются обратно по модулю размера, поэтому
край текстуры обрабатывается так же, как середина, и результат тайлится.
"""

import os
import threading

import numpy as np

import depthnet as DN

BUILD = 25          # версия набора файлов, проверяется при запуске

MODEL = {
    'title': 'DeepBump 256 (Hugo Tini)',
    'file': 'deepbump256.onnx',
    'url': 'https://raw.githubusercontent.com/HugoTini/DeepBump/master/deepbump256.onnx',
    'size': 26706979,
    'license': 'GPL-3.0',
}
TILE = 256

# Зелёный канал. Сеть выдаёт нормаль в раскладке OpenGL (зелёный вверх):
# так её понимает Blender, для которого она сделана. Проверено в _check.py
# на кладке: высота, проинтегрированная из нормали в раскладке OpenGL,
# совпадает с яркостной высотой (камень светлее шва), в раскладке DirectX —
# выходит вывернутой.
NATIVE_DIRECTX = False

_sess = None
_lock = threading.Lock()


def model_path():
    """Full path of the ONNX file in the shared model folder (depthnet.model_dir)."""
    return os.path.join(DN.model_dir(), MODEL['file'])


def have_model():
    """True if the model file exists and has exactly the expected size."""
    try:
        return os.path.getsize(model_path()) == MODEL['size']
    except OSError:
        return False


def download(progress=None, chunk=1 << 18):
    """Download the model into the model folder.

    Same scheme as depthnet.download: write to ``.part``, check the exact size,
    then rename; on any error or cancel the partial file is removed.

    Args:
        progress: optional callable(done_bytes, total_bytes); returning False
            aborts (surfaces as KeyboardInterrupt).
        chunk: read block size in bytes.

    Returns:
        Path of the model file.

    Raises:
        IOError('bad size') when the downloaded size is wrong; network errors.
    """
    dst = model_path()
    tmp = dst + '.part'
    DN.fetch(MODEL['url'], tmp, MODEL['size'], progress, chunk=chunk)
    if os.path.getsize(tmp) != MODEL['size']:
        os.remove(tmp)
        raise IOError('bad size')
    if os.path.exists(dst):
        os.remove(dst)
    os.replace(tmp, dst)
    return dst


def session():
    """Open (once) the onnxruntime session; GPU providers first, CPU fallback."""
    global _sess
    with _lock:
        if _sess is not None:
            return _sess
        import onnxruntime as ort
        DN.preload()
        prov = DN.available_providers() or ['CPUExecutionProvider']
        try:
            s = ort.InferenceSession(model_path(), providers=prov)
        except Exception:
            s = ort.InferenceSession(model_path(), providers=['CPUExecutionProvider'])
        DN._used['deepbump'] = (s.get_providers() or ['?'])[0]
        _sess = s
        return s


def unload():
    """Drop the cached session."""
    global _sess
    with _lock:
        _sess = None


def _weight(n):
    """n×n blending weight: a raised cosine that is 0 at the window edge.

    Squared sine in each direction; strictly positive inside, so a pixel
    covered by at least one window interior always has a non-zero total
    weight."""
    t = (np.arange(n, dtype=np.float32) + 0.5) / n
    w = np.sin(np.pi * t) ** 2
    return np.outer(w, w).astype(np.float32)


def _positions(size, stride):
    """Evenly spaced window origins covering a period of ``size`` pixels.

    The number of windows is chosen so that the actual step does not exceed
    ``stride``; origins are rounded to whole pixels."""
    n = max(1, int(np.ceil(size / float(stride))))
    return [int(round(i * size / n)) for i in range(n)]


def run_gray(gray, overlap=0.5, progress=None, sess=None):
    """Run the network over a grey picture on the torus.

    Args:
        gray: H×W float, 0…1 (DeepBump's input is the mean of the sRGB
            channels, as in the original add-on).
        overlap: fraction of the window shared by neighbours, 0.25…0.75.
            More overlap — smoother blending and more passes.
        progress: optional callable(done, total).
        sess: an open session or anything with ``run`` (for tests).

    Returns:
        H×W×3 float32 unit vectors in the network's own layout (OpenGL green,
        see NATIVE_DIRECTX).
    """
    s = sess if sess is not None else session()
    g = np.ascontiguousarray(np.clip(gray, 0, 1), np.float32)
    H, W = g.shape
    # маленькую текстуру сначала размножаем: окно 256 должно целиком
    # заполняться картинкой, а тайл — это она же, повторённая
    rep_y, rep_x = int(np.ceil(TILE / H)), int(np.ceil(TILE / W))
    big = np.tile(g, (rep_y + 1, rep_x + 1))
    stride = max(8, int(TILE * (1.0 - float(np.clip(overlap, 0.0, 0.9)))))
    ys, xs = _positions(H, stride), _positions(W, stride)
    wgt = _weight(TILE)
    acc = np.zeros((H, W, 3), np.float64)
    wsum = np.zeros((H, W), np.float64)
    total, k = len(ys) * len(xs), 0
    for y0 in ys:
        for x0 in xs:
            win = big[y0:y0 + TILE, x0:x0 + TILE]
            res = s.run(None, {'input': win[None, None]})[0][0]      # 3×T×T, 0…1
            res = np.transpose(res, (1, 2, 0)) * 2.0 - 1.0
            iy = (np.arange(TILE) + y0) % H
            ix = (np.arange(TILE) + x0) % W
            # окно может заходить на ту же клетку несколько раз (текстура
            # меньше окна) — np.add.at складывает все попадания
            np.add.at(acc, (iy[:, None], ix[None, :]), res * wgt[..., None])
            np.add.at(wsum, (iy[:, None], ix[None, :]), wgt)
            k += 1
            if progress:
                progress(k, total)
    n = (acc / np.maximum(wsum, 1e-9)[..., None]).astype(np.float32)
    n[..., 2] = np.maximum(n[..., 2], 1e-3)
    return n / np.sqrt((n * n).sum(-1, keepdims=True))


def normal(rgb, overlap=0.5, directx=True, progress=None, sess=None):
    """Normal map for an sRGB texture.

    Args:
        rgb: H×W×3 sRGB 0…1 (alpha ignored).
        overlap: see run_gray.
        directx: layout of the result (True: DirectX, green down).
        progress, sess: see run_gray.

    Returns:
        H×W×3 float32 unit vectors in the requested layout.
    """
    gray = np.asarray(rgb, np.float32)[..., :3].mean(-1)
    n = run_gray(gray, overlap, progress, sess)
    if bool(directx) != NATIVE_DIRECTX:
        n = n.copy()
        n[..., 1] = -n[..., 1]
    return n
