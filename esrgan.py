# -*- coding: utf-8 -*-
"""
Texelith — ESRGAN material-map models (normal map and "franken map").

Two models by Joey (JoeyBallentine), trained on CC0 textures:

  1x NormalMapGenerator-CX-Lite  — a normal map straight from the texture;
  1x FrankenMapGenerator-CX-Lite — a "franken map": R holds the texture in
                                   greyscale, G roughness, B height.

The weights are licensed CC-BY-NC-SA-4.0: attribution is required and they may
be used for non-commercial purposes only. That is why they are not shipped
with the program and are downloaded with a button from their original
location (the user sees the licence and the author before downloading); the
code of this module is under the same licence as the rest of the program.

The models are stored in PyTorch format (.pth). Pulling in PyTorch means
gigabytes, so here the weights are read without it (a .pth file is a pickle
with references to raw arrays, which can be parsed by hand), converted to ONNX
once and from then on computed by the same onnxruntime as depth
(depthnet.py) — on the GPU if there is one. Both .pth layouts are understood:
the current zip archive (data.pkl plus one file per storage) and the legacy
single-stream format; tensors are resolved through the pickle's persistent ids
and torch._utils._rebuild_tensor_v2 is replaced by a stand-in, so no torch code
is ever imported (see load_pth).

Architecture — the old ESRGAN (RRDB_Net from BasicSR/traiNNer):
  first convolution → 12 RRDB blocks (three dense blocks of five convolutions
  each; the output of each dense block ×0.2 plus its input, the RRDB output
  ×0.2 plus its input) → convolution → added to the trunk input → two output
  convolutions.
Input and output are RGB in 0…1 (that is how the original generate.py
prepares them).

Inside a dense block every convolution sees the concatenation of the block
input and all previous outputs (dense connections), with LeakyReLU of slope 0.2
after the first four; the residual scaling by 0.2 is what keeps such a deep
stack stable. Both models here are 1x, i.e. without upsampling layers: the map
has the size of the texture. The converter also accepts the upsampling layers
of other ESRGAN models (nearest ×2 + convolution each), but the program does
not use them.

Seamlessness: run() wraps the texture around itself before cutting it into
windows, so the texture border is treated like any other place inside the
tile and the map tiles (see run for the exact claim).

Reference: X. Wang et al., "ESRGAN: Enhanced Super-Resolution Generative
Adversarial Networks", ECCV Workshops, 2018.

Texelith — ESRGAN material-map models (normal map and "franken map").
Карты из нейросетей ESRGAN: нормаль и «франкен-карта».

Две модели Joey (JoeyBallentine), обученные на текстурах CC0:

  1x NormalMapGenerator-CX-Lite  — карта нормалей прямо из текстуры;
  1x FrankenMapGenerator-CX-Lite — «франкен-карта»: в R текстура в сером,
                                   в G шероховатость, в B высота.

Лицензия весов — CC-BY-NC-SA-4.0: с указанием автора и только для
некоммерческого использования. Поэтому в поставку они не входят и
скачиваются по кнопке с исходного места; код этого модуля — под той же
лицензией, что и вся программа.

Модели сохранены в формате PyTorch (.pth). Тянуть PyTorch — это гигабайты,
поэтому здесь веса читаются без него (формат .pth — это pickle со
ссылками на сырые массивы, его можно разобрать самому), один раз
переводятся в ONNX и дальше считаются тем же onnxruntime, что и глубина
(depthnet.py), — с видеокартой, если она есть.

Архитектура — старый ESRGAN (RRDB_Net из BasicSR/traiNNer):
  первая свёртка → 12 блоков RRDB (по три плотных блока из пяти свёрток,
  выход каждого плотного блока ×0.2 плюс вход, выход RRDB ×0.2 плюс вход)
  → свёртка → сложение со входом ствола → две свёртки на выходе.
Вход и выход — RGB в 0…1 (так их готовит родной generate.py).
"""

import io
import os
import pickle
import struct
import zipfile

import numpy as np

import depthnet as DN

BUILD = 25          # версия набора файлов, проверяется при запуске

_REPO = ('https://github.com/joeyballentine/Material-Map-Generator/raw/master/'
         'utils/models/')
MODELS = {
    'normal': dict(
        title='1x NormalMapGenerator-CX-Lite',
        file='1x_NormalMapGenerator-CX-Lite_200000_G.pth',
        url=_REPO + '1x_NormalMapGenerator-CX-Lite_200000_G.pth',
        page='https://openmodeldb.info/models/1x-NormalMapGenerator-CX-Lite',
        author='Joey (JoeyBallentine)', license='CC-BY-NC-SA-4.0',
        size=19 * (1 << 20)),
    'franken': dict(
        title='1x FrankenMapGenerator-CX-Lite',
        file='1x_FrankenMapGenerator-CX-Lite_215000_G.pth',
        url=_REPO + '1x_FrankenMapGenerator-CX-Lite_215000_G.pth',
        page='https://openmodeldb.info/models/1x-FrankenMapGenerator-CX-Lite',
        author='Joey (JoeyBallentine)', license='CC-BY-NC-SA-4.0',
        size=19 * (1 << 20)),
}
MODEL_IDS = ('normal', 'franken')

# Имена файлов с OpenModelDB отличаются от имён в репозитории автора; оба
# варианта принимаются, если положить файл в папку моделей руками.
ALIASES = {'normal': ('1x-NormalMapGenerator-CX-Lite.pth',),
           'franken': ('1x-FrankenMapGenerator-CX-Lite.pth',)}


# ------------------------------------------------------------ файлы -------
def pth_path(key):
    """Return the path of the .pth weights file for model ``key``.

    Two file names are accepted: the one from the author's repository
    (``MODELS[key]['file']``) and the one used on OpenModelDB (``ALIASES[key]``),
    because the user may put either of them into the model folder by hand. The
    first one that exists wins.

    Args:
        key: 'normal' or 'franken'.

    Returns:
        Absolute path (str): of an existing file if one is found, otherwise the
        path under the repository name where download() will put it.
    """
    d = DN.model_dir()
    for name in (MODELS[key]['file'],) + ALIASES.get(key, ()):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return os.path.join(d, MODELS[key]['file'])


def onnx_path(key):
    """Return the path of the converted ONNX file for model ``key``.

    The file is ``esrgan_<key>.onnx`` in the common model folder
    (depthnet.model_dir()); it may not exist yet — prepare() creates it.
    """
    return os.path.join(DN.model_dir(), 'esrgan_%s.onnx' % key)


def have_model(key):
    """Tell whether model ``key`` is ready: converted ONNX or at least the .pth exists.

    Only existence is checked, not the size (the expected size in MODELS is
    approximate): a downloaded .pth is validated by parsing it with load_pth()
    before it is renamed into place, see download().

    Returns:
        True if either file exists, else False.
    """
    return os.path.isfile(onnx_path(key)) or os.path.isfile(pth_path(key))


def have_onnx_lib():
    """Return True if the optional ``onnx`` package can be imported.

    It is needed only once per model, to convert the .pth weights into an ONNX
    graph (to_onnx); running the converted model needs only onnxruntime.
    """
    try:
        import onnx  # noqa: F401
        return True
    except Exception:
        return False


def download(key, progress=None):
    """Download the .pth into the model folder.

    Same logic as depthnet.download: first into a temporary file
    (``<name>.part``), then a rename, so that an interrupted download does not
    look like a finished model. Before the rename the file is parsed with
    load_pth(), so a broken or truncated file is rejected here and not later when
    computing. The exact byte size is not compared (unlike depthnet), because the
    size in MODELS is only approximate; it serves as the progress total when the
    server sends no Content-Length. The request has a 60 s timeout and is read in
    64 KiB chunks. On failure the ``.part`` file stays behind and is overwritten by
    the next attempt.

    Args:
        key: 'normal' or 'franken'.
        progress: optional callable(got_bytes, total_bytes), called after every
            chunk (from the calling thread).

    Returns:
        Path (str) of the downloaded file under its repository name.

    Raises:
        urllib / OSError on network or disk errors; ValueError (or other
        unpickling errors) from load_pth if the file is not valid ESRGAN weights.
    """
    m = MODELS[key]
    dst = os.path.join(DN.model_dir(), m['file'])
    tmp = dst + '.part'
    DN.fetch(m['url'], tmp, 0, progress)       # точный размер неизвестен
    load_pth(tmp)                 # битый файл отсеется здесь, а не при счёте
    os.replace(tmp, dst)
    return dst


# ---------------------------------------------- чтение .pth без torch -----
_DTYPES = {'FloatStorage': np.float32, 'HalfStorage': np.float16,
           'DoubleStorage': np.float64, 'BFloat16Storage': None,
           'LongStorage': np.int64, 'IntStorage': np.int32,
           'ShortStorage': np.int16, 'CharStorage': np.int8,
           'ByteStorage': np.uint8, 'BoolStorage': np.bool_}


class _Storage:
    """Reference to a raw array inside the file; the data are filled in later.

    Attributes:
        dtype: numpy dtype of the elements (from _DTYPES).
        key: storage key (str): the file name under ``data/`` in the zip format,
            or the key from the list that follows the pickle in the legacy format.
        size: unused, always None.
        data: raw bytes of the storage, set by load_pth() after unpickling.
    """

    def __init__(self, dtype, key, size=None):
        """Remember dtype and key; ``data`` stays None until load_pth() fills it."""
        self.dtype, self.key, self.size = dtype, key, size
        self.data = None


class _StorageType:
    """Stand-in for a torch ``*Storage`` class named in the pickle.

    Only the class name (e.g. 'FloatStorage') is kept; _Unpickler.persistent_load
    maps it to a numpy dtype through _DTYPES.
    """
    def __init__(self, name):
        """Store the storage class name (str)."""
        self.name = name


def _rebuild_tensor(storage, offset, size, stride, *rest):
    """Stand-in for torch._utils._rebuild_tensor_v2 / _rebuild_tensor.

    Instead of building a tensor it returns a description
    ``('tensor', storage, offset, size, stride)`` that _materialize() later turns
    into a numpy array. ``offset`` and ``stride`` are in elements, not bytes; the
    remaining arguments (requires_grad, backward hooks, metadata) are ignored.
    """
    return ('tensor', storage, int(offset), tuple(size), tuple(stride))


class _Unpickler(pickle.Unpickler):
    """Parses the pickle from a .pth knowing nothing about torch.

    Tensors become descriptions (where the data lie and what shape they have),
    and the data themselves are read separately (load_pth). Every global the
    pickle asks for is resolved by find_class to a harmless stand-in; no module
    named in the file is imported, so nothing from torch (or anything else) is
    executed while loading.
    """

    def __init__(self, f, storages):
        """Wrap binary file ``f`` (positioned at the pickle).

        Args:
            f: binary file object.
            storages: dict key → _Storage shared with the caller; it is filled as
                persistent ids are met, and load_pth() then reads the bytes into it.
        """
        super().__init__(f)
        self._st = storages

    def find_class(self, module, name):
        """Map a global referenced by the pickle to a stand-in.

        _rebuild_tensor_v2 / _rebuild_tensor → _rebuild_tensor (a tensor
        description); torch.*Storage → _StorageType; collections.OrderedDict → the
        real OrderedDict (state dicts are ordered dicts); _rebuild_parameter → returns
        the wrapped tensor description; torch.device / torch.Size → a tuple of the
        arguments. Everything else (optimiser state and so on) becomes a function that
        returns None — it is not needed.
        """
        if module == 'torch._utils' and name in ('_rebuild_tensor_v2', '_rebuild_tensor'):
            return _rebuild_tensor
        if module == 'torch' and name.endswith('Storage'):
            return _StorageType(name)
        if module == 'collections' and name == 'OrderedDict':
            import collections
            return collections.OrderedDict
        if module == 'torch._utils' and name == '_rebuild_parameter':
            return lambda data, requires_grad, hooks, *a: data
        if module == 'torch' and name in ('device', 'Size'):
            return lambda *a, **k: a
        # всё остальное (состояние оптимизатора и т. п.) нам не нужно
        return lambda *a, **k: None

    def persistent_load(self, pid):
        """Resolve a persistent id (a reference to a tensor storage) into a _Storage.

        The id is ('storage', type, key, location, numel) in the zip format and
        ('storage', type, key, location, size, view_metadata) in the legacy one; only
        the type and the key are used. One _Storage is kept per key, so tensors that
        share a storage share the object.

        Raises:
            ValueError: for storage types numpy cannot represent (BFloat16) or
                unknown ones.
        """
        # zip-формат: ('storage', тип, ключ, место, число элементов)
        # старый формат: ('storage', тип, ключ, место, размер, view)
        typ, key = pid[1], str(pid[2])
        name = typ.name if isinstance(typ, _StorageType) else str(typ)
        dt = _DTYPES.get(name.split('.')[-1])
        if dt is None:
            raise ValueError('unsupported tensor type: %s' % name)
        st = self._st.get(key)
        if st is None:
            st = self._st[key] = _Storage(dt, key)
        return st


def _materialize(obj):
    """Recursively replace tensor descriptions by numpy arrays.

    A description from _rebuild_tensor is resolved against its storage bytes:
    the flat array is viewed with the given offset, size and stride (strides are
    converted from elements to bytes) and copied. float16 and float64 become
    float32, other dtypes are kept. A scalar tensor (empty size) becomes a 1-D
    array of one element with the storage dtype. Dicts are processed value by
    value (keys kept); anything else, including lists, is returned as is.
    """
    if isinstance(obj, tuple) and obj and obj[0] == 'tensor':
        _, st, off, size, stride = obj
        a = np.frombuffer(st.data, dtype=st.dtype)
        n = int(np.prod(size)) if size else 1
        if size:
            item = a.itemsize
            out = np.lib.stride_tricks.as_strided(
                a[off:], shape=size, strides=tuple(s * item for s in stride))
            return np.array(out, dtype=np.float32 if st.dtype in (np.float16, np.float64)
                            else st.dtype, copy=True)
        return np.array(a[off:off + n], copy=True)
    if isinstance(obj, dict):
        return {k: _materialize(v) for k, v in obj.items()}
    return obj


def load_pth(path):
    """Read the weights from a .pth without PyTorch.

    Both formats are understood: the new one (a zip with data.pkl and separate
    data files, one per storage under ``data/``) and the old one (a pickle, then a
    list of keys and the raw data one after another). The old one is a sequence
    of pickles — magic number 0x1950a86a20f9469cfc6c, protocol version, system
    info, the object itself, the list of storage keys — followed, for every key,
    by an 8-byte little-endian element count and the raw bytes. Checkpoints
    saved by BasicSR wrap the weights into 'params_ema' (preferred) or 'params';
    both are unwrapped.

    Args:
        path: path of the .pth file.

    Returns:
        Dict layer name → numpy array (float weights as float32); entries that
        are not arrays are dropped.

    Raises:
        ValueError: 'not a PyTorch file', an unsupported tensor type, or 'no
            weights in the file'. A zip without data.pkl raises StopIteration.
    """
    storages = {}
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            pkl = next(n for n in names if n.endswith('data.pkl'))
            root = pkl[:-len('data.pkl')]
            obj = _Unpickler(io.BytesIO(z.read(pkl)), storages).load()
            for key, st in storages.items():
                st.data = z.read(root + 'data/' + key)
    else:
        with open(path, 'rb') as f:
            magic = pickle.load(f)
            if magic != 0x1950a86a20f9469cfc6c:
                raise ValueError('not a PyTorch file')
            pickle.load(f)                       # версия протокола
            pickle.load(f)                       # сведения о системе
            obj = _Unpickler(f, storages).load()
            keys = pickle.load(f)
            for key in keys:
                st = storages[str(key)]
                n = struct.unpack('<q', f.read(8))[0]
                st.data = f.read(n * np.dtype(st.dtype).itemsize)
    sd = _materialize(obj)
    if isinstance(sd, dict) and 'params_ema' in sd:
        sd = sd['params_ema']
    elif isinstance(sd, dict) and 'params' in sd:
        sd = sd['params']
    if not isinstance(sd, dict) or not sd:
        raise ValueError('no weights in the file')
    return {k: v for k, v in sd.items() if isinstance(v, np.ndarray)}


# ------------------------------------------------ устройство сети ---------
def describe(sd):
    """Work out how the network is built from the names and shapes of the weights.

    Both layouts are understood: the old one (model.0, model.1.sub…) and the new
    one (conv_first, RRDB_trunk…). A state dict saved without the ``model.``
    wrapper (keys '0.weight', …) is renamed in place to the old layout, so ``sd``
    may be modified — callers that need the original pass a copy.

    In the old layout model.0 is the first convolution, model.1.sub.<i> are the
    RRDB blocks and model.1.sub.<nb> the trunk convolution; of the following
    numbered layers the last two are the HR convolution and the output
    convolution, and any before them are upsampling convolutions.

    Args:
        sd: dict name → array from load_pth().

    Returns:
        Dict with the layer names by role: 'first', 'nb' (number of RRDB blocks),
        'rdb' (callable (block, dense block 1…3, conv 1…5) → layer name prefix),
        'trunk', 'ups' (list; empty for 1x models), 'hr0', 'last'.

    Raises:
        ValueError: if the layout is not an ESRGAN (RRDB) model.
    """
    if '0.weight' in sd and 'model.0.weight' not in sd:
        # сохранено без обёртки: те же имена без приставки model.
        sd.update({'model.' + k: sd.pop(k) for k in list(sd) if k[:1].isdigit()})
    if 'model.0.weight' in sd:
        pre = 'model.1.sub.'
        nb = 0
        while (pre + '%d.RDB1.conv1.0.weight' % nb) in sd:
            nb += 1
        tail = sorted({int(k.split('.')[1]) for k in sd
                       if k.startswith('model.') and k.split('.')[1].isdigit()
                       and int(k.split('.')[1]) > 1 and k.endswith('.weight')})
        if len(tail) < 2:
            raise ValueError('unexpected ESRGAN layout')
        rdb = lambda b, r, c: pre + '%d.RDB%d.conv%d.0' % (b, r, c)
        return dict(first='model.0', nb=nb, rdb=rdb, trunk=pre + '%d' % nb,
                    ups=['model.%d' % i for i in tail[:-2]],
                    hr0='model.%d' % tail[-2], last='model.%d' % tail[-1])
    if 'conv_first.weight' in sd:
        nb = 0
        while ('RRDB_trunk.%d.RDB1.conv1.weight' % nb) in sd:
            nb += 1
        rdb = lambda b, r, c: 'RRDB_trunk.%d.RDB%d.conv%d' % (b, r, c)
        ups = [n for n in ('upconv1', 'upconv2', 'upconv3') if n + '.weight' in sd]
        return dict(first='conv_first', nb=nb, rdb=rdb, trunk='trunk_conv', ups=ups,
                    hr0='HRconv', last='conv_last')
    raise ValueError('not an ESRGAN (RRDB) model')


def to_onnx(sd, path):
    """Build an ONNX graph from the weights and save it.

    Convolutions with zero padding of one pixel (k // 2, i.e. 1 for the 3×3
    kernels), LeakyReLU 0.2, dense connections via Concat — exactly as in the
    architecture. The picture size is not fixed in the graph. Residual scaling is
    a Mul by the constant 0.2 ('k02'); every upsampling layer, if present, is a
    nearest ×2 Resize followed by convolution and LeakyReLU. The zero padding only
    affects the outer margin, which run() adds by wrapping and then cuts away.

    The graph has input 'x' and output 'y', both float32 [1, 3, H, W], RGB 0…1;
    opset 13, IR version 7. It is validated with onnx.checker, written to
    ``path + '.part'`` and renamed, so an interrupted conversion never leaves a
    half-written file that would pass for a ready model. Requires the ``onnx``
    package.

    Args:
        sd: dict name → array from load_pth() (may be renamed in place by
            describe()).
        path: destination .onnx path.

    Returns:
        The layout dict from describe().
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    a = describe(sd)
    nodes, inits = [], []
    cnt = [0]

    def uid(p):
        """Return a fresh tensor name ``<p>_<n>`` (one counter for the whole graph)."""
        cnt[0] += 1
        return '%s_%d' % (p, cnt[0])

    def conv(x, name):
        """Append a Conv node for layer ``name`` and return its output name.

        The weights ``name.weight`` and, if present, ``name.bias`` are added as
        float32 initialisers; padding is k // 2 on every side (same size, zeros).
        """
        w, b = sd[name + '.weight'], sd.get(name + '.bias')
        wn, bn = name + '.weight', name + '.bias'
        inits.append(numpy_helper.from_array(w.astype(np.float32), wn))
        ins = [x, wn]
        if b is not None:
            inits.append(numpy_helper.from_array(b.astype(np.float32), bn))
            ins.append(bn)
        k = w.shape[-1]
        y = uid('conv')
        nodes.append(helper.make_node('Conv', ins, [y], pads=[k // 2] * 4,
                                      kernel_shape=[k, k]))
        return y

    def lrelu(x):
        """Append LeakyRelu with slope 0.2 (as in ESRGAN) and return its output name."""
        y = uid('lrelu')
        nodes.append(helper.make_node('LeakyRelu', [x], [y], alpha=0.2))
        return y

    def cat(xs):
        """Append a Concat on the channel axis (dense links); return its output name."""
        y = uid('cat')
        nodes.append(helper.make_node('Concat', xs, [y], axis=1))
        return y

    def add(x, z):
        """Append an element-wise Add of ``x`` and ``z``; return its output name."""
        y = uid('add')
        nodes.append(helper.make_node('Add', [x, z], [y]))
        return y

    inits.append(numpy_helper.from_array(np.array(0.2, np.float32), 'k02'))

    def scale02(x):
        """Append a Mul by 0.2 (ESRGAN residual scaling); return its output name."""
        y = uid('mul')
        nodes.append(helper.make_node('Mul', [x, 'k02'], [y]))
        return y

    fea = conv('x', a['first'])
    h = fea
    for b in range(a['nb']):
        rin = h
        for r in (1, 2, 3):
            x0 = h
            feats = [x0]
            for c in (1, 2, 3, 4):
                y = lrelu(conv(cat(feats) if len(feats) > 1 else feats[0],
                               a['rdb'](b, r, c)))
                feats.append(y)
            x5 = conv(cat(feats), a['rdb'](b, r, 5))
            h = add(scale02(x5), x0)
        h = add(scale02(h), rin)
    h = add(conv(h, a['trunk']), fea)
    if a['ups']:
        inits.append(numpy_helper.from_array(
            np.array([1, 1, 2, 2], np.float32), 'up2'))
        for u in a['ups']:
            y = uid('up')
            nodes.append(helper.make_node('Resize', [h, '', 'up2'], [y], mode='nearest'))
            h = lrelu(conv(y, u))
    h = lrelu(conv(h, a['hr0']))
    out = conv(h, a['last'])
    nodes.append(helper.make_node('Identity', [out], ['y']))
    g = helper.make_graph(
        nodes, 'esrgan', [helper.make_tensor_value_info('x', TensorProto.FLOAT,
                                                        [1, 3, None, None])],
        [helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 3, None, None])],
        inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)],
                          producer_name='Texelith')
    m.ir_version = 7
    onnx.checker.check_model(m)
    tmp = path + '.part'
    onnx.save(m, tmp)
    os.replace(tmp, path)
    return a


def prepare(key, progress=None):
    """Bring the model to a ready ONNX: download the .pth if missing, then convert.

    Blocking: may access the network (download) and needs the ``onnx`` package
    for the conversion. An existing ONNX is returned as is, without looking at
    the .pth.

    Args:
        key: 'normal' or 'franken'.
        progress: optional download progress callable(got_bytes, total_bytes).

    Returns:
        Path (str) to the ONNX file.
    """
    op = onnx_path(key)
    if os.path.isfile(op):
        return op
    pp = pth_path(key)
    if not os.path.isfile(pp):
        pp = download(key, progress)
    to_onnx(load_pth(pp), op)
    return op


_SESS = {}


def session(key):
    """Open (once) the onnxruntime session for model ``key``.

    Same accelerator choice as for depth (depthnet.PROVIDERS): the GPU if there
    is one, otherwise the CPU. If the session cannot be created with the GPU
    providers, it is silently recreated on the CPU. Sessions are cached in _SESS
    per key until unload(). Calls prepare(), so the first call may download and
    convert the model. Requires onnxruntime.

    Returns:
        onnxruntime.InferenceSession.
    """
    if key in _SESS:
        return _SESS[key]
    import onnxruntime as ort
    op = prepare(key)
    DN.preload()
    avail = ort.get_available_providers()
    prov = [p for p in DN.PROVIDERS if p in avail] or ['CPUExecutionProvider']
    try:
        s = ort.InferenceSession(op, providers=prov)
    except Exception:
        s = ort.InferenceSession(op, providers=['CPUExecutionProvider'])
    DN._used['esr_' + key] = (s.get_providers() or ['?'])[0]
    _SESS[key] = s
    return s


def unload():
    """Drop all cached sessions, so the next session() call reopens the ONNX file.

    Used after a model file has been replaced; also frees the memory held by the
    sessions.
    """
    _SESS.clear()


# ------------------------------------------------------------ счёт ---------
def run(key, rgb, tile=384, pad=48, progress=None):
    """Run the network over the picture window by window, seamlessly.

    Args:
        key: model name ('normal' or 'franken'), or an already opened session
            (anything with a ``run`` method) — convenient for testing.
        rgb: sRGB 0…1, H×W×3 (as the model was trained: without converting to
            linear). Extra channels (alpha) are ignored; values are clipped to
            0…1.
        tile: window side in pixels (at least 32); memory grows with the square
            of the window (with a 384 window the network holds about 150 MB of
            intermediate data).
        pad: how much to take around the window, in pixels. The picture is first
            wrapped around its edges (like a tile), so there are neighbours at
            the texture edges too, and the result comes out seamless: a shift of
            the picture gives the same shift of the map. The network sees far,
            but noticeably only a few dozen pixels; 48 is more than enough
            (checked in _check.py).
        progress: optional callable(done_windows, total_windows).

    The texture border is thus handled exactly like a border between two windows
    inside the texture. Shift equivariance is exact when the network's receptive
    field fits into ``pad`` (_check.py verifies it this way with a small test
    network); with the real, deeper network the context beyond ``pad`` has a tiny,
    in practice invisible, influence.

    Returns:
        H×W×3 float32 in 0…1.
    """
    # key — имя модели или уже открытая сессия (так удобнее проверять)
    s = key if hasattr(key, 'run') else session(key)
    x = np.ascontiguousarray(np.clip(rgb[..., :3], 0, 1), np.float32)
    H, W = x.shape[:2]
    P = int(pad)
    big = np.pad(x, ((P, P), (P, P), (0, 0)), mode='wrap')
    out = np.zeros((H, W, 3), np.float32)
    T = max(32, int(tile))
    boxes = [(y, xx) for y in range(0, H, T) for xx in range(0, W, T)]
    for i, (y0, x0) in enumerate(boxes):
        y1, x1 = min(y0 + T, H), min(x0 + T, W)
        win = big[y0:y1 + 2 * P, x0:x1 + 2 * P]
        inp = np.transpose(win, (2, 0, 1))[None]
        res = s.run(None, {'x': inp})[0][0]
        out[y0:y1, x0:x1] = np.transpose(res, (1, 2, 0))[P:P + (y1 - y0),
                                                          P:P + (x1 - x0)]
        if progress:
            progress(i + 1, len(boxes))
    return np.clip(out, 0.0, 1.0)


def normal_from_output(out, directx=True, flip_y=False):
    """Turn the NormalMapGenerator output into unit vectors in −1…1.

    The model outputs an RGB normal; flip_y reverses green if the model's
    convention did not match the one needed.

    Args:
        out: H×W×3 output of run(), 0…1.
        directx: True keeps the green channel as the model gives it, False
            negates it — i.e. the raw output is treated as DirectX (Y down).
        flip_y: additionally negate green.

    z is floored at 1e-3 so that no vector faces away from the viewer and the
    normalisation never divides by zero.

    Returns:
        H×W×3 float32 unit vectors.
    """
    n = out * 2.0 - 1.0
    if flip_y:
        n[..., 1] = -n[..., 1]
    if not directx:
        n[..., 1] = -n[..., 1]
    n[..., 2] = np.maximum(n[..., 2], 1e-3)
    return (n / np.sqrt((n * n).sum(-1, keepdims=True))).astype(np.float32)


def franken_split(out):
    """Split the FrankenMapGenerator output into (grey, roughness, height), all 0…1.

    The channel layout follows the author's description: R — the texture in
    greyscale, G — roughness, B — height (displacement).

    Args:
        out: H×W×3 output of run(), 0…1.

    Returns:
        Tuple of three H×W float32 arrays: (R grey, G roughness, B height).
    """
    return (out[..., 0].astype(np.float32), out[..., 1].astype(np.float32),
            out[..., 2].astype(np.float32))
