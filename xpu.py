# -*- coding: utf-8 -*-
"""
Texelith — where the arrays live: graphics card (CuPy) or processor (numpy).

The image maths in matgen.py and view3d.py is written once, for "an array
module" ``xp``: numpy on the processor, CuPy on an NVIDIA graphics card (CuPy
repeats the numpy and scipy.ndimage interfaces, so the same code runs on
either). A function looks at the array it was given and uses that array's
module; whoever hands over device arrays gets device arrays back.

The node graph moves the source texture to the card once, keeps every
intermediate result there, and brings back to the processor only what must be
seen or saved: thumbnails, the 2D view, files. Parts that exist only on the
processor (damage repair, the networks' input and output, PIL) receive host
arrays; the node wrapper in nodes.py converts at those borders.

CuPy is optional (``pip install cupy-cuda13x[ctk]``: MIT itself, the CUDA
libraries it brings are NVIDIA's). Without it, or without a working NVIDIA
card, everything runs on the processor as before.

For the regression suite there is also a *strict stand-in* for a device
module (``fake=True`` in use()): numpy under the hood, but its arrays refuse
to be converted to host arrays implicitly and refuse to mix with host arrays —
exactly the two ways code that works with numpy breaks with CuPy. It lets the
GPU path be exercised on a machine without a graphics card.

Texelith — где лежат массивы: на видеокарте (CuPy) или на процессоре (numpy).
Математика написана один раз для «модуля массивов» xp; функция берёт модуль
того массива, который ей дали. Граф узлов держит промежуточные результаты на
видеокарте и возвращает на процессор только то, что надо показать или
записать.
"""

import numpy as np
from scipy import ndimage as _sndi

BUILD = 25          # версия набора файлов, проверяется при запуске

_state = {'on': False, 'cp': None, 'cndi': None, 'name': '', 'err': '', 'fake': False}


# ------------------------------------------------------------- включение --
def use(on=True, fake=False):
    """Turn device arrays on (if CuPy and a card work) or off.

    Returns (on, text): whether device arrays are now used, and the device
    name or the reason they are not.
    """
    import os
    _state.update(on=False, cp=None, cndi=None, name='', err='', fake=False)
    if not on:
        return False, ''
    # для проверок без видеокарты: TEXELITH_FAKE_GPU=1 включает строгую подмену
    # (старое имя TESSERA_FAKE_GPU тоже понимается)
    fake = fake or '1' in (os.environ.get('TEXELITH_FAKE_GPU'), os.environ.get('TESSERA_FAKE_GPU'))
    if fake:
        _state.update(on=True, cp=_FakeXP, cndi=_FakeNDI, name='strict stand-in', fake=True)
        return True, _state['name']
    try:
        import cupy as cp
        import cupyx.scipy.ndimage as cndi
        cp.cuda.runtime.getDeviceCount()
        a = cp.arange(4, dtype=cp.float32)
        float((a * 2).sum())                     # ядро действительно запускается
        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        name = props.get('name', b'GPU')
        name = name.decode() if isinstance(name, bytes) else str(name)
    except Exception as e:                       # noqa: BLE001
        _state['err'] = '%s: %s' % (e.__class__.__name__, e)
        return False, _state['err']
    _state.update(on=True, cp=cp, cndi=cndi, name=name)
    return True, name


def active():
    """True while device arrays are in use."""
    return _state['on']


def device_name():
    """Name of the card in use, '' when on the processor."""
    return _state['name'] if _state['on'] else ''


def last_error():
    """Why the card could not be used ('' if it was not tried or works)."""
    return _state['err']


def have_cupy():
    """True if the cupy package is installed (without importing it)."""
    import importlib.util
    try:
        return importlib.util.find_spec('cupy') is not None
    except Exception:
        return False


# ------------------------------------------------------------ модули ------
def is_dev(a):
    """True if ``a`` is a device array (looks inside dicts, lists and tuples)."""
    cp = _state['cp']
    if cp is None:
        return False
    if isinstance(a, dict):
        return any(is_dev(v) for v in a.values())
    if isinstance(a, (list, tuple)):
        return any(is_dev(v) for v in a)
    return isinstance(a, cp.ndarray)


def xp(*arrays):
    """Array module for these arrays: the device module if any of them lives
    on the card, numpy otherwise."""
    for a in arrays:
        if is_dev(a):
            return _state['cp']
    return np


def ndi(*arrays):
    """scipy.ndimage, or its device twin when an array lives on the card."""
    for a in arrays:
        if is_dev(a):
            return _state['cndi']
    return _sndi


def to_dev(a):
    """Move a host array to the card (no-op if already there or if off)."""
    if not _state['on'] or a is None or is_dev(a) or not isinstance(a, np.ndarray):
        return a
    return _state['cp'].asarray(a)


def to_host(a):
    """Bring a device array back as numpy (no-op for host arrays and others)."""
    if is_dev(a) and not isinstance(a, (dict, list, tuple)):
        cp = _state['cp']
        return cp.asnumpy(a) if cp is not _FakeXP else a._a.copy()
    return a


def host_tree(v):
    """to_host applied through dicts, lists and tuples."""
    if isinstance(v, dict):
        return {k: host_tree(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return type(v)(host_tree(x) for x in v)
    return to_host(v)


def dev_tree(v):
    """to_dev applied through dicts, lists and tuples."""
    if isinstance(v, dict):
        return {k: dev_tree(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return type(v)(dev_tree(x) for x in v)
    return to_dev(v)


_kernels = {}


def kernel(src, name):
    """A compiled CUDA kernel (cupy.RawKernel) for real device arrays, or None
    — on the processor and on the strict stand-in, where the caller runs its
    numpy version instead. Kernels are compiled once per (source, name) and
    kept; a compile error is remembered and None returned from then on (the
    reason is in last_error()), so a broken kernel never stops the program."""
    cp = _state['cp']
    if not _state['on'] or _state['fake'] or cp is None:
        return None
    key = (name, hash(src))
    if key not in _kernels:
        try:
            k = cp.RawKernel(src, name)
            k.compile()
            _kernels[key] = k
        except Exception as e:                   # noqa: BLE001
            _state['err'] = 'kernel %s: %s: %s' % (name, e.__class__.__name__, e)
            _kernels[key] = None
    return _kernels[key]


def f32(a, like=None):
    """``a`` as float32 in the module of ``a`` (or of ``like`` for lists and
    numbers)."""
    m = xp(a, like) if like is not None else xp(a)
    return m.asarray(a, dtype=m.float32)


def num(x):
    """A scalar (0-d array of either module) as a Python float."""
    return float(to_host(x)) if is_dev(x) else float(x)


# ================================================= строгая подмена CuPy ===
class _FakeArray(np.lib.mixins.NDArrayOperatorsMixin):
    """numpy array in a wrapper that behaves like a CuPy array where it
    matters for bugs: no implicit conversion to numpy, no mixing with numpy
    arrays in arithmetic, numpy functions dispatch to the "device"."""

    __slots__ = ('_a',)

    def __init__(self, a):
        self._a = a

    # --- то, что CuPy запрещает
    def __array__(self, *a, **k):
        raise TypeError('implicit conversion of a device array to numpy')

    # --- свойства и методы ndarray
    shape = property(lambda s: s._a.shape)
    ndim = property(lambda s: s._a.ndim)
    dtype = property(lambda s: s._a.dtype)
    size = property(lambda s: s._a.size)
    T = property(lambda s: _FakeArray(s._a.T))
    real = property(lambda s: _FakeArray(s._a.real))
    imag = property(lambda s: _FakeArray(s._a.imag))

    def __len__(self):
        return len(self._a)

    def __float__(self):
        return float(self._a)

    def __int__(self):
        return int(self._a)

    def __bool__(self):
        return bool(self._a)

    def __iter__(self):
        for x in self._a:
            yield _wrap(x)

    def __repr__(self):
        return 'fake_device(%r)' % (self._a,)

    def __getitem__(self, k):
        return _wrap(self._a[_unwrap(k)])

    def __setitem__(self, k, v):
        if isinstance(v, np.ndarray):
            raise TypeError('assigning a host array into a device array')
        self._a[_unwrap(k)] = _unwrap(v)

    def __getattr__(self, name):
        attr = getattr(self._a, name)
        if callable(attr):
            def f(*a, **k):
                return _wrap(attr(*_unwrap(a), **_unwrap(k)))
            return f
        return _wrap(attr)

    def get(self):
        """CuPy's explicit transfer to the host."""
        return self._a.copy()

    def __array_ufunc__(self, ufunc, method, *inputs, **kw):
        for x in inputs:
            if isinstance(x, np.ndarray):
                raise TypeError('mixing a host array with a device array in %s' % ufunc.__name__)
        return _wrap(getattr(ufunc, method)(*_unwrap(inputs), **_unwrap(kw)))

    def __array_function__(self, func, types, args, kwargs):
        for x in list(args) + list(kwargs.values()):
            if isinstance(x, np.ndarray):
                raise TypeError('mixing a host array with a device array in %s' % func.__name__)
        return _wrap(func(*_unwrap(args), **_unwrap(kwargs)))


def _unwrap(v):
    """Fake device arrays (also inside tuples, lists, dicts) → the numpy arrays
    inside them."""
    if isinstance(v, _FakeArray):
        return v._a
    if isinstance(v, tuple):
        return tuple(_unwrap(x) for x in v)
    if isinstance(v, list):
        return [_unwrap(x) for x in v]
    if isinstance(v, dict):
        return {k: _unwrap(x) for k, x in v.items()}
    return v


def _wrap(v):
    # как CuPy: и скаляры (sum, mean, sqrt от числа) — это 0-мерные массивы на
    # видеокарте, у них нет round() и прочего, что есть у numpy-скаляров
    """numpy results → fake device arrays; numpy scalars become 0-d arrays, as
    in CuPy."""
    if isinstance(v, np.generic):
        return _FakeArray(np.asarray(v))
    if isinstance(v, np.ndarray):
        return _FakeArray(v)
    if isinstance(v, tuple):
        return tuple(_wrap(x) for x in v)
    if isinstance(v, list):
        return [_wrap(x) for x in v]
    return v


# функции, которым и в CuPy можно дать число первым аргументом
_SCALAR_OK = {'asarray', 'array', 'zeros', 'ones', 'full', 'empty', 'arange', 'linspace',
              'eye', 'identity', 'float32', 'float64', 'int32', 'int64', 'uint8', 'dtype',
              'radians', 'degrees', 'deg2rad', 'rad2deg', 'isscalar', 'ndim', 'fftfreq',
              'rfftfreq'}


class _FakeModule:
    """numpy (or scipy.ndimage) whose functions take and return fake device
    arrays and refuse host arrays."""

    def __init__(self, mod, strict=True):
        self._m = mod
        self._strict = strict

    def __getattr__(self, name):
        """Module attribute as the fake device sees it: functions refuse host
        arrays (and, like CuPy, a plain number as first argument) and return
        fake device arrays."""
        attr = getattr(self._m, name)
        if name == 'ndarray':
            return _FakeArray
        if isinstance(attr, type(np)):
            return _FakeModule(attr, self._strict)
        if callable(attr) and not isinstance(attr, type):
            def f(*a, **k):
                for x in list(a) + list(k.values()):
                    if self._strict and isinstance(x, np.ndarray):
                        raise TypeError('host array passed to device %s' % name)
                # как CuPy: кроме ufunc (sqrt, exp, maximum…) и создателей
                # массивов, функции модуля не принимают простое число первым
                # аргументом — cupy.clip(0.5, …) падает, numpy.clip нет
                if (self._strict and a and isinstance(a[0], (int, float, np.generic))
                        and not isinstance(a[0], bool) and not isinstance(attr, np.ufunc)
                        and name not in _SCALAR_OK):
                    raise TypeError('device %s got a plain number %r, not an array'
                                    % (name, a[0]))
                # как cupyx: морфология с iterations > 1 есть только перебором
                if (self._strict and name in ('binary_dilation', 'binary_erosion')
                        and int(k.get('iterations', a[2] if len(a) > 2 else 1)) != 1
                        and not k.get('brute_force', False)):
                    raise NotImplementedError('only brute_force iteration has been implemented')
                return _wrap(attr(*_unwrap(a), **_unwrap(k)))
            return f
        return attr

    def asarray(self, a, dtype=None):
        """Like cupy.asarray: host arrays and lists are moved to the device."""
        return _FakeArray(np.asarray(_unwrap(a), dtype=dtype))

    def array(self, a, dtype=None, copy=True):
        """Like cupy.array."""
        return _FakeArray(np.array(_unwrap(a), dtype=dtype, copy=copy))

    def asnumpy(self, a):
        """Like cupy.asnumpy."""
        return _unwrap(a).copy() if isinstance(a, _FakeArray) else np.asarray(a)

    ndarray = _FakeArray


_FakeXP = _FakeModule(np)
_FakeNDI = _FakeModule(_sndi)
