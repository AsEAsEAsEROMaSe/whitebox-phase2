"""gpufnp (19.10.2026, white-box-89): шим flopscope / flopscope.numpy -> cupy для дампу ознак ланцюга на GPU.
Лише для дампу (без обліку бюджету); сабміт ним НЕ запускати. install() підміняє модулі в sys.modules
на час імпорту естиматора; whestbench треба імпортувати ДО install()."""
import sys, types, math
import numpy as np
sys.path.append("/home/trim/gpupkg")   # cupy; numpy of the venv stays first on the path
import cupy as cp
from cupyx.scipy import special as _csp

_INV_SQRT2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _einsum(*args, out=None, optimize=None, **kw):
    r = cp.einsum(*args, **kw)
    if out is not None:
        cp.copyto(out, r)
        return out
    return r


class _Rng:
    """numpy Generator, results moved to the GPU (the same numbers as the CPU chain)."""

    def __init__(self, seed):
        self._g = np.random.default_rng(seed)

    def __getattr__(self, name):
        f = getattr(self._g, name)

        def w(*a, **k):
            r = f(*a, **k)
            return cp.asarray(r) if isinstance(r, np.ndarray) else r
        return w


def _load(path, *a, **k):
    d = np.load(path, *a, **k)
    if isinstance(d, np.ndarray):
        return cp.asarray(d)
    return {key: cp.asarray(d[key]) for key in d.files}


def _array(x, dtype=None, **k):
    return cp.array(x, dtype=dtype)


def _asarray(x, dtype=None, **k):
    return cp.asarray(x, dtype=dtype)


fnp = types.ModuleType("flopscope.numpy")
fnp.__getattr__ = lambda name: getattr(cp, name)
fnp.einsum = _einsum
fnp.load = _load
fnp.array = _array
fnp.asarray = _asarray
fnp.random = types.SimpleNamespace(default_rng=_Rng)


def _svd(a, full_matrices=True, compute_uv=True, k=None, **kw):
    """flopscope svd(k=r): the leading r triplets (cupy: full thin svd, sliced)."""
    if k is not None and compute_uv and not full_matrices and k <= 64 and a.ndim == 2:
        # top-k through eigh of the float64 Gram (cusolver gesvd ~1.3 s per 1024^2 call, this ~0.08 s);
        # svd_bench.py: rank-k reconstruction rel 1.9e-7 against LAPACK
        a64 = a.astype(cp.float64)
        w, V = cp.linalg.eigh(a64.T @ a64)
        V = V[:, ::-1][:, :k]
        s_ = cp.sqrt(cp.maximum(w[::-1][:k], 0.0))
        u = (a64 @ V) / cp.where(s_ > 0, s_, 1.0)
        return u.astype(a.dtype), s_.astype(a.dtype), V.T.astype(a.dtype)
    r = cp.linalg.svd(a, full_matrices=full_matrices, compute_uv=compute_uv)
    if k is None:
        return r
    if not compute_uv:
        return r[:k]
    u, s_, vh = r
    return u[:, :k], s_[:k], vh[:k]


fnp.linalg = types.SimpleNamespace(svd=_svd, qr=cp.linalg.qr, solve=cp.linalg.solve, eigh=cp.linalg.eigh,
                                   cholesky=cp.linalg.cholesky, inv=cp.linalg.inv, norm=cp.linalg.norm)
fnp.ndarray = cp.ndarray

flops = types.ModuleType("flopscope")
flops.numpy = fnp
flops.as_symmetric = lambda x, symmetry=None, **k: x
flops.stats = types.SimpleNamespace(norm=types.SimpleNamespace(
    cdf=lambda x: _csp.ndtr(x.astype(cp.float64) if x.dtype == cp.float32 else x),
    pdf=lambda x: cp.exp(-0.5 * (x.astype(cp.float64) if x.dtype == cp.float32 else x) ** 2) * _INV_SQRT2PI))


def install():
    saved = {k: sys.modules.get(k) for k in ("flopscope", "flopscope.numpy")}
    sys.modules["flopscope"] = flops
    sys.modules["flopscope.numpy"] = fnp
    return saved


def restore(saved):
    for k, v in saved.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v
