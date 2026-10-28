"""Двигун (стенд 26.09): перенесений на S (32 нейрони) 4-тензор від діаграм κ₄(h) з n²-блоків.
Діаграма = вершини (внутрішні індекси n) + блоки-пари (матриці n×n, рядки=u, стовпці=v) + векторні множники на вершинах
(wick, var, D3, K4). Зовнішні осі a,b,c,d прикріплені до вершин множниками W_S[ext, n] (перенос) або I_S (пломба).
Значення = Σ_внутрішніх Π множників Π блоків; результат тензор (32,32,32,32) з осями a,b,c,d.
Маска «різні індекси» через включення-виключення по несуміжних парах вершин (злиття вершин)."""
import itertools, torch

EXT = "abcd"


class Diagram:
    def __init__(self, nverts):
        self.factors = [[] for _ in range(nverts)]   # список (ext_names tuple, tensor (*32, n))
        self.blocks = []                              # (u, v, M) M[u, v]

    def add_vec(self, u, vec):
        self.factors[u].append(((), vec))

    def add_ext(self, u, name, WS):
        self.factors[u].append(((name,), WS))

    def add_block(self, u, v, M):
        self.blocks.append((u, v, M))

    def copy(self):
        d = Diagram(len(self.factors)); d.factors = [list(f) for f in self.factors]; d.blocks = list(self.blocks); return d


def _materialize(factors, n, limit=3):
    """Добуток множників вершини у один тензор (ext..., n); повертає (names, tensor). Припускає Σ ext ≤ limit."""
    names = (); T = None
    for nm, t in factors:
        if T is None:
            names = nm; T = t
        else:
            T = T[..., None, :] * t if False else _outer(T, names, t, nm)
            names = names + nm
    assert len(names) <= limit, names
    return names, T


def _outer(T, names, t, nm):
    # T (*names, n), t (*nm, n) -> (*names, *nm, n)
    k = len(names); m = len(nm)
    T2 = T.reshape(T.shape[:k] + (1,) * m + (T.shape[-1],))
    t2 = t.reshape((1,) * k + t.shape)
    return T2 * t2


def _split_materialize(factors, n):
    """Вершина з Σ ext = 4: дві групи ≤ 2 ext; повертає (namesA, A (*,n)), (namesB, B (*,n))."""
    fa, fb, ea, eb = [], [], 0, 0
    for nm, t in sorted(factors, key=lambda x: -len(x[0])):
        if ea + len(nm) <= 2:
            fa.append((nm, t)); ea += len(nm)
        else:
            fb.append((nm, t)); eb += len(nm)
    assert eb <= 2, (ea, eb)
    return _materialize(fa, n, 2), _materialize(fb, n, 2)


def contract(d, n):
    """Повна контракція; повертає (names, tensor)."""
    d = d.copy()
    # злити паралельні блоки
    merged = {}
    for u, v, M in d.blocks:
        if u == v:
            return None
        key = (min(u, v), max(u, v)); Mo = M if u <= v else M.T
        merged[key] = Mo if key not in merged else merged[key] * Mo
    blocks = {k: v for k, v in merged.items()}
    alive = set(range(len(d.factors)))
    while len(alive) > 1:
        deg = {u: 0 for u in alive}
        for (u, v) in blocks:
            deg[u] += 1; deg[v] += 1
        leaves = [u for u in alive if deg[u] == 1]
        if leaves:
            leaf = min(leaves, key=lambda u: sum(len(nm) for nm, _ in d.factors[u]))
            (key,) = [k for k in blocks if leaf in k]
            M = blocks.pop(key); nb = key[0] if key[1] == leaf else key[1]
            Mo = M if key[0] == nb else M.T            # Mo[nb, leaf]
            names, L = _materialize(d.factors[leaf], n)  # (*ext, leaf)
            F = (L.reshape(-1, n) @ Mo.T).reshape(L.shape)   # (*ext, nb)
            d.factors[nb].append((names, F)); alive.discard(leaf)
            continue
        # цикл на 3 вершинах: вилучити вершину з найменшою кількістю ext
        assert len(alive) == 3 and len(blocks) == 3, (alive, blocks)
        i = min(alive, key=lambda u: sum(len(nm) for nm, _ in d.factors[u]))
        k, l = sorted(alive - {i})
        ni, Ui = _materialize(d.factors[i], n, 1); Ui = Ui.reshape(-1, n)
        Mik = blocks[(min(i, k), max(i, k))]; Mik = Mik if i < k else Mik.T
        Mil = blocks[(min(i, l), max(i, l))]; Mil = Mil if i < l else Mil.T
        Mkl = blocks[(k, l)]
        X = torch.einsum("ei,ik,il->ekl", Ui, Mik, Mil) * Mkl[None]
        nk, Uk = _materialize(d.factors[k], n); nl, Ul = _materialize(d.factors[l], n)
        Y = torch.einsum("ekl,fk->efl", X, Uk.reshape(-1, n))
        R = torch.einsum("efl,gl->efg", Y, Ul.reshape(-1, n))
        names = ni + nk + nl
        return names, R.reshape(tuple(32 for _ in names))
    (u,) = alive
    tot = sum(len(nm) for nm, _ in d.factors[u])
    if tot <= 3:
        names, T = _materialize(d.factors[u], n); return names, T.sum(-1)
    (na, A), (nb, B) = _split_materialize(d.factors[u], n)
    R = A.reshape(-1, n) @ B.reshape(-1, n).T
    return na + nb, R.reshape(tuple(32 for _ in na + nb))


def to_abcd(names, T):
    """Переставити осі у порядок a,b,c,d (осі, яких нема, відсутні)."""
    order = [names.index(c) for c in EXT if c in names]
    return T.permute(*order)


def masked_value(d, n, adjacency_zero=True):
    """Σ по підмножинах несуміжних пар вершин (−1)^|S| · значення зі злитими вершинами (маска різних індексів)."""
    nv = len(d.factors)
    adj = set((min(u, v), max(u, v)) for u, v, _ in d.blocks)
    na = [(u, v) for u in range(nv) for v in range(u + 1, nv) if (u, v) not in adj]
    out = None
    for r in range(len(na) + 1):
        for sub in itertools.combinations(na, r):
            # union-find злиття
            root = list(range(nv))
            def find(x):
                while root[x] != x:
                    x = root[x]
                return x
            for u, v in sub:
                ru, rv = find(u), find(v)
                if ru != rv:
                    root[max(ru, rv)] = min(ru, rv)
            reps = sorted(set(find(x) for x in range(nv))); idx = {rp: t for t, rp in enumerate(reps)}
            dm = Diagram(len(reps)); zero = False
            for x in range(nv):
                dm.factors[idx[find(x)]].extend(d.factors[x])
            for u, v, M in d.blocks:
                a_, b_ = idx[find(u)], idx[find(v)]
                if a_ == b_:
                    zero = True; break
                dm.blocks.append((a_, b_, M))
            if zero:
                continue
            res = contract(dm, n)
            if res is None:
                continue
            names, T = res; T = to_abcd(names, T)
            sgn = -1.0 if r % 2 else 1.0
            out = sgn * T if out is None else out + sgn * T
    return out
