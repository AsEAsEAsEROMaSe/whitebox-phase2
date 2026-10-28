"""19.10.2026 white-box-16: коректор LT2 Susupro (MIT, #336537) у torch, біт у біт за _lt2_layer з estimator_sb2.py.
Вхід шару l (природний порядок нейронів, X з дампу: [LT_FE 27 | W prev8 | WW prev8 | LT_FE2 8]):
  own35 = tf(X[:, :27] ++ X[:, 43:51]) стандартизовані (mz, iz [:35]); v16 = tf(X[:, 27:43]) (mz, iz [35:]);
  emb 64 = [h_prev @ W_l, h_prev @ W_l^2] (0 на шарі 0); uin 3 = [Wu, Phi*Wu, u^2 @ W_l^2] * usc[l] (0 на шарі 0), Wu = u_prev @ W_l.
  118 -> 128 GELU(erf) -> 128 GELU -> 128 GELU -> 33: g = out0 * es[l]; h = tanh(out[1:]); u = Phi*Wu + g.
Вихід: pred_15 + u_15. tf(x) = sign(x) log1p(1e3 |x|)."""
import numpy as np
import torch

OWN = list(range(27)) + list(range(43, 51))
V16 = list(range(27, 43))


def tf(x):
    return torch.sign(x) * torch.log1p(torch.abs(x) * 1e3)


def gelu(h):
    return h * 0.5 * (1.0 + torch.erf(h * 0.7071067811865476))


class LT2(torch.nn.Module):
    def __init__(self, npz, dev):
        super().__init__()
        d = {k: torch.tensor(np.asarray(v), dtype=torch.float32 if v.dtype != np.int64 else torch.int64) for k, v in np.load(npz).items()}
        P = lambda x: torch.nn.Parameter(x.to(dev).clone())
        self.W1 = P(d["W1"]); self.b1 = P(d["b1"])
        self.Wm0 = P(d["Wm0"]); self.bm0 = P(d["bm0"]); self.Wm1 = P(d["Wm1"]); self.bm1 = P(d["bm1"])
        self.Wo = P(d["Wo"]); self.bo = P(d["bo"])
        for k in ("usc", "es", "mz", "iz"):
            self.register_buffer(k, d[k].to(dev))

    def layer(self, l, X, Phi, Wl, st):
        """X (B,n,51), Phi (B,n), Wl (B,n,n) natural (pre = h @ Wl), st = None or (u, h). Returns g, u, h."""
        B, n, _ = X.shape
        Z0 = (tf(X[..., OWN]) - self.mz[l, :35]) * self.iz[l, :35]
        Zv = (tf(X[..., V16]) - self.mz[l, 35:]) * self.iz[l, 35:]
        if st is None:
            emb = X.new_zeros(B, n, 64); uin = X.new_zeros(B, n, 3); Wu = None
        else:
            u, h = st
            W2 = Wl * Wl
            emb = torch.cat([torch.bmm(h.transpose(1, 2), Wl).transpose(1, 2),
                             torch.bmm(h.transpose(1, 2), W2).transpose(1, 2)], dim=2)
            Wu = torch.bmm(u[:, None, :], Wl)[:, 0]
            uu = torch.bmm((u * u)[:, None, :], W2)[:, 0]
            uin = torch.stack([Wu, Phi * Wu, uu], dim=2) * self.usc[l]
        x = torch.cat([Z0, Zv, emb, uin], dim=2)
        a = gelu(x @ self.W1 + self.b1[l])
        a = gelu(a @ self.Wm0 + self.bm0)
        a = gelu(a @ self.Wm1 + self.bm1)
        o = a @ self.Wo + self.bo
        g = o[..., 0] * self.es[l]
        hn = torch.tanh(o[..., 1:])
        un = g if Wu is None else Phi * Wu + g
        return g, un, hn

    def run(self, X, Phi, Ws):
        """X (B,16,n,51), Phi (B,16,n), Ws list of 16 tensors (B,n,n). Returns list of g_l and u_l."""
        st = None; gs = []; us = []
        for l in range(16):
            g, u, h = self.layer(l, X[:, l], Phi[:, l], Ws[l], st)
            gs.append(g); us.append(u); st = (u, h)
        return gs, us
