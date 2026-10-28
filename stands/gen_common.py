"""Спільне для стендів масштабу даних (27.09): детермінована генерація мереж фази 2 (He-init, як whestbench.generation.sample_mlp:
N(0, 2/n) fp32, конвенція x @ W, форма (L, n, n)) із сіду; вагу не зберігаємо (64 МБ/мережу), регенеруємо."""
import math, torch
L, N = 16, 1024
SEED0 = 5_000_000
def make_w(i, device="cpu"):
    g = torch.Generator(device="cpu"); g.manual_seed(SEED0 + int(i))
    w = torch.randn(L, N, N, generator=g, dtype=torch.float32) * math.sqrt(2.0 / N)
    return w.to(device)
