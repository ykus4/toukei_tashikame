"""回帰の4つの仮定を1つずつ壊し、係数のバイアスと95%区間の被覆確率を対応表にする。

「線形性・独立性・等分散性・残差の正規性」は並べて教えられるが、破ったときに何が起きるかは
まったく違う。壊す仮定を1つに絞って 10,000 回ずつ数えると、その違いが表になる。

- **線形性**を壊すと係数自体が偏る。区間をいくら正しく作っても中心がずれている
- **等分散性・独立性**を壊しても係数は偏らない。壊れるのは標準誤差のほう＝区間の幅
- **正規性**を壊しても n=100 では区間はほぼ持ちこたえる（中心極限定理）

つまり優先順位がある。まず線形性、次に独立性、その次に等分散性で、正規性は最後である。
残差のヒストグラムだけを睨むのは、優先順位を逆から見ていることになる。

    uv run python examples/ch12/assumption_breakage_table.py
"""

import unicodedata

import numpy as np
from scipy import stats

from toukei_tashikame import plots

N, TRIALS, SEED = 100, 10_000, 124
B0, B1 = 1.0, 1.0
PHI = 0.7          # 独立性を壊すときの AR(1) 係数
CURVE = 0.15       # 線形性を壊すときの2次の係数
CHUNK = 1_000


def ar1(trials: int, n: int, phi: float, rng) -> np.ndarray:
    """周辺分散が 1 になる AR(1) 系列を (trials, n) で作る。"""
    u = rng.normal(0.0, 1.0, size=(trials, n))
    out = np.empty_like(u)
    out[:, 0] = u[:, 0]
    for t in range(1, n):
        out[:, t] = phi * out[:, t - 1] + np.sqrt(1.0 - phi**2) * u[:, t]
    return out


def _width(s: str) -> int:
    """全角を2桁と数えた表示幅。表の桁を日本語のまま揃えるためだけの補助。"""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def make(condition: str, trials: int, rng):
    """条件ごとに (x, y) を trials 本まとめて作る。真の b1 はどれも B1。"""
    x = rng.normal(1.0, 1.0, size=(trials, N))
    eps = rng.normal(0.0, 1.0, size=(trials, N))
    if condition == "等分散性の破れ":
        # 誤差の大きさが x に比例する（右へ行くほど散らばる、よくある形）
        eps = eps * (0.3 + np.abs(x))
    elif condition == "独立性の破れ":
        # 時系列データ。x も誤差も前の時点を引きずる
        x = 1.0 + ar1(trials, N, PHI, rng)
        eps = ar1(trials, N, PHI, rng)
    elif condition == "正規性の破れ":
        # 平均0・分散1にそろえた指数分布（右に強く歪む）
        eps = rng.exponential(1.0, size=(trials, N)) - 1.0
    y = B0 + B1 * x + eps
    if condition == "線形性の破れ":
        y = y + CURVE * x**2      # 真は曲線なのに直線を当てる
    return x, y


def fit(x: np.ndarray, y: np.ndarray):
    """単回帰の (b1, 95%区間の下限, 上限) をまとめて返す。"""
    xc = x - x.mean(axis=1, keepdims=True)
    yc = y - y.mean(axis=1, keepdims=True)
    sxx = (xc * xc).sum(axis=1)
    b1 = (xc * yc).sum(axis=1) / sxx
    rss = (yc * yc).sum(axis=1) - b1 * (xc * yc).sum(axis=1)
    se = np.sqrt(rss / (N - 2) / sxx)
    half = stats.t.ppf(0.975, N - 2) * se
    return b1, b1 - half, b1 + half


CONDITIONS = ("全仮定が成立", "等分散性の破れ", "独立性の破れ", "線形性の破れ", "正規性の破れ")


def main() -> None:
    plots.setup()
    rng = np.random.default_rng(SEED)

    print(f"--- 真の b1 = {B1}、n={N}、各条件 {TRIALS:,} 回 ---")
    print("  条件              b1の平均   バイアス   b1の標準偏差   平均の区間幅   被覆確率")
    table = {}
    for cond in CONDITIONS:
        b1s, los, his = [], [], []
        for i in range(0, TRIALS, CHUNK):
            x, y = make(cond, min(CHUNK, TRIALS - i), rng)
            b, lo, hi = fit(x, y)
            b1s.append(b)
            los.append(lo)
            his.append(hi)
        b1 = np.concatenate(b1s)
        lo, hi = np.concatenate(los), np.concatenate(his)
        cover = float(((lo <= B1) & (B1 <= hi)).mean())
        se_cover = np.sqrt(cover * (1 - cover) / TRIALS)
        table[cond] = (b1, cover, se_cover)
        pad = " " * (16 - _width(cond))
        print(f"  {cond}{pad}{b1.mean():.4f}    {b1.mean() - B1:+.4f}"
              f"       {b1.std(ddof=1):.4f}        {(hi - lo).mean():.4f}"
              f"       {cover:.4f} ± {1.96 * se_cover:.4f}")

    print("\n  係数が偏るのは線形性を壊したときだけ。他の3つは平均としては真値に当たる。")
    print("  壊れるのは区間のほうで、独立性の破れが最も重い（幅が狭すぎる＝自信を持ちすぎる）。")
    print("  正規性の破れは n=100 ではほとんど効かない。よく点検される割に、効きは小さい。")

    # --- 図 ---
    fig, axes = plots.figure(1, 2, w=2.0, h=1.0)
    pal = plots.PALETTE

    ax = axes[0]
    names = list(CONDITIONS)
    covers = [table[c][1] for c in names]
    errs = [1.96 * table[c][2] for c in names]
    ypos = np.arange(len(names))[::-1]
    ax.barh(ypos, covers, height=0.55, color=pal["estimate"], alpha=0.75,
            xerr=errs, error_kw={"lw": 0.8, "ecolor": pal["ink2"]})
    plots.mark_truth(ax, 0.95, "名目 95%")
    for yp, c in zip(ypos, covers, strict=True):
        ax.annotate(f"{c:.4f}", xy=(c, yp), xytext=(3, 0), textcoords="offset points",
                    va="center", fontsize=6.0, color=pal["ink2"])
    ax.set_yticks(ypos, names, fontsize=6.2)
    ax.set_xlim(0.0, 1.08)
    ax.set_xlabel("95%区間が真値を包んだ割合")
    ax.set_title("① 区間はどの破れで壊れるか")

    ax = axes[1]
    bins = np.linspace(0.5, 1.8, 70)
    for cond, color, alpha in [("全仮定が成立", pal["estimate"], 0.55),
                               ("線形性の破れ", pal["truth"], 0.45)]:
        ax.hist(table[cond][0], bins=bins, color=color, alpha=alpha, lw=0)
        ax.annotate(f"{cond} {table[cond][0].mean():.3f}",
                    xy=(table[cond][0].mean(), 0.9 if cond == "全仮定が成立" else 0.75),
                    xycoords=("data", "axes fraction"), xytext=(3, 0),
                    textcoords="offset points", fontsize=6.0, color=color)
    plots.mark_truth(ax, B1, f"真値 {B1}")
    ax.set_xlabel("$\\hat{b}_1$")
    ax.set_ylabel(f"{TRIALS:,} 回のうちの回数")
    ax.set_title("② 線形性の破れだけが中心をずらす")

    fig.tight_layout()
    plots.save(fig, "fig-12-4-assumption-breakage.png")


if __name__ == "__main__":
    main()
