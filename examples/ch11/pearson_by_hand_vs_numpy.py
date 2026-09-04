"""相関係数を共分散から組み立てる — 標準化しているだけだと確かめる。

Pearson の相関係数は「共分散を、2つの標準偏差で割ったもの」でしかない。式を写す
かわりに numpy で3行に分解して組み立て、``np.corrcoef`` と ``scipy.stats.pearsonr``
に 1,000 標本ぶん突き合わせる。差が浮動小数の丸め（1e-16 の桁）に収まるなら、
ライブラリがやっているのは本当にこの3行だと言ってよい。

そのうえで単位を変える。x をメートルからミリメートルに直すと共分散は 1,000 倍に
なるが、相関は1桁も動かない。共分散の大きさに意味を読もうとしても読めない理由が
ここにある——相関は共分散から単位を割り落とした残りである。

    uv run python examples/ch11/pearson_by_hand_vs_numpy.py
"""

import numpy as np
from scipy import stats

from toukei_tashikame import sim

N, RHO, SEED, TRIALS = 100, 0.7, 110, 1_000


def cov_by_hand(x: np.ndarray, y: np.ndarray) -> float:
    """共分散。「偏差の積の平均」を ddof=1 で。"""
    n = x.size
    return float(((x - x.mean()) * (y - y.mean())).sum() / (n - 1))


def pearson_by_hand(x: np.ndarray, y: np.ndarray) -> float:
    """共分散を2つの標準偏差で割る。これが標準化の中身。"""
    return cov_by_hand(x, y) / (x.std(ddof=1) * y.std(ddof=1))


def bivariate(rng: np.random.Generator, n: int = N, rho: float = RHO):
    """相関 rho の2変量正規を1組。x に独立成分を足して y を作る。"""
    x = rng.normal(size=n)
    y = rho * x + np.sqrt(1 - rho**2) * rng.normal(size=n)
    return x, y


def three_ways(rng: np.random.Generator) -> tuple[float, float, float]:
    """同じ標本に3つの実装をかけ、3つの r を並べて返す。"""
    x, y = bivariate(rng)
    return (
        pearson_by_hand(x, y),
        float(np.corrcoef(x, y)[0, 1]),
        float(stats.pearsonr(x, y).statistic),
    )


def main() -> None:
    print(f"--- 11-1 共分散から相関へ（n={N}, 真の相関 ρ={RHO}, seed={SEED}）---")
    rng = np.random.default_rng(SEED)
    x, y = bivariate(rng)

    sxy = cov_by_hand(x, y)
    r_hand = pearson_by_hand(x, y)
    print(f"  共分散 s_xy      {sxy: .6f}   ← 単位は (x の単位)×(y の単位)")
    print(f"  標準偏差 s_x     {x.std(ddof=1): .6f}")
    print(f"  標準偏差 s_y     {y.std(ddof=1): .6f}")
    print(f"  手書き r         {r_hand: .6f}   ← s_xy / (s_x s_y)")
    print(f"  np.corrcoef      {np.corrcoef(x, y)[0, 1]: .6f}")
    print(f"  scipy.pearsonr   {stats.pearsonr(x, y).statistic: .6f}")

    print("\n--- 単位を変える（x をメートル → ミリメートル、つまり 1000 倍）---")
    xm = x * 1000.0
    print(f"  共分散  もとの単位 {sxy: .6f}  →  変換後 {cov_by_hand(xm, y): .6f}"
          f"（{cov_by_hand(xm, y) / sxy:.1f} 倍）")
    print(f"  相関    もとの単位 {r_hand: .12f}  →  変換後 {pearson_by_hand(xm, y): .12f}"
          f"（差 {abs(pearson_by_hand(xm, y) - r_hand):.3e}）")
    print("  ← 共分散は単位を持つので、値の大小をそのまま「関係の強さ」と読めない。")
    print("    2つの標準偏差で割って単位を消したものが相関で、これは常に [-1, 1] に入る。")

    print(f"\n--- {TRIALS:,} 標本で3実装を突き合わせる ---")
    rs = sim.repeat(three_ways, trials=TRIALS, seed=SEED, progress=False)
    d_numpy = np.abs(rs[:, 0] - rs[:, 1])
    d_scipy = np.abs(rs[:, 0] - rs[:, 2])
    print(f"  手書き r の平均            {rs[:, 0].mean():.6f}（真値 {RHO}）")
    print(f"  手書き r の標準偏差        {rs[:, 0].std(ddof=1):.6f}"
          f"（理論 {(1 - RHO**2) / np.sqrt(N - 1):.6f}）")
    print(f"  手書き vs np.corrcoef      最大絶対差 {d_numpy.max():.3e}")
    print(f"  手書き vs scipy.pearsonr   最大絶対差 {d_scipy.max():.3e}")
    print(f"  倍精度の刻み eps           {np.finfo(float).eps:.3e}")
    print("  ← 差は丸めの桁。3つは同じ計算をしている")


if __name__ == "__main__":
    main()
