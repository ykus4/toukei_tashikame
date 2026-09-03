"""検出力とサンプルサイズ設計。

検出力は「対立仮説が本当なら、どれくらいの割合で棄却できるか」。第一種の誤りと同じ
手続きで数えられる——違うのはデータの生成側で帰無が真かどうかだけである。だから
:func:`power_sim` は ``sim.rejection_rate`` をそのまま呼ぶ。

解析解（:func:`power_ttest`）を併置してあるのは、数え上げと式が一致することを毎回
確かめられるようにするため。式だけだと近似の質が見えず、数え上げだけだと式の意味が
身につかない。

**検出力の不足は、有意にならないことだけの問題ではない。** 検出力が低い設計で有意に
なった効果量は、系統的に過大になる（:func:`winners_curse`）。第9章の落とし穴の根は
ほとんどここにある。
"""

from __future__ import annotations

import numpy as np
from scipy import optimize, stats

from . import sim

__all__ = [
    "mde", "n_for_power", "n_for_proportions", "power_curve", "power_sim",
    "power_ttest", "winners_curse",
]


def power_ttest(n: int, d: float, alpha: float = 0.05, kind: str = "two-sample") -> float:
    """t 検定の検出力（解析解）。非心 t 分布の裾を積む。

    ``kind`` は ``two-sample``（各群 n）/ ``one-sample`` / ``paired``。
    ``d`` は Cohen の d。
    """
    if kind == "two-sample":
        df = 2 * (n - 1)
        ncp = d * np.sqrt(n / 2)
    elif kind in ("one-sample", "paired"):
        df = n - 1
        ncp = d * np.sqrt(n)
    else:
        raise ValueError(f"unknown kind: {kind}")
    crit = stats.t.ppf(1 - alpha / 2, df)
    # 両側。反対側の裾も足すが、実用上は片方が支配する。
    value = stats.nct.sf(crit, df, ncp) + stats.nct.cdf(-crit, df, ncp)
    if np.isfinite(value):
        return float(value)
    # scipy の非心 t は ncp と df が大きいと nan を返す。そこは正規近似で十分に
    # 精確な領域（df が大きいほど t は正規に近い）なので、素直に切り替える。
    # ここで nan を返すと、n_for_power の二分探索が「未達」と読んで壊れる。
    z = stats.norm.ppf(1 - alpha / 2)
    return float(stats.norm.sf(z - ncp) + stats.norm.cdf(-z - ncp))


def power_sim(n: int, d: float, alpha: float = 0.05, trials: int = 10_000,
              seed: int = 0, equal_var: bool = False):
    """検出力を数え上げで求める。``sim.rejection_rate`` をそのまま使う。

    第一種の誤りを数えるコードと1文字も違わない。違うのは ``d`` が 0 かどうかだけで、
    そこが「検出力と第一種の誤りは同じ手続きの裏表」という話の実装上の姿である。
    """

    def pvalue(rng):
        a = rng.normal(0.0, 1.0, size=n)
        b = rng.normal(d, 1.0, size=n)
        return float(stats.ttest_ind(a, b, equal_var=equal_var).pvalue)

    return sim.rejection_rate(pvalue, alpha=alpha, trials=trials, seed=seed, progress=False)


def n_for_power(d: float, power: float = 0.8, alpha: float = 0.05,
                kind: str = "two-sample", n_max: int = 100_000) -> int:
    """目標の検出力に届く最小の n を二分探索で求める。

    返すのは ``kind="two-sample"`` なら**各群の** n。総数ではない。
    """
    if power_ttest(n_max, d, alpha, kind) < power:
        raise ValueError(f"n={n_max} でも検出力 {power} に届かない（d={d} が小さすぎる）")
    lo, hi = 2, n_max
    while lo < hi:
        mid = (lo + hi) // 2
        if power_ttest(mid, d, alpha, kind) >= power:
            hi = mid
        else:
            lo = mid + 1
    return int(lo)


def n_for_proportions(p1: float, p2: float, power: float = 0.8, alpha: float = 0.05,
                      ratio: float = 1.0) -> int:
    """2群の比率を比べるのに要る n（群1あたり）。A/Bテストの設計に使う。

    ``ratio`` は n2/n1。プールした分散と各群の分散を両方使う標準的な式で、
    3.0% を 3.3% にする（相対10%）程度の差にどれだけ要るかが、これで出る。
    """
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    p_bar = (p1 + ratio * p2) / (1 + ratio)
    num = (z_a * np.sqrt((1 + 1 / ratio) * p_bar * (1 - p_bar))
           + z_b * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2) / ratio)) ** 2
    return int(np.ceil(num / (p1 - p2) ** 2))


def power_curve(ns, d: float, alpha: float = 0.05, kind: str = "two-sample"):
    """n を振ったときの検出力の曲線。"""
    import pandas as pd

    return pd.DataFrame({
        "n": list(ns),
        "power": [power_ttest(int(n), d, alpha, kind) for n in ns],
    })


def mde(n: int, power: float = 0.8, alpha: float = 0.05, kind: str = "two-sample") -> float:
    """最小検出可能効果。「この n で拾えるのはどれくらいの効果までか」。

    サンプルサイズ設計を逆から見たもの。n が先に決まっている（実験期間が決まっている、
    ユーザー数が動かせない）ときは、こちらが実務的な問いになる。
    """
    def gap(d):
        return power_ttest(n, d, alpha, kind) - power

    return float(optimize.brentq(gap, 1e-6, 10.0))


def winners_curse(n: int, d_true: float, alpha: float = 0.05, trials: int = 10_000,
                  seed: int = 0) -> dict:
    """有意になった試行**だけ**を集めたときの効果量の平均。

    検出力が低いほど、有意になるのは「たまたま大きく出た」試行だけになる。だから
    発表される効果量は系統的に過大になる。これは不正でも p ハッキングでもなく、
    足切りをした標本の性質そのものである。

    返り値は真値・全試行の平均・有意だった試行だけの平均・その比・検出力。
    """
    def one(rng):
        a = rng.normal(0.0, 1.0, size=n)
        b = rng.normal(d_true, 1.0, size=n)
        res = stats.ttest_ind(a, b, equal_var=False)
        sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        return float(res.pvalue), float((b.mean() - a.mean()) / sp)

    out = sim.repeat(one, trials=trials, seed=seed, progress=False)
    p, d_hat = out[:, 0], out[:, 1]
    sig = p < alpha
    return {
        "d_true": d_true,
        "d_all": float(d_hat.mean()),
        "d_significant": float(d_hat[sig].mean()) if sig.any() else float("nan"),
        "inflation": float(d_hat[sig].mean() / d_true) if sig.any() and d_true else float("nan"),
        "power": float(sig.mean()),
        "n_significant": int(sig.sum()),
    }
