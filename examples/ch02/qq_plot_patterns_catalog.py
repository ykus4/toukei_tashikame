"""Q-Qプロットの曲がり方のカタログ。歪み・裾の重さ・二峰を、形で見分ける。

Q-Qプロットは「正規かどうか」の○×装置ではない。外れているときに**どう**外れて
いるかが形に出るのが本体である。右に反れば右歪み、両端が持ち上がれば裾が重い、
S字なら二峰。真の分布を知っている4種を並べて、形と原因を対応づける。

    uv run python examples/ch02/qq_plot_patterns_catalog.py
"""

import numpy as np
from scipy import stats

from toukei_tashikame import datasets, describe, plots

N, SEED = 200, 4

CASES = {
    "正規 N(0,1)": (lambda: datasets.normal_sample(N, mu=0.0, sigma=1.0, seed=SEED),
                    "直線に乗る"),
    "右歪み 対数正規": (lambda: datasets.skewed_sample(N, kind="lognormal", seed=SEED),
                   "右端が上に跳ねる"),
    "裾重 t(3)": (lambda: datasets.heavy_tailed(N, kind="t3", seed=SEED),
                "両端が反対向きに外れる"),
    "二峰 混合正規": (lambda: datasets.bimodal(N, sep=4.0, seed=SEED),
                 "中央が横に寝てS字"),
}


def tail_gap(x: np.ndarray, frac: float = 0.05) -> tuple[float, float]:
    """両端 ``frac`` の点が、当てはめた直線からどれだけ外れているか（SD 単位）。

    Q-Qプロットで目が拾っている「反り」を数にする。左端の平均残差と右端の平均残差
    を返す。歪みなら片側だけ、裾の重さなら両側が符号違いで大きくなる。
    """
    xs = np.sort(np.asarray(x, dtype=float))
    n = xs.size
    p = (np.arange(1, n + 1) - 0.375) / (n + 0.25)     # Blom のプロット位置
    q = stats.norm.ppf(p)
    slope, intercept = np.polyfit(q, xs, 1)
    resid = (xs - (slope * q + intercept)) / slope     # x 軸と同じ単位に戻す
    k = max(round(n * frac), 1)
    return float(resid[:k].mean()), float(resid[-k:].mean())


def main() -> None:
    plots.setup()
    print("--- 2-6 Q-Qプロットの曲がり方 ---")
    print(f"各 n={N}、seed={SEED}。真の分布はこちらが知っている。")

    print(f"\n{'ケース':<16}{'歪度':>8}{'超過尖度':>10}{'左端の反り':>11}"
          f"{'右端の反り':>11}{'Shapiro-Wilk p':>16}")
    samples = {}
    for label, (make, _) in CASES.items():
        x = make()
        samples[label] = x
        lo, hi = tail_gap(x)
        p = float(stats.shapiro(x).pvalue)
        p_txt = f"{p:.4f}" if p >= 1e-4 else f"{p:.1e}"
        print(f"{label:<16}{describe.skewness(x):>8.3f}{describe.kurtosis(x):>10.3f}"
              f"{lo:>11.3f}{hi:>11.3f}{p_txt:>16}")

    print("\n  形と原因の対応:")
    for label, (_, shape) in CASES.items():
        print(f"    {label:<16} {shape}")
    print("\n  ← 正規だけが p > 0.05 で、残り3つは棄却される。だが「棄却された」から"
          "\n    分かることは何もない。どちらに反っているかを見て初めて、"
          "\n    対数変換すべきか・ロバストな方法に替えるべきか・層別すべきかが決まる")

    print("\n  n を変えると Shapiro-Wilk の答えは変わる（同じ t(3) で）:")
    for n in (20, 50, 200, 1000):
        x = datasets.heavy_tailed(n, kind="t3", seed=SEED)
        p = float(stats.shapiro(x).pvalue)
        verdict = "棄却しない" if p >= 0.05 else "棄却"
        print(f"    n={n:>5}  p={p:.2e}  → {verdict}")
    print("  ← 分布は同じなのに、n が増えれば必ず棄却される。"
          "検定は「ずれの大きさ」ではなく\n    「ずれを見分けられるか」を答えている。だから図を見る")

    fig, axes = plots.figure(2, 2, h=1.7, w=1.6)
    for ax, (label, x) in zip(axes.ravel(), samples.items(), strict=True):
        plots.qq(ax, x)
        lo, hi = tail_gap(x)
        ax.set_title(f"{label}   左{lo:+.2f} / 右{hi:+.2f}")
    plots.save(fig, "fig-2-6-qq-patterns.png")


if __name__ == "__main__":
    main()
