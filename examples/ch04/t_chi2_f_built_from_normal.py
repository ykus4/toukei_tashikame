"""t・カイ二乗・F は、正規乱数だけを組み合わせて作れる。定義どおりに200,000本組む。

第7章以降で出てくる3つの分布は、天から降ってくる表ではなく、正規乱数の組み合わせに
名前をつけたものである。

    $\\chi^2(k) = \\sum_{i=1}^{k} Z_i^2$
    $t(k) = Z / \\sqrt{V/k}$            （$V \\sim \\chi^2(k)$、$Z$ と独立）
    $F(k_1, k_2) = (V_1/k_1) / (V_2/k_2)$

``scipy.stats`` を一切使わずに右辺を200,000回組み立て、できあがった標本が
``stats.chi2 / t / f`` と一致するかを KS 検定で突き合わせる。一致すれば、t 検定の
「t 分布表」も分散分析の「F 表」も、正規乱数の足し算と割り算の言い換えでしかない
ことになる。

最後に、この3つが実際のデータ解析でどこから現れるかも1つだけ見ておく。
$\\mathcal{N}(\\mu,\\sigma^2)$ から n 個引いたときの $(n-1)s^2/\\sigma^2$ が $\\chi^2(n-1)$、
$\\sqrt{n}(\\bar{x}-\\mu)/s$ が $t(n-1)$ になる——σ を s で置き換えた代償が t 分布である。

    uv run python examples/ch04/t_chi2_f_built_from_normal.py
"""

import numpy as np
from scipy import stats

from toukei_tashikame import plots

DRAWS = 200_000
K = 5           # t と chi2 の自由度
K1, K2 = 5, 10  # F の分子・分母の自由度
N_SAMPLE = 6    # 「標本から現れる」ほうの標本サイズ（自由度 N_SAMPLE-1 = K）
SEED = 16


def build_from_normal(rng) -> dict[str, np.ndarray]:
    """正規乱数だけから chi2 / t / F を定義どおりに組み立てる。scipy は使わない。"""
    # chi2(k) = Z_1^2 + ... + Z_k^2
    v = (rng.normal(size=(DRAWS, K)) ** 2).sum(axis=1)

    # t(k) = Z / sqrt(V/k)。分子の Z は分母の V と独立に引き直す。
    z = rng.normal(size=DRAWS)
    v_for_t = (rng.normal(size=(DRAWS, K)) ** 2).sum(axis=1)
    t = z / np.sqrt(v_for_t / K)

    # F(k1, k2) = (V1/k1) / (V2/k2)
    v1 = (rng.normal(size=(DRAWS, K1)) ** 2).sum(axis=1)
    v2 = (rng.normal(size=(DRAWS, K2)) ** 2).sum(axis=1)
    f = (v1 / K1) / (v2 / K2)

    return {"chi2": v, "t": t, "f": f}


def compare(built: dict[str, np.ndarray]) -> dict[str, object]:
    """組み立てた標本と scipy の分布を KS 検定で突き合わせる。"""
    targets = {
        "chi2": (f"chi2({K})", stats.chi2(df=K)),
        "t": (f"t({K})", stats.t(df=K)),
        "f": (f"f({K1}, {K2})", stats.f(dfn=K1, dfd=K2)),
    }
    print(f"--- 正規乱数だけで組んだ {DRAWS:,} 本 vs scipy.stats ---")
    print(f"{'作ったもの':<14}{'突き合わせ先':<16}{'KS距離':>10}{'p値':>10}"
          f"{'実測の平均':>12}{'理論':>10}")
    out = {}
    for key, (name, dist) in targets.items():
        x = built[key]
        res = stats.kstest(x, dist.cdf)
        print(f"{key:<14}{name:<16}{res.statistic:>10.4f}{res.pvalue:>10.4f}"
              f"{x.mean():>12.4f}{dist.mean():>10.4f}")
        out[key] = (name, dist, res)
    print(f"  KS距離の目安は 1/√n = {1 / np.sqrt(DRAWS):.4f}。"
          "3つとも p が大きく、「別の分布だ」とは言えない")
    print("  ← 定義式の右辺を組み立てただけで、表と同じ分布が出てくる")
    return out


def where_they_come_from(rng) -> None:
    """標本から実際に現れる形。σ を s に置き換えた代償が t 分布になる。"""
    x = rng.normal(loc=50.0, scale=10.0, size=(DRAWS, N_SAMPLE))
    xbar = x.mean(axis=1)
    s = x.std(axis=1, ddof=1)
    df = N_SAMPLE - 1

    scaled_var = df * s**2 / 10.0**2               # (n-1)s^2/σ^2
    z_stat = np.sqrt(N_SAMPLE) * (xbar - 50.0) / 10.0   # σ を知っている場合
    t_stat = np.sqrt(N_SAMPLE) * (xbar - 50.0) / s      # σ を s で置き換えた場合

    print(f"\n--- N(50, 10^2) から n={N_SAMPLE} を {DRAWS:,} 回引くと、そこから現れる ---")
    for label, value, dist in (
        (f"(n−1)s²/σ²   → chi2({df})", scaled_var, stats.chi2(df=df)),
        (f"√n(x̄−μ)/s    → t({df})", t_stat, stats.t(df=df)),
    ):
        res = stats.kstest(value, dist.cdf)
        print(f"  {label:<26} KS距離 {res.statistic:.4f}  p = {res.pvalue:.4f}")

    print(f"\n  σ を知っている √n(x̄−μ)/σ : SD {z_stat.std(ddof=1):.4f}"
          f"（理論 1.0000）、|値|>1.96 の割合 {(np.abs(z_stat) > 1.96).mean():.4f}")
    print(f"  σ を s で置き換えた √n(x̄−μ)/s : SD {t_stat.std(ddof=1):.4f}"
          f"（理論 {np.sqrt(df / (df - 2)):.4f}）、|値|>1.96 の割合 "
          f"{(np.abs(t_stat) > 1.96).mean():.4f}")
    print(f"  1.96 で切ると {(np.abs(t_stat) > 1.96).mean():.4f} が外に出る。"
          f"5% で切りたいなら t の分位点 {stats.t.ppf(0.975, df):.4f} を使う")
    print("  ← s も標本ごとに動く。その「ばらつきの分のばらつき」が t の裾の重さ")

    print("\n  自由度を上げると t は正規に近づく（|値|>1.96 の理論値）:")
    for k in (2, 5, 10, 30, 100, 1000):
        print(f"    t({k:>4}) {2 * stats.t.sf(1.96, k):.4f}   "
              f"上側2.5%点 {stats.t.ppf(0.975, k):.4f}")
    print(f"    正規      {2 * stats.norm.sf(1.96):.4f}   上側2.5%点 "
          f"{stats.norm.ppf(0.975):.4f}")


def make_figure(built: dict[str, np.ndarray], compared: dict[str, object]) -> None:
    plots.setup()
    fig, axes = plots.figure(1, 3, w=1.0, h=0.72, constrained_layout=True)

    ranges = {"chi2": (0.0, 20.0), "t": (-5.0, 5.0), "f": (0.0, 5.0)}
    for ax, key in zip(axes, ("chi2", "t", "f"), strict=True):
        name, dist, res = compared[key]
        lo, hi = ranges[key]
        x = built[key]
        grid = np.linspace(lo, hi, 400)
        plots.sim_hist(ax, x[(x >= lo) & (x <= hi)], theory=(grid, dist.pdf(grid)),
                       bins=70, theory_label=f"{name} の pdf")
        ax.set_xlim(lo, hi)
        ax.set_xlabel("値")
        ax.set_title(f"KS距離 {res.statistic:.4f} (p={res.pvalue:.2f})")
    axes[0].set_ylabel("密度")

    plots.save(fig, "fig-4-8-built-from-normal.png")


def main() -> None:
    rng = np.random.default_rng(SEED)
    built = build_from_normal(rng)
    compared = compare(built)
    where_they_come_from(rng)
    make_figure(built, compared)


if __name__ == "__main__":
    main()
