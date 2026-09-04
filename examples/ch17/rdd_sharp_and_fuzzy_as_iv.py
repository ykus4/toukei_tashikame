"""閾値のまわりだけを見る — シャープ RDD と、遵守率が下がったときのファジー RDD。

スコア 50 で線を引き、上なら適用・下なら非適用とする制度を作る。49.9 の人と 50.1 の人は
ほとんど同じ人なのに扱いが違う。この「ほとんど同じ」を使って効果を測るのが RDD で、
仕込んだ段差は 1.0 である。

帯域幅を変えると答えが動く。狭くすれば偏りは小さいが区間が広がり、広げれば区間は
狭まるが曲線の形を段差と読み違える。**帯域を1つだけ報告する RDD は読めない。**

閾値を越えても全員が実際に処置を受けるとは限らない。この場合の RDD は「閾値を越えたか」
を操作変数とする局所 IV（Wald 比）になる。遵守率が低いと第一段が弱くなり、
2SLS の偏りは OLS の偏りより大きくなりうる——第一段の F を必ず見る理由である。

    uv run python examples/ch17/rdd_sharp_and_fuzzy_as_iv.py
"""

import numpy as np

from toukei_tashikame import causal, plots, sim

N, CUTOFF, JUMP, SEED = 3000, 50.0, 1.0, 179
CURVE = 0.006          # 閾値より上だけに入れた曲がり。広い帯域で「段差」に化ける
BANDWIDTHS = (5.0, 10.0, 25.0)   # 走行変数の範囲（±50）に対して 0.1 / 0.2 / 0.5
REPS = 400             # ファジー版を回す回数


def sharp_data(rng, n: int = N):
    """閾値をまたぐと全員が処置される世界（シャープ）。"""
    x = rng.uniform(0.0, 100.0, size=n)
    d = (x >= CUTOFF).astype(float)
    base = 20.0 + 0.05 * (x - CUTOFF) + CURVE * (x - CUTOFF) ** 2 * (x >= CUTOFF)
    y = base + JUMP * d + rng.normal(0.0, 1.0, size=n)
    return x, d, y


def fuzzy_data(rng, p_low: float, p_high: float, n: int = N):
    """閾値を越えても一部しか処置されない世界（ファジー）。

    未観測の熱心さ $U$ が「処置を受けるか」と「結果」の両方を押し上げる。だから
    素朴に $Y \\sim D$ と回帰すると上に偏る。閾値をまたぐことだけが外生な変動である。
    """
    x = rng.uniform(0.0, 100.0, size=n)
    u = rng.normal(0.0, 1.0, size=n)                     # 未観測の交絡
    p = np.where(x >= CUTOFF, p_high, p_low) + 0.15 * (u > 0)
    d = (rng.random(n) < p).astype(float)
    base = 20.0 + 0.05 * (x - CUTOFF)
    y = base + JUMP * d + 1.0 * u + rng.normal(0.0, 1.0, size=n)
    return x, d, y


def wald_ratio(x, d, y, bandwidth: float):
    """ファジー RDD ＝ 局所 IV。「結果の段差 ÷ 処置率の段差」が Wald 比。"""
    first = causal.rdd(x, d, cutoff=CUTOFF, bandwidth=bandwidth)     # 第一段
    reduced = causal.rdd(x, y, cutoff=CUTOFF, bandwidth=bandwidth)   # 誘導形
    est = reduced.estimate / first.estimate
    f_stat = (first.estimate / first.se) ** 2                        # 第一段の F（= t²）
    # デルタ法。比の分散は、分子と分母の相対誤差を足したもの。
    se = abs(est) * np.sqrt((reduced.se / reduced.estimate) ** 2
                            + (first.se / first.estimate) ** 2)
    return est, se, float(f_stat), first.estimate, reduced.estimate


def naive_ols(x, d, y, bandwidth: float) -> float:
    """帯域内で素朴に Y ~ D と回すとどうなるか（比較用）。"""
    keep = np.abs(x - CUTOFF) <= bandwidth
    return float(y[keep][d[keep] == 1].mean() - y[keep][d[keep] == 0].mean())


def draw(x, y, results) -> None:
    fig, axes = plots.figure(1, 2, w=2.0)
    pal = plots.PALETTE

    ax = axes[0]
    edges = np.arange(30.0, 70.1, 2.0)     # 2点刻みのビン平均。生の散布は密すぎる
    idx = np.digitize(x, edges) - 1
    ok = (idx >= 0) & (idx < len(edges) - 1)
    centers = (edges[:-1] + edges[1:]) / 2
    means = np.array([y[ok & (idx == i)].mean() for i in range(len(edges) - 1)])
    ax.scatter(centers, means, s=10, color=pal["data"], lw=0, zorder=3)
    edge = {}
    for side in (0, 1):
        keep = (np.abs(x - CUTOFF) <= BANDWIDTHS[1]) & ((x >= CUTOFF) == bool(side))
        b = np.polyfit(x[keep], y[keep], 1)
        xs = np.linspace(x[keep].min(), x[keep].max(), 50)
        ax.plot(xs, np.polyval(b, xs), color=pal["estimate"], lw=1.3, zorder=4)
        edge[side] = float(np.polyval(b, CUTOFF))
    gap = edge[1] - edge[0]
    ax.plot([CUTOFF, CUTOFF], [edge[0], edge[1]], color=pal["truth"], lw=1.6, zorder=5)
    ax.axvline(CUTOFF, color=pal["reject"], lw=0.9, ls="--", dashes=(3, 2.0), zorder=2)
    ax.annotate(f"閾値 {CUTOFF:g}", xy=(CUTOFF, 0.03), xycoords=("data", "axes fraction"),
                fontsize=6.0, color=pal["reject"], xytext=(3, 0), textcoords="offset points")
    ax.annotate(f"帯域 ±{BANDWIDTHS[1]:g} での段差 {gap:.2f}\n（真値は {JUMP:g}）",
                xy=(CUTOFF, (edge[0] + edge[1]) / 2), fontsize=6.2, color=pal["truth"],
                ha="right", va="center", xytext=(-5, 0), textcoords="offset points")
    ax.set_xlim(30, 70)
    ax.set_xlabel("走行変数（スコア）")
    ax.set_ylabel("結果 Y のビン平均")

    ax = axes[1]
    pos = np.arange(len(results))
    for i, (_h, est, se, n_eff) in enumerate(results):
        ax.plot([est - 1.96 * se, est + 1.96 * se], [i, i], color=pal["interval"], lw=1.6,
                solid_capstyle="butt", zorder=3)
        ax.scatter(est, i, s=16, color=pal["estimate"], zorder=4, lw=0)
        ax.annotate(f"n={n_eff:,}", xy=(est, i), fontsize=6.0, color=pal["ink2"],
                    ha="center", va="bottom", xytext=(0, 5), textcoords="offset points")
    plots.mark_truth(ax, JUMP, f"真の段差 = {JUMP:g}")
    ax.set_yticks(pos)
    ax.set_yticklabels([f"帯域 ±{h:g}" for h, *_ in results], fontsize=6.2)
    ax.set_ylim(-0.6, len(results) - 0.4)
    ax.set_xlabel("シャープ RDD の推定値と 95%CI")
    fig.tight_layout()
    plots.save(fig, "fig-17-9-rdd-discontinuity.png")


def main() -> None:
    plots.setup()
    rng = np.random.default_rng(SEED)
    x, _, y = sharp_data(rng)

    print(f"--- シャープ RDD（n={N:,}, 閾値 {CUTOFF:g}, 真の段差 {JUMP:g}, seed={SEED}）---\n")
    print(f"{'帯域幅':<20}{'推定値':>9}{'SE':>8}{'95%CI':>22}{'有効標本':>10}")
    results = []
    for h in BANDWIDTHS:
        res = causal.rdd(x, y, cutoff=CUTOFF, bandwidth=h)
        n_eff = int((np.abs(x - CUTOFF) <= h).sum())
        lo, hi = res.ci
        frac = h / 50.0
        print(f"{f'±{h:g}（範囲の {frac:.1f}）':<20}{res.estimate:>9.3f}{res.se:>8.3f}"
              f"   [{lo:>6.3f}, {hi:>6.3f}]{n_eff:>10,}")
        results.append((h, res.estimate, res.se, n_eff))
    full = causal.rdd(x, y, cutoff=CUTOFF)
    print(f"{'全データ（帯域なし）':<20}{full.estimate:>9.3f}{full.se:>8.3f}"
          f"   [{full.ci[0]:>6.3f}, {full.ci[1]:>6.3f}]{N:>10,}")

    print("\n  帯域を広げると SE は縮み、推定値は真値から離れる。これが偏りと分散の綱引きで、")
    print("  ここでは閾値の上側だけに曲がりを仕込んである。直線を当てると、その曲がりの一部が")
    print("  段差として読み取られる。曲線の形を段差と誤読するのが RDD の典型的な失敗である。")
    print(f"  帯域 ±{BANDWIDTHS[0]:g} と ±{BANDWIDTHS[-1]:g} で推定値は "
          f"{results[0][1]:.3f} → {results[-1][1]:.3f} と動く。どれを報告するかで結論が変わる。")

    print(f"\n--- ファジー RDD ＝ 局所 IV（帯域 ±{BANDWIDTHS[1]:g}, {REPS} 回ずつ）---\n")
    print(f"{'設計':<28}{'第一段の段差':>12}{'第一段F':>10}{'Wald比の中央値':>15}{'同 平均':>10}{'素朴OLS':>9}")
    designs = (("遵守率が低い（0.10 → 0.25）", 0.10, 0.25),
               ("遵守率が高い（0.10 → 0.70）", 0.10, 0.70))
    summary = {}
    with sim.Timer(f"ファジー {REPS * len(designs)} 本") as timer:
        for label, p_lo, p_hi in designs:
            def one(r, p_lo=p_lo, p_hi=p_hi):
                xf, df, yf = fuzzy_data(r, p_lo, p_hi)
                est, _, f_stat, first, _ = wald_ratio(xf, df, yf, BANDWIDTHS[1])
                return est, f_stat, first, naive_ols(xf, df, yf, BANDWIDTHS[1])
            out = sim.repeat(one, trials=REPS, seed=SEED, progress=False)
            summary[label] = out
            print(f"{label:<28}{out[:, 2].mean():>12.3f}{out[:, 1].mean():>10.1f}"
                  f"{np.median(out[:, 0]):>15.3f}{out[:, 0].mean():>10.3f}"
                  f"{out[:, 3].mean():>9.3f}")

    weak, strong = (summary[label] for label, *_ in designs)
    print(f"\n  真の効果は両方とも {JUMP:g}。素朴な OLS は未観測の熱心さのぶんだけ上に偏る"
          f"（{weak[:, 3].mean():.3f}）。")
    print(f"  遵守率が低いほうは第一段の F が平均 {weak[:, 1].mean():.1f} しかない。"
          f"Wald 比の平均は {weak[:, 0].mean():.3f}、")
    print(f"  ばらつきは SD {weak[:, 0].std(ddof=1):.3f}、"
          f"中央値 {np.median(weak[:, 0]):.3f}、"
          f"{REPS} 回のうち推定値が真値の2倍を超えたのが "
          f"{int((weak[:, 0] > 2 * JUMP).sum())} 回。")
    print(f"  遵守率を上げると F は {strong[:, 1].mean():.1f} まで上がり、"
          f"平均 {strong[:, 0].mean():.3f}（SD {strong[:, 0].std(ddof=1):.3f}）に落ち着く。")
    print("  弱い操作変数の怖さは平均が少しずれることではなく、"
          "小さい分母で割った値が暴れることにある。")
    print(f"  F<10 の回が {int((weak[:, 1] < 10).sum()) / REPS:.0%} ある設計の推定値を、"
          "点推定として報告してはいけない。")

    rng2 = np.random.default_rng(SEED + 1)
    xf, df, yf = fuzzy_data(rng2, 0.10, 0.25)
    est, se, f_stat, first, reduced = wald_ratio(xf, df, yf, BANDWIDTHS[1])
    print(f"\n  1本ぶんの内訳（seed={SEED + 1}, 遵守率が低い設計）:")
    print(f"    誘導形の段差 {reduced:.3f} ÷ 第一段の段差 {first:.3f} = {est:.3f}"
          f"（SE {se:.3f}, 第一段 F {f_stat:.1f}）")
    print("    分母が小さいほど、分子のわずかな揺れが答えを大きく振る。")
    print(f"\n  （ファジー {REPS * 2} 本で {timer.elapsed:.1f} 秒）")

    draw(x, y, results)


if __name__ == "__main__":
    main()
