"""真の係数が全部0の20変数に前向きステップワイズをかけると、何本が「有意」に選ばれるか。

y はどの説明変数とも無関係な純粋な雑音である。正しい答えは「1本も選ばない」。
それでも p<0.05 を採択基準にした前向き選択を 1,000 回まわすと、平均で1本前後が
選ばれ、3回に2回は「何もないところから」1本以上が拾われる。

理由は多重比較そのものである。20本から最も p の小さい1本を選ぶ行為は、20回の検定の
最小値を取ることに等しく、その最小 p はもはや一様分布ではない。そして選んだあとに
表示される p 値は、「選ばれたから小さい」という選択の結果を反映していない。
**選択後の p 値は p 値ではない。**

    uv run python examples/ch12/stepwise_picks_fake_variables.py
"""

import numpy as np

from toukei_tashikame import plots, regression, sim

N, K, TRIALS, SEED = 100, 20, 1_000, 129
ALPHA = 0.05      # 変数を採用する基準


def one_trial(rng) -> tuple[int, float, float, float]:
    """1回ぶん。(採用本数, 選択後の最小p, p<0.05だった係数の割合, 選択後のR²)。

    ``y`` は X とまったく無関係に作る。真のモデルは「切片だけ」である。
    """
    X = rng.normal(0.0, 1.0, size=(N, K))
    y = rng.normal(0.0, 1.0, size=N)          # どの列とも関係がない
    picked = regression.stepwise(X, y, criterion="p", threshold=ALPHA)
    if not picked:
        return 0, np.nan, np.nan, 0.0
    res = regression.ols(X[:, picked], y)
    p_slopes = res.pvalues[1:]                # 切片を除く
    return len(picked), float(p_slopes.min()), float((p_slopes < ALPHA).mean()), res.r2


def main() -> None:
    plots.setup()

    with sim.Timer(f"ステップワイズ {TRIALS:,} 回"):
        out = sim.repeat(one_trial, trials=TRIALS, seed=SEED, progress=False)
    n_picked = out[:, 0]
    min_p = out[:, 1]
    frac_sig = out[:, 2]
    r2 = out[:, 3]

    print(f"\n--- 真の係数がすべて0の説明変数 {K} 本、n={N}、{TRIALS:,} 回 ---")
    print(f"  採用された変数の平均     {n_picked.mean():.3f} 本"
          f"（中央値 {np.median(n_picked):.0f}、最大 {int(n_picked.max())}）")
    print(f"  1本以上採用した割合      {float((n_picked > 0).mean()):.4f}")
    print(f"  1本も採用しなかった割合  {float((n_picked == 0).mean()):.4f}"
          f"   ← これが正解である")
    print("  採用本数の分布:")
    for k in range(0, 8):
        cnt = int((n_picked == k).sum())
        bar = "█" * int(40 * cnt / TRIALS)
        print(f"    {k} 本  {cnt:>4} 回 {bar}")
    print(f"    8本以上  {int((n_picked >= 8).sum()):>2} 回")

    sel = n_picked > 0
    print(f"\n  選択後のモデルで p<{ALPHA} だった係数の割合 "
          f"{np.nanmean(frac_sig[sel]):.4f}")
    print(f"  選択後の最小 p の中央値 {np.nanmedian(min_p[sel]):.5f}"
          f"（名目 α={ALPHA} より桁が小さい）")
    print(f"  選択後の R² の平均 {r2.mean():.4f}"
          f"（真の R² は 0。y は X と無関係である）")
    print(f"  R² が 0.2 を超えた割合 {float((r2 > 0.2).mean()):.4f}")

    print("\n  選ぶという行為が既に検定であり、20本から最良の1本を選ぶのは20回の")
    print("  多重比較にあたる。にもかかわらず、出力される p 値は1回ぶんの検定として")
    print("  計算されている。ステップワイズの表を読むときは、この二重帳簿に注意する。")

    # --- 図 ---
    fig, axes = plots.figure(1, 2, w=2.0, h=1.0)
    pal = plots.PALETTE

    ax = axes[0]
    counts = np.bincount(n_picked.astype(int), minlength=9)[:9]
    ax.bar(np.arange(9), counts, color=pal["reject"], alpha=0.75, width=0.7)
    ax.bar([0], [counts[0]], color=pal["estimate"], alpha=0.85, width=0.7)
    ax.annotate(f"正しい答え（0本）は {counts[0] / TRIALS:.1%} しかない",
                xy=(0.3, counts[0]), xytext=(4, 4), textcoords="offset points",
                fontsize=6.0, color=pal["estimate"])
    plots.mark_truth(ax, 0.0, "真の本数 = 0")
    ax.annotate(f"平均 {n_picked.mean():.2f} 本", xy=(0.72, 0.72),
                xycoords="axes fraction", fontsize=6.4, color=pal["reject"])
    ax.set_xlabel(f"採用された変数の本数（真の係数は {K} 本ともすべて0）")
    ax.set_ylabel(f"{TRIALS:,} 回のうちの回数")
    ax.set_title("① 何もないところから1本前後が選ばれる")

    ax = axes[1]
    ax.hist(min_p[sel], bins=np.linspace(0, ALPHA, 30), color=pal["reject"],
            alpha=0.7, lw=0, density=True)
    ax.axhline(1.0, color=pal["truth"], lw=1.1)
    ax.annotate("帰無が真なら p は一様分布（密度1）のはず",
                xy=(0.5, 1.0), xycoords=("axes fraction", "data"), xytext=(0, 4),
                textcoords="offset points", ha="center", fontsize=6.0, color=pal["truth"])
    ax.set_xlabel(f"選択後のモデルの最小 p 値（採用があった {int(sel.sum()):,} 回）")
    ax.set_ylabel("密度")
    ax.set_title("② 選んだあとの p 値は p 値として読めない")

    fig.tight_layout()
    plots.save(fig, "fig-12-9-stepwise-false-selection.png")


if __name__ == "__main__":
    main()
