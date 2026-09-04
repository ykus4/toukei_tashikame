"""平方和の分解を numpy 8行で書いて F 統計量を組み立て、scipy と突き合わせる。

分散分析は「群間のばらつき ÷ 群内のばらつき」でしかない。全体の変動 SST が
SSB（群平均が全体平均からどれだけ離れているか）と SSW（各点が自分の群平均から
どれだけ離れているか）にちょうど分かれ、それぞれを自由度で割った平均平方の比が F になる。

分解が「ちょうど」であること — SST = SSB + SSW が誤差なしで成り立つこと — は
ピタゴラスの定理そのもので、偶然ではない。数字で確かめると納得が早い。

    uv run python examples/ch14/one_way_anova_by_hand.py
"""

import numpy as np
from scipy import stats

from toukei_tashikame import datasets, plots

MEANS = (10.0, 11.0, 13.0)
N_PER, SD, SEED = 25, 2.5, 142


def main() -> None:
    plots.setup()
    y, g = datasets.anova_data(MEANS, n_per_group=N_PER, sd=SD, seed=SEED)
    groups = [y[g == k] for k in np.unique(g)]
    k, n = len(groups), y.size

    # --- 平方和の分解、ここが8行 ---
    grand = y.mean()                                              # 全体平均
    m = np.array([a.mean() for a in groups])                      # 群平均
    ss_b = N_PER * ((m - grand) ** 2).sum()                       # 群間平方和
    ss_w = sum(((a - a.mean()) ** 2).sum() for a in groups)       # 群内平方和
    ss_t = ((y - grand) ** 2).sum()                               # 全平方和
    df_b, df_w = k - 1, n - k
    f_hand = (ss_b / df_b) / (ss_w / df_w)                        # 平均平方の比
    p_hand = stats.f.sf(f_hand, df_b, df_w)

    ref = stats.f_oneway(*groups)

    print(f"--- 14-2 平方和の分解（{k}群, 各 n={N_PER}, sd={SD}, seed={SEED}）---")
    print(f"  真の群平均 {MEANS} / 観測された群平均 {np.round(m, 4)}\n")
    print("  要因      平方和 SS      自由度 df   平均平方 MS      F")
    print(f"  群間      {ss_b:10.4f}      {df_b:>3}     {ss_b / df_b:10.4f}   {f_hand:8.4f}")
    print(f"  群内      {ss_w:10.4f}      {df_w:>3}     {ss_w / df_w:10.4f}")
    print(f"  全体      {ss_t:10.4f}      {n - 1:>3}")
    print(f"\n  分解の検算 SSB + SSW = {ss_b + ss_w:.10f}")
    print(f"            SST       = {ss_t:.10f}   （差 {abs(ss_b + ss_w - ss_t):.2e}）")

    print(f"\n  自作      F = {f_hand:.4f}   p = {p_hand:.4g}")
    print(f"  scipy     F = {ref.statistic:.4f}   p = {ref.pvalue:.4g}")
    print(f"  差        F の差 {abs(f_hand - ref.statistic):.2e} / "
          f"p の差 {abs(p_hand - ref.pvalue):.2e}")
    print(f"\n  効果量 η² = SSB / SST = {ss_b / ss_t:.4f}"
          "  ← 全変動のうち群の違いで説明できた割合")
    print(f"  MSW = {ss_w / df_w:.4f} は誤差分散の推定値で、真の値 sd² = {SD**2:.4f} に対応する")
    print("  F が大きいとは「群平均のばらつきが、群内のばらつきから期待される量を超えた」の意味。")
    print("  帰無仮説が正しければ MSB も MSW も同じ誤差分散を推定するので、比は 1 のまわりに来る")

    # --- 図: 群間と群内、それぞれの偏差を同じ絵に描く ---
    fig, axes = plots.figure(1, 2, w=1.8, h=1.0)
    x = np.arange(k)
    rng = np.random.default_rng(SEED)
    jitter = rng.uniform(-0.16, 0.16, size=n)

    for ax, which in zip(axes, ["within", "between"], strict=True):
        ax.scatter(x[g] + jitter, y, s=5, color=plots.PALETTE["data"], lw=0, alpha=0.7, zorder=3)
        if which == "within":
            for j in range(n):
                ax.plot([x[g[j]] + jitter[j]] * 2, [y[j], m[g[j]]],
                        color=plots.PALETTE["data"], lw=0.4, alpha=0.5, zorder=2)
            for i in x:
                ax.plot([i - 0.3, i + 0.3], [m[i]] * 2, color=plots.PALETTE["estimate"],
                        lw=1.6, zorder=5)
            ax.set_title(f"群内 SSW = {ss_w:.2f}（df {df_w}）")
        else:
            for i in x:
                ax.plot([i - 0.3, i + 0.3], [m[i]] * 2, color=plots.PALETTE["estimate"],
                        lw=1.6, zorder=5)
                ax.plot([i, i], [grand, m[i]], color=plots.PALETTE["reject"], lw=1.8, zorder=4)
            ax.set_title(f"群間 SSB = {ss_b:.2f}（df {df_b}）")
        plots.mark_truth(ax, grand, f"全体平均 = {grand:.2f}", axis="y")
        ax.set_xticks(x)
        ax.set_xticklabels([f"群{i + 1}" for i in x])
    axes[0].set_ylabel("y")
    fig.tight_layout()
    plots.save(fig, "fig-14-2-between-within-variance.png")


if __name__ == "__main__":
    main()
