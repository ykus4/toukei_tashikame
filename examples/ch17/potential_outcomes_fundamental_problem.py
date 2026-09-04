"""因果推論の根本問題 — 一人につき片方の結果しか見られない。

処置を受けた人の「受けなかったらどうなっていたか」は、原理的に観測できない。
ここでは両方の潜在結果 $Y^{(1)}, Y^{(0)}$ を持つ合成データを作り、真の ATE と ATT を
直接計算してから、片方を消して「現実の」データにする。消したあとで何が復元できて、
何が復元できないのかを、同じ 5,000 人について並べて見る。

要点は3つ。(1) 個体ごとの効果 $\\tau_i = Y_i^{(1)} - Y_i^{(0)}$ は一度も観測できない。
(2) それでも平均 ATE は、割付が結果と独立なら推定できる。(3) ATE と ATT は別物で、
効果に個人差があり、かつ効果の大きい人ほど処置されやすいなら、両者はずれる。

    uv run python examples/ch17/potential_outcomes_fundamental_problem.py
"""

import numpy as np

from toukei_tashikame import plots

N, SEED = 5000, 172


def make_world(rng):
    """両方の潜在結果を持つ世界を作る。現実にはこの表の半分しか見えない。"""
    x = rng.normal(0.0, 1.0, size=N)                    # 観測できる共変量（利用歴など）
    y0 = 10.0 + 2.0 * x + rng.normal(0.0, 1.0, size=N)  # 処置しなかったときの結果
    tau = 2.5 + 1.0 * x + rng.normal(0.0, 1.5, size=N)  # 個体ごとの効果。x で変わる
    y1 = y0 + tau                                       # 処置したときの結果

    # 割付は x に依存する。効果が大きい人（x が大きい人）ほど処置されやすい世界。
    p = 1.0 / (1.0 + np.exp(-1.2 * x))
    t = (rng.random(N) < p).astype(int)

    y_obs = np.where(t == 1, y1, y0)                    # 現実に見える1列
    return x, y0, y1, tau, t, y_obs


def draw(y0, y1, tau, t, ate, att) -> None:
    fig, axes = plots.figure(1, 2, w=2.0)
    pal = plots.PALETTE

    ax = axes[0]
    ax.hist(tau, bins=50, color=pal["data"], alpha=0.55, lw=0)
    plots.mark_truth(ax, ate, f"真の ATE = {ate:.3f}")
    ax.axvline(att, color=pal["alt"], lw=1.1, ls="--", dashes=(4, 2.0), zorder=5)
    ax.annotate(f"真の ATT = {att:.3f}", xy=(att, 0.86), xycoords=("data", "axes fraction"),
                ha="right", va="top", fontsize=6.0, color=pal["alt"],
                xytext=(-3, 0), textcoords="offset points")
    ax.axvline(0.0, color=pal["ink2"], lw=0.6, zorder=2)
    ax.set_xlabel("個体ごとの効果 $\\tau_i = Y_i^{(1)} - Y_i^{(0)}$（一度も観測できない）")
    ax.set_ylabel("人数")

    ax = axes[1]
    for i in range(24):  # 先頭24人ぶんを、観測できた点と欠測の点で描き分ける
        ax.plot([y0[i], y1[i]], [i, i], color=pal["grid"], lw=0.8, zorder=1)
        for value, arm in ((y0[i], 0), (y1[i], 1)):
            color = pal["estimate"] if arm == 1 else pal["data"]
            observed = t[i] == arm
            ax.scatter(value, i, s=11, zorder=3, linewidths=0.7, edgecolors=color,
                       facecolors=color if observed else "white")
    ax.set_yticks([])
    ax.set_ylabel("先頭24人")
    ax.set_xlabel("塗り＝観測できた結果、白抜き＝反実仮想（欠測）")
    fig.tight_layout()
    plots.save(fig, "fig-17-2-potential-outcomes-missing.png")


def main() -> None:
    plots.setup()
    rng = np.random.default_rng(SEED)
    x, y0, y1, tau, t, y_obs = make_world(rng)

    ate = float(tau.mean())
    att = float(tau[t == 1].mean())
    atc = float(tau[t == 0].mean())

    print(f"--- 神の視点: 両方の潜在結果が見える表（n={N:,}, seed={SEED}）---\n")
    print(f"{'i':>4}{'x':>8}{'Y(0)':>9}{'Y(1)':>9}{'τ_i':>9}{'T':>4}{'観測される Y':>14}")
    for i in range(6):
        print(f"{i:>4}{x[i]:>8.3f}{y0[i]:>9.3f}{y1[i]:>9.3f}{tau[i]:>9.3f}"
              f"{t[i]:>4}{y_obs[i]:>14.3f}")
    print("     …（残り 4,994 人）")

    print(f"\n  真の ATE  = E[τ]        = {ate:.3f}")
    print(f"  真の ATT  = E[τ | T=1]  = {att:.3f}   （処置された {int(t.sum()):,} 人）")
    print(f"  真の ATC  = E[τ | T=0]  = {atc:.3f}   （処置されなかった {int((1 - t).sum()):,} 人）")
    print(f"  個体効果のばらつき SD(τ) = {tau.std(ddof=1):.3f}"
          f"（最小 {tau.min():.2f} 〜 最大 {tau.max():.2f}）")
    print("  効果が全員同じなら ATE=ATT で悩みは無い。ここでは τ_i が人によって違い、")
    print("  しかも効果の大きい人ほど処置されやすいので、ATT が ATE より大きい。")

    print("\n--- 現実の視点: 半分が欠測になる ---\n")
    n1, n0 = int(t.sum()), int((1 - t).sum())
    print(f"  処置群 T=1（{n1:,}人）:  Y(1) 観測ずみ  |  Y(0) 欠測      |  τ_i 観測不能")
    print(f"  対照群 T=0（{n0:,}人）:  Y(1) 欠測      |  Y(0) 観測ずみ  |  τ_i 観測不能")
    seen = 2 * N
    print(f"\n  潜在結果は全部で {seen:,} マス。そのうち観測できるのは {N:,} マス "
          f"= {N / seen:.0%} だけ。")
    print("  τ_i の列は1マスも埋まらない。これが Holland の言う「因果推論の根本問題」で、")
    print("  欠測が「たまたま」ではなく設計上必ず起きるところが、ふつうの欠測と違う。")

    naive = float(y_obs[t == 1].mean() - y_obs[t == 0].mean())
    print("\n--- 観測できるデータから何が言えるか ---\n")
    print(f"  素朴な群間差 E[Y|T=1] - E[Y|T=0] = {naive:.3f}")
    print(f"    その正体は  ATT {att:.3f}  +  選択バイアス "
          f"{float(y0[t == 1].mean() - y0[t == 0].mean()):.3f}")
    print("    選択バイアス = E[Y(0)|T=1] - E[Y(0)|T=0]。"
          "処置群は「処置しなくても元から高かった」")
    print(f"  真の ATE {ate:.3f} からのずれ … {naive - ate:+.3f}")
    print("\n  個人の効果は誰一人ぶんも分からない。それでも平均になら手が届く——")
    print("  ただし「割付が潜在結果と独立」という条件つきで、その条件は 17-4 で作る。")

    draw(y0, y1, tau, t, ate, att)


if __name__ == "__main__":
    main()
