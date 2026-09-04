"""無作為化はバイアスを消し、ブロック化はばらつきを削る — 別々の仕事である。

個体差の大きい60人（個体効果の SD=8）に、効果 3.0 の処置を当てる。誤差の源は
「誰が処置群に入ったか」で、完全無作為化ではこれが丸ごと推定量のばらつきになる。

前もって測ってある個体の値で60人を並べ、隣どうしを2人1組にして組の中で
コインを投げる（ブロック化無作為化）と、組の中では個体差がほとんど揃っているので、
差を取った瞬間に個体差が消える。無作為化は保ったまま、ばらつきだけが減る。

どちらも真の効果 3.0 を偏りなく当てる。違うのは推定量の SD と、その帰結である検出力。
「サンプルを増やす」以外に検出力を上げる道があることが、この節の要点である。

    uv run python examples/ch14/randomization_and_blocking_reduce_variance.py
"""

import numpy as np
from scipy import stats

from toukei_tashikame import plots, sim

N, SD_SUBJECT, SD_NOISE, TAU = 60, 8.0, 3.0, 3.0
SEED, TRIALS, ALPHA = 147, 10_000, 0.05
N_PAIR = N // 2

# 60人の個体効果は最初に1度だけ引く。動くのは割り付けのほうで、人は動かない。
BASELINE = np.sort(np.random.default_rng(SEED).normal(0.0, SD_SUBJECT, size=N))
PAIR_OF = np.repeat(np.arange(N_PAIR), 2)     # 並べた順に2人ずつ組にする


def one_trial(rng: np.random.Generator) -> tuple[float, float, float, float]:
    """同じ60人に、2通りの割り付けをそれぞれ1回。(推定値, p) を2組返す。"""
    noise = rng.normal(0.0, SD_NOISE, size=N)     # 測定誤差は割り付けと無関係

    # --- (A) 完全無作為化。60人から30人を選んで処置 ---
    treat = np.zeros(N, dtype=bool)
    treat[rng.permutation(N)[: N // 2]] = True
    y = BASELINE + TAU * treat + noise
    a, b = y[treat], y[~treat]
    est_crd = a.mean() - b.mean()
    se_crd = np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
    p_crd = 2 * stats.t.sf(abs(est_crd) / se_crd, df=N - 2)

    # --- (B) ブロック化。組の中で1人だけを処置に ---
    first_is_treated = rng.random(N_PAIR) < 0.5
    treat_b = np.empty(N, dtype=bool)
    treat_b[0::2] = first_is_treated
    treat_b[1::2] = ~first_is_treated
    yb = BASELINE + TAU * treat_b + noise
    diff = np.where(first_is_treated, yb[0::2] - yb[1::2], yb[1::2] - yb[0::2])
    est_blk = diff.mean()
    se_blk = diff.std(ddof=1) / np.sqrt(N_PAIR)
    p_blk = 2 * stats.t.sf(abs(est_blk) / se_blk, df=N_PAIR - 1)   # 組の差の1標本t検定

    return est_crd, p_crd, est_blk, p_blk


def main() -> None:
    plots.setup()
    gaps = BASELINE[1::2] - BASELINE[0::2]

    print(f"--- 14-7 個体差の大きい {N} 人（個体効果 SD={SD_SUBJECT}, 測定誤差 SD={SD_NOISE}）---")
    print(f"  真の効果 τ = {TAU}、seed={SEED}、割り付けを {TRIALS:,} 回引き直す")
    print(f"  60人の個体効果の実際の SD = {BASELINE.std(ddof=1):.4f}"
          f"（範囲 {BASELINE.min():.2f} 〜 {BASELINE.max():.2f}）")
    print(f"  並べて組にすると、組の中の個体差は平均 {np.abs(gaps).mean():.4f} まで縮む")

    with sim.Timer("14-7 の 10,000 回"):
        out = sim.repeat(one_trial, trials=TRIALS, seed=SEED, progress=False)
    est_crd, p_crd, est_blk, p_blk = out.T

    rows = [("完全無作為化", est_crd, p_crd), ("ブロック化無作為化", est_blk, p_blk)]
    print("\n  割り付けの仕方            平均       バイアス      SD       検出力")
    stats_by_row = []
    for label, est, pv in rows:
        bias = est.mean() - TAU
        sd = est.std(ddof=1)
        power = float((pv < ALPHA).mean())
        stats_by_row.append((label, est.mean(), bias, sd, power))
        print(f"  {label:<22} {est.mean():.4f}    {bias:+.4f}    {sd:.4f}    {power:.4f}")

    sd_crd, sd_blk = stats_by_row[0][3], stats_by_row[1][3]
    print("\n  バイアスはどちらも 0 付近。無作為化しているかぎり、"
          "どんな割り付けでも推定量は真値を中心に来る")
    print(f"  変わったのは SD だけで {sd_crd:.4f} → {sd_blk:.4f}"
          f"（{sd_blk / sd_crd:.3f} 倍、分散で {(sd_blk / sd_crd) ** 2:.3f} 倍）")
    print(f"  検出力は {stats_by_row[0][4]:.4f} → {stats_by_row[1][4]:.4f}。"
          "人を増やさずに上げた検出力である")
    print(f"  完全無作為化で同じ SD を出すには n を約 {(sd_crd / sd_blk) ** 2:.1f} 倍"
          f"（{N} 人 → {N * (sd_crd / sd_blk) ** 2:.0f} 人）にする必要がある")

    s2 = BASELINE.var(ddof=1)
    theory_crd = np.sqrt(2 * (s2 + SD_NOISE**2) / (N // 2))
    theory_blk = np.sqrt((gaps**2).sum() / N_PAIR**2 + 2 * SD_NOISE**2 / N_PAIR)
    print(f"\n  理論値との突き合わせ  完全無作為化 √(2(s²_個体+σ²_誤差)/30) = {theory_crd:.4f}"
          f"（実測 {sd_crd:.4f}）")
    print(f"                        ブロック化   組内の差から = {theory_blk:.4f}"
          f"（実測 {sd_blk:.4f}）")
    print("  ブロック化の分散から σ_個体² が消えている。これがブロック化のしていること")
    print("\n  無作為化とブロック化は役割が違う。無作為化はバイアスを消し（比較可能にし）、")
    print("  ブロック化はばらつきを削る（同じ n で細かい差を見えるようにする）。"
          "どちらか一方では足りない")

    # --- 図: 10,000 本の推定値の分布と、検出力の比較 ---
    fig, axes = plots.figure(1, 2, w=1.9, h=1.0)

    ax = axes[0]
    bins = np.linspace(min(est_crd.min(), est_blk.min()), max(est_crd.max(), est_blk.max()), 60)
    ax.hist(est_crd, bins=bins, density=True, color=plots.PALETTE["data"], alpha=0.55, lw=0,
            zorder=2)
    ax.hist(est_blk, bins=bins, density=True, color=plots.PALETTE["estimate"], alpha=0.65, lw=0,
            zorder=3)
    plots.mark_truth(ax, TAU, f"真の効果 = {TAU}")
    ax.annotate(f"完全無作為化 SD {sd_crd:.2f}", xy=(TAU - 2.2 * sd_crd, 0.05),
                fontsize=6.2, color=plots.PALETTE["data"], ha="center", va="bottom")
    ax.annotate(f"ブロック化 SD {sd_blk:.2f}", xy=(TAU - 1.3, 0.32), fontsize=6.2,
                color=plots.PALETTE["estimate"], ha="right", va="bottom")
    ax.set_xlabel("推定された処置効果")
    ax.set_ylabel("密度")
    ax.set_title(f"同じ{N}人・同じ真値、{TRIALS:,} 回の割り付け")

    ax = axes[1]
    x = np.arange(2)
    powers = [stats_by_row[0][4], stats_by_row[1][4]]
    ax.bar(x, powers, width=0.5, color=[plots.PALETTE["data"], plots.PALETTE["estimate"]],
           lw=0, zorder=3)
    for xi, pw in zip(x, powers, strict=True):
        ax.annotate(f"{pw:.4f}", xy=(xi, pw), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6.2)
    ax.axhline(0.8, color=plots.PALETTE["reject"], lw=0.9, ls="--", dashes=(4, 2.2), zorder=4)
    ax.annotate("よく使われる目標 0.8", xy=(0.02, 0.8), xycoords=("axes fraction", "data"),
                ha="left", va="bottom", fontsize=6.0, color=plots.PALETTE["reject"])
    ax.set_xticks(x)
    ax.set_xticklabels(["完全無作為化", "ブロック化"])
    ax.set_ylabel(f"検出力（α={ALPHA}）")
    ax.set_ylim(0, 1.05)
    ax.set_title("人を増やさずに上がった検出力")
    fig.tight_layout()
    plots.save(fig, "fig-14-7-blocking-vs-crd.png")


if __name__ == "__main__":
    main()
