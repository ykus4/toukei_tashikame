"""@slow — A/Bテストをベイズで締める。「Bが勝つ確率」と「間違えたときの損」。

第10章で数えたとおり、この規模（各5,000件、真のリフト+10%）は検出力がまるで足りない。
たまたま有意になることもあるが、p 値が答えるのは「差が無いとしたらこのデータは珍しいか」
だけで、有意にならなければ「分からない」で終わる。意思決定はそこから始まるのに、である。

事後分布があれば、知りたい形の問いにそのまま答えられる。

  Pr[p_B > p_A]      B のほうが良い確率
  期待損失            B を採ったのに実は A が良かったときに失う CVR の期待値
  リフトの信用区間    どれくらい良いのか

期待損失には「これを下回ったら決める」という閾値を置ける。ここでは 0.0005（0.05pt）を
使う。p 値の 0.05 と違い、この閾値は事業側の言葉で決められる量である。

    uv sync --extra slow
    uv run python examples/ch16/bayesian_ab_prob_b_wins_expected_loss.py
"""

import numpy as np
import pymc as pm

from toukei_tashikame import bayes, datasets, plots, power, sim, testing

SEED = 168
DRAWS, TUNE, CHAINS = 1_000, 1_000, 4
HDI_MASS = 0.94
LOSS_THRESHOLD = 0.0005   # 0.05pt。これを下回ったら採択してよい、と先に決めておく


def fit_hierarchical(k, n, seed: int):
    """階層ベータ二項。2群の CVR に共通の親分布を置く。

    群ごとに独立な Beta(1,1) を置くのと違い、親（全体の CVR とそのばらつき）も
    データから推す。群が2つだけなら効きは小さいが、群が増えるほど 16-7 の縮小が
    そのまま効く形になっている。
    """
    with pm.Model():
        mu = pm.Beta("mu", 1.0, 1.0)                       # 全体の CVR
        kappa = pm.Exponential("kappa", 1.0 / 200.0) + 2.0  # 群間のまとまり具合
        p = pm.Beta("p", alpha=mu * kappa, beta=(1.0 - mu) * kappa, shape=2)
        pm.Binomial("k", n=n, p=p, observed=k)
        idata = pm.sample(DRAWS, tune=TUNE, chains=CHAINS, random_seed=seed,
                          target_accept=0.90, progressbar=False)
    return idata


def draw(diff_pt, prob_b, loss_a, loss_b, hdi_pt) -> None:
    fig, axes = plots.figure(1, 2, w=2.0, h=1.0, gridspec_kw={"width_ratios": [1.6, 1.0]})
    pal = plots.PALETTE

    ax = axes[0]
    counts, bins = np.histogram(diff_pt, bins=70, density=True)
    centers = 0.5 * (bins[1:] + bins[:-1])
    ax.fill_between(centers, counts, where=centers > 0, color=pal["posterior"],
                    alpha=0.35, lw=0, step="mid", zorder=2)
    ax.fill_between(centers, counts, where=centers <= 0, color=pal["reject"],
                    alpha=0.45, lw=0, step="mid", zorder=2)
    ax.plot(centers, counts, color=pal["posterior"], lw=1.2, zorder=4)
    ax.axvline(0.0, color=pal["ink"], lw=0.8, zorder=5)
    plots.mark_interval(ax, hdi_pt[0], hdi_pt[1],
                        label=f"{HDI_MASS:.0%} HDI [{hdi_pt[0]:+.2f}, {hdi_pt[1]:+.2f}]pt",
                        y=counts.max() * 0.06)
    ax.annotate(f"青の面積 = Pr[$p_B > p_A$] = {prob_b:.3f}",
                xy=(0.55, 0.92), xycoords="axes fraction", fontsize=6.0,
                color=pal["posterior"])
    ax.annotate("オレンジ = B が負けている側", xy=(0.02, 0.92), xycoords="axes fraction",
                fontsize=6.0, color=pal["reject"])
    ax.set_xlabel("リフト $p_B - p_A$（パーセントポイント）")
    ax.set_ylabel("事後密度")
    ax.set_title("① 差そのものの事後分布")

    ax = axes[1]
    names = ["A を選ぶ", "B を選ぶ"]
    vals = [loss_a, loss_b]
    ax.bar(names, vals, color=[pal["data"], pal["posterior"]], width=0.55, zorder=3)
    ax.axhline(LOSS_THRESHOLD, color=pal["truth"], lw=1.1, zorder=5)
    ax.annotate(f"採択の閾値 {LOSS_THRESHOLD:.4f}", xy=(0.98, LOSS_THRESHOLD),
                xycoords=("axes fraction", "data"), ha="right", va="bottom",
                fontsize=6.0, color=pal["truth"])
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.5f}", xy=(i, v), ha="center", va="bottom", fontsize=6.0,
                    color=pal["ink2"])
    ax.set_ylabel("期待損失（CVR）")
    ax.set_title("② 選び間違えたときに失う量")

    plots.save(fig, "fig-16-8-prob-b-wins-and-loss.png")


def main() -> None:
    plots.setup()
    d = datasets.ab_test(seed=SEED)
    k = np.array([d.a.sum(), d.b.sum()], dtype=int)
    n = np.array([d.a.size, d.b.size], dtype=int)

    print(f"--- datasets.ab_test(seed={SEED}) のログ ---")
    print(f"  A: {k[0]:,} / {n[0]:,} = {k[0] / n[0]:.4f}"
          f"（真値 {d.p_a:.4f}）")
    print(f"  B: {k[1]:,} / {n[1]:,} = {k[1] / n[1]:.4f}"
          f"（真値 {d.p_b:.4f}、真のリフト {d.lift:+.1%}）\n")

    with sim.Timer("  階層ベータ二項のサンプリング"):
        idata = fit_hierarchical(k, n, SEED + 1)

    post = idata.posterior["p"].values.reshape(-1, 2)
    p_a, p_b = post[:, 0], post[:, 1]
    diff = p_b - p_a
    prob_b = float((diff > 0).mean())
    # 期待損失。「選んだほうが実は劣っていた」ぶんだけを、事後標本ごとに数えて平均する。
    loss_b = float(np.maximum(p_a - p_b, 0.0).mean())
    loss_a = float(np.maximum(p_b - p_a, 0.0).mean())
    hdi = bayes.credible_interval(diff, HDI_MASS, kind="hdi")
    div = int(idata.sample_stats["diverging"].values.sum())

    # 比較用: 階層を置かず、群ごとに独立な一様事前で更新した場合（第15章の共役解）。
    flat_a = bayes.beta_binomial(int(k[0]), int(n[0]))
    flat_b = bayes.beta_binomial(int(k[1]), int(n[1]))
    prob_b_flat = bayes.prob_b_beats_a(flat_a, flat_b, draws=200_000, seed=SEED + 2)
    _, loss_b_flat = bayes.expected_loss(flat_a, flat_b, draws=200_000, seed=SEED + 2)

    freq = testing.prop_2samp(int(k[1]), int(n[1]), int(k[0]), int(n[0]))
    need = power.n_for_proportions(d.p_a, d.p_b)

    print(f"\n  発散 {div} 本、事後標本 {post.shape[0]:,} 本\n")
    print("  ベイズの答え（階層ベータ二項）:")
    print(f"    Pr[p_B > p_A]              {prob_b:.4f}")
    verdict = "下回る → 採択してよい" if loss_b < LOSS_THRESHOLD else "上回る → まだ決めない"
    print(f"    B を選んだときの期待損失   {loss_b:.6f}"
          f"   閾値 {LOSS_THRESHOLD:.4f} を{verdict}")
    print(f"    A を選んだときの期待損失   {loss_a:.6f}")
    print(f"    リフトの {HDI_MASS:.0%} HDI       "
          f"[{hdi[0] * 100:+.2f}pt, {hdi[1] * 100:+.2f}pt]"
          f"（点推定 {diff.mean() * 100:+.2f}pt、真値 {(d.p_b - d.p_a) * 100:+.2f}pt）")
    print(f"    参考: 階層を置かない Beta(1,1) では Pr={prob_b_flat:.4f}、"
          f"期待損失 {loss_b_flat:.6f}")

    print("\n  頻度論の答え（第10章と同じ設計）:")
    print(f"    {freq}")
    print(f"    p = {freq.pvalue:.4f} "
          f"{'→ α=0.05 で有意' if freq.pvalue < 0.05 else '→ α=0.05 では有意でない'}")
    print(f"    この差（{d.p_a:.1%} → {d.p_b:.1%}）を検出力80%で見るには "
          f"1群 {need:,} 件が要る。実際は {n[0]:,} 件。")

    print("\n  p 値と事後確率は、同じ向きの結論になることもあれば、割れることもある。")
    print("  割れても矛盾ではない。答えている問いが違うからである。")
    print("    p 値      「差が無いとしたら、これくらいの差が出るのは珍しいか」")
    print("    事後確率  「今の情報のもとで、B が良い確率はいくらか」")
    print("  意思決定に直接つながるのは後者で、しかも期待損失を併せると"
          "「決めてよいか」まで言える。")
    print("  ただしこれは事前分布と生成モデルを引き受けた上での数字である。"
          "そこは無料ではない。")

    draw(diff * 100, prob_b, loss_a, loss_b, (hdi[0] * 100, hdi[1] * 100))


if __name__ == "__main__":
    main()
