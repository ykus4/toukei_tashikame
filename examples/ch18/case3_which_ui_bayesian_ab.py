"""依頼3「どっちのUIが良い？」に、勝率と期待損失で答える。@slow

「有意差はありませんでした」は、この依頼に対する答えになっていない。聞かれているのは
どちらを本番に出すかであって、帰無仮説が棄却できるかどうかではない。ベータ二項モデル
を PyMC で解けば $\\Pr[p_B > p_A]$ がそのまま出るし、**選び間違えたときに失う量の
期待値**（期待損失）まで出せる。決めるのはこの2つ目のほうである。

勝率が閾値に届かなくても、外したときの損が十分小さければ切り替えてよい。逆に勝率
99% でも損が大きければ待つ。ここで使う標本は、まさにその判断が要る位置に落ちる。

PyMC が要る（``uv sync --extra slow``）。共役解と突き合わせて、MCMC が正しい事後を
返していることも確かめる。

    uv run python examples/ch18/case3_which_ui_bayesian_ab.py
"""

import numpy as np

from toukei_tashikame import bayes, datasets, plots, sim, testing

N_A, N_B = 6000, 6000
P_A, LIFT, SEED = 0.030, 0.20, 182
DRAWS, TUNE, CHAINS = 2000, 1000, 2
HDI_PROB = 0.94                 # arviz の既定。94% に意味は無い、と明示するための値
WIN_THRESHOLD = 0.95            # 「勝率がこれを超えたら採択」という運用ルール
LOSS_THRESHOLD = 0.0005         # 「期待損失がこれを下回ったら採択」＝0.05pt


def fit_pymc(k_a: int, k_b: int):
    """ベータ二項モデルを MCMC で解く。事前は Beta(1,1)（一様）。"""
    import pymc as pm

    with pm.Model():
        p_a = pm.Beta("p_a", 1.0, 1.0)
        p_b = pm.Beta("p_b", 1.0, 1.0)
        pm.Deterministic("lift", p_b - p_a)
        pm.Binomial("obs_a", n=N_A, p=p_a, observed=k_a)
        pm.Binomial("obs_b", n=N_B, p=p_b, observed=k_b)
        idata = pm.sample(DRAWS, tune=TUNE, chains=CHAINS, cores=1,
                          random_seed=SEED, progressbar=False)
    return idata


def hdi(samples: np.ndarray, prob: float) -> tuple[float, float]:
    """最高密度区間。``bayes.credible_interval(kind="hdi")`` と同じものを使う。"""
    return bayes.credible_interval(samples, conf=prob, kind="hdi")


def draw(lift: np.ndarray, prob_b: float, lo: float, hi: float) -> None:
    fig, axes = plots.figure(1, 2, w=2.0, h=0.95)
    pal = plots.PALETTE

    ax = axes[0]
    counts, edges = np.histogram(100 * lift, bins=70, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax.plot(centers, counts, color=pal["posterior"], lw=1.2, zorder=4)
    ax.fill_between(centers, counts, where=centers > 0, color=pal["posterior"],
                    alpha=0.35, lw=0, zorder=2)
    ax.fill_between(centers, counts, where=centers <= 0, color=pal["reject"],
                    alpha=0.45, lw=0, zorder=2)
    ax.axvline(0.0, color=pal["ink2"], lw=0.8, zorder=5)
    ax.annotate(f"$\\Pr[p_B > p_A] = {prob_b:.4f}$", xy=(0.04, 0.92),
                xycoords="axes fraction", fontsize=6.4, color=pal["estimate"],
                va="top")
    ax.annotate("B を選んで\n外す側", xy=(0.04, 0.55), xycoords="axes fraction",
                fontsize=6.0, color=pal["reject"], va="top")
    plots.mark_interval(ax, 100 * lo, 100 * hi, label=f"{HDI_PROB:.0%} HDI")
    plots.mark_truth(ax, 100 * P_A * LIFT, f"真の差 {100 * P_A * LIFT:+.2f}pt")
    ax.set_xlabel("$p_B - p_A$（pt）")
    ax.set_ylabel("事後密度")
    ax.set_title("① 差の事後分布。0 より右の面積が「B が勝つ確率」")

    # ② 打ち切り基準を2つ並べる。勝率だけでは決まらないことを1枚で見せる。
    ax = axes[1]
    loss_b = np.maximum(-lift, 0.0)
    grid = np.linspace(0.5, 1.0, 200)
    # 勝率の閾値を動かしたとき、この標本で採択されるか
    ax.plot(grid, np.where(prob_b >= grid, 1.0, 0.0), color=pal["estimate"],
            lw=1.4, zorder=4)
    ax.axvline(WIN_THRESHOLD, color=pal["reject"], lw=0.9, ls="--",
               dashes=(4, 2.2), zorder=5)
    ax.annotate(f"運用ルール {WIN_THRESHOLD:.2f}", xy=(WIN_THRESHOLD, 0.55),
                xytext=(-3, 0), textcoords="offset points", ha="right",
                fontsize=6.0, color=pal["reject"])
    ax.annotate(f"この標本の勝率 {prob_b:.4f}\n→ 勝率基準では「見送り」",
                xy=(0.52, 0.42), xycoords="axes fraction", fontsize=6.2,
                color=pal["ink2"], va="top")
    ax.annotate(f"期待損失 {100 * loss_b.mean():.4f}pt < 閾値 "
                f"{100 * LOSS_THRESHOLD:.2f}pt\n→ 損失基準では「採択」",
                xy=(0.52, 0.22), xycoords="axes fraction", fontsize=6.2,
                color=pal["estimate"], va="top")
    ax.set_ylim(-0.1, 1.25)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["見送り", "採択"])
    ax.set_xlabel("採択に要求する勝率の閾値")
    ax.set_title("② 同じ事後でも、決め方で結論が変わる")

    fig.tight_layout()
    plots.save(fig, "fig-18-4-posterior-decision.png")


def main() -> None:
    plots.setup()
    d = datasets.ab_test(n_a=N_A, n_b=N_B, p_a=P_A, lift=LIFT, seed=SEED)
    k_a, k_b = int(d.a.sum()), int(d.b.sum())

    print(f"--- 18-4 依頼3「どっちのUIが良い？」（n={N_A:,}/群, seed={SEED}）---\n")
    print(f"  A（現行UI）  {k_a:>4} / {N_A:,} = {100 * k_a / N_A:.3f}%"
          f"    真値 {100 * d.p_a:.1f}%")
    print(f"  B（新UI）    {k_b:>4} / {N_B:,} = {100 * k_b / N_B:.3f}%"
          f"    真値 {100 * d.p_b:.1f}%")

    with sim.Timer("  PyMC（ベータ二項、NUTS）"):
        idata = fit_pymc(k_a, k_b)
    import arviz as az

    post = idata.posterior
    lift = post["lift"].to_numpy().ravel()
    p_a_s = post["p_a"].to_numpy().ravel()
    p_b_s = post["p_b"].to_numpy().ravel()
    prob_b = float((lift > 0).mean())
    lo, hi = hdi(lift, HDI_PROB)
    loss_b = float(np.maximum(-lift, 0.0).mean())   # B を採って外したときに失う CVR
    loss_a = float(np.maximum(lift, 0.0).mean())    # A に留まって外したときに失う CVR

    print(f"\n① 事後分布（{CHAINS}本 × {DRAWS:,}draws、事前は Beta(1,1)）")
    print(f"  p_A の事後平均 {100 * p_a_s.mean():.3f}%   "
          f"p_B の事後平均 {100 * p_b_s.mean():.3f}%")
    print(f"  差の事後平均   {100 * lift.mean():+.4f}pt"
          f"（真の差 {100 * (d.p_b - d.p_a):+.2f}pt）")
    print(f"  {HDI_PROB:.0%} HDI      [{100 * lo:+.4f}pt, {100 * hi:+.4f}pt]"
          "   ← 0 を含む")
    print(f"  R-hat {float(az.rhat(idata)['lift']):.4f} / "
          f"ESS {float(az.ess(idata)['lift']):.0f}   ← 収束の確認")

    # 共役解と突き合わせる。ベータ二項は手で解けるので、MCMC の答え合わせができる。
    post_a = bayes.beta_binomial(k_a, N_A)
    post_b = bayes.beta_binomial(k_b, N_B)
    prob_conj = bayes.prob_b_beats_a(post_a, post_b, draws=200_000, seed=SEED)
    # 返り値は (A に留まって外したときの損, B に切り替えて外したときの損)
    _loss_conj_a, loss_conj_b = bayes.expected_loss(post_a, post_b, draws=200_000, seed=SEED)
    print("\n② 共役解との突き合わせ（ベータ二項は手でも解ける）")
    print(f"  Pr[p_B > p_A]   MCMC {prob_b:.4f}  /  共役 {prob_conj:.4f}"
          f"   差 {abs(prob_b - prob_conj):.4f}")
    print(f"  期待損失(B採択) MCMC {100 * loss_b:.5f}pt  /  "
          f"共役 {100 * loss_conj_b:.5f}pt")

    print("\n③ 意思決定")
    win_ok = prob_b >= WIN_THRESHOLD
    loss_ok = loss_b < LOSS_THRESHOLD
    print(f"  Pr[B > A] = {prob_b:.4f}      閾値 {WIN_THRESHOLD:.2f} "
          f"→ {'採択' if win_ok else '見送り'}")
    print(f"  B を採って外したとき失う CVR   {100 * loss_b:.5f}pt "
          f"（閾値 {100 * LOSS_THRESHOLD:.2f}pt → {'採択' if loss_ok else '見送り'}）")
    print(f"  A に留まって外したとき失う CVR {100 * loss_a:.5f}pt")
    print(f"  ← 損の非対称: 間違えて B にしたときの損は、"
          f"間違えて A のままにしたときの {loss_a / max(loss_b, 1e-12):.1f}分の1")
    print("  実装コストが小さいなら、勝率が閾値に届かなくても切り替えてよい。"
          "決めているのは期待損失であって p 値ではない")

    # ④ 頻度論との比較
    freq = testing.prop_2samp(k_b, N_B, k_a, N_A, method="score")
    print("\n④ 頻度論との比較表")
    print("  指標              値         答えている問い")
    print(f"  p 値          {freq.pvalue:>9.4f}     "
          "「差が無いとき、この差以上が出る確率」（主語はデータ）")
    print(f"  Pr[p_B>p_A]   {prob_b:>9.4f}     "
          "「B のほうが良い確率」（主語はパラメータ）")
    print(f"  期待損失      {100 * loss_b:>9.5f}     "
          "「B を選んで外したとき、平均で失う CVR」（pt）")
    print("  p 値は意思決定の入力にならない。3行目だけが、そのまま稟議に書ける形をしている")

    print("\n⑤ 報告（18-7 のテンプレート）")
    print(f"  効果量: 差の事後平均 {100 * lift.mean():+.3f}pt"
          f"（相対 {100 * (p_b_s.mean() / p_a_s.mean() - 1):+.1f}%）")
    print(f"  区間  : {HDI_PROB:.0%} HDI [{100 * lo:+.3f}pt, {100 * hi:+.3f}pt]、"
          f"Pr[B > A] = {prob_b:.4f}")
    print("  仮定  : 事前分布 Beta(1,1)／割付はランダム／各ユーザは独立／"
          "見ている指標は初回コンバージョンのみ")
    print(f"  限界  : 事前分布の取り方で勝率は動く。閾値 {WIN_THRESHOLD:.2f} と "
          f"{100 * LOSS_THRESHOLD:.2f}pt は事業側の合意事項であって、"
          "データから決まるものではない")

    draw(lift, prob_b, lo, hi)


if __name__ == "__main__":
    main()
