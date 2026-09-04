"""@slow — R̂ が 1.00 でも、モデルが間違っていることはある。

収束診断（16-5）が答えるのは「サンプラーがちゃんと回ったか」だけで、「そのモデルが
データを説明できているか」ではない。この2つは別の問いである。

ここでは、ゼロが多いカウントデータ（1日の問い合わせ件数など、そもそも発生しない日が
混ざる類のもの）に素のポアソンを当てる。R̂ は 1.000、発散もゼロ、ESS も十分。それでも
モデルからデータを生成し直すと、実データのゼロの多さをまったく再現しない。

事後予測検査 (posterior predictive check) は、その食い違いを1つの図と1つの数字にする。
最後に、ゼロ過剰ポアソンに替えると再現するところまで見る。

    uv sync --extra slow
    uv run python examples/ch16/posterior_predictive_check.py
"""

import arviz as az
import numpy as np
import pymc as pm

from toukei_tashikame import plots, sim

SEED = 166
N = 400
LAM_TRUE = 3.0        # 発生する日の平均件数
PSI_TRUE = 0.75       # 「そもそも発生しうる日」の割合（25% は構造的にゼロ）
DRAWS, TUNE, CHAINS = 1_000, 1_000, 4   # 事後予測は 4,000 本のデータセットを作り直す


def make_data(seed: int) -> np.ndarray:
    """ゼロ過剰なカウント。ゼロには2種類ある（構造的なゼロと、たまたまのゼロ）。"""
    rng = np.random.default_rng(seed)
    active = rng.random(N) < PSI_TRUE
    return np.where(active, rng.poisson(LAM_TRUE, size=N), 0).astype(float)


def fit_poisson(y, seed: int):
    """素のポアソン。パラメータは1個だけ。"""
    with pm.Model():
        lam = pm.Exponential("lam", 1.0 / 5.0)
        pm.Poisson("y", mu=lam, observed=y)
        idata = pm.sample(DRAWS, tune=TUNE, chains=CHAINS, random_seed=seed,
                          progressbar=False)
        # 事後標本1本ごとに、同じ n のデータを丸ごと1セット作り直す。
        idata.extend(pm.sample_posterior_predictive(idata, random_seed=seed,
                                                    progressbar=False))
    return idata


def fit_zip(y, seed: int):
    """ゼロ過剰ポアソン。「発生しうるか」と「何件か」を分けて書く。"""
    with pm.Model():
        psi = pm.Beta("psi", 1.0, 1.0)
        lam = pm.Exponential("lam", 1.0 / 5.0)
        pm.ZeroInflatedPoisson("y", psi=psi, mu=lam, observed=y)
        idata = pm.sample(DRAWS, tune=TUNE, chains=CHAINS, random_seed=seed,
                          progressbar=False)
        # 事後標本1本ごとに、同じ n のデータを丸ごと1セット作り直す。
        idata.extend(pm.sample_posterior_predictive(idata, random_seed=seed,
                                                    progressbar=False))
    return idata


def zero_fraction(rep: np.ndarray) -> np.ndarray:
    """事後予測データセット1本ごとの「ゼロの割合」。これを検査統計量にする。"""
    return (rep == 0).mean(axis=-1)


def check(idata, y) -> dict:
    """R̂ と、ゼロ割合のベイジアン p 値。収束と当てはまりを並べて出す。"""
    summary = az.summary(idata)
    rep = idata.posterior_predictive["y"].values.reshape(-1, y.size)
    t_rep = zero_fraction(rep)
    t_obs = float((y == 0).mean())
    # ベイジアン p 値。「モデルから作ったデータのうち、実データ以上に極端なものの割合」。
    p_value = float((t_rep >= t_obs).mean())
    return {
        "rhat": float(summary["r_hat"].max()),
        "ess": float(summary["ess_bulk"].min()),
        "div": int(idata.sample_stats["diverging"].values.sum()),
        "t_rep": t_rep,
        "mean_zero": float(t_rep.mean()),
        "p": p_value,
        "rep": rep,
        "summary": summary,
    }


def draw(y, poi, zip_) -> None:
    fig, axes = plots.figure(1, 2, w=2.0, h=1.0)
    pal = plots.PALETTE
    t_obs = float((y == 0).mean())

    ax = axes[0]
    bins = np.linspace(0.02, 0.40, 60)
    ax.hist(poi["t_rep"], bins=bins, density=True, color=pal["data"], alpha=0.55, lw=0)
    ax.hist(zip_["t_rep"], bins=bins, density=True, color=pal["posterior"], alpha=0.45,
            lw=0)
    plots.mark_truth(ax, t_obs, f"実データ {t_obs:.3f}")
    ax.annotate(f"素のポアソン\n平均 {poi['mean_zero']:.3f}\np = {poi['p']:.3f}",
                xy=(poi["mean_zero"], ax.get_ylim()[1] * 0.55), ha="center",
                fontsize=6.0, color=pal["data"])
    ax.annotate(f"ゼロ過剰\n平均 {zip_['mean_zero']:.3f}\np = {zip_['p']:.2f}",
                xy=(zip_["mean_zero"], ax.get_ylim()[1] * 0.30), ha="center",
                fontsize=6.0, color=pal["posterior"])
    ax.set_xlabel("ゼロの割合（事後予測データ1本ごと）")
    ax.set_ylabel("密度")
    ax.set_title("① 検査統計量: モデルはゼロの多さを再現できるか")

    ax = axes[1]
    kmax = 12
    ks = np.arange(kmax + 1)
    obs_counts = np.array([(y == k).mean() for k in ks])
    poi_counts = np.array([(poi["rep"] == k).mean() for k in ks])
    zip_counts = np.array([(zip_["rep"] == k).mean() for k in ks])
    ax.bar(ks, obs_counts, width=0.8, color=pal["data"], alpha=0.55, lw=0, zorder=2)
    ax.plot(ks, poi_counts, color=pal["truth"], lw=1.2, marker="o", ms=2.4, zorder=4)
    ax.plot(ks, zip_counts, color=pal["posterior"], lw=1.2, marker="s", ms=2.4,
            ls="--", dashes=(4, 2.0), zorder=4)
    ax.annotate("実データ", xy=(6.0, obs_counts[6] + 0.02), fontsize=6.0,
                color=pal["ink2"])
    ax.annotate("素のポアソン", xy=(2.6, poi_counts[3] + 0.03), fontsize=6.0,
                color=pal["truth"])
    ax.annotate("ゼロ過剰ポアソン", xy=(4.0, zip_counts[5] + 0.06), fontsize=6.0,
                color=pal["posterior"])
    ax.set_xlabel("件数")
    ax.set_ylabel("割合")
    ax.set_title("② 事後予測の分布と実データ")

    plots.save(fig, "fig-16-6-posterior-predictive-check.png")


def main() -> None:
    plots.setup()
    y = make_data(SEED)
    t_obs = float((y == 0).mean())

    print(f"--- ゼロ過剰なカウントデータ n={N}（seed={SEED}）---")
    print(f"  真の生成過程: {1 - PSI_TRUE:.0%} は構造的にゼロ、"
          f"残りが Poisson({LAM_TRUE})")
    print(f"  実データ: 平均 {y.mean():.3f}、分散 {y.var(ddof=1):.3f}"
          f"（ポアソンなら等しいはず）、ゼロの割合 {t_obs:.3f}\n")

    with sim.Timer("  素のポアソン"):
        poi = check(fit_poisson(y, SEED + 1), y)
    with sim.Timer("  ゼロ過剰ポアソン"):
        zip_ = check(fit_zip(y, SEED + 2), y)

    print(f"\n{'':>18}{'R̂':>8}{'ESS':>10}{'発散':>8}"
          f"{'事後予測のゼロ割合':>20}{'ベイジアン p 値':>18}")
    for name, r in (("素のポアソン", poi), ("ゼロ過剰ポアソン", zip_)):
        print(f"  {name:<16}{r['rhat']:>8.3f}{r['ess']:>10,.0f}{r['div']:>8}"
              f"{r['mean_zero']:>20.3f}{r['p']:>18.3f}")
    print(f"  {'実データ':<16}{'—':>8}{'—':>10}{'—':>8}{t_obs:>20.3f}")

    print(f"\n  素のポアソンは R̂={poi['rhat']:.3f}、発散 {poi['div']} 本。"
          "収束診断は全部きれいに通っている。")
    print(f"  それでも事後予測のゼロ割合は {poi['mean_zero']:.3f} で、実データの "
          f"{t_obs:.3f} に届かない。")
    print(f"  ベイジアン p 値 {poi['p']:.3f}"
          f"（{poi['t_rep'].size:,} 本の事後予測のうち、実データ以上にゼロが多かった割合）。")
    print("  「回った」と「合っている」は別である、というのがこの節の全部である。\n")

    print(f"  ゼロ過剰ポアソンに替えると {zip_['mean_zero']:.3f}、"
          f"p 値 {zip_['p']:.3f} で、実データが事後予測の真ん中に来る。")
    psi_post = zip_["summary"].loc["psi", "mean"]
    lam_post = zip_["summary"].loc["lam", "mean"]
    print(f"  推定された ψ={psi_post:.3f}（真値 {PSI_TRUE}）、"
          f"λ={lam_post:.3f}（真値 {LAM_TRUE}）。")
    print(f"  素のポアソンの λ は {poi['summary'].loc['lam', 'mean']:.3f} で、"
          "ゼロの多さを平均の低さとして飲み込んでいた。\n")

    print("  ベイジアン p 値は 0.5 に近いほど良い、という向きの数字ではない。")
    print("  0 や 1 に張り付いたときに「そこが再現できていない」と教えるためのもので、")
    print("  どの統計量で測るかは、モデルに何をさせたいかで決める（ここではゼロの割合）。")

    draw(y, poi, zip_)


if __name__ == "__main__":
    main()
