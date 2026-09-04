"""@slow — R̂・ESS・発散。同じ事後を、書き方だけ変えて壊す。

MCMC は「動いたから正しい」ではない。サンプラーが行けなかった場所があっても、鎖は
何食わぬ顔で数字を返す。それを捕まえるための3つの目印がある。

  R̂    複数の鎖が同じ場所に集まったか。1.01 を超えたら疑う
  ESS  自己相関のぶんだけ目減りした実質の標本数。目安は 400 以上
  発散  ハミルトニアンの積分が壊れた回数。1本でもあれば事後の一部を取り逃している

ここでは**同じ階層モデルを2通りに書く**。数学的には同じ事後だが、中心化
（theta ~ Normal(mu, tau)）は tau が小さいところで漏斗のように細くなり、NUTS が
そこへ入れずに発散する。非中心化（theta = mu + tau * z）は同じ事後を素直な形に
書き直すだけで、発散が消える。

    uv sync --extra slow
    uv run python examples/ch16/rhat_ess_divergences_diagnostics.py
"""

import arviz as az
import numpy as np
import pymc as pm

from toukei_tashikame import plots, sim

# 8つの施策の効果推定値と、その標準誤差（各施策のA/Bテストは別々の規模で回した）。
# 階層モデルの教科書的な例（eight schools）と同じ数値を使う。
EFFECT = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
SE = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])

SEED = 165
DRAWS, TUNE, CHAINS = 1_000, 1_000, 4
TARGET_ACCEPT = 0.90   # 両方に同じ設定を使う。違うのはモデルの書き方だけ


def fit_centered(seed: int):
    """中心化。素直に書くとこうなる。そして、これが壊れる。"""
    with pm.Model():
        mu = pm.Normal("mu", 0.0, 10.0)
        tau = pm.HalfNormal("tau", 10.0)
        theta = pm.Normal("theta", mu=mu, sigma=tau, shape=EFFECT.size)
        pm.Normal("y", mu=theta, sigma=SE, observed=EFFECT)
        return pm.sample(DRAWS, tune=TUNE, chains=CHAINS, random_seed=seed,
                         target_accept=TARGET_ACCEPT, progressbar=False)


def fit_noncentered(seed: int):
    """非中心化。theta を「mu から z 標準偏差ぶん離れた場所」として書く。

    事後は中心化と同じである。変わるのはサンプラーが歩く座標だけで、
    z と tau が独立になるぶん、漏斗の首が消える。
    """
    with pm.Model():
        mu = pm.Normal("mu", 0.0, 10.0)
        tau = pm.HalfNormal("tau", 10.0)
        z = pm.Normal("z", 0.0, 1.0, shape=EFFECT.size)
        theta = pm.Deterministic("theta", mu + tau * z)
        pm.Normal("y", mu=theta, sigma=SE, observed=EFFECT)
        return pm.sample(DRAWS, tune=TUNE, chains=CHAINS, random_seed=seed,
                         target_accept=TARGET_ACCEPT, progressbar=False)


def diagnostics(idata) -> dict:
    """R̂ の最大・ESS の最小・発散の本数。見るのはこの3つだけでよい。"""
    summary = az.summary(idata, var_names=["mu", "tau", "theta"])
    div = idata.sample_stats["diverging"].values
    return {
        "rhat": float(summary["r_hat"].max()),
        "ess": float(summary["ess_bulk"].min()),
        "div": int(div.sum()),
        "div_rate": float(div.mean()),
        "tau_mean": float(idata.posterior["tau"].values.mean()),
        "theta1": idata.posterior["theta"].values[..., 0].ravel(),
        "log_tau": np.log(idata.posterior["tau"].values).ravel(),
        "div_flat": div.ravel(),
        "summary": summary,
    }


def draw(cen, non) -> None:
    fig, axes = plots.figure(1, 2, w=2.0, h=1.0, sharey=True)
    pal = plots.PALETTE

    for ax, res, title in ((axes[0], cen, "中心化"), (axes[1], non, "非中心化")):
        ok = ~res["div_flat"].astype(bool)
        ax.scatter(res["theta1"][ok], res["log_tau"][ok], s=1.2, lw=0,
                   color=pal["posterior"], alpha=0.35, zorder=2)
        bad = res["div_flat"].astype(bool)
        ax.scatter(res["theta1"][bad], res["log_tau"][bad], s=5.0, lw=0,
                   color=pal["reject"], alpha=0.9, zorder=4)
        ax.set_xlim(-25, 45)
        ax.set_ylim(-3.5, 4.0)
        ax.set_xlabel("$\\theta_1$（施策1の効果）")
        ax.set_title(f"{title}: 発散 {res['div']} 本 / R̂={res['rhat']:.3f}")
        if res["div"]:
            ax.annotate("オレンジ＝発散した点。\n漏斗の首に集まっている",
                        xy=(0.03, 0.05), xycoords="axes fraction", fontsize=6.0,
                        color=pal["reject"], va="bottom")
        else:
            ax.annotate("発散なし。首の奥まで\nサンプラーが入れている",
                        xy=(0.03, 0.05), xycoords="axes fraction", fontsize=6.0,
                        color=pal["estimate"], va="bottom")
    axes[0].set_ylabel("$\\log \\tau$（施策間のばらつき）")

    plots.save(fig, "fig-16-5-divergences-funnel.png")


def main() -> None:
    plots.setup()
    print("--- 8つの施策の効果を階層モデルで推定する（同じ事後を2通りに書く）---")
    print(f"  観測: {np.array2string(EFFECT, precision=0)}")
    print(f"  標準誤差: {np.array2string(SE, precision=0)}\n")

    with sim.Timer("  中心化のサンプリング"):
        cen = diagnostics(fit_centered(SEED))
    with sim.Timer("  非中心化のサンプリング"):
        non = diagnostics(fit_noncentered(SEED))

    total = CHAINS * DRAWS
    print(f"\n  各 {CHAINS} 鎖 × {DRAWS:,} draws = {total:,} 本\n")
    print(f"{'':>12}{'R̂ の最大':>12}{'ESS の最小':>13}{'発散':>10}"
          f"{'発散率':>10}{'τ の事後平均':>14}")
    for name, r in (("中心化", cen), ("非中心化", non)):
        print(f"  {name:<10}{r['rhat']:>12.3f}{r['ess']:>13,.0f}{r['div']:>10,}"
              f"{r['div_rate']:>10.3f}{r['tau_mean']:>14.3f}")

    print("\n  非中心化の arviz.summary()（抜粋）:")
    print("    " + non["summary"].head(4).to_string().replace("\n", "\n    "))

    print(f"\n  中心化は発散を {cen['div']:,} 本出している。発散は「行けなかった」の記録で、")
    print("  取り逃しているのは τ が小さい領域、つまり『施策間に差がない』側の事後である。")
    print(f"  そのぶん τ の事後平均が {cen['tau_mean']:.3f} と "
          f"{non['tau_mean']:.3f} でずれる（中心化のほうが差を大きく見積もる）。")
    print(f"  ESS も {cen['ess']:,.0f} → {non['ess']:,.0f} と "
          f"{non['ess'] / cen['ess']:.1f} 倍に増えている。\n")

    print(f"  R̂ だけを見ていると見落としやすい。中心化の R̂ は {cen['rhat']:.3f} で、"
          "目安の 1.01 を")
    print("  わずかに外れる程度にしか壊れない。はっきり出るのは発散の本数のほうである。")
    print("  **発散はゼロでなければならない**、が実務の線引き。")
    print("  発散が出たときの手当ては3つ: 非中心化に書き直す / target_accept を上げる /")
    print("  事前を締める。まず試すのは、いつも書き直しである。")

    draw(cen, non)


if __name__ == "__main__":
    main()
