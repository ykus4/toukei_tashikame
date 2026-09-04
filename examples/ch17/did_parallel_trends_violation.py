"""差分の差分は、平行トレンドが破れた瞬間に壊れる。8期のパネルで 10,000 回ずつ数える。

DID は「処置がなければ2群は同じ傾きで動いたはず」という反実仮想に全体重を預けている。
この仮定は観測できない。だから確かめる方法は、仮定が成り立つ世界と破れた世界を自分で
作って、同じ推定量に同じ手順を通すことしかない。

ここでは処置群だけ元から年 0.4 ずつ速く伸びる世界を用意する。処置の真の効果は
どちらの世界でも 2.0 のままである。破れた世界では推定値が上に寄り、95%区間の被覆が
崩れる。事前トレンド検定は破れをある程度は捕まえるが、取りこぼしも多い——
**検定を通ったことは、平行トレンドの証明にはならない。**

    uv run python examples/ch17/did_parallel_trends_violation.py
"""

import numpy as np
from scipy import stats

from toukei_tashikame import plots, sim

N_UNIT, T_PERIOD, T_START = 100, 8, 4     # 100 店舗 × 8 期、5期目から処置
EFFECT, EXTRA_SLOPE = 2.0, 0.4            # 真の効果と、破れた世界の傾きの差
NOISE_SD = 1.5                            # 期ごとのばらつき（事前トレンド検定の検出力を決める）
TRIALS, SEED, ALPHA = 10_000, 178, 0.05

PERIODS = np.arange(T_PERIOD)
PRE, POST = PERIODS < T_START, PERIODS >= T_START


def make_panel(rng, parallel: bool):
    """(N_UNIT, T_PERIOD) のパネルを作る。前半の店舗が処置群。"""
    treated = np.arange(N_UNIT) < N_UNIT // 2
    unit_effect = rng.normal(0.0, 1.0, size=(N_UNIT, 1))       # 店舗の水準差
    time_effect = 0.5 * PERIODS                                # 全店に共通の時間トレンド
    y = unit_effect + time_effect + rng.normal(0.0, NOISE_SD, size=(N_UNIT, T_PERIOD))
    y = y + EFFECT * np.outer(treated, POST)                   # 処置の効果（真値）
    if not parallel:
        y = y + EXTRA_SLOPE * np.outer(treated, PERIODS)       # 処置群だけ元から速い
    return y, treated


def did_interval(y, treated):
    """DID を手で計算する。店舗ごとに「後 − 前」を作り、その群間差を取るだけ。"""
    delta = y[:, POST].mean(axis=1) - y[:, PRE].mean(axis=1)   # 第1の差分（時間）
    a, b = delta[treated], delta[~treated]                     # 第2の差分（群）
    est = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
    h = stats.t.ppf(0.975, a.size + b.size - 2) * se
    return est - h, est + h


def pretrend_pvalue(y, treated):
    """処置前だけを使って、2群の傾きが同じかを検定する（プラセボ検定）。"""
    pre_y = y[:, PRE]
    x = PERIODS[PRE] - PERIODS[PRE].mean()
    slope = (pre_y * x).sum(axis=1) / (x**2).sum()             # 店舗ごとの処置前の傾き
    a, b = slope[treated], slope[~treated]
    se = np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
    return float(2 * stats.t.sf(abs(a.mean() - b.mean()) / se, df=a.size + b.size - 2))


def one_trial(rng, parallel: bool):
    y, treated = make_panel(rng, parallel)
    lo, hi = did_interval(y, treated)
    return lo, hi, pretrend_pvalue(y, treated)


def summarize(out, label: str) -> dict:
    est = out[:, :2].mean(axis=1)
    covered = (out[:, 0] <= EFFECT) & (EFFECT <= out[:, 1])
    rate = float(covered.mean())
    pre_rej = float((out[:, 2] < ALPHA).mean())
    return {"label": label, "est": float(est.mean()), "sd": float(est.std(ddof=1)),
            "cov": rate, "cov_se": float(np.sqrt(rate * (1 - rate) / len(out))),
            "pre": pre_rej, "pre_se": float(np.sqrt(pre_rej * (1 - pre_rej) / len(out))),
            }


def draw() -> None:
    fig, axes = plots.figure(1, 2, w=2.0)
    pal = plots.PALETTE

    for ax, parallel, title in ((axes[0], True, "平行トレンド成立"),
                                (axes[1], False, "処置群だけ年 0.4 速い")):
        y, treated = make_panel(np.random.default_rng(SEED), parallel)
        m1, m0 = y[treated].mean(axis=0), y[~treated].mean(axis=0)
        ax.plot(PERIODS, m0, color=pal["data"], lw=1.2, marker="o", ms=3, zorder=3)
        ax.plot(PERIODS, m1, color=pal["estimate"], lw=1.2, marker="o", ms=3, zorder=3)
        # 処置がなかった場合の処置群（反実仮想）。観測できないので破線で描く。
        cf = m1.copy()
        cf[POST] = cf[POST] - EFFECT
        ax.plot(PERIODS[POST], cf[POST], color=pal["truth"], lw=1.0, ls="--",
                dashes=(4, 2.0), zorder=4)
        ax.axvline(T_START - 0.5, color=pal["reject"], lw=0.9, ls="--", dashes=(3, 2.0),
                   zorder=2)
        ax.annotate("処置開始", xy=(T_START - 0.5, 0.03), xycoords=("data", "axes fraction"),
                    fontsize=6.0, color=pal["reject"], xytext=(3, 0),
                    textcoords="offset points")
        ax.annotate("処置群", xy=(PERIODS[-1], m1[-1]), fontsize=6.2, color=pal["estimate"],
                    ha="right", va="bottom", xytext=(-2, 3), textcoords="offset points")
        ax.annotate("対照群", xy=(PERIODS[-1], m0[-1]), fontsize=6.2, color=pal["data"],
                    ha="right", va="top", xytext=(-2, -4), textcoords="offset points")
        ax.annotate("処置が無かった場合", xy=(PERIODS[T_START], cf[T_START]), fontsize=6.2,
                    color=pal["truth"], ha="left", va="bottom",
                    xytext=(3, 4), textcoords="offset points")
        ax.set_title(title, fontsize=7)
        ax.set_xlabel("期")
        ax.set_ylabel("平均の結果")
    fig.tight_layout()
    plots.save(fig, "fig-17-8-did-parallel-trends.png")


def main() -> None:
    plots.setup()
    with sim.Timer(f"{TRIALS:,} 回 × 2つの世界") as timer:
        rows = [
            summarize(sim.repeat(lambda r: one_trial(r, True), trials=TRIALS, seed=SEED,
                                 progress=False), "平行トレンド成立"),
            summarize(sim.repeat(lambda r: one_trial(r, False), trials=TRIALS, seed=SEED,
                                 progress=False), "平行トレンドが破れる"),
        ]

    print(f"\n--- {N_UNIT} 店舗 × {T_PERIOD} 期のパネル、{T_START + 1} 期目から処置"
          f"（真の効果 {EFFECT}, seed={SEED}）---\n")
    print(f"{'世界':<24}{'平均推定':>10}{'バイアス':>10}{'SD':>8}{'被覆':>18}")
    for r in rows:
        print(f"{r['label']:<24}{r['est']:>10.3f}{r['est'] - EFFECT:>+10.3f}"
              f"{r['sd']:>8.3f}{r['cov']:>14.4f} ± {1.96 * r['cov_se']:.4f}")

    bias = rows[1]["est"] - EFFECT
    drift = EXTRA_SLOPE * (PERIODS[POST].mean() - PERIODS[PRE].mean())
    print(f"\n  破れた世界のバイアス {bias:+.3f} は、傾きの差 {EXTRA_SLOPE} ×"
          f"（処置後の平均期 {PERIODS[POST].mean():.1f} − 処置前 {PERIODS[PRE].mean():.1f}）"
          f" = {drift:.3f} と一致する。")
    print("  DID は「処置後に開いた差」を全部処置のせいにするので、"
          "元から開きつつあったぶんがそのまま乗る。")

    print(f"\n--- 事前トレンド検定（処置前 {T_START} 期の傾きが等しいか, α={ALPHA}）---\n")
    print(f"{'世界':<24}{'棄却率':>18}{'読み方':>10}")
    for r in rows:
        note = "第一種の誤り" if r["label"].endswith("成立") else "破れの検出力"
        print(f"{r['label']:<24}{r['pre']:>14.4f} ± {1.96 * r['pre_se']:.4f}   {note}")
    print(f"\n  破れているのに検定が通ってしまう確率は {1 - rows[1]['pre']:.4f}。"
          "処置前が平行に見えたからといって、")
    print("  処置後も平行だった保証はどこにもない。事前トレンド検定は反証の道具であって、")
    print("  「通ったから DID を使ってよい」という許可証ではない。")
    print(f"  しかも検出力は処置前の期数で決まる。ここでは {T_START} 期しかない。")

    print(f"\n  平行トレンドが成り立つときの被覆 {rows[0]['cov']:.4f} は名目どおり。"
          "推定量が悪いのではなく、")
    print("  仮定が成り立つかどうかだけで結果が決まる。DID の中身は「後−前」の引き算2回で、")
    print(f"  そこに難しいことは何もない。難しいのは仮定のほうである。（{timer.elapsed:.1f} 秒）")

    draw()


if __name__ == "__main__":
    main()
