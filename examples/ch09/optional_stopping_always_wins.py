"""有意になるまでデータを足し続ければ、効果がゼロでもいつかは必ず勝つ。

「まだ有意じゃないから、もう少しデータを集めよう」は、実務でいちばんよく踏まれる罠で
ある。悪意はないし、追加のデータは本物である。それでも第一種の誤りは名目の α を大きく
超える。止めどきをデータの側に決めさせた瞬間、「何回検定したか」が観測されなくなる
からだ。

真の差がゼロのデータを n=20 から10件ずつ足し、そのつど検定して有意になったら止める。
上限を n=100 / 1,000 / 10,000 と変えて、棄却率がどこまで伸びるかを 10,000 回数える。

    uv run python examples/ch09/optional_stopping_always_wins.py
"""

import numpy as np
from scipy import special

from toukei_tashikame import plots, sim

MIN_N, STEP, N_MAX = 20, 10, 10_000
CAPS = (100, 1_000, 10_000)
ALPHA = 0.05
TRIALS = 10_000
SEED = 92

LOOKS = np.arange(MIN_N, N_MAX + 1, STEP)   # 覗くタイミング（999 回）


def one_trial(rng) -> tuple[float, float]:
    """1本の実験を n=10,000 まで走らせ、最初に有意になった n を返す。

    A/B の差の系列を1本の標本と見て、μ=0 の1標本 t 検定を覗くたびに当てる。
    累積和を先に作っておけば、999 回ぶんの t 統計量が一度のベクトル演算で出る。
    """
    x = rng.normal(0.0, 1.0, size=N_MAX)
    csum = np.cumsum(x)
    csum2 = np.cumsum(x * x)

    n = LOOKS
    mean = csum[n - 1] / n
    var = (csum2[n - 1] - n * mean * mean) / (n - 1)
    t = mean / np.sqrt(var / n)
    p = 2 * special.stdtr(n - 1, -np.abs(t))

    hit = np.flatnonzero(p < ALPHA)
    stop = float(n[hit[0]]) if hit.size else np.inf
    return stop, float(p[-1])         # 止めた n と、覗かず n=10,000 まで行ったときの p


def draw(stop: np.ndarray, curve: np.ndarray) -> None:
    fig, axes = plots.figure(1, 2, w=2.0, h=0.95)

    ax = axes[0]
    ax.plot(LOOKS, curve, color=plots.PALETTE["reject"], lw=1.3, zorder=4)
    ax.fill_between(LOOKS, ALPHA, curve, color=plots.PALETTE["reject"], alpha=0.20, lw=0)
    plots.mark_truth(ax, ALPHA, "名目 α = 0.05", axis="y")
    ax.set_xscale("log")
    ax.set_xlabel("上限とした n（対数目盛）")
    ax.set_ylabel("棄却率（真の差はゼロ）")
    ax.set_ylim(0, max(curve) * 1.15)
    ax.set_title("覗き続けるほど棄却率は伸びる")
    for cap in CAPS:
        y = curve[np.searchsorted(LOOKS, cap)]
        ax.annotate(f"n≤{cap:,} で {y:.3f}", xy=(cap, y), xytext=(-2, 4),
                    textcoords="offset points", ha="right", fontsize=6.0,
                    color=plots.PALETTE["ink2"])

    ax = axes[1]
    stopped = stop[np.isfinite(stop)]
    ax.hist(np.log10(stopped), bins=40, color=plots.PALETTE["data"], alpha=0.6, lw=0)
    med = float(np.median(stopped))
    ax.axvline(np.log10(med), color=plots.PALETTE["estimate"], lw=1.2, zorder=5)
    ax.annotate(f"中央値 n = {med:,.0f}", xy=(np.log10(med), 0.95),
                xycoords=("data", "axes fraction"), xytext=(3, 0),
                textcoords="offset points", fontsize=6.0, color=plots.PALETTE["estimate"])
    ax.set_xlabel("止めたときの n（$\\log_{10}$）")
    ax.set_ylabel("試行数")
    ax.set_title("止まるのは早い。だから気づかない")

    plots.save(fig, "fig-9-2-optional-stopping.png")


def main() -> None:
    plots.setup()
    with sim.Timer("9-2 逐次的な覗き見"):
        out = sim.repeat(one_trial, trials=TRIALS, seed=SEED, progress=False)
    stop, p_fixed = out[:, 0], out[:, 1]

    # 覗く上限を変えるのは、止めた n をその上限で切るのと同じこと。
    curve = np.array([(stop <= n).mean() for n in LOOKS])

    print(f"真の差はゼロ。n={MIN_N} から {STEP} 件ずつ足し、有意になったら止める"
          f"（{TRIALS:,} 回）\n")
    print(f"{'上限とした n':>12}{'覗いた回数':>10}{'棄却率':>10}{'±95%':>9}"
          f"{'止めた n の中央値':>18}")
    for cap in CAPS:
        rate = float((stop <= cap).mean())
        se = np.sqrt(rate * (1 - rate) / TRIALS)
        med = np.median(stop[stop <= cap]) if rate > 0 else np.nan
        looks = int((cap - MIN_N) // STEP + 1)
        print(f"{cap:>12,}{looks:>10}{rate:>10.4f}{1.96 * se:>9.4f}{med:>18,.0f}")

    fixed = float((p_fixed < ALPHA).mean())
    print(f"\n覗かずに n={N_MAX:,} で1回だけ検定すると {fixed:.4f}（設計どおり 5%）。")
    print("同じデータ、同じ検定、同じ α。違うのは「いつ止めたか」を誰が決めたかだけである。")
    draw(stop, curve)


if __name__ == "__main__":
    main()
