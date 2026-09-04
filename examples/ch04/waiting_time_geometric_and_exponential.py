"""待ち時間の分布は「これまで待った時間」を覚えていない — 無記憶性を数え上げる。

成功確率 $p=0.05$ のガチャを初回成功まで引く回数（幾何分布）と、平均20分に1回起きる
故障までの時間（指数分布）。離散と連続の違いはあるが、どちらも
$\\Pr[T>s+t \\mid T>s] = \\Pr[T>t]$ を満たす。つまり「もう10回外した」ことは、
これから何回かかるかの見込みを1ミリも変えない。

「そろそろ当たるはず」が誤りである理由を、200,000回の待ち時間を数え上げて確かめる。

    uv run python examples/ch04/waiting_time_geometric_and_exponential.py
"""

import numpy as np

from toukei_tashikame import plots

P = 0.05
MEAN = 20.0
DRAWS = 200_000
SEED = 13


def main() -> None:
    rng = np.random.default_rng(SEED)
    geo = rng.geometric(P, size=DRAWS).astype(float)   # 初回成功までの試行回数（1,2,...）
    exp = rng.exponential(MEAN, size=DRAWS)            # 次の事象までの時間

    print(f"--- 初回成功までの回数（幾何分布 p={P}）と、次の事象までの時間"
          f"（指数分布 平均{MEAN:g}）を各 {DRAWS:,} 回 ---")
    print(f"{'':<14}{'平均':>10}{'理論':>10}{'中央値':>10}{'理論':>10}{'SD':>10}{'理論':>10}")
    print(f"{'幾何分布':<14}{geo.mean():>10.2f}{1 / P:>10.2f}{np.median(geo):>10.2f}"
          f"{np.ceil(np.log(0.5) / np.log(1 - P)):>10.2f}"
          f"{geo.std(ddof=1):>10.2f}{np.sqrt(1 - P) / P:>10.2f}")
    print(f"{'指数分布':<14}{exp.mean():>10.2f}{MEAN:>10.2f}{np.median(exp):>10.2f}"
          f"{MEAN * np.log(2):>10.2f}{exp.std(ddof=1):>10.2f}{MEAN:>10.2f}")
    print(f"  平均 {MEAN:g} なのに中央値は {MEAN * np.log(2):.1f}。"
          "半分以上は平均より早く終わり、残りが長く尾を引く")

    print("\n--- 無記憶性: すでに10だけ待った人の、その先の待ち時間 ---")
    print(f"{'':<14}{'Pr[T>30 | T>10]':>18}{'Pr[T>20]':>12}{'差':>10}")
    for name, x in (("幾何分布", geo), ("指数分布", exp)):
        cond = float((x[x > 10] > 30).mean())
        plain = float((x > 20).mean())
        print(f"{name:<14}{cond:>18.4f}{plain:>12.4f}{cond - plain:>+10.4f}")
    print(f"  指数分布の理論値は e^-1 = {np.exp(-1):.4f}。"
          f"幾何分布は (1-p)^20 = {(1 - P) ** 20:.4f}")
    print("  「10だけ待った」という条件をつけても、その先の分布は元のまま。"
          "分布は過去を覚えていない")

    print("\n--- 待つほど有利にはならない ---")
    for s in (0, 10, 20, 50, 100):
        rest = exp[exp > s] - s
        print(f"  すでに {s:>3} 待った人（{rest.size:>6,} 人）の残り待ち時間の平均 "
              f"{rest.mean():>6.2f}")
    print("  条件をどこで切っても平均は 20 のまま。これが「そろそろ当たる」が"
          "成り立たないということ")

    plots.setup()
    fig, axes = plots.figure(1, 2, w=1.0, h=0.85, constrained_layout=True)

    ax = axes[0]
    t = np.linspace(0, 80, 400)
    surv = np.array([(exp > u).mean() for u in t])
    ax.plot(t, surv, color=plots.PALETTE["data"], lw=1.2)
    ax.annotate("Pr[T > t]（実測）", xy=(30, surv[150]), fontsize=6.0,
                color=plots.PALETTE["data"], xytext=(4, 4), textcoords="offset points")
    shifted = np.array([(exp[exp > 10] > 10 + u).mean() for u in t])
    ax.plot(t, shifted, color=plots.PALETTE["truth"], lw=1.1, ls="--", dashes=(4, 2.0))
    ax.annotate("Pr[T > 10+t | T > 10]", xy=(42, shifted[210]), fontsize=6.0,
                color=plots.PALETTE["truth"], xytext=(2, 6), textcoords="offset points")
    ax.set_yscale("log")
    ax.set_xlabel("さらに待つ時間 t")
    ax.set_ylabel("生存確率")
    ax.set_title("2本は重なる（無記憶性）")

    ax = axes[1]
    ax.hist(exp[exp < 100], bins=60, density=True, color=plots.PALETTE["data"],
            alpha=0.55, lw=0)
    rest = exp[exp > 10] - 10
    ax.hist(rest[rest < 100], bins=60, density=True, histtype="step",
            color=plots.PALETTE["truth"], lw=1.1)
    ax.annotate("10だけ待った後の残り時間", xy=(30, 0.022), fontsize=6.0,
                color=plots.PALETTE["truth"])
    ax.set_xlabel("待ち時間")
    ax.set_ylabel("密度")
    ax.set_title("分布そのものが同じ形")

    plots.save(fig, "fig-4-5-memoryless.png")


if __name__ == "__main__":
    main()
