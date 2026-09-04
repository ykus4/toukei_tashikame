"""相関の信頼区間はなぜ z 変換を挟むのか — 被覆確率を数えて確かめる。

r は −1 と 1 で頭打ちになる。真の相関が 0.5、n=25 という小標本だと r の分布は右で
詰まって左に裾を引き、正規分布の形をしていない。それでも $r \\pm 1.96 \\times SE$ という
素朴な区間を作ると、対称な区間を非対称な分布にかぶせることになる。

Fisher の z 変換 $z=\\mathrm{arctanh}\\,r$ は、この $[-1,1]$ を実数全体へ引き伸ばす。
z の上では分布はほぼ正規で、標準誤差は $1/\\sqrt{n-3}$ という r に依らない定数になる。
z で区間を作って tanh で戻すと、r の側では非対称な区間が出てくる。

真値を知っているのはシミュレーションだけなので、10,000 回引いて「区間が 0.5 を包んだ
割合」を両方式について数える。95% と名乗る区間が本当に 95% かは、数えれば分かる。

    uv run python examples/ch11/fisher_z_interval_coverage.py
"""

import numpy as np
from scipy import stats

from toukei_tashikame import plots, sim, testing

RHO, N, TRIALS, SEED, CONF = 0.5, 25, 10_000, 114, 0.95


def sample_r(rng: np.random.Generator, n: int = N, rho: float = RHO) -> float:
    """相関 rho の2変量正規から n 個引いて、標本相関を1つ返す。"""
    x = rng.normal(size=n)
    y = rho * x + np.sqrt(1 - rho**2) * rng.normal(size=n)
    return float(np.corrcoef(x, y)[0, 1])


def naive_ci(r: float, n: int, conf: float = CONF) -> tuple[float, float]:
    """z 変換なしの素朴な区間。SE=(1-r²)/√(n-3) を使って r のまま対称に取る。"""
    half = stats.norm.ppf(0.5 + conf / 2) * (1 - r**2) / np.sqrt(n - 3)
    return r - half, r + half


def fisher_interval(rng: np.random.Generator) -> tuple[float, float]:
    return testing.fisher_z_ci(sample_r(rng), N, conf=CONF)


def naive_interval(rng: np.random.Generator) -> tuple[float, float]:
    return naive_ci(sample_r(rng), N, conf=CONF)


def main() -> None:
    plots.setup()
    print(f"--- 11-4 相関の 95% 区間（真の相関 ρ={RHO}, n={N}, {TRIALS:,}回, seed={SEED}）---")

    fisher = sim.coverage(fisher_interval, truth=RHO, trials=TRIALS, seed=SEED, progress=False)
    naive = sim.coverage(naive_interval, truth=RHO, trials=TRIALS, seed=SEED, progress=False)

    for name, res in [("Fisher の z 変換", fisher), ("素朴な正規近似  ", naive)]:
        w = res.intervals[:, 1] - res.intervals[:, 0]
        lo_miss = float((res.intervals[:, 1] < RHO).mean())
        hi_miss = float((res.intervals[:, 0] > RHO).mean())
        print(f"  {name} 被覆 {res.rate:.4f} ± {1.96 * res.se:.4f}"
              f"   区間幅の中央値 {np.median(w):.4f}"
              f"   下に外す {lo_miss:.4f} / 上に外す {hi_miss:.4f}")
    print(f"  名目は {CONF:.0%}。z 変換のほうが名目に当たり、素朴な区間は "
          f"{(CONF - naive.rate) * 100:.1f} ポイント足りない")
    print("  外し方の内訳も違う。素朴な区間は左右対称に作るせいで、片側に偏って外す")

    # 区間が [-1, 1] を飛び出すのは素朴な区間だけ。相関が 1 に近いほど起きやすい。
    print("\n--- 区間が [-1, 1] を飛び出すか（素朴な区間だけが持つ病気）---")
    for rho, n in [(RHO, N), (0.9, 10), (0.95, 8)]:
        rs_ = np.array([sample_r(np.random.default_rng(s_), n=n, rho=rho)
                        for s_ in np.random.SeedSequence(SEED).spawn(2_000)])
        hi = np.array([naive_ci(r_, n)[1] for r_ in rs_])
        print(f"  ρ={rho}, n={n:<3} 素朴な区間の上端が 1 を超えた回数 "
              f"{int((hi > 1).sum()):>5,} / 2,000（最大 {hi.max():.4f}）")
    print("  z 変換の区間は tanh で戻すので、定義上 [-1, 1] を出ない")

    r_ex = sample_r(np.random.default_rng(SEED))
    print(f"\n--- 1標本での見え方（r={r_ex:.4f}）---")
    lo_f, hi_f = testing.fisher_z_ci(r_ex, N, conf=CONF)
    lo_n, hi_n = naive_ci(r_ex, N, conf=CONF)
    print(f"  Fisher   [{lo_f: .4f}, {hi_f: .4f}]   点推定からの距離 "
          f"下 {r_ex - lo_f:.4f} / 上 {hi_f - r_ex:.4f}   ← 非対称")
    print(f"  素朴     [{lo_n: .4f}, {hi_n: .4f}]   点推定からの距離 "
          f"下 {r_ex - lo_n:.4f} / 上 {hi_n - r_ex:.4f}   ← 対称")

    # --- 図 ---
    rs = np.array([sample_r(np.random.default_rng(s))
                   for s in np.random.SeedSequence(SEED + 1).spawn(TRIALS)])
    fig, axes = plots.figure(1, 3, w=1.95, h=0.95)

    ax = axes[0]
    ax.hist(rs, bins=50, density=True, color=plots.PALETTE["data"], alpha=0.55, lw=0)
    plots.mark_truth(ax, RHO, f"真値 ρ = {RHO}")
    ax.set_title(f"r の分布（n={N}）  歪度 {stats.skew(rs):.3f}")
    ax.set_xlabel("標本相関 r")
    ax.set_ylabel("密度")

    ax = axes[1]
    z = np.arctanh(rs)
    ax.hist(z, bins=50, density=True, color=plots.PALETTE["data"], alpha=0.55, lw=0)
    zs = np.linspace(z.min(), z.max(), 200)
    ax.plot(zs, stats.norm.pdf(zs, np.arctanh(RHO), 1 / np.sqrt(N - 3)),
            color=plots.PALETTE["truth"], lw=1.2, ls="--", dashes=(4, 2.0), zorder=5)
    ax.set_title(f"z = arctanh r の分布  歪度 {stats.skew(z):.3f}")
    ax.set_xlabel("z（赤破線は正規, SD=1/√(n-3)）")

    ax = axes[2]
    missed = plots.coverage_stripes(ax, fisher.intervals, RHO, n_show=100)
    ax.set_title(f"Fisher 区間 100 本（外した {missed} 本が赤）")
    ax.set_xlabel("相関")
    fig.tight_layout()
    plots.save(fig, "fig-11-4-fisher-z-coverage.png")


if __name__ == "__main__":
    main()
