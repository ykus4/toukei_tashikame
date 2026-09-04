"""中心極限定理 — どんな形から出発しても、標本平均は正規に寄っていく。

一様・指数・対数正規・ベルヌーイ(p=0.1) の4つ。左右対称なもの、右に裾を引くもの、
極端に歪んだもの、0と1しか取らないもの。この4つで n=1,2,5,30,100 の標本平均を各20,000回
作り、標準化して正規分布と重ねる。

見るべきは「近づく」ことではなく**近づく速さが元の分布の歪みで決まる**こと。一様分布は
n=2 でもう三角形、n=5 でほぼ正規。対数正規は n=100 でもまだ右に尾を引いている。
「n≥30 なら正規でよい」という経験則がどこで破れるかは、この表の歪度の列に出る。

    uv run python examples/ch05/clt_four_distributions_grid.py
"""

import unicodedata

import numpy as np

from toukei_tashikame import describe, plots

TRIALS = 20_000
N_LIST = [1, 2, 5, 30, 100]
SEED = 19

# 名前 → (標本を引く関数, 母平均, 母標準偏差, 母集団の歪度)
DISTS = {
    "一様 U(0,1)": (lambda rng, size: rng.uniform(0.0, 1.0, size=size),
                    0.5, np.sqrt(1 / 12), 0.0),
    "指数 Exp(1)": (lambda rng, size: rng.exponential(1.0, size=size),
                    1.0, 1.0, 2.0),
    "対数正規 σ=1": (lambda rng, size: rng.lognormal(0.0, 1.0, size=size),
                     np.exp(0.5), np.sqrt((np.e - 1) * np.e), 6.1849),
    "ベルヌーイ p=0.1": (lambda rng, size: (rng.random(size) < 0.1).astype(float),
                        0.1, np.sqrt(0.09), 0.8 / 0.3),
}


def pad(text: str, width: int) -> str:
    """全角を2桁として数えて左詰めする。日本語の見出しでも表の桁が揃う。"""
    w = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)
    return text + " " * max(width - w, 0)


def main() -> None:
    plots.setup()
    rng = np.random.default_rng(SEED)
    skew = {}
    fig, axes = plots.figure(len(DISTS), len(N_LIST), h=2.6, w=2.0, sharex=True)

    for i, (name, (draw, mu, sigma, _)) in enumerate(DISTS.items()):
        skew[name] = []
        for j, n in enumerate(N_LIST):
            # (試行数, n) をまとめて引いて行ごとに平均する。1行が1つの標本。
            means = draw(rng, (TRIALS, n)).mean(axis=1)
            z = (means - mu) / (sigma / np.sqrt(n))
            skew[name].append(describe.skewness(z))

            ax = axes[i, j]
            ax.hist(z, bins=60, density=True, range=(-4, 4),
                    color=plots.PALETTE["data"], alpha=0.55, lw=0)
            g = np.linspace(-4, 4, 200)
            ax.plot(g, np.exp(-0.5 * g**2) / np.sqrt(2 * np.pi),
                    color=plots.PALETTE["truth"], lw=1.0, ls="--", dashes=(4, 2.0))
            ax.set_xlim(-4, 4)
            ax.set_yticks([])
            ax.set_xticks([-2, 0, 2])
            ax.set_title(f"n={n}  歪度 {skew[name][-1]:+.2f}", fontsize=6.0)
            if j == 0:
                ax.set_ylabel(name, fontsize=6.2)

    print(f"--- 標準化した標本平均の歪度（各 {TRIALS:,}回）---")
    print("  " + pad("分布", 20) + f"{'母集団':>6}"
          + "".join(f"{'n=' + str(n):>9}" for n in N_LIST))
    for name, (_, _, _, pop_skew) in DISTS.items():
        cells = "".join(f"{v:>9.2f}" for v in skew[name])
        print("  " + pad(name, 20) + f"{pop_skew:>8.2f}{cells}")
    print("  ← 正規分布の歪度は 0。0 への近づき方は 1/√n（母集団の歪度 ÷ √n が目安）")

    ln = skew["対数正規 σ=1"]
    print(f"\n  対数正規: n=1 で {ln[0]:.2f} → n=30 で {ln[3]:.2f} → n=100 で {ln[4]:.2f}")
    print(f"  一様:     n=5 ですでに {skew['一様 U(0,1)'][2]:+.2f}")
    print("  同じ n=30 でも、正規に見えるかどうかは元の分布で決まる")

    fig.suptitle("中心極限定理 — 4つの分布 × 5つの n（赤の破線は N(0,1)）", fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    plots.save(fig, "fig-5-5-clt-grid.png")


if __name__ == "__main__":
    main()
