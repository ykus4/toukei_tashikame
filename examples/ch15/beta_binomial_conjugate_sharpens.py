"""共役事前なら、更新は足し算で終わる。データが増えるほど事後は尖る。

ベータ事前に二項の尤度を掛けると、また別のベータになる。これが共役ということで、
$\\mathrm{Beta}(a, b)$ に「$k$ 成功 $n-k$ 失敗」を見せた後の事後は
$\\mathrm{Beta}(a+k,\\ b+n-k)$ でしかない。積分も最適化も出てこない。

同じデータ列を n=1 から n=5000 まで伸ばしながら更新すると、事後が一点に集まっていく
過程がそのまま絵になる。尖り方は $1/\\sqrt{n}$ で、これは標準誤差と同じ速さである。
ベイズと頻度論はここで同じ景色を見ている。

    uv run python examples/ch15/beta_binomial_conjugate_sharpens.py
"""

import numpy as np

from toukei_tashikame import bayes, plots

P_TRUE, SEED = 0.28, 155
PRIOR_A, PRIOR_B = 1.0, 1.0
NS = (1, 5, 10, 25, 50, 100, 250, 500, 1000, 5000)


def posterior_sd(post: bayes.BetaPosterior) -> float:
    a, b = post.a, post.b
    return float(np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1))))


def draw(posts) -> None:
    fig, axes = plots.figure(2, 5, w=2.4, h=1.5)
    pal = plots.PALETTE
    grid = np.linspace(0.0, 1.0, 500)

    for ax, post in zip(axes.ravel(), posts, strict=True):
        y = post.pdf(grid)
        ax.plot(grid, y, color=pal["posterior"], lw=1.2, zorder=4)
        ax.fill_between(grid, y, color=pal["posterior"], alpha=0.20, lw=0, zorder=1)
        ax.axvline(P_TRUE, color=pal["truth"], lw=1.0, zorder=5)
        ax.set_title(f"n={post.n:,}  SD={posterior_sd(post):.4f}")
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_yticks([])
    fig.suptitle(f"事前 Beta(1,1) を同じ列で更新していく（赤い線が真値 {P_TRUE}）",
                 fontsize=7)
    fig.tight_layout()
    plots.save(fig, "fig-15-5-conjugate-updating.png")


def main() -> None:
    plots.setup()
    rng = np.random.default_rng(SEED)
    stream = (rng.random(max(NS)) < P_TRUE).astype(int)   # 1本の列を伸ばして使う

    posts = [bayes.beta_binomial(int(stream[:n].sum()), n, PRIOR_A, PRIOR_B) for n in NS]

    print(f"--- 事前 Beta({PRIOR_A:g}, {PRIOR_B:g})、真の CVR = {P_TRUE}、"
          f"同じ列の先頭 n 件で更新（seed={SEED}）---\n")
    print(f"{'n':>7}{'k':>7}{'事後':>18}{'事後平均':>11}{'事後SD':>10}"
          f"{'√(p(1-p)/n)':>13}{'比':>7}{'95%信用区間':>22}")
    for post in posts:
        sd = posterior_sd(post)
        approx = np.sqrt(P_TRUE * (1 - P_TRUE) / post.n)
        lo, hi = post.interval(0.95)
        print(f"{post.n:>7,}{post.k:>7}"
              f"{f'Beta({post.a:g}, {post.b:g})':>18}{post.mean:>11.4f}{sd:>10.4f}"
              f"{approx:>13.4f}{sd / approx:>7.2f}   [{lo:.4f}, {hi:.4f}]")

    sd10 = posterior_sd(posts[NS.index(10)])
    sd100 = posterior_sd(posts[NS.index(100)])
    sd1000 = posterior_sd(posts[NS.index(1000)])
    sd5000 = posterior_sd(posts[NS.index(5000)])
    print(f"\n  事後SD は n=10 で {sd10:.4f}、n=100 で {sd100:.4f}、"
          f"n=1000 で {sd1000:.4f}、n=5000 で {sd5000:.4f}。")
    print(f"  n を 100 倍したときの理屈は 10 倍。実測は n=10→1000 で "
          f"{sd10 / sd1000:.2f} 倍、n=50→5000 で "
          f"{posterior_sd(posts[NS.index(50)]) / sd5000:.2f} 倍。")
    print("  小さい n でずれるのは、事前 Beta(1,1) が「成功1・失敗1」ぶんの重さを")
    print("  まだ持っているから。表の右2列（事後SD と √(p(1-p)/n) の比）を見ると、")
    print("  n=25 を過ぎたあたりから 1.0 に貼りつく。事後の尖り方は標準誤差と同じ速さで、")
    print("  ベイズと頻度論はここで同じ $1/\\sqrt{n}$ を見ている。\n")

    print("  更新の中身は「成功を a に、失敗を b に足す」だけである:")
    p0 = posts[0]
    p1 = posts[1]
    print(f"    n={p0.n} まで  Beta({p0.a:g}, {p0.b:g})")
    print(f"    n={p1.n} まで  Beta({p1.a:g}, {p1.b:g})"
          f"   ← 成功 {p1.k - p0.k} 件と失敗 {(p1.n - p1.k) - (p0.n - p0.k)} 件を足しただけ")
    print("\n  逐次に更新しても、全データを一度に見せても、同じ事後になる。")
    print("  「昨日までの事後を今日の事前にする」がそのまま成り立つのが共役の効用で、")
    print("  ログを貯めずに (a, b) の2つだけ持ち回れば済む。")
    draw(posts)


if __name__ == "__main__":
    main()
