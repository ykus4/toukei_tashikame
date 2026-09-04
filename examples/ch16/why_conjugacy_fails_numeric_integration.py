"""なぜ MCMC が要るのか — 正規化定数の積分は次元とともに手に負えなくなる。

事後分布は「事前 × 尤度 を正規化したもの」でしかない（第15章）。問題は正規化、つまり
分母の積分である。1次元なら 1000 点に刻んで足すだけで済む。ところが刻みの細かさを
保ったままパラメータを増やすと、格子点は次元の指数で増える。2次元で 10^6、5次元で
10^15、10次元で 10^30。

ここでは実際にグリッド積分を回して1格子点あたりの秒数を測り、その実測レートから
高次元の所要時間とメモリを見積もる。グリッド近似が壊れる地点を数字で見ておくと、
MCMC が「難しい方法」ではなく「他に手がないから使う方法」であることが分かる。

    uv run python examples/ch16/why_conjugacy_fails_numeric_integration.py
"""

import time

import numpy as np

from toukei_tashikame import plots, sim

GRID_PER_DIM = 1000        # 1辺あたりの刻み数。この細かさを保ったまま次元を上げる
LO, HI = -6.0, 6.0         # 積分区間（事前が実質ゼロになるところまで）
# 実測は「メモリに載る範囲」で。1辺の刻み数を落として、1点あたりの秒数だけ取り出す。
MEASURE = ((1, 1000), (2, 700), (3, 200))
REPORT_DIMS = (1, 2, 3, 5, 10)

SECONDS_PER_YEAR = 365.25 * 24 * 3600
AGE_OF_UNIVERSE_YEARS = 1.38e10


def log_posterior_kernel(theta: np.ndarray) -> np.ndarray:
    """正規化していない事後の対数。

    パラメータ d 個が独立に N(0,1) の事前を持ち、各パラメータについて観測 y=0.5 を
    分散 1 で1個見た、という一番簡単な設定。積分の値は解析的に分かるので、
    グリッドの答え合わせができる。
    """
    return -0.5 * theta**2 - 0.5 * (0.5 - theta) ** 2


def exact_normalizing_constant(dim: int) -> float:
    """解析解。1次元の積分値の dim 乗（積の積分は積分の積）。"""
    one = np.sqrt(np.pi) * np.exp(-0.0625)
    return float(one**dim)


def grid_integrate(dim: int, m: int) -> tuple[float, int, float]:
    """d 次元の格子で正規化定数を数値積分する。``(値, 格子点数, 秒)`` を返す。"""
    axis = np.linspace(LO, HI, m)
    dx = axis[1] - axis[0]
    kernel_1d = np.exp(log_posterior_kernel(axis))

    t0 = time.perf_counter()
    # 各軸の寄与を外積で積み上げる。d 重ループは書かずに済むが、配列の要素数は
    # m**d そのもので、時間もメモリもそこで決まる。
    total = kernel_1d
    for _ in range(dim - 1):
        total = np.multiply.outer(total, kernel_1d)
    value = float(total.sum() * dx**dim)
    return value, m**dim, time.perf_counter() - t0


def human_time(seconds: float) -> str:
    """秒を人間の尺度に直す。年を超えたら年で書く。"""
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} μs"
    if seconds < 1.0:
        return f"{seconds * 1e3:.2f} ms"
    if seconds < 120:
        return f"{seconds:.2f} 秒"
    if seconds < 3 * SECONDS_PER_YEAR:
        return f"{seconds / 3600:.1f} 時間"
    years = seconds / SECONDS_PER_YEAR
    return f"{years:,.0f} 年" if years < 1e6 else f"{years:.1e} 年"


def human_bytes(nbytes: float) -> str:
    """バイト数を単位つきで。10^30 点の格子は GB では書けない。"""
    for unit, scale in (("KB", 1e3), ("MB", 1e6), ("GB", 1e9), ("TB", 1e12),
                        ("PB", 1e15), ("EB", 1e18)):
        if nbytes < scale * 1000:
            return f"{nbytes / scale:,.0f} {unit}"
    return f"{nbytes:.0e} B"


def draw(measured, rate) -> None:
    fig, axes = plots.figure(1, 2, w=2.0, h=1.0)
    pal = plots.PALETTE
    d = np.arange(1, 11)
    meas_dims = [row[0] for row in measured]
    meas_pts = [row[2] for row in measured]
    meas_sec = [row[3] for row in measured]

    ax = axes[0]
    ax.plot(d, float(GRID_PER_DIM) ** d, color=pal["truth"], lw=1.4, zorder=4)
    ax.plot(d, 4000.0 * d, color=pal["estimate"], lw=1.4, ls="--", dashes=(4, 2.0),
            zorder=4)
    ax.scatter(meas_dims, meas_pts, s=14, color=pal["truth"], zorder=5)
    ax.annotate("グリッド積分 $1000^d$", xy=(2.6, 1e14), fontsize=6.0, color=pal["truth"])
    ax.annotate("MCMC（次元に比例）", xy=(4.6, 3e2), fontsize=6.0, color=pal["estimate"])
    ax.annotate("● 実際に回した点", xy=(3.4, 2e5), fontsize=6.0, color=pal["ink2"])
    ax.set_yscale("log")
    ax.set_xlabel("パラメータの数 $d$")
    ax.set_ylabel("必要な評価点数")
    ax.set_title("① 格子は次元の指数で増える")

    ax = axes[1]
    est_years = np.array([float(GRID_PER_DIM) ** k * rate for k in d]) / SECONDS_PER_YEAR
    ax.plot(d, est_years, color=pal["truth"], lw=1.4, zorder=4)
    ax.scatter(meas_dims, np.asarray(meas_sec) / SECONDS_PER_YEAR, s=14,
               color=pal["truth"], zorder=5)
    ax.axhline(1.0, color=pal["reject"], lw=0.9, ls="--", dashes=(4, 2.2), zorder=3)
    ax.annotate("1年", xy=(1.2, 3.0), fontsize=6.0, color=pal["reject"])
    ax.axhline(AGE_OF_UNIVERSE_YEARS, color=pal["ink2"], lw=0.9, ls=":", zorder=3)
    ax.annotate("宇宙の年齢", xy=(1.2, AGE_OF_UNIVERSE_YEARS * 3), fontsize=6.0,
                color=pal["ink2"])
    ax.set_yscale("log")
    ax.set_ylim(1e-16, 1e20)
    ax.set_xlabel("パラメータの数 $d$")
    ax.set_ylabel("所要時間（年）")
    ax.set_title("② 実測レートから外挿した所要時間")

    plots.save(fig, "fig-16-1-curse-of-dimensionality.png")


def main() -> None:
    plots.setup()
    print(f"--- 正規化定数 ∫ p(θ)p(y|θ) dθ を1辺 {GRID_PER_DIM} 点の格子で積分する ---\n")

    measured = []
    with sim.Timer("  実測ぶん（1〜3次元）合計"):
        for dim, m in MEASURE:
            value, points, seconds = grid_integrate(dim, m)
            err = abs(value / exact_normalizing_constant(dim) - 1.0)
            measured.append((dim, m, points, seconds, err))

    print("\n  メモリに載る範囲で実際に回す:")
    print("    次元   1辺の刻み       格子点数        実測時間     解析解との相対誤差")
    for dim, m, points, seconds, err in measured:
        print(f"     {dim:<5}  {m:>6}      {points:>10.1e}   {human_time(seconds):>10}"
              f"   {err:>14.1e}")

    # 1格子点あたりの秒数。一番大きい格子（＝呼び出しの固定費が薄まる）から取る。
    biggest = max(measured, key=lambda r: r[2])
    rate = biggest[3] / biggest[2]
    print(f"\n  この計算機のレート: 1格子点あたり {rate * 1e9:.2f} ns\n")

    print(f"  1辺 {GRID_PER_DIM} 点の細かさを保ったまま次元を上げると:")
    print("    次元        格子点数     所要時間（上のレートから）      必要メモリ")
    for dim in REPORT_DIMS:
        points = float(GRID_PER_DIM) ** dim
        print(f"     {dim:<5}    {points:>10.1e}   {human_time(points * rate):>18}"
              f"   {human_bytes(points * 8):>18}")

    est5 = float(GRID_PER_DIM) ** 5 * rate
    est10 = float(GRID_PER_DIM) ** 10 * rate
    print(f"\n  5次元で {human_time(est5)}。ここまでは「待てば終わる」が、"
          f"格子を置く場所（{human_bytes(8e15)}）がもう無い。")
    print(f"  10次元は {human_time(est10)} で、宇宙の年齢の "
          f"{est10 / SECONDS_PER_YEAR / AGE_OF_UNIVERSE_YEARS:.0e} 倍。ここで打ち切る。\n")

    print("  刻みを粗くすれば点は減るが、そのぶん積分の誤差が増える。1辺 10 点まで")
    print("  落としても10次元で 10^10 点であり、しかも実務のモデル（回帰係数が20個、")
    print("  階層が3段）ではパラメータが100個を超える。グリッドはそこには行けない。")
    print("  MCMC の評価回数は次元に比例する程度で済む。指数と比例の差がすべてである。")

    draw(measured, rate)


if __name__ == "__main__":
    main()
