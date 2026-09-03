"""本書の最初の1本。95%信頼区間の「95%」を数え上げる。

μ=50 の母集団から n=20 を引いて t 区間を作る、を 10,000 回。そのうち何本が真の μ を
包んだかを数える。定義文を読むかわりに数えると、95% が何の割合なのかが一度で分かる。
動いているのは区間のほうで、μ は一度も動かない。

この数字が出れば環境構築は完了である。

    uv run python examples/ch00/hello_confidence_interval.py
"""

import numpy as np
from scipy import stats

from toukei_tashikame import sim

MU, SIGMA, N = 50.0, 10.0, 20


def one_trial(rng):
    """標本を1つ引いて、t 分布に基づく 95% 信頼区間を返す。"""
    x = rng.normal(MU, SIGMA, size=N)
    se = x.std(ddof=1) / np.sqrt(N)
    half = stats.t.ppf(0.975, df=N - 1) * se
    return x.mean() - half, x.mean() + half


def main() -> None:
    res = sim.coverage(one_trial, truth=MU, trials=10_000, seed=0, progress=False)
    print(f"{res.rate:.4f} ± {1.96 * res.se:.4f}")


if __name__ == "__main__":
    main()
