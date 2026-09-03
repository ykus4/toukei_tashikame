"""「10,000回まわして数える」を一行にする。本書の確かめる段はすべてここを通る。

95%信頼区間の95%が何なのかは、定義を読むより数えたほうが速い。ただし数える手続きを
110本のスクリプトで書き直すと、何を数えているのかがループの記述に埋もれる。このモジュール
はその手続きだけを引き受け、呼び出し側には「1回ぶんの試行」だけを書かせる。

    from toukei_tashikame import sim

    def one_trial(rng):
        x = rng.normal(50.0, 10.0, size=20)
        se = x.std(ddof=1) / np.sqrt(20)
        h = stats.t.ppf(0.975, 19) * se
        return x.mean() - h, x.mean() + h

    res = sim.coverage(one_trial, truth=50.0, trials=10_000, seed=0)
    print(f"{res.rate:.4f} ± {1.96 * res.se:.4f}")   # -> 0.9477 ± 0.0044

試行関数は ``rng`` を受け取る。本書のコードは ``np.random.seed`` を一度も呼ばない。
グローバルな乱数状態から引くと、並列にした瞬間に結果が変わり、「同じ数字が出る」という
この本の前提が崩れるからである。子シードは ``SeedSequence(seed).spawn(trials)`` で
試行ごとに作るので、``workers`` を変えても、途中で打ち切っても、i 番目の試行は必ず同じ
乱数を見る。
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

__all__ = [
    "CoverageResult",
    "RejectionResult",
    "Timer",
    "coverage",
    "progress_bar",
    "rejection_rate",
    "repeat",
    "sweep",
]


# ---------------------------------------------------------------------------
# 進捗表示
# ---------------------------------------------------------------------------


class progress_bar:
    """依存を増やさないための最小の進捗表示。TTY でなければ何も出さない。

    tqdm を入れれば済む話ではある。入れないのは、読者の最初の ``uv sync`` を軽くする
    ためと、CI のログに制御文字を吐かせないため。端末でなければ黙る。
    """

    def __init__(self, total: int, label: str = "") -> None:
        self.total = max(int(total), 1)
        self.label = label
        self.enabled = sys.stderr.isatty()
        self._done = 0
        self._last = -1.0

    def __enter__(self) -> progress_bar:
        return self

    def update(self, n: int = 1) -> None:
        self._done += n
        if not self.enabled:
            return
        now = time.monotonic()
        # 更新しすぎると、進捗の描画そのものが試行より重くなる。
        if now - self._last < 0.1 and self._done < self.total:
            return
        self._last = now
        frac = self._done / self.total
        width = 24
        filled = int(width * frac)
        bar = "█" * filled + "·" * (width - filled)
        print(f"\r{self.label} [{bar}] {self._done}/{self.total}", end="", file=sys.stderr)

    def __exit__(self, *exc: object) -> None:
        if self.enabled:
            print(file=sys.stderr)


class Timer:
    """経過秒を出す。5秒を超えたら警告する。

    本書は所要時間を紙に書く方針なので、書くべきものが出たことを実行時に知らせる。
    ここで黙っていると、読者の環境で3分かかる節が本文では一言も断られていない、
    という事故になる。
    """

    THRESHOLD_SEC = 5.0

    def __init__(self, label: str = "") -> None:
        self.label = label
        self.elapsed = 0.0

    def __enter__(self) -> Timer:
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed = time.perf_counter() - self._t0
        print(f"{self.label}: {self.elapsed:.2f} 秒")
        if self.elapsed > self.THRESHOLD_SEC:
            print(
                f"  ※ {self.THRESHOLD_SEC:.0f} 秒を超えた。本文に所要時間を書くこと",
                file=sys.stderr,
            )


# ---------------------------------------------------------------------------
# 中核
# ---------------------------------------------------------------------------


def _child_seeds(seed: int, trials: int) -> list[np.random.SeedSequence]:
    """試行 i のシードを、trials や workers に依存しない形で決める。

    ``spawn`` は呼ばれた順に子を作るので、まとめて一度に取り出す。ここで
    ``default_rng(seed + i)`` のような作り方をすると、隣り合うストリームが相関しうる。
    """
    return np.random.SeedSequence(seed).spawn(trials)


def _run_chunk(fn: Callable, seeds: Sequence[np.random.SeedSequence]) -> list:
    return [fn(np.random.default_rng(s)) for s in seeds]


def repeat(
    fn: Callable[[np.random.Generator], object],
    trials: int = 10_000,
    seed: int = 0,
    *,
    progress: bool = True,
    workers: int = 1,
    label: str = "",
) -> np.ndarray:
    """``fn(rng)`` を ``trials`` 回呼び、結果を配列にして返す。

    返り値の形は ``fn`` が返したものに従う。スカラを返せば ``(trials,)``、2要素の
    タプルを返せば ``(trials, 2)`` になる。区間を返す試行をそのまま
    :func:`coverage` に渡せるのはこのため。
    """
    seeds = _child_seeds(seed, trials)

    if workers > 1:
        # チャンクに割ってから配る。試行1回が軽い（本書のほとんどがそう）ときは、
        # プロセス間の往復のほうが試行より高くつく。
        chunks = np.array_split(np.arange(trials), workers * 4)
        out: list = [None] * trials
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_run_chunk, fn, [seeds[i] for i in idx]): idx
                for idx in chunks
                if len(idx)
            }
            with progress_bar(len(futures), label or "並列試行") as bar:
                for fut, idx in futures.items():
                    for pos, value in zip(idx, fut.result(), strict=False):
                        out[pos] = value
                    bar.update(1) if progress else None
        return np.asarray(out)

    values: list = []
    with progress_bar(trials, label or "試行") as bar:
        for s in seeds:
            values.append(fn(np.random.default_rng(s)))
            if progress:
                bar.update(1)
    return np.asarray(values)


@dataclass(frozen=True)
class CoverageResult:
    """被覆確率と、その推定自体の誤差。"""

    rate: float
    se: float
    covered: np.ndarray
    intervals: np.ndarray
    truth: float
    trials: int

    @property
    def ci(self) -> tuple[float, float]:
        """被覆率そのものの 95% 区間。"""
        h = 1.96 * self.se
        return self.rate - h, self.rate + h

    def __str__(self) -> str:
        return f"{self.rate:.4f} ± {1.96 * self.se:.4f}（{self.trials:,}回）"


@dataclass(frozen=True)
class RejectionResult:
    """棄却率。帰無が真なら第一種の誤り、対立が真なら検出力。"""

    rate: float
    se: float
    pvalues: np.ndarray
    alpha: float
    trials: int

    @property
    def ci(self) -> tuple[float, float]:
        h = 1.96 * self.se
        return self.rate - h, self.rate + h

    def __str__(self) -> str:
        return f"{self.rate:.4f} ± {1.96 * self.se:.4f}（α={self.alpha}, {self.trials:,}回）"


def _binomial_se(rate: float, trials: int) -> float:
    """シミュレーションそのものの標準誤差。

    0.9502 と 0.9487 の差を実装の違いだと読まれないために、必ず併記する。10,000回なら
    95%付近で ±0.0043 程度で、この幅より小さい差は数え直しただけで動く。
    """
    return float(np.sqrt(rate * (1.0 - rate) / trials))


def coverage(
    interval_fn: Callable[[np.random.Generator], tuple[float, float]],
    truth: float,
    trials: int = 10_000,
    seed: int = 0,
    *,
    progress: bool = True,
    workers: int = 1,
) -> CoverageResult:
    """``interval_fn(rng) -> (lo, hi)`` が ``truth`` を包んだ割合を数える。

    真値を引数に取るのが要点で、被覆確率は真値を知っている者にしか数えられない。
    実データで信頼区間の当たり外れを数えられないのはそのためであり、この本が合成データ
    から始まる理由でもある。
    """
    intervals = repeat(
        interval_fn, trials=trials, seed=seed, progress=progress, workers=workers,
        label="被覆を数える",
    )
    intervals = np.asarray(intervals, dtype=float)
    if intervals.ndim != 2 or intervals.shape[1] != 2:
        raise ValueError("interval_fn は (lo, hi) の2要素を返すこと")

    lo, hi = intervals[:, 0], intervals[:, 1]
    covered = (lo <= truth) & (truth <= hi)
    rate = float(covered.mean())
    return CoverageResult(
        rate=rate,
        se=_binomial_se(rate, trials),
        covered=covered,
        intervals=intervals,
        truth=float(truth),
        trials=trials,
    )


def rejection_rate(
    pvalue_fn: Callable[[np.random.Generator], float],
    alpha: float = 0.05,
    trials: int = 10_000,
    seed: int = 0,
    *,
    progress: bool = True,
    workers: int = 1,
) -> RejectionResult:
    """``pvalue_fn(rng) -> p`` が ``alpha`` を下回った割合を数える。

    同じ関数が第一種の誤りにも検出力にもなる。違うのはデータの生成側で帰無が真かどうか
    だけで、数える手続きは1つしかない。本書がこの2つを同じ節で並べるのはそのため。
    """
    pvalues = np.asarray(
        repeat(pvalue_fn, trials=trials, seed=seed, progress=progress, workers=workers,
               label="棄却を数える"),
        dtype=float,
    )
    rate = float((pvalues < alpha).mean())
    return RejectionResult(
        rate=rate,
        se=_binomial_se(rate, trials),
        pvalues=pvalues,
        alpha=alpha,
        trials=trials,
    )


def sweep(
    fn: Callable[..., object],
    over: Mapping[str, Iterable],
    trials: int = 2_000,
    seed: int = 0,
    *,
    progress: bool = True,
) -> object:
    """パラメータを振りながら :func:`coverage` / :func:`rejection_rate` を繰り返す。

    ``fn(value, trials=..., seed=...)`` が ``CoverageResult`` か ``RejectionResult``
    を返すこと。検出力曲線も被覆曲線も、横軸が何であれ形は同じなので1つにまとめてある。

    掃引の各点で seed をずらす。同じ乱数列を使い回すと、曲線の上下動が「パラメータの
    効果」ではなく「たまたま引いた1本」の形をなぞってしまう。
    """
    import pandas as pd

    if len(over) != 1:
        raise ValueError("掃引するパラメータは1つにすること（曲線が読めなくなる）")
    (name, values), = over.items()
    values = list(values)

    rows = []
    with progress_bar(len(values), f"{name} を掃引") as bar:
        for i, v in enumerate(values):
            res = fn(v, trials=trials, seed=seed + 1000 * i)
            rows.append({name: v, "rate": res.rate, "se": res.se,
                         "lo": res.ci[0], "hi": res.ci[1]})
            if progress:
                bar.update(1)
    return pd.DataFrame(rows)
