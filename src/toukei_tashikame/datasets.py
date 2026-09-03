"""合成データ生成器。本書の図と数字の8割はここから出る。

設計の要は2つある。

**すべての関数が ``seed`` を取る。** 既定値があっても省略できるだけで、内部では必ず
``np.random.default_rng(seed)`` を作る。グローバルな乱数状態は一度も触らない。読者の
手元で同じ数字が出ることが、この本の主張の前提になっている。

**真値を返り値に含める。** 被覆確率も第一種の誤りも、真値を知らなければ数えられない。
実データでは信頼区間の当たり外れを数えられない——これは道具の限界ではなく、真値が
隠れているという状況そのものの帰結である。合成データだけが持っている情報を、返り値の
かたちで明示する。

``anscombe`` と ``datasaurus`` だけは乱数を使わない。座標は ``_coords/`` に同梱して
あり、外部リポジトリのリンク切れで第2章が死なないようにしてある。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "ABTestData",
    "HierData",
    "ObsData",
    "ab_test",
    "anova_data",
    "anscombe",
    "bimodal",
    "bivariate_normal",
    "clustered",
    "collinear_data",
    "confounded_xy",
    "contaminated",
    "count_data",
    "datasaurus",
    "did_panel",
    "heavy_tailed",
    "income",
    "latency",
    "logistic_data",
    "normal_sample",
    "observational",
    "paired",
    "penguins",
    "rdd_data",
    "regression_data",
    "skewed_sample",
    "store_conversions",
    "tips",
    "two_groups",
]

_COORDS = Path(__file__).resolve().parent / "_coords"


# ---------------------------------------------------------------------------
# 真値つきの返り値
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ABTestData:
    """A/Bテストの観測と、合成データだけが知っている真のCVR。"""

    a: np.ndarray  # 0/1 のコンバージョン
    b: np.ndarray
    p_a: float
    p_b: float
    seed: int

    @property
    def lift(self) -> float:
        """真の相対リフト。観測から推定する対象であって、答えではない。"""
        return self.p_b / self.p_a - 1.0

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "group": np.r_[np.repeat("A", self.a.size), np.repeat("B", self.b.size)],
                "converted": np.r_[self.a, self.b],
            }
        )


@dataclass(frozen=True)
class ObsData:
    """観察データ。交絡があるので、素朴な差は ATE ではない。"""

    x: np.ndarray  # 処置 0/1
    y: np.ndarray  # 結果
    z: np.ndarray  # 交絡因子（実務では観測できないこともある）
    ate: float  # 真の平均処置効果
    seed: int

    @property
    def naive_diff(self) -> float:
        """処置群と対照群の単純な平均差。ATE から交絡のぶんだけずれる。"""
        return float(self.y[self.x == 1].mean() - self.y[self.x == 0].mean())

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"x": self.x, "y": self.y, "z": self.z})


@dataclass(frozen=True)
class HierData:
    """店舗ごとのコンバージョン。真の店舗別確率を持つ（階層モデルの検算用）。"""

    trials: np.ndarray  # 店舗ごとの試行数
    successes: np.ndarray
    p_store: np.ndarray  # 店舗ごとの真の確率
    mu: float  # 全体平均（ロジット尺度）
    tau: float  # 店舗間のばらつき
    seed: int

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "store": np.arange(self.trials.size),
                "trials": self.trials,
                "successes": self.successes,
                "rate": self.successes / self.trials,
            }
        )


# ---------------------------------------------------------------------------
# 1変量の標本
# ---------------------------------------------------------------------------


def normal_sample(n: int, mu: float = 50.0, sigma: float = 10.0, seed: int = 0) -> np.ndarray:
    """正規標本。本書の「うまくいく側」の基準線。"""
    return np.random.default_rng(seed).normal(mu, sigma, size=n)


def skewed_sample(n: int, kind: str = "lognormal", seed: int = 0) -> np.ndarray:
    """右に裾を引く標本。平均と中央値が離れる例に使う。

    ``kind`` は ``lognormal`` / ``exponential`` / ``chi2``。どれも平均 > 中央値になる。
    """
    rng = np.random.default_rng(seed)
    if kind == "lognormal":
        return rng.lognormal(mean=0.0, sigma=1.0, size=n)
    if kind == "exponential":
        return rng.exponential(scale=1.0, size=n)
    if kind == "chi2":
        return rng.chisquare(df=3, size=n)
    raise ValueError(f"unknown kind: {kind}")


def income(n: int = 200, median: float = 400.0, sigma: float = 0.8, seed: int = 0) -> np.ndarray:
    """年収（万円）。対数正規なので、平均が中央値より上に引かれる。

    ``median`` をそのまま中央値として指定できるように、対数正規の位置母数を
    ``log(median)`` に取る。対数正規の中央値は ``exp(μ)`` なので、σ を変えても
    中央値は動かない。平均だけが動く——それがこの生成器の見せ場である。
    """
    rng = np.random.default_rng(seed)
    return rng.lognormal(mean=np.log(median), sigma=sigma, size=n)


def contaminated(n: int, eps: float = 0.05, scale: float = 10.0, seed: int = 0) -> np.ndarray:
    """割合 ``eps`` だけ広い正規が混ざった標本。平均と標準偏差が壊れる例。

    混合の正体を隠さないために、汚染は「別の正規から引いた」という形にしてある。
    外れ値を後から足すのではなく、最初から2つの分布の混合として作る。
    """
    rng = np.random.default_rng(seed)
    contaminated_mask = rng.random(n) < eps
    x = rng.normal(0.0, 1.0, size=n)
    x[contaminated_mask] = rng.normal(0.0, scale, size=int(contaminated_mask.sum()))
    return x


def bimodal(n: int, sep: float = 4.0, seed: int = 0) -> np.ndarray:
    """2つの山。平均が「誰もいない谷」を指す例に使う。"""
    rng = np.random.default_rng(seed)
    which = rng.random(n) < 0.5
    x = rng.normal(-sep / 2, 1.0, size=n)
    x[which] = rng.normal(sep / 2, 1.0, size=int(which.sum()))
    return x


def latency(n: int, median_ms: float = 200.0, sigma: float = 0.8, seed: int = 0) -> np.ndarray:
    """レイテンシ（ミリ秒）。対数正規。平均ではなく分位点で語るべき量の代表。"""
    rng = np.random.default_rng(seed)
    return rng.lognormal(mean=np.log(median_ms), sigma=sigma, size=n)


def heavy_tailed(n: int, kind: str = "cauchy", seed: int = 0) -> np.ndarray:
    """裾の重い標本。``cauchy`` は平均も分散も持たない。

    第5章の「破れ目」で使う。コーシーでは標本平均が n を増やしても収束しないので、
    中心極限定理が効かない例になる。``t3`` は分散を持つが裾が重い中間の例。
    """
    rng = np.random.default_rng(seed)
    if kind == "cauchy":
        return rng.standard_cauchy(size=n)
    if kind == "t3":
        return rng.standard_t(df=3, size=n)
    raise ValueError(f"unknown kind: {kind}")


# ---------------------------------------------------------------------------
# 群の比較
# ---------------------------------------------------------------------------


def two_groups(
    n1: int, n2: int, delta: float = 0.0, sd_ratio: float = 1.0, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """2群。``delta`` が真の差、``sd_ratio`` が分散比。

    ``delta=0`` なら帰無が真なので第一種の誤りを、``delta>0`` なら検出力を数えられる。
    ``sd_ratio != 1`` は等分散の仮定を破るためにある（Welch と Student の差が出る）。
    """
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, 1.0, size=n1)
    b = rng.normal(delta, sd_ratio, size=n2)
    return a, b


def paired(
    n: int, delta: float = 0.0, rho: float = 0.7, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """対応のある2測定。``rho`` が個体内相関。

    対応を無視して独立2標本の検定にかけると検出力を捨てることになる、という比較に使う。
    """
    rng = np.random.default_rng(seed)
    subject = rng.normal(0.0, np.sqrt(rho), size=n)
    before = subject + rng.normal(0.0, np.sqrt(1.0 - rho), size=n)
    after = subject + delta + rng.normal(0.0, np.sqrt(1.0 - rho), size=n)
    return before, after


def clustered(
    n_cluster: int, per_cluster: int, icc: float = 0.3, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """クラスタ内相関のある観測。``(y, cluster_id)`` を返す。

    独立と思って検定にかけると第一種の誤りが名目の α を大きく超える。「n はいくつか」
    という問いが、行数ではなくクラスタ数で決まる例。
    """
    rng = np.random.default_rng(seed)
    cluster_effect = rng.normal(0.0, np.sqrt(icc), size=n_cluster)
    within = rng.normal(0.0, np.sqrt(1.0 - icc), size=(n_cluster, per_cluster))
    y = (cluster_effect[:, None] + within).ravel()
    g = np.repeat(np.arange(n_cluster), per_cluster)
    return y, g


# ---------------------------------------------------------------------------
# 2変量・回帰
# ---------------------------------------------------------------------------


def bivariate_normal(n: int, rho: float = 0.5, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """相関 ``rho`` の2変量正規。"""
    rng = np.random.default_rng(seed)
    cov = np.array([[1.0, rho], [rho, 1.0]])
    xy = rng.multivariate_normal(np.zeros(2), cov, size=n)
    return xy[:, 0], xy[:, 1]


def confounded_xy(
    n: int, effect: float = 0.0, z_strength: float = 1.0, simpson: bool = False, seed: int = 0
) -> pd.DataFrame:
    """x と y の相関が z から来ている状況。``simpson=True`` で符号が反転する。

    ``effect`` が x → y の真の効果。0 のまま相関だけが立つのが交絡であり、
    ``simpson=True`` では層別すると各層の傾きが全体と逆向きになる。
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 1.0, size=n)
    x = z_strength * z + rng.normal(0.0, 1.0, size=n)
    if simpson:
        # 層内では負、層をまたぐと z の効果が勝って正に見える。
        y = -1.0 * x + 3.0 * z_strength * z + rng.normal(0.0, 0.5, size=n)
    else:
        y = effect * x + z_strength * z + rng.normal(0.0, 1.0, size=n)
    return pd.DataFrame({"x": x, "y": y, "z": z})


def regression_data(
    n: int, b: tuple[float, ...] = (1.0, 2.0), sigma: float = 1.0, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(X, y, b_true)``。X は切片列を含む計画行列。

    ``b`` の先頭が切片。真の係数を返すので、推定値との差をそのまま数えられる。
    """
    rng = np.random.default_rng(seed)
    b_true = np.asarray(b, dtype=float)
    k = b_true.size - 1
    X = np.column_stack([np.ones(n), rng.normal(0.0, 1.0, size=(n, k))])
    y = X @ b_true + rng.normal(0.0, sigma, size=n)
    return X, y, b_true


def collinear_data(
    n: int, rho: float = 0.99, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ほぼ共線な2列。係数の推定値が暴れる（VIF の節で使う）。"""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0.0, 1.0, size=n)
    x2 = rho * x1 + np.sqrt(1.0 - rho**2) * rng.normal(0.0, 1.0, size=n)
    X = np.column_stack([np.ones(n), x1, x2])
    b_true = np.array([0.0, 1.0, 1.0])
    y = X @ b_true + rng.normal(0.0, 1.0, size=n)
    return X, y, b_true


def logistic_data(
    n: int, b: tuple[float, ...] = (-1.0, 0.8), seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ロジスティック回帰用。``(X, y, b_true)``、y は 0/1。"""
    rng = np.random.default_rng(seed)
    b_true = np.asarray(b, dtype=float)
    k = b_true.size - 1
    X = np.column_stack([np.ones(n), rng.normal(0.0, 1.0, size=(n, k))])
    p = 1.0 / (1.0 + np.exp(-(X @ b_true)))
    y = (rng.random(n) < p).astype(float)
    return X, y, b_true


def count_data(
    n: int, b: tuple[float, ...] = (0.5, 0.3), overdispersion: float = 1.0, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ポアソン回帰用。``overdispersion > 1`` で負の二項に切り替わる。

    ポアソンは平均と分散が等しいことを仮定する。過分散はその仮定の破れであり、
    標準誤差が小さく出すぎる（＝有意になりすぎる）という形で現れる。
    """
    rng = np.random.default_rng(seed)
    b_true = np.asarray(b, dtype=float)
    k = b_true.size - 1
    X = np.column_stack([np.ones(n), rng.normal(0.0, 1.0, size=(n, k))])
    mu = np.exp(X @ b_true)
    if overdispersion <= 1.0:
        y = rng.poisson(mu).astype(float)
    else:
        # 分散が overdispersion * mu になるようにガンマ混合の形状母数を決める。
        shape = mu / (overdispersion - 1.0)
        y = rng.poisson(rng.gamma(shape, scale=(overdispersion - 1.0))).astype(float)
    return X, y, b_true


def anova_data(
    group_means: tuple[float, ...], n_per_group: int, sd: float = 1.0, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """一元配置。``(y, g)`` を返す。``group_means`` が真の群平均。"""
    rng = np.random.default_rng(seed)
    means = np.asarray(group_means, dtype=float)
    y = np.concatenate([rng.normal(m, sd, size=n_per_group) for m in means])
    g = np.repeat(np.arange(means.size), n_per_group)
    return y, g


# ---------------------------------------------------------------------------
# 応用領域
# ---------------------------------------------------------------------------


def ab_test(
    n_a: int = 5000,
    n_b: int = 5000,
    p_a: float = 0.030,
    lift: float = 0.10,
    seed: int = 0,
) -> ABTestData:
    """A/Bテスト。``lift`` は相対リフト（0.10 なら 10% 改善）。

    既定の 3.0% と 5,000件ずつは、実務でよくある「小さい効果を小さい標本で見に行く」
    状況にしてある。この設定では検出力が足りない——それを第10章で数える。
    """
    rng = np.random.default_rng(seed)
    p_b = p_a * (1.0 + lift)
    a = (rng.random(n_a) < p_a).astype(float)
    b = (rng.random(n_b) < p_b).astype(float)
    return ABTestData(a=a, b=b, p_a=p_a, p_b=p_b, seed=seed)


def observational(
    n: int = 2000, ate: float = 2.0, confounding: float = 1.5, seed: int = 0
) -> ObsData:
    """交絡のある観察データ。素朴な平均差は ATE からずれる。

    z が処置と結果の両方を押し上げるので、``naive_diff`` は ``ate`` より大きく出る。
    調整して ate に戻せるか、が第17章の主題。
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 1.0, size=n)
    p_treat = 1.0 / (1.0 + np.exp(-(confounding * z)))
    x = (rng.random(n) < p_treat).astype(float)
    y = ate * x + confounding * z + rng.normal(0.0, 1.0, size=n)
    return ObsData(x=x, y=y, z=z, ate=ate, seed=seed)


def store_conversions(n_store: int = 20, seed: int = 0) -> HierData:
    """店舗ごとのコンバージョン。試行数が店舗で大きく違う（階層モデルの動機）。

    試行の少ない店舗の観測率は極端に振れる。完全プーリングでは店舗差が消え、
    ノープーリングでは少数店舗のノイズを信じてしまう——その中間が第16章の主題。
    """
    rng = np.random.default_rng(seed)
    mu, tau = -2.5, 0.6  # ロジット尺度。exp(-2.5) ≈ 7.6%
    logit_p = rng.normal(mu, tau, size=n_store)
    p_store = 1.0 / (1.0 + np.exp(-logit_p))
    trials = rng.integers(30, 800, size=n_store)
    successes = rng.binomial(trials, p_store)
    return HierData(
        trials=trials, successes=successes, p_store=p_store, mu=mu, tau=tau, seed=seed
    )


def did_panel(n: int, effect: float = 1.5, parallel: bool = True, seed: int = 0) -> pd.DataFrame:
    """差分の差分用のパネル。``parallel=False`` で平行トレンドの仮定を破る。

    DID が寄りかかっているのは「処置がなければ2群は平行に動いたはず」という、
    観測できない反実仮想である。``parallel=False`` はそこだけを壊す。
    """
    rng = np.random.default_rng(seed)
    unit = np.repeat(np.arange(n), 2)
    period = np.tile([0, 1], n)
    treated = np.repeat((np.arange(n) < n // 2).astype(float), 2)
    unit_effect = np.repeat(rng.normal(0.0, 1.0, size=n), 2)
    trend = 1.0 * period
    if not parallel:
        trend = trend + 0.8 * period * treated  # 処置群だけ元から傾きが違う
    y = unit_effect + trend + effect * treated * period + rng.normal(0.0, 0.5, size=2 * n)
    return pd.DataFrame({"unit": unit, "period": period, "treated": treated, "y": y})


def rdd_data(n: int, cutoff: float = 0.0, jump: float = 2.0, seed: int = 0) -> pd.DataFrame:
    """回帰不連続。``cutoff`` を境に処置が入り、``jump`` だけ跳ぶ。"""
    rng = np.random.default_rng(seed)
    running = rng.uniform(cutoff - 3.0, cutoff + 3.0, size=n)
    treated = (running >= cutoff).astype(float)
    y = 1.0 + 0.7 * running + jump * treated + rng.normal(0.0, 1.0, size=n)
    return pd.DataFrame({"running": running, "treated": treated, "y": y})


# ---------------------------------------------------------------------------
# 同梱データ（乱数を使わない）
# ---------------------------------------------------------------------------


def anscombe() -> pd.DataFrame:
    """Anscombe の4組。要約統計量が同じで、散布図が違う。

    座標は同梱してある。要約が同じであることは丸め誤差の範囲まで成り立つので、
    生成しなおすのではなく原典の値をそのまま持つ。
    """
    return pd.read_csv(_COORDS / "anscombe.csv")


def datasaurus(which: str = "dino") -> pd.DataFrame:
    """Datasaurus Dozen。平均・分散・相関が全て同じで、絵が違う。

    ``which="all"`` で13組すべて、それ以外は該当する1組だけを返す。
    """
    df = pd.read_csv(_COORDS / "datasaurus.csv")
    if which == "all":
        return df
    subset = df[df["dataset"] == which]
    if subset.empty:
        raise ValueError(f"unknown dataset: {which}（{sorted(df['dataset'].unique())}）")
    return subset.reset_index(drop=True)


def penguins() -> pd.DataFrame:
    """Palmer Penguins。欠測の落とし方をここで固定する。

    seaborn 側の既定に任せると、章によって行数が変わって本文の数字と合わなくなる。
    本書は「解析に使う4列のいずれかが欠けている行を落とす」に固定する。
    """
    import seaborn as sns

    df = sns.load_dataset("penguins")
    cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    return df.dropna(subset=cols).reset_index(drop=True)


def tips() -> pd.DataFrame:
    """レストランのチップ。回帰と相関の実データ例。"""
    import seaborn as sns

    return sns.load_dataset("tips")
