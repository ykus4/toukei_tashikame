"""因果推論。調整すると真値に寄ること、そして仮定が必ず付いてくること。"""

from __future__ import annotations

import numpy as np
import pytest

from toukei_tashikame import bayes, causal, datasets


def test_adjustment_moves_towards_the_truth():
    """素朴な差は交絡でずれ、IPW は真値に寄る。第17章の骨格。"""
    o = datasets.observational(n=4000, ate=2.0, confounding=1.5, seed=0)
    ps = causal.propensity_score(o.z[:, None], o.x)
    naive = causal.naive_diff(o.y, o.x).estimate
    ipw = causal.ipw_ate(o.y, o.x, ps).estimate
    assert abs(naive - o.ate) > 1.0
    assert abs(ipw - o.ate) < 0.35


def test_matching_also_recovers_the_effect():
    o = datasets.observational(n=3000, ate=2.0, seed=1)
    ps = causal.propensity_score(o.z[:, None], o.x)
    assert abs(causal.match_ate(o.y, o.x, ps).estimate - o.ate) < 0.5


def test_balance_improves_after_weighting():
    o = datasets.observational(n=4000, ate=2.0, seed=0)
    ps = causal.propensity_score(o.z[:, None], o.x)
    w = np.where(o.x == 1, 1 / ps, 1 / (1 - ps))
    before = abs(causal.balance_table(o.z[:, None], o.x)["SMD"].iloc[0])
    after = abs(causal.balance_table(o.z[:, None], o.x, weights=w)["SMD"].iloc[0])
    assert after < before / 2


def test_did_recovers_effect_when_trends_are_parallel():
    d = causal.did(datasets.did_panel(600, effect=1.5, parallel=True, seed=0))
    assert d.estimate == pytest.approx(1.5, abs=0.2)


def test_did_is_biased_when_trends_are_not_parallel():
    """平行トレンドが破れれば DID は外す。仮定が効いていることの確認。"""
    d = causal.did(datasets.did_panel(600, effect=1.5, parallel=False, seed=0))
    assert abs(d.estimate - 1.5) > 0.5


def test_rdd_recovers_the_jump():
    df = datasets.rdd_data(3000, cutoff=0.0, jump=2.0, seed=0)
    r = causal.rdd(df["running"], df["y"], cutoff=0.0, bandwidth=1.0)
    assert r.estimate == pytest.approx(2.0, abs=0.4)


def test_iv_reports_first_stage_f():
    rng = np.random.default_rng(0)
    n = 2000
    z = rng.normal(size=n)
    u = rng.normal(size=n)                      # 未観測の交絡
    d = 0.8 * z + u + rng.normal(size=n) * 0.5
    y = 1.5 * d + u + rng.normal(size=n) * 0.5  # 真の効果 1.5
    res = causal.iv_2sls(y, d, z)
    assert res.estimate == pytest.approx(1.5, abs=0.2)
    assert "第1段のF" in res.extra
    assert float(res.extra["第1段のF"]) > 10


def test_every_causal_result_carries_assumptions():
    """推定値だけを出して仮定を落とせない、という設計を型で固定する。"""
    o = datasets.observational(n=1000, seed=0)
    ps = causal.propensity_score(o.z[:, None], o.x)
    for res in (causal.naive_diff(o.y, o.x),
                causal.ipw_ate(o.y, o.x, ps),
                causal.match_ate(o.y, o.x, ps),
                causal.did(datasets.did_panel(200, seed=0)),
                causal.rdd(*datasets.rdd_data(500, seed=0)[["running", "y"]].to_numpy().T)):
        assert res.assumptions, f"{res.name} が仮定を持っていない"
        assert "仮定" in str(res)


def test_e_value_grows_with_the_association():
    assert causal.e_value(2.0) < causal.e_value(4.0)
    assert causal.e_value(1.0) == pytest.approx(1.0)


class TestBayes:
    def test_conjugate_update_is_addition(self):
        post = bayes.beta_binomial(30, 100, a=1.0, b=1.0)
        assert (post.a, post.b) == (31.0, 71.0)
        assert post.mean == pytest.approx(31 / 102)

    def test_grid_posterior_matches_conjugate(self):
        """刻んで掛けるのと、共役の式は同じ答えに行き着く。"""
        from scipy import stats as st

        grid = np.linspace(1e-4, 1 - 1e-4, 4000)
        k, n = 30, 100
        loglik = st.binom.logpmf(k, n, grid)
        post = bayes.grid_posterior(loglik, np.ones_like(grid), grid)
        mean_grid = float(np.trapezoid(grid * post, grid))
        assert mean_grid == pytest.approx(bayes.beta_binomial(k, n).mean, abs=1e-4)

    def test_metropolis_recovers_the_posterior(self):
        from scipy import stats as st

        def logpost(p):
            if not 0 < p < 1:
                return -np.inf
            return float(st.binom.logpmf(30, 100, p))

        res = bayes.metropolis_hastings(logpost, init=0.5, n=20_000, step=0.08, seed=0)
        chain = res.burned(2000)
        assert 0.15 < res.accept_rate < 0.75
        assert chain.mean() == pytest.approx(bayes.beta_binomial(30, 100).mean, abs=0.01)

    def test_rhat_detects_unconverged_chains(self):
        rng = np.random.default_rng(0)
        good = rng.normal(size=(4, 2000))
        bad = good + np.array([[0.0], [3.0], [-3.0], [6.0]])  # 鎖が別の場所にいる
        assert bayes.rhat(good) < 1.01
        assert bayes.rhat(bad) > 1.1

    def test_prob_b_beats_a_and_expected_loss_agree(self):
        a = bayes.beta_binomial(300, 10_000)
        b = bayes.beta_binomial(360, 10_000)
        p = bayes.prob_b_beats_a(a, b, draws=50_000, seed=0)
        loss_a, loss_b = bayes.expected_loss(a, b, draws=50_000, seed=0)
        assert p > 0.9                 # B が勝つ確率は高い
        assert loss_b < loss_a         # B を選んで外したときの損は小さい

    def test_hdi_is_narrower_than_eti_for_skewed_posterior(self):
        samples = bayes.beta_binomial(2, 50).rvs(200_000, seed=0)
        eti = bayes.credible_interval(samples, kind="eti")
        hdi = bayes.credible_interval(samples, kind="hdi")
        assert (hdi[1] - hdi[0]) <= (eti[1] - eti[0]) + 1e-9
