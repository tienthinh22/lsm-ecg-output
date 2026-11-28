"""Statistics and decision rules on synthetic run tables."""
import math
import numpy as np
import pandas as pd
import pytest
from scipy import stats as sstats

from lsm2 import stats as S

SPLITS = [f"cv-r{r}f{f}" for r in (1, 2, 3) for f in (1, 2, 3, 4)]


def runs(base, shift=0.0, seeds=(2026, 69, 420), noise=0.01, seed_free=False, rng=None, ds2=None):
    rng = rng or np.random.default_rng(0)
    rows = []
    for sd in ([None] if seed_free else seeds):
        for i, sp in enumerate(SPLITS + ["final"]):
            v = base[i] + shift + noise * rng.standard_normal()
            if sp == "final" and ds2 is not None:
                v = ds2
            rows.append({"spec_split": sp, "spec_seed": sd, "macro_f1": v, "macro_f1_no232": v,
                         "recall_SVEB": v, "precision_SVEB": v, "recall_VEB": v, "precision_VEB": v})
    return pd.DataFrame(rows)


def test_nb_test_matches_formula():
    rng = np.random.default_rng(1)
    base = 0.6 + 0.05 * rng.standard_normal(13)
    a, b = runs(base, 0.03, rng=rng), runs(base, 0.0, rng=rng)
    r = S.compare(a, b)
    A = a.groupby(["spec_split", "spec_seed"]).macro_f1.mean().groupby(level=0).mean()
    B = b.groupby(["spec_split", "spec_seed"]).macro_f1.mean().groupby(level=0).mean()
    d = (A[SPLITS] - B[SPLITS]).values
    se = math.sqrt(np.var(d, ddof=1) * (1 / 12 + 0.336))
    assert r["n_splits"] == 12 and r["cv_diff"] == pytest.approx(d.mean())
    assert r["t"] == pytest.approx(d.mean() / se)
    assert r["p"] == pytest.approx(2 * sstats.t.sf(abs(d.mean() / se), 11))
    assert r["ci_high"] - r["cv_diff"] == pytest.approx(sstats.t.ppf(0.975, 11) * se)
    assert r["ds2_diff"] == pytest.approx(A["final"] - B["final"])


def test_verdicts():
    big = {"cv_diff": 0.05, "ci_low": 0.03, "ci_high": 0.07, "ds2_diff": 0.04, "p_holm": 0.001, "same_sign_all_seeds": True}
    assert S.verdict(big) == "meaningful"
    assert S.verdict({**big, "ds2_diff": -0.01}) == "inconclusive"            # DS2 must have the same sign
    assert S.verdict({**big, "same_sign_all_seeds": False}, need_seed_sign=True) == "inconclusive"
    assert S.verdict({**big, "cv_diff": 0.008, "ci_low": 0.007, "ci_high": 0.009, "ds2_diff": 0.005}) == "comparable"
    small = {"cv_diff": 0.001, "ci_low": -0.015, "ci_high": 0.017, "ds2_diff": 0.01, "p_holm": 0.8}
    assert S.verdict(small) == "comparable"
    assert S.verdict({**small, "ds2_diff": 0.03}) == "inconclusive"
    assert S.verdict(None) == "missing"


def test_holm():
    rows = S.holm([{"p": 0.01}, {"p": 0.04}, {"p": 0.03}, None])
    assert rows[0]["p_holm"] == pytest.approx(0.03)
    assert rows[2]["p_holm"] == pytest.approx(0.06)
    assert rows[1]["p_holm"] == pytest.approx(0.06)          # monotone


def test_seed_sign_and_seed_free_control():
    rng = np.random.default_rng(2)
    base = 0.6 + 0.05 * rng.standard_normal(13)
    a = runs(base, 0.02, rng=rng, noise=0.001)
    c = runs(base, 0.0, rng=rng, noise=0.001, seed_free=True)
    r = S.compare(a, c, require_seed_sign=True)
    assert r is not None and r["cv_diff"] == pytest.approx(0.02, abs=0.002)


def test_compare_diff_f7():
    rng = np.random.default_rng(3)
    base = 0.6 + 0.05 * rng.standard_normal(13)
    on_full, off_full = runs(base, 0.04, rng=rng, noise=0.001), runs(base, 0.0, rng=rng, noise=0.001)
    on_abl, off_abl = runs(base, 0.01, rng=rng, noise=0.001), runs(base, 0.0, rng=rng, noise=0.001)
    r = S.compare_diff(on_full, off_full, on_abl, off_abl)
    assert r["cv_diff"] == pytest.approx(0.03, abs=0.002) and r["p"] < 1e-6


def test_family_applies_holm():
    rng = np.random.default_rng(4)
    base = 0.6 + 0.05 * rng.standard_normal(13)
    items = [(f"c{i}", S.compare(runs(base, 0.03 * i, rng=rng), runs(base, 0.0, rng=rng))) for i in range(3)]
    fam = S.family("F", items)
    assert [f["comparison"] for f in fam] == ["c0", "c1", "c2"] and all("p_holm" in f for f in fam)
