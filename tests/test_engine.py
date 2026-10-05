"""Unit tests for the estimators and the router. No database."""
import numpy as np
import pytest

from wl import engine, world


def test_occupancy_recovers_known_parameters_and_naive_count_is_too_low():
    rng = np.random.default_rng(0)
    used = rng.random(400) < 0.5
    y = (rng.random((400, 10)) < 0.12) & used[:, None]                    # present at half the sites, seen on 12% of days there
    est = engine.occupancy(y.astype(int), boot=20)
    assert est["naive"] < 0.42 and abs(est["psi"] - 0.5) < 0.08 and abs(est["p"] - 0.12) < 0.03 and est["low"] <= est["psi"] <= est["high"]
    assert engine.occupancy(np.zeros((30, 10), dtype=int))["psi"] == 0.0
    everywhere = engine.occupancy(np.ones((30, 10), dtype=int), boot=0)
    assert everywhere["psi"] > 0.99 and everywhere["p"] > 0.99


def test_histories_keep_only_the_period_asked_for():
    y = engine.histories([(0, 1), (0, 1), (2, 16), (1, 3)], 3, range(0, 15))
    assert y.shape == (3, 15) and y.sum() == 2 and y[0, 1] == 1 and y[2].sum() == 0


def test_shift_is_detected_when_real_and_not_when_absent():
    rng = np.random.default_rng(1)
    moved = engine.shift(rng.normal(8, 2, 60), rng.normal(14, 2, 60), perms=500)
    same = engine.shift(rng.normal(8, 2, 60), rng.normal(8, 2, 60), perms=500)
    assert 5 < moved["shift_km"] < 7 and moved["p_value"] < 0.01 and same["p_value"] > 0.05
    assert engine.shift([1, 2], [3, 4])["p_value"] is None                 # too few detections to say anything


def test_night_belongs_to_the_evening_before():
    assert engine.night_of(17, 23.5) == 17 and engine.night_of(18, 1.5) == 17 and engine.is_night(2) and engine.is_night(22) and not engine.is_night(12)


def test_vehicle_clusters_need_confidence_repetition_and_far_more_than_chance():
    cams = [(1, 1), (2, 2), (20, 20), (10, 10)] + [(k, 23) for k in range(26)]
    repeat = [(c, d, 23.5, 0.9) for c in (0, 1) for d in (3, 5, 7, 9)]
    unsure = [(2, d, 23.5, 0.5) for d in (3, 5, 7, 9)]                     # same pattern, low confidence: ignored
    once = [(3, 4, 2.0, 0.95)]                                             # one confident night: not a pattern
    daytime = [(3, d, 12.0, 0.95) for d in range(20)]
    clusters, counts = engine.vehicle_clusters(repeat + unsure + once + daytime, cams, sounds=[((2, 1), 5, "engine"), ((22, 22), 5, "gunshot")])
    assert len(clusters) == 1 and clusters[0]["cameras"] == [0, 1] and clusters[0]["nights"] == [3, 5, 7, 9] and clusters[0]["confidence"] == "high"
    assert clusters[0]["corroborating_sounds"] == [{"night": 5, "sound": "engine"}] and clusters[0]["chance_p_value"] < 0.001
    assert counts["night_vehicle_detections"] == 13 and counts["confident"] == 9 and counts["cameras_flagged"] == 2
    assert engine.vehicle_clusters(once + daytime, cams)[0] == []


def test_the_gunshot_threshold_is_set_for_the_field_not_for_the_test_set():
    model, thr, m = engine.train_sounds(world.sounds(1), world.sounds(2, 12000))
    assert m["at_0.5"]["precision_on_balanced_test"] > 0.85 and m["at_0.5"]["precision_expected_in_field"] < 0.1      # looks fine on balanced data, useless at one in a thousand
    assert thr > 0.8 and m["chosen"]["precision_expected_in_field"] >= engine.FIELD_PRECISION and 0 < m["chosen"]["recall"] < m["at_0.5"]["recall"]
    assert engine.loud_and_short([0.15, 0.005, 900, 3000, 85]) and not engine.loud_and_short([2.5, 0.2, 120, 400, 80])


def test_risk_map_peaks_at_the_evidence_and_patrols_stay_within_their_hours():
    cluster = [{"centre": (20.0, 4.0)}]
    risk = engine.risk_tiles(cluster, [(20.5, 3.5)], [(14.0, 22.0)])
    iy, ix = np.unravel_index(risk.argmax(), risk.shape)
    assert risk.max() == 1.0 and abs(ix + 0.5 - 20) <= 1 and abs(iy + 0.5 - 4) <= 1
    for hours in (4.0, 8.0):
        routes = engine.patrols(risk, teams=2, hours=hours, ms=300)
        assert len(routes) == 2 and all(r[0] == world.HQ == r[-1] and engine.path_hours(r) <= hours + 0.01 for r in routes)
    long_, short = engine.covered(risk, engine.patrols(risk, teams=2, hours=8.0, ms=300)), engine.covered(risk, engine.patrols(risk, teams=2, hours=3.0, ms=300))
    assert long_ >= short and long_ > engine.covered(risk, engine.perimeter(2, 8.0)) and 0 <= short <= 1
    assert all(engine.path_hours(p, dwell=0) <= 8.0 + 0.01 for p in engine.perimeter(3, 8.0))
    assert engine.covered(np.zeros((24, 24)), [[(1, 1), (2, 2)]]) == 0.0 and engine.patrols(np.zeros((24, 24)), teams=2) == [[world.HQ, world.HQ]] * 2
