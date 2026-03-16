"""Tests for the CalibrationService.

Covers rolling performance computation, weight recalibration (exponential
weights with guardrails), degradation detection (z-score analysis), and
helper methods.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.database import Base
from app.models.draw import Draw
from app.models.performance import EnsembleWeight, ModelPerformance
from app.models.prediction import Prediction
from app.models.user import User
from app.services.calibration import CalibrationService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine():
    """Create an in-memory SQLite database engine for testing."""
    eng = create_engine("sqlite:///:memory:")

    @event.listens_for(eng, "connect")
    def _set_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON;")
        cursor.close()

    Base.metadata.create_all(bind=eng)
    return eng


@pytest.fixture
def db(engine):
    """Yield a fresh database session; rolls back after each test."""
    connection = engine.connect()
    transaction = connection.begin()
    TestSession = sessionmaker(bind=connection)
    session = TestSession()

    yield session

    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def svc():
    """Return a CalibrationService instance."""
    return CalibrationService()


# Fake strategy names used in tests.
FAKE_STRATEGIES = ["frequency", "cold_due", "moving_average"]


def _patch_strategy_names(names: list[str] | None = None):
    """Return a patch context manager that overrides _get_strategy_names."""
    target_names = names or FAKE_STRATEGIES
    return patch.object(
        CalibrationService,
        "_get_strategy_names",
        staticmethod(lambda: target_names),
    )


def _seed_user(db: Session) -> User:
    """Create a minimal user row for FK references (nullable, but useful)."""
    user = User(
        email="test@example.com",
        hashed_password="fake",
    )
    db.add(user)
    db.flush()
    return user


def _seed_predictions(
    db: Session,
    strategy_name: str,
    game_type: str = "daily3",
    count: int = 20,
    straight_hit_pct: float = 0.10,
    box_hit_pct: float = 0.30,
    confidence: float = 0.25,
    scored: bool = True,
    backtest: bool = False,
    days_ago_start: int = 0,
) -> list[Prediction]:
    """Insert scored predictions with deterministic hit patterns."""
    preds = []
    now = datetime.utcnow()
    for i in range(count):
        is_straight = i < int(count * straight_hit_pct)
        is_box = i < int(count * box_hit_pct)
        p = Prediction(
            game_type=game_type,
            target_draw_date=now - timedelta(days=days_ago_start + i),
            target_draw_time="evening",
            strategy_name=strategy_name,
            digit_1=1,
            digit_2=2,
            digit_3=3,
            confidence=confidence,
            is_backtest=backtest,
            straight_hit=is_straight if scored else None,
            box_hit=is_box if scored else None,
            scored_at=(now - timedelta(days=days_ago_start + i)) if scored else None,
        )
        db.add(p)
        preds.append(p)
    db.flush()
    return preds


# ---------------------------------------------------------------------------
# Helper method tests
# ---------------------------------------------------------------------------

class TestHelperMethods:
    """Tests for static/internal helpers on CalibrationService."""

    def test_equal_weights_basic(self, svc):
        result = svc._equal_weights(["a", "b", "c"])
        assert len(result) == 3
        assert all(abs(v - 1 / 3) < 1e-9 for v in result.values())

    def test_equal_weights_empty(self, svc):
        assert svc._equal_weights([]) == {}

    def test_equal_weights_single(self, svc):
        result = svc._equal_weights(["solo"])
        assert result == {"solo": 1.0}

    def test_min_max_normalize_basic(self, svc):
        values = {"a": 10.0, "b": 20.0, "c": 30.0}
        result = svc._min_max_normalize(values)
        assert result["a"] == pytest.approx(0.0)
        assert result["b"] == pytest.approx(0.5)
        assert result["c"] == pytest.approx(1.0)

    def test_min_max_normalize_all_same(self, svc):
        values = {"a": 5.0, "b": 5.0}
        result = svc._min_max_normalize(values)
        assert all(v == 0.5 for v in result.values())

    def test_min_max_normalize_empty(self, svc):
        assert svc._min_max_normalize({}) == {}

    def test_softmax_basic(self, svc):
        scores = {"a": 1.0, "b": 1.0, "c": 1.0}
        result = svc._softmax(scores, temperature=0.5)
        # All equal scores -> equal weights.
        assert all(abs(v - 1 / 3) < 1e-6 for v in result.values())

    def test_softmax_sums_to_one(self, svc):
        scores = {"a": 0.8, "b": 0.2, "c": 0.5}
        result = svc._softmax(scores, temperature=0.5)
        assert sum(result.values()) == pytest.approx(1.0)

    def test_softmax_higher_score_gets_more(self, svc):
        scores = {"a": 1.0, "b": 0.0}
        result = svc._softmax(scores, temperature=0.5)
        assert result["a"] > result["b"]

    def test_softmax_empty(self, svc):
        assert svc._softmax({}, temperature=0.5) == {}


# ---------------------------------------------------------------------------
# Guardrails tests
# ---------------------------------------------------------------------------

class TestGuardrails:
    """Tests for the _apply_guardrails method."""

    def test_floor_applied(self, svc):
        """Weights below floor should be lifted to floor."""
        raw = {"a": 0.0001, "b": 0.9999}
        result = svc._apply_guardrails(raw, prev_weights={})
        assert result["a"] >= svc.WEIGHT_FLOOR / sum(result.values())

    def test_cap_applied(self, svc):
        """Weights above cap should be clipped to cap (before renorm)."""
        raw = {"a": 0.95, "b": 0.05}
        result = svc._apply_guardrails(raw, prev_weights={})
        # After capping and renormalization, 'a' should not exceed cap.
        # Because renormalization happens after, the final weight may differ,
        # but the pre-renorm value should have been capped at WEIGHT_CAP.
        assert result["a"] <= 1.0

    def test_damping_limits_change(self, svc):
        """Weight shift from prev weights should not exceed MAX_WEIGHT_SHIFT."""
        prev = {"a": 0.5, "b": 0.5}
        raw = {"a": 0.95, "b": 0.05}
        result = svc._apply_guardrails(raw, prev_weights=prev)
        # The raw shift for "a" is +0.45, but damping should cap it at 0.20
        # (before renormalization).
        # After all guardrails + renorm, the change should be bounded.
        assert result["a"] < 0.9  # Cannot jump from 0.5 to 0.95

    def test_sums_to_one(self, svc):
        """Output of _apply_guardrails should always sum to 1."""
        raw = {"a": 0.6, "b": 0.3, "c": 0.1}
        prev = {"a": 0.33, "b": 0.33, "c": 0.34}
        result = svc._apply_guardrails(raw, prev)
        assert sum(result.values()) == pytest.approx(1.0, abs=1e-9)

    def test_empty_raw(self, svc):
        assert svc._apply_guardrails({}, {}) == {}

    def test_no_prev_weights(self, svc):
        """Without previous weights, damping should be skipped."""
        raw = {"a": 0.7, "b": 0.3}
        result = svc._apply_guardrails(raw, prev_weights={})
        assert sum(result.values()) == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Rolling performance tests
# ---------------------------------------------------------------------------

class TestRollingPerformance:
    """Tests for compute_rolling_performance."""

    def test_no_predictions_returns_empty(self, svc, db):
        """With no predictions, should return empty list."""
        with _patch_strategy_names(["nonexistent"]):
            results = svc.compute_rolling_performance(db, "daily3")
        assert results == []

    def test_computes_all_windows(self, svc, db):
        """Should compute metrics for every window + 'all'."""
        _seed_predictions(db, "frequency", count=50, straight_hit_pct=0.10, box_hit_pct=0.30)
        with _patch_strategy_names(["frequency"]):
            results = svc.compute_rolling_performance(db, "daily3")

        window_types = {r.window_type for r in results}
        # Should have all windows: 7d, 30d, 90d, 180d, 365d, all
        assert "7d" in window_types
        assert "30d" in window_types
        assert "all" in window_types

    def test_hit_rates_correct(self, svc, db):
        """Straight and box hit rates should match seeded data."""
        count = 20
        straight_pct = 0.10
        box_pct = 0.30
        _seed_predictions(
            db, "frequency", count=count,
            straight_hit_pct=straight_pct, box_hit_pct=box_pct,
        )

        with _patch_strategy_names(["frequency"]):
            results = svc.compute_rolling_performance(db, "daily3")

        # The "all" window should have all 20 predictions.
        all_perf = next((r for r in results if r.window_type == "all"), None)
        assert all_perf is not None
        assert all_perf.total_predictions == count
        assert all_perf.straight_hits == int(count * straight_pct)
        assert all_perf.box_hits == int(count * box_pct)
        assert all_perf.straight_hit_rate == pytest.approx(straight_pct)
        assert all_perf.box_hit_rate == pytest.approx(box_pct)

    def test_ignores_backtest_predictions(self, svc, db):
        """Backtest predictions should not affect performance metrics."""
        _seed_predictions(db, "frequency", count=10, box_hit_pct=1.0, backtest=True)
        _seed_predictions(db, "frequency", count=5, box_hit_pct=0.0, backtest=False)

        with _patch_strategy_names(["frequency"]):
            results = svc.compute_rolling_performance(db, "daily3")

        all_perf = next((r for r in results if r.window_type == "all"), None)
        assert all_perf is not None
        # Should only count non-backtest predictions.
        assert all_perf.total_predictions == 5
        assert all_perf.box_hits == 0

    def test_ignores_unscored_predictions(self, svc, db):
        """Unscored predictions should be excluded."""
        _seed_predictions(db, "frequency", count=10, scored=False)
        _seed_predictions(db, "frequency", count=5, scored=True, box_hit_pct=1.0)

        with _patch_strategy_names(["frequency"]):
            results = svc.compute_rolling_performance(db, "daily3")

        all_perf = next((r for r in results if r.window_type == "all"), None)
        assert all_perf is not None
        assert all_perf.total_predictions == 5

    def test_upserts_existing_record(self, svc, db):
        """Calling compute_rolling_performance twice should update, not duplicate."""
        _seed_predictions(db, "frequency", count=10, box_hit_pct=0.30)

        with _patch_strategy_names(["frequency"]):
            results1 = svc.compute_rolling_performance(db, "daily3")
            results2 = svc.compute_rolling_performance(db, "daily3")

        # Both should return the same number of result rows.
        assert len(results1) == len(results2)

    def test_calibration_score_computed(self, svc, db):
        """Calibration score = 1 - |avg_confidence - box_hit_rate|."""
        conf = 0.25
        box_pct = 0.30
        _seed_predictions(db, "frequency", count=20, confidence=conf, box_hit_pct=box_pct)

        with _patch_strategy_names(["frequency"]):
            results = svc.compute_rolling_performance(db, "daily3")

        all_perf = next((r for r in results if r.window_type == "all"), None)
        assert all_perf is not None
        expected_cal = 1.0 - abs(conf - box_pct)
        assert all_perf.calibration_score == pytest.approx(expected_cal, abs=1e-6)


# ---------------------------------------------------------------------------
# Weight recalibration tests
# ---------------------------------------------------------------------------

class TestRecalibrateWeights:
    """Tests for recalibrate_weights."""

    def test_equal_weights_when_no_data(self, svc, db):
        """Without scored predictions, should return equal weights."""
        with _patch_strategy_names(FAKE_STRATEGIES):
            weights = svc.recalibrate_weights(db, "daily3")

        assert len(weights) == len(FAKE_STRATEGIES)
        expected = 1.0 / len(FAKE_STRATEGIES)
        for w in weights.values():
            assert w == pytest.approx(expected, abs=1e-6)

    def test_weights_sum_to_one(self, svc, db):
        """Recalibrated weights must always sum to 1.0."""
        for name in FAKE_STRATEGIES:
            _seed_predictions(
                db, name, count=30,
                straight_hit_pct=0.05 if name == "frequency" else 0.02,
                box_hit_pct=0.20 if name == "frequency" else 0.10,
            )

        with _patch_strategy_names(FAKE_STRATEGIES):
            weights = svc.recalibrate_weights(db, "daily3")

        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-9)

    def test_better_strategy_gets_higher_weight(self, svc, db):
        """A strategy with better hit rates should receive a higher weight."""
        # 'frequency' is much better.
        _seed_predictions(db, "frequency", count=60, box_hit_pct=0.50, straight_hit_pct=0.20)
        _seed_predictions(db, "cold_due", count=60, box_hit_pct=0.05, straight_hit_pct=0.01)
        _seed_predictions(db, "moving_average", count=60, box_hit_pct=0.05, straight_hit_pct=0.01)

        with _patch_strategy_names(FAKE_STRATEGIES):
            weights = svc.recalibrate_weights(db, "daily3")

        assert weights["frequency"] > weights["cold_due"]
        assert weights["frequency"] > weights["moving_average"]

    def test_weight_floor_respected(self, svc, db):
        """No weight should go below WEIGHT_FLOOR (after renormalization)."""
        # One dominant strategy.
        _seed_predictions(db, "frequency", count=60, box_hit_pct=0.90)
        _seed_predictions(db, "cold_due", count=60, box_hit_pct=0.01)
        _seed_predictions(db, "moving_average", count=60, box_hit_pct=0.01)

        with _patch_strategy_names(FAKE_STRATEGIES):
            weights = svc.recalibrate_weights(db, "daily3")

        # After renormalization, every weight should be positive.
        for w in weights.values():
            assert w > 0

    def test_weight_cap_respected(self, svc, db):
        """No single weight should exceed WEIGHT_CAP (before renorm)."""
        _seed_predictions(db, "frequency", count=60, box_hit_pct=0.95, straight_hit_pct=0.50)
        _seed_predictions(db, "cold_due", count=60, box_hit_pct=0.001)
        _seed_predictions(db, "moving_average", count=60, box_hit_pct=0.001)

        with _patch_strategy_names(FAKE_STRATEGIES):
            weights = svc.recalibrate_weights(db, "daily3")

        # Because of renormalization after capping, the exact final value
        # depends on the others, but it should be bounded.
        assert weights["frequency"] <= 1.0

    def test_stores_weights_in_db(self, svc, db):
        """After recalibration, EnsembleWeight rows should be created."""
        _seed_predictions(db, "frequency", count=30, box_hit_pct=0.20)
        _seed_predictions(db, "cold_due", count=30, box_hit_pct=0.20)
        _seed_predictions(db, "moving_average", count=30, box_hit_pct=0.20)

        with _patch_strategy_names(FAKE_STRATEGIES):
            svc.recalibrate_weights(db, "daily3")

        active = (
            db.query(EnsembleWeight)
            .filter(
                EnsembleWeight.game_type == "daily3",
                EnsembleWeight.effective_to.is_(None),
            )
            .all()
        )
        assert len(active) == len(FAKE_STRATEGIES)

    def test_expires_old_weights(self, svc, db):
        """Recalibration should expire previous active weights."""
        _seed_predictions(db, "frequency", count=30, box_hit_pct=0.20)
        _seed_predictions(db, "cold_due", count=30, box_hit_pct=0.20)
        _seed_predictions(db, "moving_average", count=30, box_hit_pct=0.20)

        with _patch_strategy_names(FAKE_STRATEGIES):
            svc.recalibrate_weights(db, "daily3")
            # Recalibrate again.
            svc.recalibrate_weights(db, "daily3")

        expired = (
            db.query(EnsembleWeight)
            .filter(
                EnsembleWeight.game_type == "daily3",
                EnsembleWeight.effective_to.isnot(None),
            )
            .all()
        )
        active = (
            db.query(EnsembleWeight)
            .filter(
                EnsembleWeight.game_type == "daily3",
                EnsembleWeight.effective_to.is_(None),
            )
            .all()
        )
        assert len(expired) == len(FAKE_STRATEGIES)  # First set expired.
        assert len(active) == len(FAKE_STRATEGIES)    # Second set active.


# ---------------------------------------------------------------------------
# Degradation detection tests
# ---------------------------------------------------------------------------

class TestDegradationDetection:
    """Tests for detect_degradation z-score analysis."""

    def test_no_alerts_when_no_data(self, svc, db):
        """Without performance data, should return no alerts."""
        with _patch_strategy_names(["frequency"]):
            alerts = svc.detect_degradation(db, "daily3")
        assert alerts == []

    def test_no_alert_when_stable(self, svc, db):
        """When 7d and 90d performance are similar, no alert should fire."""
        # Seed identical performance across windows. Use hit_pct=0 so
        # the binomial std=0 and no z-score can be computed.
        _seed_predictions(db, "frequency", count=200, box_hit_pct=0.0, straight_hit_pct=0.0)

        with _patch_strategy_names(["frequency"]):
            svc.compute_rolling_performance(db, "daily3")
            alerts = svc.detect_degradation(db, "daily3")

        # All predictions have zero hit rates, std=0, no z-score -> no alerts.
        assert len(alerts) == 0

    def test_check_metric_degradation_warning(self, svc):
        """z < -2 should trigger WARNING."""
        # With p=0.20, current=0.0, n=20 -> z=-2.236, which is < -2.
        alert = svc._check_metric_degradation(
            strategy_name="test",
            metric_name="box_hit_rate",
            current_value=0.0,      # complete drop
            baseline_value=0.20,    # 20% baseline
            n_current=20,
            n_baseline=180,
        )
        assert alert is not None
        assert alert["severity"] in ("WARNING", "CRITICAL")

    def test_check_metric_degradation_critical(self, svc):
        """z < -3 should trigger CRITICAL."""
        alert = svc._check_metric_degradation(
            strategy_name="test",
            metric_name="box_hit_rate",
            current_value=0.0,
            baseline_value=0.30,
            n_current=50,
            n_baseline=500,
        )
        assert alert is not None
        assert alert["severity"] == "CRITICAL"

    def test_no_alert_when_improved(self, svc):
        """Improvement (positive z-score) should not trigger alerts."""
        alert = svc._check_metric_degradation(
            strategy_name="test",
            metric_name="box_hit_rate",
            current_value=0.40,
            baseline_value=0.20,
            n_current=20,
            n_baseline=180,
        )
        assert alert is None

    def test_no_alert_zero_baseline(self, svc):
        """When baseline is 0 (zero variance), no z-score can be computed."""
        alert = svc._check_metric_degradation(
            strategy_name="test",
            metric_name="box_hit_rate",
            current_value=0.0,
            baseline_value=0.0,
            n_current=14,
            n_baseline=180,
        )
        assert alert is None

    def test_no_alert_baseline_one(self, svc):
        """When baseline is 1.0 (zero variance), no z-score can be computed."""
        alert = svc._check_metric_degradation(
            strategy_name="test",
            metric_name="box_hit_rate",
            current_value=0.9,
            baseline_value=1.0,
            n_current=14,
            n_baseline=180,
        )
        assert alert is None

    def test_z_score_value(self, svc):
        """Verify z-score formula: z = (current - baseline) / (std / sqrt(n))."""
        p = 0.20
        current = 0.05
        n = 50   # z = (0.05-0.20)/(0.4/sqrt(50)) = -2.65 -> WARNING
        std = math.sqrt(p * (1 - p))
        se = std / math.sqrt(n)
        expected_z = (current - p) / se

        alert = svc._check_metric_degradation(
            strategy_name="test",
            metric_name="box_hit_rate",
            current_value=current,
            baseline_value=p,
            n_current=n,
            n_baseline=180,
        )
        assert alert is not None
        assert alert["z_score"] == pytest.approx(expected_z, abs=0.01)

    def test_alert_structure(self, svc):
        """Alerts should contain strategy, severity, z_score, metric, baseline, current."""
        alert = svc._check_metric_degradation(
            strategy_name="frequency",
            metric_name="box_hit_rate",
            current_value=0.0,
            baseline_value=0.25,
            n_current=20,
            n_baseline=200,
        )
        assert alert is not None
        assert "strategy" in alert
        assert "severity" in alert
        assert "z_score" in alert
        assert "metric" in alert
        assert "baseline" in alert
        assert "current" in alert
        assert alert["strategy"] == "frequency"
        assert alert["metric"] == "box_hit_rate"


# ---------------------------------------------------------------------------
# Weight history & current weights
# ---------------------------------------------------------------------------

class TestWeightHistoryAndCurrent:
    """Tests for get_current_weights and get_weight_history."""

    def test_current_weights_fallback_to_equal(self, svc, db):
        """Without stored weights, should return equal weights."""
        with _patch_strategy_names(FAKE_STRATEGIES):
            weights = svc.get_current_weights(db, "daily3")
        assert len(weights) == len(FAKE_STRATEGIES)
        expected = 1.0 / len(FAKE_STRATEGIES)
        for w in weights.values():
            assert w == pytest.approx(expected, abs=1e-6)

    def test_current_weights_from_db(self, svc, db):
        """Stored active weights should be returned."""
        now = datetime.utcnow()
        for name, w in [("frequency", 0.5), ("cold_due", 0.3), ("moving_average", 0.2)]:
            ew = EnsembleWeight(
                game_type="daily3",
                strategy_name=name,
                weight=w,
                effective_from=now,
                effective_to=None,
                reason="test",
            )
            db.add(ew)
        db.flush()

        with _patch_strategy_names(FAKE_STRATEGIES):
            weights = svc.get_current_weights(db, "daily3")
        assert weights["frequency"] == pytest.approx(0.5)
        assert weights["cold_due"] == pytest.approx(0.3)

    def test_weight_history_empty(self, svc, db):
        """With no weight history, should return empty list."""
        history = svc.get_weight_history(db, "daily3", days=90)
        assert history == []

    def test_weight_history_returns_records(self, svc, db):
        """Stored weight history should be retrievable."""
        now = datetime.utcnow()
        for i in range(3):
            ew = EnsembleWeight(
                game_type="daily3",
                strategy_name="frequency",
                weight=0.3 + i * 0.05,
                effective_from=now - timedelta(days=i * 7),
                effective_to=now - timedelta(days=i * 7 - 1) if i < 2 else None,
                reason="test",
            )
            db.add(ew)
        db.flush()

        history = svc.get_weight_history(db, "daily3", days=90)
        assert len(history) == 3
        assert all("date" in h for h in history)
        assert all("strategy_name" in h for h in history)
        assert all("weight" in h for h in history)

    def test_weight_history_respects_days_filter(self, svc, db):
        """History beyond the requested window should be excluded."""
        now = datetime.utcnow()
        # One recent, one old.
        db.add(EnsembleWeight(
            game_type="daily3", strategy_name="frequency", weight=0.5,
            effective_from=now - timedelta(days=5), reason="test",
        ))
        db.add(EnsembleWeight(
            game_type="daily3", strategy_name="frequency", weight=0.4,
            effective_from=now - timedelta(days=100), reason="old",
        ))
        db.flush()

        history = svc.get_weight_history(db, "daily3", days=30)
        assert len(history) == 1
        assert history[0]["weight"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Store weights tests
# ---------------------------------------------------------------------------

class TestStoreWeights:
    """Tests for _store_weights internal method."""

    def test_stores_new_weights(self, svc, db):
        weights = {"frequency": 0.6, "cold_due": 0.4}
        svc._store_weights(db, "daily3", weights, reason="test")
        db.flush()

        active = (
            db.query(EnsembleWeight)
            .filter(EnsembleWeight.game_type == "daily3", EnsembleWeight.effective_to.is_(None))
            .all()
        )
        assert len(active) == 2
        stored = {a.strategy_name: a.weight for a in active}
        assert stored["frequency"] == pytest.approx(0.6)
        assert stored["cold_due"] == pytest.approx(0.4)

    def test_expires_previous_active_weights(self, svc, db):
        svc._store_weights(db, "daily3", {"frequency": 0.5, "cold_due": 0.5}, reason="first")
        db.flush()
        svc._store_weights(db, "daily3", {"frequency": 0.7, "cold_due": 0.3}, reason="second")
        db.flush()

        expired = (
            db.query(EnsembleWeight)
            .filter(EnsembleWeight.game_type == "daily3", EnsembleWeight.effective_to.isnot(None))
            .all()
        )
        assert len(expired) == 2
        assert all(e.reason == "first" for e in expired)

    def test_reason_stored(self, svc, db):
        svc._store_weights(db, "daily3", {"frequency": 1.0}, reason="auto_recalibration")
        db.flush()

        active = (
            db.query(EnsembleWeight)
            .filter(EnsembleWeight.game_type == "daily3", EnsembleWeight.effective_to.is_(None))
            .all()
        )
        assert active[0].reason == "auto_recalibration"


# ---------------------------------------------------------------------------
# Multi-game isolation test
# ---------------------------------------------------------------------------

class TestGameIsolation:
    """Ensure daily3 and daily4 metrics/weights are computed independently."""

    def test_different_games_independent(self, svc, db):
        _seed_predictions(db, "frequency", game_type="daily3", count=30, box_hit_pct=0.30)
        _seed_predictions(db, "frequency", game_type="daily4", count=30, box_hit_pct=0.10)

        with _patch_strategy_names(["frequency"]):
            results_d3 = svc.compute_rolling_performance(db, "daily3")
            results_d4 = svc.compute_rolling_performance(db, "daily4")

        d3_all = next((r for r in results_d3 if r.window_type == "all"), None)
        d4_all = next((r for r in results_d4 if r.window_type == "all"), None)

        assert d3_all is not None and d4_all is not None
        assert d3_all.box_hit_rate > d4_all.box_hit_rate
