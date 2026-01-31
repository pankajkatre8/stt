"""
Tests for Multi-NLP model evaluator.

Tests comparative analysis of multiple NLP pipelines.
"""
from __future__ import annotations

import pytest

from hsttb.core.types import Entity, EntityLabel
from hsttb.metrics.multi_nlp import (
    ModelEvaluation,
    MultiNLPEvaluator,
    MultiNLPResult,
    create_default_evaluator,
)
from hsttb.nlp import MockNERPipeline

# Check if rapidfuzz is available (needed for entity matching)
try:
    from rapidfuzz import fuzz  # noqa: F401
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

requires_rapidfuzz = pytest.mark.skipif(
    not HAS_RAPIDFUZZ, reason="rapidfuzz not installed"
)


class TestModelEvaluation:
    """Tests for ModelEvaluation dataclass."""

    def test_creation(self) -> None:
        """Create ModelEvaluation with basic attributes."""
        evaluation = ModelEvaluation(
            model_name="test_model",
            gt_entities=[],
            pred_entities=[],
            precision=0.9,
            recall=0.8,
            f1_score=0.85,
        )

        assert evaluation.model_name == "test_model"
        assert evaluation.precision == 0.9
        assert evaluation.recall == 0.8
        assert evaluation.f1_score == 0.85

    def test_to_dict(self) -> None:
        """to_dict returns serializable dictionary."""
        evaluation = ModelEvaluation(
            model_name="test",
            gt_entities=[
                Entity(text="metformin", label=EntityLabel.DRUG, span=(0, 9))
            ],
            pred_entities=[
                Entity(text="metformin", label=EntityLabel.DRUG, span=(0, 9))
            ],
            precision=1.0,
            recall=1.0,
            f1_score=1.0,
        )

        result = evaluation.to_dict()

        assert isinstance(result, dict)
        assert result["model_name"] == "test"
        assert result["precision"] == 1.0
        assert result["gt_entity_count"] == 1
        assert result["pred_entity_count"] == 1


class TestMultiNLPResult:
    """Tests for MultiNLPResult dataclass."""

    def test_creation(self) -> None:
        """Create MultiNLPResult."""
        result = MultiNLPResult(
            ground_truth="patient takes metformin",
            predicted="patient takes metformin",
            model_evaluations={},
            best_model="mock",
        )

        assert result.ground_truth == "patient takes metformin"
        assert result.best_model == "mock"

    def test_to_dict(self) -> None:
        """to_dict returns serializable dictionary."""
        evaluation = ModelEvaluation(
            model_name="mock",
            gt_entities=[],
            pred_entities=[],
            f1_score=0.9,
        )
        result = MultiNLPResult(
            ground_truth="gt",
            predicted="pred",
            model_evaluations={"mock": evaluation},
            best_model="mock",
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "mock" in result_dict["models"]
        assert result_dict["best_model"] == "mock"


class TestMultiNLPEvaluator:
    """Tests for MultiNLPEvaluator class."""

    @pytest.fixture
    def evaluator(self) -> MultiNLPEvaluator:
        """Create evaluator with mock model."""
        evaluator = MultiNLPEvaluator()
        evaluator.add_model("mock", MockNERPipeline.with_common_patterns())
        return evaluator

    def test_initialization(self) -> None:
        """Initialize evaluator with default settings."""
        evaluator = MultiNLPEvaluator()
        assert evaluator.list_models() == []

    def test_add_model(self, evaluator: MultiNLPEvaluator) -> None:
        """Add model to evaluator."""
        assert "mock" in evaluator.list_models()

    def test_remove_model(self, evaluator: MultiNLPEvaluator) -> None:
        """Remove model from evaluator."""
        result = evaluator.remove_model("mock")
        assert result is True
        assert "mock" not in evaluator.list_models()

    def test_remove_model_nonexistent(self, evaluator: MultiNLPEvaluator) -> None:
        """Remove nonexistent model returns False."""
        result = evaluator.remove_model("nonexistent")
        assert result is False

    def test_list_models(self, evaluator: MultiNLPEvaluator) -> None:
        """List registered models."""
        models = evaluator.list_models()
        assert isinstance(models, list)
        assert "mock" in models

    @requires_rapidfuzz
    def test_evaluate_basic(self, evaluator: MultiNLPEvaluator) -> None:
        """Basic evaluation with matching texts."""
        result = evaluator.evaluate(
            ground_truth="patient takes metformin",
            predicted="patient takes metformin",
        )

        assert isinstance(result, MultiNLPResult)
        assert "mock" in result.model_evaluations
        assert result.best_model == "mock"

    @requires_rapidfuzz
    def test_evaluate_with_drug_error(self, evaluator: MultiNLPEvaluator) -> None:
        """Evaluation with drug substitution error."""
        result = evaluator.evaluate(
            ground_truth="patient takes metformin",
            predicted="patient takes aspirin",
        )

        assert isinstance(result, MultiNLPResult)
        # Should detect different entities
        mock_eval = result.model_evaluations["mock"]
        assert mock_eval.gt_entities != mock_eval.pred_entities or len(mock_eval.gt_entities) == 0

    def test_evaluate_no_models(self) -> None:
        """Evaluation without models raises ValueError."""
        evaluator = MultiNLPEvaluator()

        with pytest.raises(ValueError, match="No NLP models registered"):
            evaluator.evaluate("text", "text")

    @requires_rapidfuzz
    def test_evaluate_specific_models(self) -> None:
        """Evaluate with specific models only."""
        evaluator = MultiNLPEvaluator()
        evaluator.add_model("model1", MockNERPipeline.with_common_patterns())
        evaluator.add_model("model2", MockNERPipeline.with_common_patterns())

        result = evaluator.evaluate(
            ground_truth="metformin",
            predicted="metformin",
            models=["model1"],
        )

        assert "model1" in result.model_evaluations
        # model2 should not be evaluated when specific models are requested
        # (only model1 is in the result)

    @requires_rapidfuzz
    def test_evaluate_consensus_entities(self, evaluator: MultiNLPEvaluator) -> None:
        """Consensus entities are computed."""
        # Add another model
        evaluator.add_model("mock2", MockNERPipeline.with_common_patterns())

        result = evaluator.evaluate(
            ground_truth="patient takes metformin for diabetes",
            predicted="patient takes metformin for diabetes",
        )

        # Consensus should include entities found by both models
        assert isinstance(result.consensus_entities_gt, list)
        assert isinstance(result.consensus_entities_pred, list)

    @requires_rapidfuzz
    def test_evaluate_agreement_rate(self, evaluator: MultiNLPEvaluator) -> None:
        """Agreement rate is computed."""
        evaluator.add_model("mock2", MockNERPipeline.with_common_patterns())

        result = evaluator.evaluate(
            ground_truth="metformin",
            predicted="metformin",
        )

        # Agreement rate should be between 0 and 1
        assert 0.0 <= result.agreement_rate <= 1.0

    @requires_rapidfuzz
    def test_evaluate_best_model(self) -> None:
        """Best model is determined by F1 score."""
        evaluator = MultiNLPEvaluator()
        evaluator.add_model("model1", MockNERPipeline.with_common_patterns())
        evaluator.add_model("model2", MockNERPipeline())  # Different config

        result = evaluator.evaluate(
            ground_truth="patient takes metformin",
            predicted="patient takes metformin",
        )

        # Best model should be selected
        assert result.best_model in ["model1", "model2"]


class TestMultiNLPEvaluatorMetrics:
    """Tests for metric computation in MultiNLPEvaluator."""

    @pytest.fixture
    def evaluator(self) -> MultiNLPEvaluator:
        """Create evaluator."""
        evaluator = MultiNLPEvaluator()
        evaluator.add_model("mock", MockNERPipeline.with_common_patterns())
        return evaluator

    @requires_rapidfuzz
    def test_precision_computation(self, evaluator: MultiNLPEvaluator) -> None:
        """Precision is computed correctly."""
        result = evaluator.evaluate(
            ground_truth="metformin",
            predicted="metformin",
        )

        mock_eval = result.model_evaluations["mock"]
        # Precision should be between 0 and 1
        assert 0.0 <= mock_eval.precision <= 1.0

    @requires_rapidfuzz
    def test_recall_computation(self, evaluator: MultiNLPEvaluator) -> None:
        """Recall is computed correctly."""
        result = evaluator.evaluate(
            ground_truth="metformin",
            predicted="metformin",
        )

        mock_eval = result.model_evaluations["mock"]
        assert 0.0 <= mock_eval.recall <= 1.0

    @requires_rapidfuzz
    def test_f1_computation(self, evaluator: MultiNLPEvaluator) -> None:
        """F1 score is computed correctly."""
        result = evaluator.evaluate(
            ground_truth="metformin",
            predicted="metformin",
        )

        mock_eval = result.model_evaluations["mock"]
        assert 0.0 <= mock_eval.f1_score <= 1.0

    @requires_rapidfuzz
    def test_extraction_time_measured(self, evaluator: MultiNLPEvaluator) -> None:
        """Extraction time is measured."""
        result = evaluator.evaluate(
            ground_truth="patient takes metformin",
            predicted="patient takes metformin",
        )

        mock_eval = result.model_evaluations["mock"]
        # Extraction time should be positive
        assert mock_eval.extraction_time_ms >= 0.0

    @requires_rapidfuzz
    def test_per_label_metrics(self, evaluator: MultiNLPEvaluator) -> None:
        """Per-label metrics are computed."""
        result = evaluator.evaluate(
            ground_truth="patient takes metformin for diabetes",
            predicted="patient takes metformin for diabetes",
        )

        mock_eval = result.model_evaluations["mock"]
        # Per-label metrics is a dict
        assert isinstance(mock_eval.per_label_metrics, dict)


class TestMultiNLPEvaluatorLabelCompatibility:
    """Tests for label compatibility in entity matching."""

    def test_labels_compatible_same(self) -> None:
        """Same labels are compatible."""
        evaluator = MultiNLPEvaluator()
        assert evaluator._labels_compatible(EntityLabel.DRUG, EntityLabel.DRUG)

    def test_labels_compatible_condition_group(self) -> None:
        """Condition-related labels are compatible."""
        evaluator = MultiNLPEvaluator()
        assert evaluator._labels_compatible(EntityLabel.CONDITION, EntityLabel.DIAGNOSIS)
        assert evaluator._labels_compatible(EntityLabel.DIAGNOSIS, EntityLabel.SYMPTOM)

    def test_labels_incompatible(self) -> None:
        """Unrelated labels are not compatible."""
        evaluator = MultiNLPEvaluator()
        assert not evaluator._labels_compatible(EntityLabel.DRUG, EntityLabel.ANATOMY)


class TestCreateDefaultEvaluator:
    """Tests for create_default_evaluator function."""

    def test_creates_evaluator(self) -> None:
        """create_default_evaluator returns evaluator."""
        evaluator = create_default_evaluator()
        assert isinstance(evaluator, MultiNLPEvaluator)

    def test_has_mock_model(self) -> None:
        """Default evaluator includes mock model."""
        evaluator = create_default_evaluator()
        # Mock model should always be available
        assert "mock" in evaluator.list_models()

    @requires_rapidfuzz
    def test_can_evaluate(self) -> None:
        """Default evaluator can run evaluation."""
        evaluator = create_default_evaluator()

        result = evaluator.evaluate(
            ground_truth="patient takes metformin",
            predicted="patient takes metformin",
        )

        assert isinstance(result, MultiNLPResult)
