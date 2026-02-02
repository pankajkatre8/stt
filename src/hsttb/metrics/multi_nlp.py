"""
Multi-NLP model evaluator for comparative analysis.

Enables side-by-side comparison of multiple NLP pipelines
for entity extraction quality assessment.

Example:
    >>> evaluator = MultiNLPEvaluator()
    >>> evaluator.add_model("scispacy", SciSpacyNERPipeline())
    >>> evaluator.add_model("biomedical", BiomedicalNERPipeline())
    >>> result = evaluator.evaluate(ground_truth, predicted)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from hsttb.core.types import Entity, EntityLabel
from hsttb.nlp.ner_pipeline import NERPipeline

logger = logging.getLogger(__name__)


@dataclass
class ModelEvaluation:
    """
    Evaluation results for a single NLP model.

    Attributes:
        model_name: Name of the model.
        gt_entities: Entities extracted from ground truth.
        pred_entities: Entities extracted from prediction.
        precision: Precision score.
        recall: Recall score.
        f1_score: F1 score.
        entity_match_rate: Rate of matching entities.
        extraction_time_ms: Time taken for extraction.
    """

    model_name: str
    gt_entities: list[Entity]
    pred_entities: list[Entity]
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    entity_match_rate: float = 0.0
    extraction_time_ms: float = 0.0

    # Per-label metrics
    per_label_metrics: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_name": self.model_name,
            "gt_entity_count": len(self.gt_entities),
            "pred_entity_count": len(self.pred_entities),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "entity_match_rate": round(self.entity_match_rate, 4),
            "extraction_time_ms": round(self.extraction_time_ms, 2),
            "per_label_metrics": {
                k: {m: round(v, 4) for m, v in metrics.items()}
                for k, metrics in self.per_label_metrics.items()
            },
            "gt_entities": [
                {"text": e.text, "label": e.label.value, "negated": e.negated}
                for e in self.gt_entities
            ],
            "pred_entities": [
                {"text": e.text, "label": e.label.value, "negated": e.negated}
                for e in self.pred_entities
            ],
        }


@dataclass
class MultiNLPResult:
    """
    Results from multi-model NLP evaluation.

    Attributes:
        ground_truth: Original ground truth text.
        predicted: Original predicted text.
        model_evaluations: Results per model.
        best_model: Name of best performing model.
        consensus_entities: Entities found by majority of models.
        agreement_rate: Rate of agreement across models.
    """

    ground_truth: str
    predicted: str
    model_evaluations: dict[str, ModelEvaluation]
    best_model: str = ""
    consensus_entities_gt: list[Entity] = field(default_factory=list)
    consensus_entities_pred: list[Entity] = field(default_factory=list)
    agreement_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "ground_truth": self.ground_truth,
            "predicted": self.predicted,
            "models": {
                name: eval.to_dict()
                for name, eval in self.model_evaluations.items()
            },
            "best_model": self.best_model,
            "consensus_gt_entities": [
                {"text": e.text, "label": e.label.value}
                for e in self.consensus_entities_gt
            ],
            "consensus_pred_entities": [
                {"text": e.text, "label": e.label.value}
                for e in self.consensus_entities_pred
            ],
            "agreement_rate": round(self.agreement_rate, 4),
        }


class MultiNLPEvaluator:
    """
    Evaluator for comparing multiple NLP models.

    Enables side-by-side comparison of entity extraction
    across different NLP pipelines to identify the best
    model for specific use cases.

    Attributes:
        models: Dictionary of registered NLP models.
        entity_matcher: Strategy for matching entities.

    Example:
        >>> evaluator = MultiNLPEvaluator()
        >>> evaluator.add_model("scispacy", SciSpacyNERPipeline())
        >>> evaluator.add_model("biomedical", BiomedicalNERPipeline())
        >>> result = evaluator.evaluate(
        ...     "Patient takes metformin for diabetes",
        ...     "Patient takes methotrexate for diabetes"
        ... )
        >>> print(f"Best model: {result.best_model}")
    """

    def __init__(
        self,
        similarity_threshold: float = 0.8,
        case_sensitive: bool = False,
    ) -> None:
        """
        Initialize the evaluator.

        Args:
            similarity_threshold: Threshold for fuzzy entity matching.
            case_sensitive: Whether entity matching is case-sensitive.
        """
        self._models: dict[str, NERPipeline] = {}
        self._similarity_threshold = similarity_threshold
        self._case_sensitive = case_sensitive

    def add_model(self, name: str, pipeline: NERPipeline) -> None:
        """
        Add an NLP model to the evaluator.

        Args:
            name: Unique name for the model.
            pipeline: NER pipeline instance.
        """
        self._models[name] = pipeline
        logger.info(f"Added NLP model: {name}")

    def remove_model(self, name: str) -> bool:
        """
        Remove an NLP model from the evaluator.

        Args:
            name: Name of the model to remove.

        Returns:
            True if removed, False if not found.
        """
        if name in self._models:
            del self._models[name]
            return True
        return False

    def list_models(self) -> list[str]:
        """List registered model names."""
        return list(self._models.keys())

    def evaluate(
        self,
        ground_truth: str,
        predicted: str,
        models: list[str] | None = None,
    ) -> MultiNLPResult:
        """
        Evaluate texts using registered NLP models.

        Args:
            ground_truth: Ground truth text.
            predicted: Predicted text.
            models: Optional list of models to use (defaults to all).

        Returns:
            MultiNLPResult with comparative analysis.
        """
        import time

        if not self._models:
            raise ValueError("No NLP models registered")

        model_names = models or list(self._models.keys())
        model_evaluations: dict[str, ModelEvaluation] = {}

        for name in model_names:
            if name not in self._models:
                logger.warning(f"Model not found: {name}")
                continue

            pipeline = self._models[name]

            # Extract entities with timing
            start = time.time()
            try:
                gt_entities = pipeline.extract_entities(ground_truth)
                pred_entities = pipeline.extract_entities(predicted)
            except Exception as e:
                logger.error(f"Model {name} extraction failed: {e}")
                continue
            extraction_time = (time.time() - start) * 1000

            # Compute metrics
            evaluation = self._compute_model_metrics(
                name,
                gt_entities,
                pred_entities,
                extraction_time,
            )
            model_evaluations[name] = evaluation

        if not model_evaluations:
            raise ValueError("No models produced valid results")

        # Find best model
        best_model = max(
            model_evaluations.keys(),
            key=lambda m: model_evaluations[m].f1_score,
        )

        # Compute consensus entities
        consensus_gt = self._compute_consensus(
            [e.gt_entities for e in model_evaluations.values()]
        )
        consensus_pred = self._compute_consensus(
            [e.pred_entities for e in model_evaluations.values()]
        )

        # Compute agreement rate
        agreement_rate = self._compute_agreement_rate(model_evaluations)

        return MultiNLPResult(
            ground_truth=ground_truth,
            predicted=predicted,
            model_evaluations=model_evaluations,
            best_model=best_model,
            consensus_entities_gt=consensus_gt,
            consensus_entities_pred=consensus_pred,
            agreement_rate=agreement_rate,
        )

    def _compute_model_metrics(
        self,
        model_name: str,
        gt_entities: list[Entity],
        pred_entities: list[Entity],
        extraction_time_ms: float,
    ) -> ModelEvaluation:
        """Compute metrics for a single model."""
        # Match entities
        matches = self._match_entities(gt_entities, pred_entities)

        # Compute precision/recall
        true_positives = len(matches)
        false_positives = len(pred_entities) - true_positives
        false_negatives = len(gt_entities) - true_positives

        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0.0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0.0
        )
        f1_score = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        # Entity match rate
        entity_match_rate = (
            len(matches) / len(gt_entities) if gt_entities else 1.0
        )

        # Per-label metrics
        per_label = self._compute_per_label_metrics(gt_entities, pred_entities)

        return ModelEvaluation(
            model_name=model_name,
            gt_entities=gt_entities,
            pred_entities=pred_entities,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            entity_match_rate=entity_match_rate,
            extraction_time_ms=extraction_time_ms,
            per_label_metrics=per_label,
        )

    def _match_entities(
        self,
        gt_entities: list[Entity],
        pred_entities: list[Entity],
    ) -> list[tuple[Entity, Entity]]:
        """Match ground truth entities to predicted entities."""
        from rapidfuzz import fuzz

        matches = []
        used_pred_indices: set[int] = set()

        for gt_ent in gt_entities:
            best_match = None
            best_score = 0.0
            best_idx = -1

            gt_text = gt_ent.text if self._case_sensitive else gt_ent.text.lower()

            for idx, pred_ent in enumerate(pred_entities):
                if idx in used_pred_indices:
                    continue

                # Must have same label (or compatible labels)
                if not self._labels_compatible(gt_ent.label, pred_ent.label):
                    continue

                pred_text = (
                    pred_ent.text if self._case_sensitive else pred_ent.text.lower()
                )

                # Compute similarity
                score = fuzz.ratio(gt_text, pred_text) / 100.0

                if score >= self._similarity_threshold and score > best_score:
                    best_match = pred_ent
                    best_score = score
                    best_idx = idx

            if best_match is not None:
                matches.append((gt_ent, best_match))
                used_pred_indices.add(best_idx)

        return matches

    def _labels_compatible(
        self,
        label1: EntityLabel,
        label2: EntityLabel,
    ) -> bool:
        """Check if two labels are compatible for matching."""
        if label1 == label2:
            return True

        # Define compatible label groups
        compatible_groups = [
            {EntityLabel.CONDITION, EntityLabel.DIAGNOSIS, EntityLabel.SYMPTOM},
            {EntityLabel.DRUG, EntityLabel.DOSAGE},
            {EntityLabel.PROCEDURE, EntityLabel.LAB_VALUE},
        ]

        for group in compatible_groups:
            if label1 in group and label2 in group:
                return True

        return False

    def _compute_per_label_metrics(
        self,
        gt_entities: list[Entity],
        pred_entities: list[Entity],
    ) -> dict[str, dict[str, float]]:
        """Compute metrics per entity label."""
        labels = set(e.label for e in gt_entities) | set(e.label for e in pred_entities)
        per_label: dict[str, dict[str, float]] = {}

        for label in labels:
            gt_label = [e for e in gt_entities if e.label == label]
            pred_label = [e for e in pred_entities if e.label == label]

            matches = self._match_entities(gt_label, pred_label)
            tp = len(matches)
            fp = len(pred_label) - tp
            fn = len(gt_label) - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            per_label[label.value] = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "gt_count": len(gt_label),
                "pred_count": len(pred_label),
            }

        return per_label

    def _compute_consensus(
        self,
        entity_lists: list[list[Entity]],
    ) -> list[Entity]:
        """Compute consensus entities across models."""
        if not entity_lists:
            return []

        # Threshold for consensus (majority)
        threshold = len(entity_lists) / 2

        # Collect all entity texts with counts
        entity_counts: dict[str, tuple[int, Entity]] = {}

        for entities in entity_lists:
            seen_in_model: set[str] = set()
            for entity in entities:
                key = (
                    entity.text.lower()
                    if not self._case_sensitive
                    else entity.text
                )
                if key not in seen_in_model:
                    if key in entity_counts:
                        count, _ = entity_counts[key]
                        entity_counts[key] = (count + 1, entity)
                    else:
                        entity_counts[key] = (1, entity)
                    seen_in_model.add(key)

        # Return entities that appear in majority of models
        consensus = [
            entity for count, entity in entity_counts.values()
            if count >= threshold
        ]

        return consensus

    def _compute_agreement_rate(
        self,
        evaluations: dict[str, ModelEvaluation],
    ) -> float:
        """Compute agreement rate across models."""
        if len(evaluations) < 2:
            return 1.0

        # Compare F1 scores across models
        f1_scores = [e.f1_score for e in evaluations.values()]
        if not f1_scores:
            return 0.0

        # Compute variance-based agreement
        mean_f1 = sum(f1_scores) / len(f1_scores)
        if mean_f1 == 0:
            return 0.0

        variance = sum((f - mean_f1) ** 2 for f in f1_scores) / len(f1_scores)
        std_dev = variance ** 0.5

        # Agreement is inverse of coefficient of variation
        cv = std_dev / mean_f1 if mean_f1 > 0 else 1.0
        agreement = max(0.0, 1.0 - cv)

        return agreement


# Convenience function for loading default models
def create_default_evaluator() -> MultiNLPEvaluator:
    """
    Create evaluator with default NLP models.

    Returns:
        MultiNLPEvaluator with scispacy, biomedical, and medspacy models.
    """
    from hsttb.nlp.registry import get_nlp_pipeline

    evaluator = MultiNLPEvaluator()

    # Add available models (production models only)
    for model_name in ["scispacy", "biomedical", "medspacy"]:
        try:
            pipeline = get_nlp_pipeline(model_name)
            evaluator.add_model(model_name, pipeline)
        except Exception as e:
            logger.warning(f"Could not load {model_name}: {e}")

    return evaluator
