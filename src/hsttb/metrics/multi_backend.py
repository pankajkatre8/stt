"""
Multi-backend TER evaluation for comparing NLP tools.

This module provides evaluation across multiple lexicon backends,
allowing comparison of term extraction accuracy between different
medical NLP tools (MedCAT, scispaCy, etc.).

Example:
    >>> from hsttb.metrics.multi_backend import MultiBackendEvaluator
    >>> evaluator = MultiBackendEvaluator()
    >>> evaluator.add_backend("medcat", medcat_lexicon)
    >>> evaluator.add_backend("scispacy", scispacy_lexicon)
    >>> results = evaluator.evaluate(ground_truth, prediction)
    >>> for name, metrics in results.items():
    ...     print(f"{name}: TER={metrics.overall_ter:.1%}")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from hsttb.metrics.ter import TEREngine, TERResult

if TYPE_CHECKING:
    from hsttb.lexicons.base import MedicalLexicon


@dataclass
class BackendMetrics:
    """
    Metrics from a single backend evaluation.

    Attributes:
        backend_name: Name of the backend (e.g., "medcat", "scispacy").
        ter_result: Full TER result from evaluation.
        terms_extracted_gt: Number of terms extracted from ground truth.
        terms_extracted_pred: Number of terms extracted from prediction.
    """

    backend_name: str
    ter_result: TERResult
    terms_extracted_gt: int
    terms_extracted_pred: int

    @property
    def ter(self) -> float:
        """Overall TER value."""
        return self.ter_result.overall_ter

    @property
    def accuracy(self) -> float:
        """Term accuracy (1 - TER)."""
        return 1.0 - min(self.ter_result.overall_ter, 1.0)


@dataclass
class MultiBackendResult:
    """
    Results from multi-backend evaluation.

    Attributes:
        ground_truth: Original ground truth text.
        prediction: Original prediction text.
        backend_metrics: Dict mapping backend name to metrics.
        consensus_terms_gt: Terms found by all backends in GT.
        consensus_terms_pred: Terms found by all backends in prediction.
    """

    ground_truth: str
    prediction: str
    backend_metrics: dict[str, BackendMetrics] = field(default_factory=dict)
    consensus_terms_gt: list[str] = field(default_factory=list)
    consensus_terms_pred: list[str] = field(default_factory=list)

    @property
    def best_backend(self) -> str | None:
        """Backend with lowest TER (best performance)."""
        if not self.backend_metrics:
            return None
        return min(
            self.backend_metrics.items(),
            key=lambda x: x[1].ter
        )[0]

    @property
    def average_ter(self) -> float:
        """Average TER across all backends."""
        if not self.backend_metrics:
            return 0.0
        total = sum(m.ter for m in self.backend_metrics.values())
        return total / len(self.backend_metrics)

    def get_comparison_table(self) -> list[dict[str, any]]:
        """
        Get comparison data for all backends.

        Returns:
            List of dicts with backend comparison data.
        """
        rows = []
        for name, metrics in sorted(self.backend_metrics.items()):
            rows.append({
                "backend": name,
                "ter": metrics.ter,
                "accuracy": metrics.accuracy,
                "gt_terms": metrics.terms_extracted_gt,
                "pred_terms": metrics.terms_extracted_pred,
                "correct": metrics.ter_result.correct_matches,
                "substitutions": len(metrics.ter_result.substitutions),
                "deletions": len(metrics.ter_result.deletions),
                "insertions": len(metrics.ter_result.insertions),
            })
        return rows


class MultiBackendEvaluator:
    """
    Evaluate TER across multiple lexicon backends.

    Allows comparing how different medical NLP tools perform
    on the same clinical text.

    Example:
        >>> evaluator = MultiBackendEvaluator()
        >>> evaluator.add_backend("medcat", MedCATLexicon())
        >>> evaluator.add_backend("scispacy", SciSpacyLexicon())
        >>> results = evaluator.evaluate("Patient has diabetes", "Patient has diabetis")
        >>> print(results.get_comparison_table())
    """

    def __init__(self) -> None:
        """Initialize multi-backend evaluator."""
        self._backends: dict[str, MedicalLexicon] = {}
        self._engines: dict[str, TEREngine] = {}

    def add_backend(
        self,
        name: str,
        lexicon: MedicalLexicon,
        fuzzy_threshold: float = 0.85,
    ) -> None:
        """
        Add a lexicon backend for evaluation.

        Args:
            name: Unique name for this backend.
            lexicon: Medical lexicon instance.
            fuzzy_threshold: TER fuzzy matching threshold.
        """
        self._backends[name] = lexicon
        self._engines[name] = TEREngine(lexicon, fuzzy_threshold=fuzzy_threshold)

    def remove_backend(self, name: str) -> bool:
        """
        Remove a backend.

        Args:
            name: Backend name to remove.

        Returns:
            True if removed, False if not found.
        """
        if name in self._backends:
            del self._backends[name]
            del self._engines[name]
            return True
        return False

    def list_backends(self) -> list[str]:
        """List all registered backend names."""
        return list(self._backends.keys())

    def evaluate(
        self,
        ground_truth: str,
        prediction: str,
    ) -> MultiBackendResult:
        """
        Evaluate text with all backends.

        Args:
            ground_truth: Reference transcript.
            prediction: Predicted transcript.

        Returns:
            MultiBackendResult with metrics from each backend.
        """
        result = MultiBackendResult(
            ground_truth=ground_truth,
            prediction=prediction,
        )

        all_gt_terms: list[set[str]] = []
        all_pred_terms: list[set[str]] = []

        for name, engine in self._engines.items():
            # Run TER evaluation
            ter_result = engine.compute(ground_truth, prediction)

            # Extract term texts for consensus calculation
            gt_term_texts = {
                t.text.lower()
                for t in engine._extract_terms(
                    engine.normalizer.normalize(ground_truth)
                )
            }
            pred_term_texts = {
                t.text.lower()
                for t in engine._extract_terms(
                    engine.normalizer.normalize(prediction)
                )
            }

            all_gt_terms.append(gt_term_texts)
            all_pred_terms.append(pred_term_texts)

            result.backend_metrics[name] = BackendMetrics(
                backend_name=name,
                ter_result=ter_result,
                terms_extracted_gt=ter_result.total_gt_terms,
                terms_extracted_pred=ter_result.total_pred_terms,
            )

        # Calculate consensus terms (found by all backends)
        if all_gt_terms:
            consensus_gt = all_gt_terms[0]
            for terms in all_gt_terms[1:]:
                consensus_gt = consensus_gt & terms
            result.consensus_terms_gt = sorted(consensus_gt)

        if all_pred_terms:
            consensus_pred = all_pred_terms[0]
            for terms in all_pred_terms[1:]:
                consensus_pred = consensus_pred & terms
            result.consensus_terms_pred = sorted(consensus_pred)

        return result

    def evaluate_batch(
        self,
        pairs: list[tuple[str, str]],
    ) -> list[MultiBackendResult]:
        """
        Evaluate multiple text pairs.

        Args:
            pairs: List of (ground_truth, prediction) tuples.

        Returns:
            List of MultiBackendResult for each pair.
        """
        return [
            self.evaluate(gt, pred)
            for gt, pred in pairs
        ]

    def get_aggregate_metrics(
        self,
        results: list[MultiBackendResult],
    ) -> dict[str, dict[str, float]]:
        """
        Calculate aggregate metrics across multiple evaluations.

        Args:
            results: List of evaluation results.

        Returns:
            Dict mapping backend name to aggregate metrics.
        """
        if not results:
            return {}

        aggregates: dict[str, dict[str, list[float]]] = {}

        for result in results:
            for name, metrics in result.backend_metrics.items():
                if name not in aggregates:
                    aggregates[name] = {
                        "ter": [],
                        "precision": [],
                        "recall": [],
                        "f1": [],
                    }
                aggregates[name]["ter"].append(metrics.ter)
                aggregates[name]["precision"].append(metrics.ter_result.precision)
                aggregates[name]["recall"].append(metrics.ter_result.recall)
                aggregates[name]["f1"].append(metrics.ter_result.f1_score)

        # Calculate averages
        return {
            name: {
                metric: sum(values) / len(values)
                for metric, values in metrics.items()
            }
            for name, metrics in aggregates.items()
        }


def create_default_evaluator() -> MultiBackendEvaluator:
    """
    Create evaluator with available backends.

    Attempts to load MedCAT and scispaCy if available,
    falls back to mock lexicon if not.

    Returns:
        MultiBackendEvaluator with available backends.
    """
    from hsttb.lexicons.mock_lexicon import MockMedicalLexicon

    evaluator = MultiBackendEvaluator()

    # Always add mock for baseline
    evaluator.add_backend("mock", MockMedicalLexicon.with_common_terms())

    # Try MedCAT
    try:
        from hsttb.lexicons.medcat_lexicon import MedCATLexicon
        # MedCAT requires a model to be loaded, so we can't add it by default
        # Just verify it's importable
        _ = MedCATLexicon
    except ImportError:
        pass

    # Try scispaCy
    try:
        from hsttb.lexicons.scispacy_lexicon import SciSpacyLexicon
        lexicon = SciSpacyLexicon()
        try:
            lexicon.load("en_ner_bc5cdr_md")
            evaluator.add_backend("scispacy_bc5cdr", lexicon)
        except OSError:
            # Model not installed
            pass
    except ImportError:
        pass

    return evaluator
