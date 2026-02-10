#!/usr/bin/env python3
"""Side-by-side comparison of baseline vs enhanced reference-free quality scoring.

This script keeps your existing `QualityEngine` behavior as the baseline and adds an
"enhanced" overlay with practical improvements for:
- disfluency handling (e.g., "uh", "um")
- missing clinical context hints (e.g., dosage present but medication absent)
- casual chitchat down-weighting for clinical-risk interpretation

It is intentionally non-invasive: no core library code is changed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import types
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DISFLUENCY_PATTERN = re.compile(r"\b(uh+|um+|hmm+|erm+)\b", flags=re.IGNORECASE)
WORD_PATTERN = re.compile(r"\b[a-zA-Z']+\b")

# Small configurable lexicons for the script-level overlay
CHATTER_WORDS = {
    "okay",
    "ok",
    "alright",
    "thanks",
    "thank",
    "hello",
    "hi",
    "morning",
    "evening",
    "please",
    "sure",
    "great",
    "good",
    "bye",
}

MEDICATION_HINT_WORDS = {
    "metformin",
    "insulin",
    "aspirin",
    "atorvastatin",
    "lisinopril",
    "amlodipine",
    "warfarin",
    "ibuprofen",
}

CONDITION_HINT_WORDS = {
    "diabetes",
    "hypertension",
    "pain",
    "asthma",
    "allergy",
}

DOSAGE_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\s*(mg|mcg|g|ml|units?)\b", re.IGNORECASE)


@dataclass
class EnhancedOverlayResult:
    """Additional signals layered on top of baseline quality output."""

    disfluency_count: int
    chatter_ratio: float
    missing_context_alerts: list[dict[str, Any]] = field(default_factory=list)
    enhanced_composite_score: float = 0.0
    enhanced_recommendation: str = "REVIEW"


@dataclass
class ComparisonResult:
    """Serializable comparison payload for one transcript."""

    input_text: str
    cleaned_text: str
    baseline: dict[str, Any]
    enhanced: dict[str, Any]


def _safe_import_quality_engine() -> tuple[Any | None, str | None]:
    """Try to import QualityEngine, with a namespace-package fallback.

    Fallback path bypasses heavy `__init__.py` side effects (like `yaml` import
    from `hsttb.core.__init__`) by injecting lightweight namespace packages.
    """
    try:
        from hsttb.metrics.quality import QualityEngine

        return QualityEngine, None
    except Exception as first_exc:  # pragma: no cover - environment dependent
        # Try direct-module loading path that avoids importing hsttb.metrics.__init__.
        try:
            repo_root = Path(__file__).resolve().parents[1]
            src_root = repo_root / "src"
            hsttb_root = src_root / "hsttb"

            if str(src_root) not in sys.path:
                sys.path.insert(0, str(src_root))

            def _ensure_namespace(name: str, path: Path) -> None:
                mod = sys.modules.get(name)
                if mod is None:
                    mod = types.ModuleType(name)
                    mod.__path__ = [str(path)]  # type: ignore[attr-defined]
                    sys.modules[name] = mod

            _ensure_namespace("hsttb", hsttb_root)
            _ensure_namespace("hsttb.metrics", hsttb_root / "metrics")
            _ensure_namespace("hsttb.core", hsttb_root / "core")
            _ensure_namespace("hsttb.nlp", hsttb_root / "nlp")
            _ensure_namespace("hsttb.lexicons", hsttb_root / "lexicons")

            quality_path = hsttb_root / "metrics" / "quality.py"
            spec = importlib.util.spec_from_file_location("hsttb.metrics.quality", quality_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Unable to create module spec for {quality_path}")

            module = importlib.util.module_from_spec(spec)
            sys.modules["hsttb.metrics.quality"] = module
            spec.loader.exec_module(module)
            QualityEngine = getattr(module, "QualityEngine")
            return QualityEngine, f"standard import failed ({first_exc}); used direct-module mode"
        except Exception as second_exc:  # pragma: no cover - environment dependent
            return None, f"standard import failed ({first_exc}); direct-module failed ({second_exc})"


def _fallback_quality_payload(text: str) -> dict[str, Any]:
    """Simple dependency-free quality approximation.

    This is only used when the real project QualityEngine cannot be imported.
    """
    words = [w.lower() for w in WORD_PATTERN.findall(text)]
    if not words:
        return {
            "composite_score": 0.0,
            "recommendation": "REJECT",
            "clinical_risk_score": 0.0,
            "clinical_recommendation": "REJECT",
            "transcription_error_score": 0.0,
            "engine_source": "fallback",
        }

    disfluencies = len(DISFLUENCY_PATTERN.findall(text))
    chatter = chatter_ratio(text)
    has_dosage = bool(DOSAGE_PATTERN.search(text))
    has_med_action = any(v in text.lower() for v in ("take", "taking", "prescribed", "on"))
    has_med_name = any(m in text.lower() for m in MEDICATION_HINT_WORDS)

    score = 0.85
    score -= min(0.25, disfluencies * 0.03)
    score -= min(0.20, chatter * 0.25)
    if has_dosage and has_med_action and not has_med_name:
        score -= 0.15

    score = max(0.0, min(1.0, score))
    reco = recommendation_from_score(score)
    return {
        "composite_score": round(score, 4),
        "recommendation": reco,
        "clinical_risk_score": round(score, 4),
        "clinical_recommendation": reco,
        "transcription_error_score": round(score, 4),
        "engine_source": "fallback",
    }


def normalize_disfluencies(text: str) -> tuple[str, int]:
    """Remove common spoken fillers and return normalized text + removed count."""
    matches = DISFLUENCY_PATTERN.findall(text)
    cleaned = DISFLUENCY_PATTERN.sub(" ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned, len(matches)


def chatter_ratio(text: str) -> float:
    """Estimate fraction of casual chitchat words in the utterance."""
    words = [w.lower() for w in WORD_PATTERN.findall(text)]
    if not words:
        return 0.0
    chatter = sum(1 for w in words if w in CHATTER_WORDS)
    return chatter / len(words)


def detect_missing_context(text: str) -> list[dict[str, Any]]:
    """Heuristic missing-context detector for clinical snippets.

    Example that should trigger:
    "I have diabetes for that I take 500 mg" (dosage + condition + no med name)
    """
    alerts: list[dict[str, Any]] = []
    lowered = text.lower()

    has_dosage = bool(DOSAGE_PATTERN.search(text))
    has_condition = any(c in lowered for c in CONDITION_HINT_WORDS)
    has_medication_name = any(m in lowered for m in MEDICATION_HINT_WORDS)
    has_medication_action = any(v in lowered for v in ("take", "taking", "prescribed", "on"))

    if has_dosage and has_condition and has_medication_action and not has_medication_name:
        alerts.append(
            {
                "type": "MISSING_MEDICATION_NAME",
                "severity": "HIGH",
                "reason": "Condition + dosage detected without a recognized medication name.",
                "suggestion": "Verify the intended drug name before scoring as clinically safe.",
            }
        )

    return alerts


def recommendation_from_score(score: float, accept: float = 0.75, review: float = 0.50) -> str:
    if score >= accept:
        return "ACCEPT"
    if score >= review:
        return "REVIEW"
    return "REJECT"


def build_comparison(text: str) -> ComparisonResult:
    """Run baseline and enhanced pipelines for side-by-side output."""
    QualityEngineCls, import_error = _safe_import_quality_engine()

    if QualityEngineCls is None:
        baseline_payload = _fallback_quality_payload(text)
        cleaned, disfluencies = normalize_disfluencies(text)
        cleaned_quality_payload = _fallback_quality_payload(cleaned) if cleaned else baseline_payload
    else:
        engine = QualityEngineCls()
        baseline_result = engine.compute(text)
        baseline_payload = {
            "composite_score": baseline_result.composite_score,
            "recommendation": baseline_result.recommendation,
            "clinical_risk_score": baseline_result.clinical_risk_score,
            "clinical_recommendation": baseline_result.clinical_recommendation,
            "transcription_error_score": baseline_result.transcription_error_score,
            "engine_source": "quality_engine",
        }
        cleaned, disfluencies = normalize_disfluencies(text)
        cleaned_result = engine.compute(cleaned) if cleaned else baseline_result
        cleaned_quality_payload = {
            "composite_score": cleaned_result.composite_score,
            "recommendation": cleaned_result.recommendation,
            "clinical_risk_score": cleaned_result.clinical_risk_score,
            "clinical_recommendation": cleaned_result.clinical_recommendation,
        }

    if import_error:
        baseline_payload["import_warning"] = import_error

    ratio = chatter_ratio(cleaned)
    alerts = detect_missing_context(cleaned)

    # Enhanced composite starts from cleaned-text quality, then applies small overlays.
    enhanced_score = float(cleaned_quality_payload["composite_score"])

    # Penalize hard missing clinical context.
    enhanced_score -= min(0.25, 0.10 * len(alerts))

    # If a lot of chatter is present, avoid over-penalizing clinical quality.
    # Blend toward baseline by up to +5 points.
    enhanced_score += min(0.05, ratio * 0.10)

    enhanced_score = max(0.0, min(1.0, enhanced_score))

    overlay = EnhancedOverlayResult(
        disfluency_count=disfluencies,
        chatter_ratio=round(ratio, 4),
        missing_context_alerts=alerts,
        enhanced_composite_score=round(enhanced_score, 4),
        enhanced_recommendation=recommendation_from_score(enhanced_score),
    )

    enhanced_payload = {
        "quality_on_cleaned_text": cleaned_quality_payload,
        "overlay": asdict(overlay),
    }

    return ComparisonResult(
        input_text=text,
        cleaned_text=cleaned,
        baseline=baseline_payload,
        enhanced=enhanced_payload,
    )


def print_side_by_side(result: ComparisonResult) -> None:
    """Human-readable terminal output."""
    baseline = result.baseline
    overlay = result.enhanced["overlay"]

    print("\n=== Transcript ===")
    print(result.input_text)
    print("\n=== Cleaned (disfluency-normalized) ===")
    print(result.cleaned_text)

    print("\n=== Baseline vs Enhanced ===")
    print(f"Baseline composite score : {baseline['composite_score']:.4f}")
    print(f"Enhanced composite score : {overlay['enhanced_composite_score']:.4f}")
    print(f"Baseline recommendation  : {baseline['recommendation']}")
    print(f"Enhanced recommendation  : {overlay['enhanced_recommendation']}")
    print(f"Baseline engine source   : {baseline.get('engine_source', 'unknown')}")
    if baseline.get("import_warning"):
        print(f"Import warning           : {baseline['import_warning']}")

    print("\n--- Supporting details ---")
    print(f"Disfluencies removed     : {overlay['disfluency_count']}")
    print(f"Chitchat ratio           : {overlay['chatter_ratio']:.4f}")

    alerts = overlay["missing_context_alerts"]
    if alerts:
        print("Missing context alerts   :")
        for i, alert in enumerate(alerts, start=1):
            print(f"  {i}. {alert['type']} [{alert['severity']}] - {alert['reason']}")
    else:
        print("Missing context alerts   : none")

    print("\n(For full payload use --json-out.)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline QualityEngine output vs enhanced overlay scoring."
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Single transcript text to compare.",
    )
    parser.add_argument(
        "--text-file",
        type=Path,
        help="Path to a UTF-8 text file (one transcript per line).",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Optional output path for JSON results.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    inputs: list[str] = []
    if args.text:
        inputs.append(args.text)

    if args.text_file:
        if not args.text_file.exists():
            raise FileNotFoundError(f"Text file not found: {args.text_file}")
        lines = [line.strip() for line in args.text_file.read_text(encoding="utf-8").splitlines()]
        inputs.extend([line for line in lines if line])

    if not inputs:
        raise ValueError("Provide --text or --text-file")

    results = [build_comparison(text) for text in inputs]

    # Print first result nicely; remaining as compact blocks.
    print_side_by_side(results[0])
    for extra in results[1:]:
        print("\n" + "=" * 72)
        print_side_by_side(extra)

    if args.json_out:
        payload = [asdict(r) for r in results]
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved JSON report to: {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
