"""
FastAPI web application for HSTTB evaluation.

Provides a simple web interface for running healthcare STT benchmarks
without authentication.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from hsttb.metrics.ter import TEREngine
from hsttb.metrics.ner import NEREngine
from hsttb.metrics.crs import CRSEngine
from hsttb.lexicons.mock_lexicon import MockMedicalLexicon
from hsttb.nlp.ner_pipeline import MockNERPipeline

# Get paths
WEBAPP_DIR = Path(__file__).parent
TEMPLATES_DIR = WEBAPP_DIR / "templates"
STATIC_DIR = WEBAPP_DIR / "static"


class EvaluationRequest(BaseModel):
    """Request model for evaluation endpoint."""

    ground_truth: str
    predicted: str
    compute_ter: bool = True
    compute_ner: bool = True
    compute_crs: bool = True


class SegmentedEvaluationRequest(BaseModel):
    """Request model for segmented CRS evaluation."""

    ground_truth_segments: list[str]
    predicted_segments: list[str]


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    application = FastAPI(
        title="HSTTB - Healthcare STT Benchmarking",
        description="Evaluate speech-to-text transcriptions for healthcare applications",
        version="1.0.0",
    )

    # Mount static files
    if STATIC_DIR.exists():
        application.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    # Set up templates
    templates = Jinja2Templates(directory=TEMPLATES_DIR)

    # Lazy-loaded engines and lexicon
    _lexicon: MockMedicalLexicon | None = None
    _ter_engine: TEREngine | None = None
    _ner_engine: NEREngine | None = None
    _crs_engine: CRSEngine | None = None

    def get_lexicon() -> MockMedicalLexicon:
        nonlocal _lexicon
        if _lexicon is None:
            _lexicon = MockMedicalLexicon()
        return _lexicon

    def get_ter_engine() -> TEREngine:
        nonlocal _ter_engine
        if _ter_engine is None:
            _ter_engine = TEREngine(get_lexicon())
        return _ter_engine

    def get_ner_engine() -> NEREngine:
        nonlocal _ner_engine
        if _ner_engine is None:
            pipeline = MockNERPipeline.with_common_patterns()
            _ner_engine = NEREngine(pipeline)
        return _ner_engine

    def get_crs_engine() -> CRSEngine:
        nonlocal _crs_engine
        if _crs_engine is None:
            _crs_engine = CRSEngine()
        return _crs_engine

    @application.get("/", response_class=HTMLResponse)
    async def landing_page(request: Request) -> HTMLResponse:
        """Render the landing page."""
        return templates.TemplateResponse("index.html", {"request": request})

    @application.get("/health")
    async def health_check() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "healthy", "service": "hsttb"}

    @application.post("/api/evaluate")
    async def evaluate(req: EvaluationRequest) -> JSONResponse:
        """
        Run evaluation on ground truth vs predicted text.

        Returns TER, NER, and CRS metrics based on request flags.
        """
        results: dict[str, Any] = {
            "ground_truth": req.ground_truth,
            "predicted": req.predicted,
        }

        try:
            # Compute TER
            if req.compute_ter:
                ter_result = get_ter_engine().compute(req.ground_truth, req.predicted)
                results["ter"] = {
                    "overall_ter": round(ter_result.overall_ter, 4),
                    "total_terms": ter_result.total_gt_terms,
                    "substitutions": len(ter_result.substitutions),
                    "deletions": len(ter_result.deletions),
                    "insertions": len(ter_result.insertions),
                    "category_ter": {
                        k: round(v, 4) for k, v in ter_result.category_ter.items()
                    },
                    "errors": [
                        {
                            "type": "substitution",
                            "ground_truth": e.ground_truth_term.text if e.ground_truth_term else None,
                            "predicted": e.predicted_term.text if e.predicted_term else None,
                            "category": e.category.value if e.category else None,
                        }
                        for e in ter_result.substitutions
                    ]
                    + [
                        {
                            "type": "deletion",
                            "ground_truth": e.ground_truth_term.text if e.ground_truth_term else None,
                            "predicted": None,
                            "category": e.category.value if e.category else None,
                        }
                        for e in ter_result.deletions
                    ]
                    + [
                        {
                            "type": "insertion",
                            "ground_truth": None,
                            "predicted": e.predicted_term.text if e.predicted_term else None,
                            "category": e.category.value if e.category else None,
                        }
                        for e in ter_result.insertions
                    ],
                }

            # Compute NER
            if req.compute_ner:
                ner_result = get_ner_engine().compute(req.ground_truth, req.predicted)
                results["ner"] = {
                    "precision": round(ner_result.precision, 4),
                    "recall": round(ner_result.recall, 4),
                    "f1_score": round(ner_result.f1_score, 4),
                    "entity_distortion_rate": round(ner_result.entity_distortion_rate, 4),
                    "entity_omission_rate": round(ner_result.entity_omission_rate, 4),
                }

            # Compute CRS (using sentences as segments)
            if req.compute_crs:
                # Split into sentences for segment comparison
                gt_segments = _split_into_segments(req.ground_truth)
                pred_segments = _split_into_segments(req.predicted)

                # Ensure same number of segments
                if len(gt_segments) != len(pred_segments):
                    # Pad shorter list or use full text as single segment
                    if len(gt_segments) == 0:
                        gt_segments = [req.ground_truth]
                    if len(pred_segments) == 0:
                        pred_segments = [req.predicted]
                    # Match segment counts
                    max_len = max(len(gt_segments), len(pred_segments))
                    while len(gt_segments) < max_len:
                        gt_segments.append("")
                    while len(pred_segments) < max_len:
                        pred_segments.append("")

                crs_result = get_crs_engine().compute(gt_segments, pred_segments)
                results["crs"] = {
                    "composite_score": round(crs_result.composite_score, 4),
                    "semantic_similarity": round(crs_result.semantic_similarity, 4),
                    "entity_continuity": round(crs_result.entity_continuity, 4),
                    "negation_consistency": round(crs_result.negation_consistency, 4),
                    "context_drift_rate": round(crs_result.context_drift_rate, 4),
                }

            # Compute overall score
            scores = []
            if "ter" in results:
                # Invert TER (lower is better)
                scores.append(1 - results["ter"]["overall_ter"])
            if "ner" in results:
                scores.append(results["ner"]["f1_score"])
            if "crs" in results:
                scores.append(results["crs"]["composite_score"])

            if scores:
                results["overall_score"] = round(sum(scores) / len(scores), 4)

            results["status"] = "success"

        except Exception as e:
            results["status"] = "error"
            results["error"] = str(e)

        return JSONResponse(content=results)

    @application.post("/api/evaluate/segments")
    async def evaluate_segments(req: SegmentedEvaluationRequest) -> JSONResponse:
        """
        Run CRS evaluation on pre-segmented text.

        Useful for streaming transcription evaluation where
        segments are already defined.
        """
        try:
            if len(req.ground_truth_segments) != len(req.predicted_segments):
                return JSONResponse(
                    content={
                        "status": "error",
                        "error": "Segment counts must match",
                    },
                    status_code=400,
                )

            crs_result = get_crs_engine().compute(
                req.ground_truth_segments, req.predicted_segments
            )

            return JSONResponse(
                content={
                    "status": "success",
                    "crs": {
                        "composite_score": round(crs_result.composite_score, 4),
                        "semantic_similarity": round(crs_result.semantic_similarity, 4),
                        "entity_continuity": round(crs_result.entity_continuity, 4),
                        "negation_consistency": round(crs_result.negation_consistency, 4),
                        "context_drift_rate": round(crs_result.context_drift_rate, 4),
                        "segment_count": len(req.ground_truth_segments),
                    },
                }
            )

        except Exception as e:
            return JSONResponse(
                content={"status": "error", "error": str(e)},
                status_code=500,
            )

    @application.get("/api/examples")
    async def get_examples() -> JSONResponse:
        """Get example inputs for testing."""
        return JSONResponse(
            content={
                "examples": [
                    {
                        "name": "Drug Substitution Error",
                        "ground_truth": "Patient takes metformin 500mg twice daily for type 2 diabetes.",
                        "predicted": "Patient takes methotrexate 500mg twice daily for type 2 diabetes.",
                        "description": "Critical error: drug name substitution",
                    },
                    {
                        "name": "Negation Flip",
                        "ground_truth": "Patient denies chest pain. No history of cardiac issues.",
                        "predicted": "Patient has chest pain. History of cardiac issues.",
                        "description": "High risk: negation lost changes clinical meaning",
                    },
                    {
                        "name": "Dosage Error",
                        "ground_truth": "Prescribed lisinopril 10mg once daily for hypertension.",
                        "predicted": "Prescribed lisinopril 100mg once daily for hypertension.",
                        "description": "High risk: dosage transcription error",
                    },
                    {
                        "name": "Perfect Match",
                        "ground_truth": "Patient presents with headache and fatigue. Blood pressure 120/80.",
                        "predicted": "Patient presents with headache and fatigue. Blood pressure 120/80.",
                        "description": "Ideal: perfect transcription",
                    },
                    {
                        "name": "Minor Variations",
                        "ground_truth": "History of diabetes mellitus type 2. Currently on metformin.",
                        "predicted": "History of type 2 diabetes mellitus. Currently taking metformin.",
                        "description": "Low risk: word order and synonym variations",
                    },
                ]
            }
        )

    return application


def _split_into_segments(text: str) -> list[str]:
    """Split text into segments (sentences)."""
    import re

    # Split on sentence boundaries
    segments = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in segments if s.strip()]


# Create default app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
