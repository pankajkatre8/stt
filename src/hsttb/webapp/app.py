"""
FastAPI web application for HSTTB evaluation.

Provides a web interface for running healthcare STT benchmarks
with support for audio uploads, real-time streaming, and multiple
NLP backends for evaluation.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from hsttb.metrics.ter import TEREngine
from hsttb.metrics.ner import NEREngine
from hsttb.metrics.crs import CRSEngine
from hsttb.metrics.multi_backend import MultiBackendEvaluator
from hsttb.lexicons.scispacy_lexicon import SciSpacyLexicon
from hsttb.nlp.scispacy_ner import SciSpacyNERPipeline
from hsttb.nlp.semantic_similarity import TransformerSimilarityEngine

logger = logging.getLogger(__name__)

# Get paths
WEBAPP_DIR = Path(__file__).parent
TEMPLATES_DIR = WEBAPP_DIR / "templates"
STATIC_DIR = WEBAPP_DIR / "static"


# ============================================================================
# Request/Response Models
# ============================================================================


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


class MultiBackendRequest(BaseModel):
    """Request model for multi-backend TER evaluation."""

    ground_truth: str
    predicted: str
    backends: list[str] | None = None  # None = use all available


class MultiNLPRequest(BaseModel):
    """Request model for multi-NLP model evaluation."""

    ground_truth: str
    predicted: str
    models: list[str] | None = None  # None = use all available


class MultiAdapterRequest(BaseModel):
    """Request model for multi-adapter STT evaluation."""

    audio_file_id: str
    adapters: list[str] | None = None  # None = use all available


class TranscribeRequest(BaseModel):
    """Request model for transcription."""

    audio_file_id: str
    adapter: str = "whisper"
    model: str | None = None
    language: str | None = None


# ============================================================================
# Application Factory
# ============================================================================


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

    # Lazy-loaded engines and services
    _lexicon: SciSpacyLexicon | None = None
    _ter_engine: TEREngine | None = None
    _ner_engine: NEREngine | None = None
    _crs_engine: CRSEngine | None = None
    _similarity_engine: TransformerSimilarityEngine | None = None
    _multi_backend: MultiBackendEvaluator | None = None
    _available_backends: dict[str, str] = {}  # name -> status

    # ========================================================================
    # Service Getters
    # ========================================================================

    def get_lexicon() -> SciSpacyLexicon:
        nonlocal _lexicon
        if _lexicon is None:
            _lexicon = SciSpacyLexicon()
            _lexicon.load("en_ner_bc5cdr_md")
        return _lexicon

    def get_similarity_engine() -> TransformerSimilarityEngine:
        nonlocal _similarity_engine
        if _similarity_engine is None:
            _similarity_engine = TransformerSimilarityEngine()
        return _similarity_engine

    def get_multi_backend_evaluator() -> MultiBackendEvaluator:
        nonlocal _multi_backend, _available_backends
        if _multi_backend is None:
            _multi_backend = MultiBackendEvaluator()
            _available_backends = {}

            # Add scispaCy lexicon (production)
            scispacy_lex = SciSpacyLexicon()
            scispacy_lex.load("en_ner_bc5cdr_md")
            _multi_backend.add_backend("scispacy", scispacy_lex)
            _available_backends["scispacy"] = "loaded (en_ner_bc5cdr_md)"

        return _multi_backend

    def get_ter_engine() -> TEREngine:
        nonlocal _ter_engine
        if _ter_engine is None:
            _ter_engine = TEREngine(get_lexicon())
        return _ter_engine

    def get_ner_engine() -> NEREngine:
        nonlocal _ner_engine
        if _ner_engine is None:
            pipeline = SciSpacyNERPipeline()
            _ner_engine = NEREngine(pipeline)
        return _ner_engine

    def get_crs_engine() -> CRSEngine:
        nonlocal _crs_engine
        if _crs_engine is None:
            _crs_engine = CRSEngine(similarity_engine=get_similarity_engine())
        return _crs_engine

    # ========================================================================
    # Core Routes
    # ========================================================================

    @application.get("/", response_class=HTMLResponse)
    async def landing_page(request: Request) -> HTMLResponse:
        """Render the landing page."""
        return templates.TemplateResponse("index.html", {"request": request})

    @application.get("/health")
    async def health_check() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "healthy", "service": "hsttb"}

    # ========================================================================
    # Evaluation Endpoints
    # ========================================================================

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
                gt_segments = _split_into_segments(req.ground_truth)
                pred_segments = _split_into_segments(req.predicted)

                # Ensure same number of segments
                if len(gt_segments) != len(pred_segments):
                    if len(gt_segments) == 0:
                        gt_segments = [req.ground_truth]
                    if len(pred_segments) == 0:
                        pred_segments = [req.predicted]
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
        """Run CRS evaluation on pre-segmented text."""
        try:
            if len(req.ground_truth_segments) != len(req.predicted_segments):
                return JSONResponse(
                    content={"status": "error", "error": "Segment counts must match"},
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

    @application.post("/api/evaluate/multi-backend")
    async def evaluate_multi_backend(req: MultiBackendRequest) -> JSONResponse:
        """Run TER evaluation with multiple NLP backends."""
        try:
            evaluator = get_multi_backend_evaluator()
            result = evaluator.evaluate(req.ground_truth, req.predicted)

            backend_results = {}
            for name, metrics in result.backend_metrics.items():
                backend_results[name] = {
                    "ter": round(metrics.ter, 4),
                    "accuracy": round(metrics.accuracy, 4),
                    "gt_terms": metrics.terms_extracted_gt,
                    "pred_terms": metrics.terms_extracted_pred,
                    "correct": metrics.ter_result.correct_matches,
                    "substitutions": len(metrics.ter_result.substitutions),
                    "deletions": len(metrics.ter_result.deletions),
                    "insertions": len(metrics.ter_result.insertions),
                    "errors": [
                        {
                            "type": "substitution",
                            "gt": e.ground_truth_term.text if e.ground_truth_term else None,
                            "pred": e.predicted_term.text if e.predicted_term else None,
                        }
                        for e in metrics.ter_result.substitutions
                    ] + [
                        {
                            "type": "deletion",
                            "gt": e.ground_truth_term.text if e.ground_truth_term else None,
                            "pred": None,
                        }
                        for e in metrics.ter_result.deletions
                    ] + [
                        {
                            "type": "insertion",
                            "gt": None,
                            "pred": e.predicted_term.text if e.predicted_term else None,
                        }
                        for e in metrics.ter_result.insertions
                    ],
                }

            return JSONResponse(content={
                "status": "success",
                "ground_truth": req.ground_truth,
                "predicted": req.predicted,
                "backends": backend_results,
                "best_backend": result.best_backend,
                "average_ter": round(result.average_ter, 4),
                "consensus_gt_terms": result.consensus_terms_gt,
                "consensus_pred_terms": result.consensus_terms_pred,
            })

        except Exception as e:
            return JSONResponse(
                content={"status": "error", "error": str(e)},
                status_code=500,
            )

    @application.post("/api/evaluate/multi-model")
    async def evaluate_multi_model(req: MultiNLPRequest) -> JSONResponse:
        """Run NER evaluation with multiple NLP models."""
        try:
            from hsttb.metrics.multi_nlp import MultiNLPEvaluator
            from hsttb.nlp.registry import get_nlp_pipeline, list_nlp_pipelines

            evaluator = MultiNLPEvaluator()

            # Add requested models
            model_names = req.models or list_nlp_pipelines()
            for name in model_names:
                try:
                    pipeline = get_nlp_pipeline(name)
                    evaluator.add_model(name, pipeline)
                except Exception as e:
                    logger.warning(f"Could not load model {name}: {e}")

            result = evaluator.evaluate(req.ground_truth, req.predicted)

            return JSONResponse(content={
                "status": "success",
                **result.to_dict(),
            })

        except Exception as e:
            logger.error(f"Multi-model evaluation error: {e}")
            return JSONResponse(
                content={"status": "error", "error": str(e)},
                status_code=500,
            )

    @application.post("/api/evaluate/multi-adapter")
    async def evaluate_multi_adapter(req: MultiAdapterRequest) -> JSONResponse:
        """Run STT evaluation with multiple adapters."""
        try:
            from hsttb.adapters import get_adapter, list_adapters
            from hsttb.webapp.audio_handler import get_audio_handler

            handler = get_audio_handler()
            audio_path = handler.get_file(req.audio_file_id)

            if not audio_path:
                return JSONResponse(
                    content={"status": "error", "error": "Audio file not found"},
                    status_code=404,
                )

            adapter_names = req.adapters or list_adapters()
            results = {}

            for name in adapter_names:
                try:
                    adapter = get_adapter(name)
                    await adapter.initialize()
                    transcript = await adapter.transcribe_file(audio_path)
                    await adapter.cleanup()

                    results[name] = {
                        "status": "success",
                        "transcript": transcript,
                    }
                except Exception as e:
                    results[name] = {
                        "status": "error",
                        "error": str(e),
                    }

            return JSONResponse(content={
                "status": "success",
                "audio_file_id": req.audio_file_id,
                "adapters": results,
            })

        except Exception as e:
            return JSONResponse(
                content={"status": "error", "error": str(e)},
                status_code=500,
            )

    # ========================================================================
    # Audio Endpoints
    # ========================================================================

    @application.post("/api/audio/upload")
    async def upload_audio(file: UploadFile = File(...)) -> JSONResponse:
        """Upload an audio file for transcription."""
        try:
            from hsttb.webapp.audio_handler import get_audio_handler, AudioValidationError

            handler = get_audio_handler()
            metadata = await handler.save_upload(file)

            return JSONResponse(content={
                "status": "success",
                **metadata.to_dict(),
            })

        except AudioValidationError as e:
            return JSONResponse(
                content={"status": "error", "error": str(e)},
                status_code=400,
            )
        except Exception as e:
            logger.error(f"Audio upload error: {e}")
            return JSONResponse(
                content={"status": "error", "error": str(e)},
                status_code=500,
            )

    @application.post("/api/audio/transcribe")
    async def transcribe_audio(req: TranscribeRequest) -> JSONResponse:
        """Transcribe an uploaded audio file."""
        try:
            from hsttb.adapters import get_adapter
            from hsttb.webapp.audio_handler import get_audio_handler

            handler = get_audio_handler()
            audio_path = handler.get_file(req.audio_file_id)

            if not audio_path:
                return JSONResponse(
                    content={"status": "error", "error": "Audio file not found"},
                    status_code=404,
                )

            # Build adapter kwargs
            kwargs: dict[str, Any] = {}
            if req.model:
                kwargs["model_size"] = req.model
            if req.language:
                kwargs["language"] = req.language

            adapter = get_adapter(req.adapter, **kwargs)
            await adapter.initialize()
            transcript = await adapter.transcribe_file(audio_path)
            await adapter.cleanup()

            return JSONResponse(content={
                "status": "success",
                "audio_file_id": req.audio_file_id,
                "adapter": req.adapter,
                "transcript": transcript,
            })

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return JSONResponse(
                content={"status": "error", "error": str(e)},
                status_code=500,
            )

    # ========================================================================
    # WebSocket Endpoints
    # ========================================================================

    @application.websocket("/ws/transcribe")
    async def websocket_transcribe(websocket: WebSocket) -> None:
        """Real-time audio streaming transcription."""
        from hsttb.webapp.websocket_handler import WebSocketHandler

        handler = WebSocketHandler(websocket)
        try:
            await handler.run()
        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")

    # ========================================================================
    # Info Endpoints
    # ========================================================================

    @application.get("/api/adapters")
    async def list_stt_adapters() -> JSONResponse:
        """List available STT adapters."""
        from hsttb.adapters import list_adapters

        adapters = list_adapters()

        # Add info about each adapter
        adapter_info = []
        for name in adapters:
            info = {"name": name, "available": True}

            # Add details based on adapter type
            if name == "whisper":
                info["description"] = "Local Whisper model for offline transcription"
                info["models"] = ["tiny", "base", "small", "medium", "large"]
                info["requires_api_key"] = False
            elif name in ("gemini", "google-cloud-speech"):
                info["description"] = "Google Cloud Speech-to-Text API"
                info["requires_api_key"] = True
            elif name == "deepgram":
                info["description"] = "Deepgram API with medical vocabulary"
                info["models"] = ["nova-2", "nova-2-medical", "nova-2-phonecall"]
                info["requires_api_key"] = True
            elif name == "mock":
                info["description"] = "Mock adapter for testing"
                info["requires_api_key"] = False

            adapter_info.append(info)

        return JSONResponse(content={
            "status": "success",
            "adapters": adapter_info,
        })

    @application.get("/api/nlp-models")
    async def list_nlp_models() -> JSONResponse:
        """List available NLP models for entity extraction."""
        from hsttb.nlp.registry import list_nlp_pipelines, get_pipeline_info

        models = list_nlp_pipelines()

        model_info = []
        for name in models:
            try:
                info = get_pipeline_info(name)
                info["available"] = True
                model_info.append(info)
            except Exception:
                model_info.append({
                    "name": name,
                    "available": False,
                })

        return JSONResponse(content={
            "status": "success",
            "models": model_info,
        })

    @application.get("/api/backends")
    async def list_backends() -> JSONResponse:
        """List available NLP backends and their status."""
        get_multi_backend_evaluator()  # Initialize to populate _available_backends
        return JSONResponse(content={
            "status": "success",
            "backends": _available_backends,
            "active": get_multi_backend_evaluator().list_backends(),
        })

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

    # ========================================================================
    # Export Endpoints
    # ========================================================================

    @application.post("/api/export/pdf")
    async def export_pdf(req: EvaluationRequest) -> JSONResponse:
        """Generate PDF report for evaluation results."""
        # TODO: Implement PDF export using reportlab or weasyprint
        return JSONResponse(
            content={
                "status": "error",
                "error": "PDF export not yet implemented",
            },
            status_code=501,
        )

    return application


def _split_into_segments(text: str) -> list[str]:
    """Split text into segments (sentences)."""
    import re

    segments = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in segments if s.strip()]


# Create default app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
