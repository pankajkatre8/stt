#!/usr/bin/env python3
"""
HSTTB Startup Script

Initializes the application by:
1. Checking if the medical lexicon database exists
2. Loading/refreshing lexicon from APIs if needed
3. Starting the web application

Usage:
    python scripts/startup.py [--refresh] [--check-only]

Options:
    --refresh     Force refresh lexicon from APIs
    --check-only  Only check lexicon status, don't start server
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import subprocess
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("hsttb.startup")


def get_lexicon_status() -> dict:
    """Check the status of the medical lexicon database."""
    from hsttb.lexicons.sqlite_lexicon import SQLiteMedicalLexicon, DEFAULT_DB_PATH

    status = {
        "db_path": str(DEFAULT_DB_PATH),
        "db_exists": DEFAULT_DB_PATH.exists(),
        "term_count": 0,
        "drug_count": 0,
        "condition_count": 0,
        "needs_refresh": True,
        "last_refresh": None,
    }

    if DEFAULT_DB_PATH.exists():
        try:
            lexicon = SQLiteMedicalLexicon()
            lexicon._init_db()

            status["term_count"] = lexicon._get_term_count()
            status["needs_refresh"] = lexicon._needs_refresh()
            status["last_refresh"] = lexicon._get_metadata("last_refresh")
            status["drug_count"] = int(lexicon._get_metadata("drug_count") or 0)
            status["condition_count"] = int(lexicon._get_metadata("condition_count") or 0)

        except Exception as e:
            logger.warning(f"Error checking lexicon status: {e}")

    return status


async def initialize_lexicon(force_refresh: bool = False) -> bool:
    """
    Initialize the medical lexicon database.

    Args:
        force_refresh: Force refresh from APIs even if data exists.

    Returns:
        True if successful, False otherwise.
    """
    from hsttb.lexicons.sqlite_lexicon import SQLiteMedicalLexicon

    logger.info("Initializing medical lexicon...")

    try:
        lexicon = SQLiteMedicalLexicon(
            drug_limit=5000,
            condition_limit=5000,
        )

        mode = "refresh" if force_refresh else "auto"
        await lexicon.load_async(mode)

        stats = lexicon.get_stats()
        if stats:
            logger.info(
                f"Lexicon ready: {stats.entry_count} terms "
                f"({stats.categories.get('drug', 0)} drugs, "
                f"{stats.categories.get('diagnosis', 0)} conditions)"
            )
            return True
        else:
            logger.warning("Lexicon loaded but no stats available")
            return False

    except Exception as e:
        logger.error(f"Failed to initialize lexicon: {e}")
        return False


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False) -> None:
    """Start the uvicorn server."""
    import uvicorn

    logger.info(f"Starting HSTTB server on {host}:{port}...")

    uvicorn.run(
        "hsttb.webapp.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="HSTTB Startup Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force refresh lexicon from APIs",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check lexicon status, don't start server",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("HOST", "0.0.0.0"),
        help="Server host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", "8000")),
        help="Server port (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )

    args = parser.parse_args()

    # Check lexicon status
    print("\n" + "=" * 60)
    print("HSTTB - Healthcare STT Benchmarking")
    print("=" * 60 + "\n")

    status = get_lexicon_status()

    print("Medical Lexicon Status:")
    print(f"  Database: {status['db_path']}")
    print(f"  Exists: {status['db_exists']}")
    print(f"  Terms: {status['term_count']}")
    print(f"  Drugs: {status['drug_count']}")
    print(f"  Conditions: {status['condition_count']}")
    print(f"  Last Refresh: {status['last_refresh'] or 'Never'}")
    print(f"  Needs Refresh: {status['needs_refresh']}")
    print()

    # Initialize lexicon if needed
    needs_init = (
        args.refresh
        or not status["db_exists"]
        or status["term_count"] < 100
        or status["needs_refresh"]
    )

    if needs_init:
        print("Initializing lexicon from APIs...")
        success = asyncio.run(initialize_lexicon(force_refresh=args.refresh))

        if not success:
            print("WARNING: Lexicon initialization failed. Using fallback data.")

        # Show updated status
        status = get_lexicon_status()
        print(f"\nUpdated: {status['term_count']} terms loaded")
    else:
        print("Lexicon is up to date.")

    if args.check_only:
        print("\nCheck complete. Exiting.")
        return 0

    # Start server
    print("\n" + "-" * 60)
    try:
        run_server(host=args.host, port=args.port, reload=args.reload)
    except KeyboardInterrupt:
        print("\nServer stopped.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
