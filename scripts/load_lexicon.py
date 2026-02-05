#!/usr/bin/env python3
"""
Load Medical Terminology into SQLite Database.

Fetches medical terms from public APIs (RxNorm, OpenFDA, ICD-10) and
stores them in a local SQLite database.

Usage:
    python scripts/load_lexicon.py [--refresh] [--counts-only]

Options:
    --refresh      Force refresh from APIs (ignore cache)
    --counts-only  Only show table counts, don't fetch
    --drug-limit   Maximum drugs to fetch (default: 5000)
    --cond-limit   Maximum conditions to fetch (default: 5000)
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sqlite3
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("hsttb.load_lexicon")


def get_table_counts(db_path: Path) -> dict[str, int]:
    """Get row counts for each table in the database."""
    counts = {}

    if not db_path.exists():
        return {"error": "Database does not exist"}

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            counts[table] = cursor.fetchone()[0]

        conn.close()

    except Exception as e:
        counts["error"] = str(e)

    return counts


def get_category_breakdown(db_path: Path) -> dict[str, int]:
    """Get term counts by category."""
    breakdown = {}

    if not db_path.exists():
        return {}

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute("""
            SELECT category, COUNT(*) as cnt
            FROM terms
            GROUP BY category
        """)

        for row in cursor.fetchall():
            breakdown[row[0]] = row[1]

        conn.close()

    except Exception as e:
        logger.warning(f"Error getting category breakdown: {e}")

    return breakdown


def get_source_breakdown(db_path: Path) -> dict[str, int]:
    """Get term counts by source."""
    breakdown = {}

    if not db_path.exists():
        return {}

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute("""
            SELECT source, COUNT(*) as cnt
            FROM terms
            GROUP BY source
        """)

        for row in cursor.fetchall():
            breakdown[row[0]] = row[1]

        conn.close()

    except Exception as e:
        logger.warning(f"Error getting source breakdown: {e}")

    return breakdown


def print_database_stats(db_path: Path) -> None:
    """Print detailed database statistics."""
    print("\n" + "=" * 60)
    print("HSTTB Medical Lexicon Database Statistics")
    print("=" * 60)
    print(f"\nDatabase path: {db_path}")
    print(f"Database exists: {db_path.exists()}")

    if not db_path.exists():
        print("\nNo database found. Run with --refresh to create.")
        return

    # Table counts
    print("\n" + "-" * 40)
    print("TABLE COUNTS")
    print("-" * 40)

    counts = get_table_counts(db_path)
    if "error" in counts:
        print(f"Error: {counts['error']}")
        return

    max_name_len = max(len(name) for name in counts.keys()) if counts else 0

    for table, count in sorted(counts.items()):
        print(f"  {table.ljust(max_name_len)}: {count:,}")

    total = sum(counts.values())
    print(f"\n  {'Total'.ljust(max_name_len)}: {total:,}")

    # Category breakdown
    print("\n" + "-" * 40)
    print("TERMS BY CATEGORY")
    print("-" * 40)

    categories = get_category_breakdown(db_path)
    for category, count in sorted(categories.items()):
        print(f"  {category.ljust(15)}: {count:,}")

    # Source breakdown
    print("\n" + "-" * 40)
    print("TERMS BY SOURCE")
    print("-" * 40)

    sources = get_source_breakdown(db_path)
    for source, count in sorted(sources.items()):
        print(f"  {source.ljust(15)}: {count:,}")

    # Metadata
    print("\n" + "-" * 40)
    print("METADATA")
    print("-" * 40)

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT key, value FROM metadata")

        for row in cursor.fetchall():
            print(f"  {row[0].ljust(20)}: {row[1]}")

        conn.close()
    except Exception as e:
        print(f"  Error reading metadata: {e}")

    print()


async def load_lexicon(
    drug_limit: int = 5000,
    condition_limit: int = 5000,
    force_refresh: bool = False,
) -> bool:
    """
    Load medical terminology into SQLite database.

    Args:
        drug_limit: Maximum drugs to fetch.
        condition_limit: Maximum conditions to fetch.
        force_refresh: Force refresh from APIs.

    Returns:
        True if successful.
    """
    from hsttb.lexicons.sqlite_lexicon import SQLiteMedicalLexicon

    print("\n" + "-" * 40)
    print("LOADING LEXICON")
    print("-" * 40)

    print(f"\nDrug limit: {drug_limit:,}")
    print(f"Condition limit: {condition_limit:,}")
    print(f"Force refresh: {force_refresh}")
    print()

    try:
        lexicon = SQLiteMedicalLexicon(
            drug_limit=drug_limit,
            condition_limit=condition_limit,
        )

        mode = "refresh" if force_refresh else "auto"
        print(f"Loading with mode: {mode}")

        await lexicon.load_async(mode)

        stats = lexicon.get_stats()
        if stats:
            print(f"\nLoaded successfully:")
            print(f"  Total terms: {stats.entry_count:,}")
            print(f"  Load time: {stats.load_time_ms:.0f}ms")
            return True
        else:
            print("Loaded but no stats available")
            return False

    except ImportError as e:
        print(f"\nMissing dependency: {e}")
        print("Install httpx for API fetching: pip install httpx")
        return False

    except Exception as e:
        logger.error(f"Failed to load lexicon: {e}")
        print(f"\nError: {e}")
        return False


async def load_drug_indications(db_path: Path, max_drugs: int = 100) -> int:
    """
    Load drug-indication relationships from API.

    Args:
        db_path: Path to database.
        max_drugs: Maximum drugs to fetch indications for.

    Returns:
        Number of relationships loaded.
    """
    from hsttb.lexicons.api_fetcher import MedicalTermFetcher
    from hsttb.lexicons.sqlite_lexicon import SQLiteMedicalLexicon

    print("\n" + "-" * 40)
    print("LOADING DRUG INDICATIONS")
    print("-" * 40)

    try:
        # Get drugs with RxCUI from database
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute("""
            SELECT term, code FROM terms
            WHERE category = 'drug' AND code IS NOT NULL AND code != ''
            LIMIT ?
        """, (max_drugs,))

        drugs_with_rxcui = [(row[0], row[1]) for row in cursor.fetchall()]
        conn.close()

        print(f"Found {len(drugs_with_rxcui)} drugs with RxCUI codes")

        if not drugs_with_rxcui:
            print("No drugs with RxCUI to look up")
            return 0

        # Fetch indications
        fetcher = MedicalTermFetcher()
        lexicon = SQLiteMedicalLexicon()

        indications_count = 0

        for i, (drug_name, rxcui) in enumerate(drugs_with_rxcui):
            if i % 10 == 0:
                print(f"  Processing {i+1}/{len(drugs_with_rxcui)}...")

            indications = await fetcher.fetch_drug_indications(rxcui)

            for indication in indications:
                if lexicon.add_drug_indication(drug_name, indication, "RxClass"):
                    indications_count += 1

            await asyncio.sleep(0.1)  # Rate limiting

        await fetcher.close()

        print(f"\nLoaded {indications_count} drug-indication relationships")
        return indications_count

    except Exception as e:
        logger.error(f"Failed to load drug indications: {e}")
        print(f"Error: {e}")
        return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Load Medical Terminology into SQLite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force refresh from APIs",
    )
    parser.add_argument(
        "--counts-only",
        action="store_true",
        help="Only show table counts",
    )
    parser.add_argument(
        "--drug-limit",
        type=int,
        default=5000,
        help="Maximum drugs to fetch (default: 5000)",
    )
    parser.add_argument(
        "--cond-limit",
        type=int,
        default=5000,
        help="Maximum conditions to fetch (default: 5000)",
    )
    parser.add_argument(
        "--with-indications",
        action="store_true",
        help="Also fetch drug-indication relationships",
    )
    parser.add_argument(
        "--indication-limit",
        type=int,
        default=100,
        help="Max drugs to fetch indications for (default: 100)",
    )

    args = parser.parse_args()

    # Get database path
    from hsttb.lexicons.sqlite_lexicon import DEFAULT_DB_PATH
    db_path = DEFAULT_DB_PATH

    # Show current stats
    print_database_stats(db_path)

    if args.counts_only:
        return 0

    # Load lexicon
    success = asyncio.run(load_lexicon(
        drug_limit=args.drug_limit,
        condition_limit=args.cond_limit,
        force_refresh=args.refresh,
    ))

    if not success:
        return 1

    # Optionally load drug indications
    if args.with_indications:
        asyncio.run(load_drug_indications(db_path, args.indication_limit))

    # Show updated stats
    print_database_stats(db_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
