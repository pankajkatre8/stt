"""
Medical terminology API fetcher.

Fetches medical terms from public APIs:
- RxNorm (NIH) - Drug names and codes
- OpenFDA - Drug information
- ICD-10-CM - Diagnosis codes
- SNOMED CT (via NLM) - Clinical terms

Example:
    >>> from hsttb.lexicons.api_fetcher import MedicalTermFetcher
    >>> fetcher = MedicalTermFetcher()
    >>> drugs = await fetcher.fetch_drugs(limit=1000)
    >>> print(f"Fetched {len(drugs)} drugs")
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Cache directory
CACHE_DIR = Path.home() / ".hsttb" / "lexicon_cache"


@dataclass
class CachedData:
    """Cached API data with metadata."""

    data: list[dict[str, Any]]
    source: str
    fetched_at: datetime
    expires_at: datetime
    version: str = "1.0"

    def is_expired(self) -> bool:
        """Check if cache has expired."""
        return datetime.now() > self.expires_at

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "data": self.data,
            "source": self.source,
            "fetched_at": self.fetched_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CachedData:
        """Create from dictionary."""
        return cls(
            data=d["data"],
            source=d["source"],
            fetched_at=datetime.fromisoformat(d["fetched_at"]),
            expires_at=datetime.fromisoformat(d["expires_at"]),
            version=d.get("version", "1.0"),
        )


@dataclass
class DrugEntry:
    """Drug information from API."""

    name: str
    rxcui: str | None = None  # RxNorm Concept Unique Identifier
    ndc: str | None = None  # National Drug Code
    brand_names: list[str] = field(default_factory=list)
    generic_name: str | None = None
    drug_class: str | None = None
    route: str | None = None


@dataclass
class DiagnosisEntry:
    """Diagnosis/condition information from API."""

    name: str
    code: str  # ICD-10 or SNOMED code
    code_system: str  # "ICD10" or "SNOMED"
    synonyms: list[str] = field(default_factory=list)
    category: str | None = None


class MedicalTermFetcher:
    """
    Fetch medical terminology from public APIs.

    Implements caching to avoid repeated API calls.
    Cache expires after 30 days by default.

    Example:
        >>> fetcher = MedicalTermFetcher()
        >>> drugs = await fetcher.fetch_drugs()
        >>> conditions = await fetcher.fetch_conditions()
    """

    # API endpoints
    RXNORM_BASE = "https://rxnav.nlm.nih.gov/REST"
    OPENFDA_BASE = "https://api.fda.gov/drug"
    ICD10_BASE = "https://clinicaltables.nlm.nih.gov/api/icd10cm/v3"

    # Cache settings
    DEFAULT_CACHE_DAYS = 30

    def __init__(
        self,
        cache_dir: Path | None = None,
        cache_days: int = DEFAULT_CACHE_DAYS,
    ) -> None:
        """
        Initialize the fetcher.

        Args:
            cache_dir: Directory for caching API responses.
            cache_days: Number of days before cache expires.
        """
        self._cache_dir = cache_dir or CACHE_DIR
        self._cache_days = cache_days
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._http_client = None

    async def _get_client(self):
        """Get or create HTTP client."""
        if self._http_client is None:
            try:
                import httpx
                self._http_client = httpx.AsyncClient(timeout=30.0)
            except ImportError:
                raise ImportError(
                    "httpx is required for API fetching. "
                    "Install with: pip install httpx"
                )
        return self._http_client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for a key."""
        hash_key = hashlib.md5(key.encode()).hexdigest()[:12]
        return self._cache_dir / f"{key}_{hash_key}.json"

    def _load_cache(self, key: str) -> CachedData | None:
        """Load data from cache if valid."""
        cache_path = self._get_cache_path(key)
        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "r") as f:
                data = json.load(f)
            cached = CachedData.from_dict(data)

            if cached.is_expired():
                logger.info(f"Cache expired for {key}")
                return None

            logger.info(f"Loaded {len(cached.data)} items from cache: {key}")
            return cached

        except Exception as e:
            logger.warning(f"Failed to load cache {key}: {e}")
            return None

    def _save_cache(self, key: str, data: list[dict[str, Any]], source: str) -> None:
        """Save data to cache."""
        cache_path = self._get_cache_path(key)
        cached = CachedData(
            data=data,
            source=source,
            fetched_at=datetime.now(),
            expires_at=datetime.now() + timedelta(days=self._cache_days),
        )

        try:
            with open(cache_path, "w") as f:
                json.dump(cached.to_dict(), f)
            logger.info(f"Saved {len(data)} items to cache: {key}")
        except Exception as e:
            logger.warning(f"Failed to save cache {key}: {e}")

    # =========================================================================
    # RxNorm API - Drug Names
    # =========================================================================

    async def fetch_drugs_rxnorm(
        self,
        limit: int = 5000,
        use_cache: bool = True,
    ) -> list[DrugEntry]:
        """
        Fetch drug names from RxNorm API.

        RxNorm is a free API from NIH/NLM providing standardized
        drug names and codes.

        Args:
            limit: Maximum number of drugs to fetch.
            use_cache: Whether to use cached data.

        Returns:
            List of DrugEntry objects.
        """
        cache_key = f"rxnorm_drugs_{limit}"

        if use_cache:
            cached = self._load_cache(cache_key)
            if cached:
                return [DrugEntry(**d) for d in cached.data]

        logger.info("Fetching drugs from RxNorm API...")
        client = await self._get_client()
        drugs: list[DrugEntry] = []

        try:
            # Get all drug classes first
            drug_classes = await self._fetch_rxnorm_drug_classes(client)

            # Fetch drugs for each class
            for drug_class in drug_classes[:50]:  # Limit classes
                class_drugs = await self._fetch_rxnorm_class_members(
                    client, drug_class
                )
                drugs.extend(class_drugs)

                if len(drugs) >= limit:
                    break

                # Rate limiting
                await asyncio.sleep(0.1)

            # Also fetch common drugs by name patterns
            common_prefixes = [
                "met", "lis", "ator", "omep", "amlo", "gaba",
                "pant", "sertr", "losart", "hydro", "pred",
            ]
            for prefix in common_prefixes:
                prefix_drugs = await self._fetch_rxnorm_by_prefix(client, prefix)
                for drug in prefix_drugs:
                    if not any(d.name.lower() == drug.name.lower() for d in drugs):
                        drugs.append(drug)

                if len(drugs) >= limit:
                    break

            drugs = drugs[:limit]

            # Cache results
            self._save_cache(
                cache_key,
                [self._drug_to_dict(d) for d in drugs],
                "RxNorm",
            )

            logger.info(f"Fetched {len(drugs)} drugs from RxNorm")
            return drugs

        except Exception as e:
            logger.error(f"Failed to fetch from RxNorm: {e}")
            return []

    async def _fetch_rxnorm_drug_classes(self, client) -> list[str]:
        """Fetch drug class names from RxNorm."""
        try:
            url = f"{self.RXNORM_BASE}/rxclass/allClasses.json"
            response = await client.get(url, params={"classTypes": "MESHPA"})
            response.raise_for_status()

            data = response.json()
            classes = []
            if "rxclassMinConceptList" in data:
                for item in data["rxclassMinConceptList"].get("rxclassMinConcept", []):
                    classes.append(item.get("className", ""))

            return classes[:100]  # Limit to 100 classes

        except Exception as e:
            logger.warning(f"Failed to fetch drug classes: {e}")
            return []

    async def _fetch_rxnorm_class_members(
        self, client, drug_class: str
    ) -> list[DrugEntry]:
        """Fetch drugs belonging to a class."""
        try:
            url = f"{self.RXNORM_BASE}/rxclass/classMembers.json"
            params = {
                "classId": drug_class,
                "relaSource": "MESHPA",
            }
            response = await client.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            drugs = []

            if "drugMemberGroup" in data:
                for member in data["drugMemberGroup"].get("drugMember", []):
                    concept = member.get("minConcept", {})
                    drugs.append(DrugEntry(
                        name=concept.get("name", ""),
                        rxcui=concept.get("rxcui"),
                        drug_class=drug_class,
                    ))

            return drugs[:50]  # Limit per class

        except Exception as e:
            logger.warning(f"Failed to fetch class members for {drug_class}: {e}")
            return []

    async def _fetch_rxnorm_by_prefix(
        self, client, prefix: str
    ) -> list[DrugEntry]:
        """Fetch drugs by name prefix."""
        try:
            url = f"{self.RXNORM_BASE}/approximateTerm.json"
            params = {"term": prefix, "maxEntries": 50}
            response = await client.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            drugs = []

            if "approximateGroup" in data:
                for candidate in data["approximateGroup"].get("candidate", []):
                    drugs.append(DrugEntry(
                        name=candidate.get("name", ""),
                        rxcui=candidate.get("rxcui"),
                    ))

            return drugs

        except Exception as e:
            logger.warning(f"Failed to fetch drugs with prefix {prefix}: {e}")
            return []

    async def fetch_drug_details(self, rxcui: str) -> DrugEntry | None:
        """Fetch detailed info for a specific drug by RxCUI."""
        client = await self._get_client()

        try:
            # Get drug properties
            url = f"{self.RXNORM_BASE}/rxcui/{rxcui}/properties.json"
            response = await client.get(url)
            response.raise_for_status()

            data = response.json()
            props = data.get("properties", {})

            drug = DrugEntry(
                name=props.get("name", ""),
                rxcui=rxcui,
            )

            # Get brand names
            url = f"{self.RXNORM_BASE}/rxcui/{rxcui}/related.json"
            params = {"tty": "BN"}  # Brand Names
            response = await client.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            if "relatedGroup" in data:
                for group in data["relatedGroup"].get("conceptGroup", []):
                    for concept in group.get("conceptProperties", []):
                        drug.brand_names.append(concept.get("name", ""))

            return drug

        except Exception as e:
            logger.warning(f"Failed to fetch drug details for {rxcui}: {e}")
            return None

    # =========================================================================
    # OpenFDA API - Drug Information
    # =========================================================================

    async def fetch_drugs_openfda(
        self,
        limit: int = 1000,
        use_cache: bool = True,
    ) -> list[DrugEntry]:
        """
        Fetch drug information from OpenFDA API.

        OpenFDA provides free access to FDA drug data including
        brand names, generic names, and NDC codes.

        Args:
            limit: Maximum number of drugs to fetch.
            use_cache: Whether to use cached data.

        Returns:
            List of DrugEntry objects.
        """
        cache_key = f"openfda_drugs_{limit}"

        if use_cache:
            cached = self._load_cache(cache_key)
            if cached:
                return [DrugEntry(**d) for d in cached.data]

        logger.info("Fetching drugs from OpenFDA API...")
        client = await self._get_client()
        drugs: list[DrugEntry] = []

        try:
            # Fetch drugs in batches
            batch_size = 100
            skip = 0

            while len(drugs) < limit:
                url = f"{self.OPENFDA_BASE}/ndc.json"
                params = {
                    "limit": min(batch_size, limit - len(drugs)),
                    "skip": skip,
                }
                response = await client.get(url, params=params)
                response.raise_for_status()

                data = response.json()
                results = data.get("results", [])

                if not results:
                    break

                for item in results:
                    generic_name = item.get("generic_name", "")
                    brand_name = item.get("brand_name", "")

                    if generic_name:
                        drugs.append(DrugEntry(
                            name=generic_name,
                            ndc=item.get("product_ndc"),
                            brand_names=[brand_name] if brand_name else [],
                            generic_name=generic_name,
                            route=item.get("route", [None])[0] if item.get("route") else None,
                        ))

                skip += batch_size
                await asyncio.sleep(0.2)  # Rate limiting

            # Deduplicate by name
            seen = set()
            unique_drugs = []
            for drug in drugs:
                name_lower = drug.name.lower()
                if name_lower not in seen:
                    seen.add(name_lower)
                    unique_drugs.append(drug)

            drugs = unique_drugs[:limit]

            # Cache results
            self._save_cache(
                cache_key,
                [self._drug_to_dict(d) for d in drugs],
                "OpenFDA",
            )

            logger.info(f"Fetched {len(drugs)} drugs from OpenFDA")
            return drugs

        except Exception as e:
            logger.error(f"Failed to fetch from OpenFDA: {e}")
            return []

    # =========================================================================
    # ICD-10-CM API - Diagnosis Codes
    # =========================================================================

    async def fetch_conditions_icd10(
        self,
        limit: int = 5000,
        use_cache: bool = True,
    ) -> list[DiagnosisEntry]:
        """
        Fetch diagnosis codes from ICD-10-CM API.

        Uses the NLM Clinical Tables API which provides free
        access to ICD-10-CM codes and descriptions.

        Args:
            limit: Maximum number of conditions to fetch.
            use_cache: Whether to use cached data.

        Returns:
            List of DiagnosisEntry objects.
        """
        cache_key = f"icd10_conditions_{limit}"

        if use_cache:
            cached = self._load_cache(cache_key)
            if cached:
                return [DiagnosisEntry(**d) for d in cached.data]

        logger.info("Fetching conditions from ICD-10-CM API...")
        client = await self._get_client()
        conditions: list[DiagnosisEntry] = []

        # Common search terms to get diverse conditions
        search_terms = [
            "diabetes", "hypertension", "heart", "lung", "kidney",
            "cancer", "infection", "pain", "disorder", "disease",
            "failure", "syndrome", "arthritis", "depression", "anxiety",
            "asthma", "copd", "stroke", "pneumonia", "anemia",
        ]

        try:
            for term in search_terms:
                url = self.ICD10_BASE + "/search"
                params = {
                    "terms": term,
                    "maxList": min(200, limit - len(conditions)),
                }
                response = await client.get(url, params=params)
                response.raise_for_status()

                data = response.json()
                # Response format: [total_count, [codes], {extra_data}, [[code, name], ...]]
                if len(data) >= 4:
                    for item in data[3]:
                        if len(item) >= 2:
                            code, name = item[0], item[1]
                            conditions.append(DiagnosisEntry(
                                name=name,
                                code=code,
                                code_system="ICD10",
                            ))

                if len(conditions) >= limit:
                    break

                await asyncio.sleep(0.1)  # Rate limiting

            # Deduplicate by code
            seen = set()
            unique_conditions = []
            for cond in conditions:
                if cond.code not in seen:
                    seen.add(cond.code)
                    unique_conditions.append(cond)

            conditions = unique_conditions[:limit]

            # Cache results
            self._save_cache(
                cache_key,
                [self._diagnosis_to_dict(d) for d in conditions],
                "ICD-10-CM",
            )

            logger.info(f"Fetched {len(conditions)} conditions from ICD-10-CM")
            return conditions

        except Exception as e:
            logger.error(f"Failed to fetch from ICD-10-CM: {e}")
            return []

    # =========================================================================
    # Combined Fetching
    # =========================================================================

    async def fetch_all(
        self,
        drug_limit: int = 2000,
        condition_limit: int = 2000,
        use_cache: bool = True,
    ) -> tuple[list[DrugEntry], list[DiagnosisEntry]]:
        """
        Fetch drugs and conditions from all sources.

        Args:
            drug_limit: Maximum drugs to fetch.
            condition_limit: Maximum conditions to fetch.
            use_cache: Whether to use cached data.

        Returns:
            Tuple of (drugs, conditions).
        """
        # Fetch in parallel
        drugs_task = self.fetch_drugs_rxnorm(drug_limit // 2, use_cache)
        drugs_fda_task = self.fetch_drugs_openfda(drug_limit // 2, use_cache)
        conditions_task = self.fetch_conditions_icd10(condition_limit, use_cache)

        drugs_rxnorm, drugs_fda, conditions = await asyncio.gather(
            drugs_task, drugs_fda_task, conditions_task
        )

        # Merge and deduplicate drugs
        all_drugs = drugs_rxnorm + drugs_fda
        seen = set()
        unique_drugs = []
        for drug in all_drugs:
            name_lower = drug.name.lower()
            if name_lower not in seen:
                seen.add(name_lower)
                unique_drugs.append(drug)

        return unique_drugs[:drug_limit], conditions[:condition_limit]

    # =========================================================================
    # Drug-Condition Relationships
    # =========================================================================

    async def fetch_drug_indications(
        self,
        rxcui: str,
    ) -> list[str]:
        """
        Fetch indications (conditions) for a drug.

        Args:
            rxcui: RxNorm Concept ID.

        Returns:
            List of indication names.
        """
        client = await self._get_client()

        try:
            # Use RxClass to get indications
            url = f"{self.RXNORM_BASE}/rxclass/class/byRxcui.json"
            params = {
                "rxcui": rxcui,
                "relaSource": "MEDRT",
                "relas": "may_treat",
            }
            response = await client.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            indications = []

            if "rxclassDrugInfoList" in data:
                for info in data["rxclassDrugInfoList"].get("rxclassDrugInfo", []):
                    class_info = info.get("rxclassMinConceptItem", {})
                    indication = class_info.get("className", "")
                    if indication:
                        indications.append(indication)

            return indications

        except Exception as e:
            logger.warning(f"Failed to fetch indications for {rxcui}: {e}")
            return []

    async def build_drug_indication_map(
        self,
        drugs: list[DrugEntry],
        max_drugs: int = 100,
    ) -> dict[str, list[str]]:
        """
        Build a map of drug names to their valid indications.

        Args:
            drugs: List of drugs to look up.
            max_drugs: Maximum drugs to process (rate limiting).

        Returns:
            Dict mapping drug name to list of indication names.
        """
        drug_indications: dict[str, list[str]] = {}

        for drug in drugs[:max_drugs]:
            if drug.rxcui:
                indications = await self.fetch_drug_indications(drug.rxcui)
                if indications:
                    drug_indications[drug.name.lower()] = indications
                await asyncio.sleep(0.1)  # Rate limiting

        return drug_indications

    # =========================================================================
    # Helpers
    # =========================================================================

    def _drug_to_dict(self, drug: DrugEntry) -> dict[str, Any]:
        """Convert DrugEntry to dictionary."""
        return {
            "name": drug.name,
            "rxcui": drug.rxcui,
            "ndc": drug.ndc,
            "brand_names": drug.brand_names,
            "generic_name": drug.generic_name,
            "drug_class": drug.drug_class,
            "route": drug.route,
        }

    def _diagnosis_to_dict(self, diag: DiagnosisEntry) -> dict[str, Any]:
        """Convert DiagnosisEntry to dictionary."""
        return {
            "name": diag.name,
            "code": diag.code,
            "code_system": diag.code_system,
            "synonyms": diag.synonyms,
            "category": diag.category,
        }


# ============================================================================
# Synchronous Wrapper
# ============================================================================


class MedicalTermFetcherSync:
    """
    Synchronous wrapper for MedicalTermFetcher.

    For use in non-async code.

    Example:
        >>> fetcher = MedicalTermFetcherSync()
        >>> drugs = fetcher.fetch_drugs()
    """

    def __init__(self, cache_dir: Path | None = None):
        """Initialize synchronous fetcher."""
        self._async_fetcher = MedicalTermFetcher(cache_dir)

    def fetch_drugs(self, limit: int = 2000) -> list[DrugEntry]:
        """Fetch drugs synchronously."""
        return asyncio.run(self._async_fetcher.fetch_drugs_rxnorm(limit))

    def fetch_conditions(self, limit: int = 2000) -> list[DiagnosisEntry]:
        """Fetch conditions synchronously."""
        return asyncio.run(self._async_fetcher.fetch_conditions_icd10(limit))

    def fetch_all(
        self,
        drug_limit: int = 2000,
        condition_limit: int = 2000,
    ) -> tuple[list[DrugEntry], list[DiagnosisEntry]]:
        """Fetch all terms synchronously."""
        return asyncio.run(
            self._async_fetcher.fetch_all(drug_limit, condition_limit)
        )


# ============================================================================
# Convenience Functions
# ============================================================================


async def fetch_medical_terms(
    drug_limit: int = 2000,
    condition_limit: int = 2000,
    use_cache: bool = True,
) -> tuple[list[DrugEntry], list[DiagnosisEntry]]:
    """
    Convenience function to fetch medical terms.

    Args:
        drug_limit: Maximum drugs to fetch.
        condition_limit: Maximum conditions to fetch.
        use_cache: Whether to use cached data.

    Returns:
        Tuple of (drugs, conditions).
    """
    fetcher = MedicalTermFetcher()
    try:
        return await fetcher.fetch_all(drug_limit, condition_limit, use_cache)
    finally:
        await fetcher.close()


def fetch_medical_terms_sync(
    drug_limit: int = 2000,
    condition_limit: int = 2000,
) -> tuple[list[DrugEntry], list[DiagnosisEntry]]:
    """
    Synchronous convenience function to fetch medical terms.

    Args:
        drug_limit: Maximum drugs to fetch.
        condition_limit: Maximum conditions to fetch.

    Returns:
        Tuple of (drugs, conditions).
    """
    fetcher = MedicalTermFetcherSync()
    return fetcher.fetch_all(drug_limit, condition_limit)
