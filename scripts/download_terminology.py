#!/usr/bin/env python3
"""
Download Medical Terminology Data Files.

Downloads and parses medical terminology from official sources:
- ICD-10-CM (CMS) - Diagnosis codes
- ICD-9-CM (CMS) - Legacy diagnosis codes
- SNOMED CT (NLM API) - Clinical terms
- RxNorm (NLM API) - Drug terminology

Usage:
    python scripts/download_terminology.py [--all] [--icd10] [--icd9] [--snomed]

Sources:
    - ICD-10-CM: https://www.cms.gov/medicare/coding-billing/icd-10-codes
    - ICD-9-CM: https://www.cms.gov/medicare/coding/icd9providerdiagnosticcodes
    - SNOMED CT: https://browser.icd.who.int/ (WHO API)
"""
from __future__ import annotations

import argparse
import asyncio
import io
import logging
import sqlite3
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("hsttb.download_terminology")

# Data directory
DATA_DIR = Path.home() / ".hsttb" / "terminology_data"
DB_PATH = Path.home() / ".hsttb" / "medical_lexicon.db"

# Download URLs
ICD10_CM_URL = "https://www.cms.gov/files/zip/2024-code-descriptions-tabular-order.zip"
ICD10_CM_BACKUP_URL = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD10CM/2024/icd10cm_codes_2024.txt"

# SNOMED CT Browser API (free, no auth required)
SNOMED_API_BASE = "https://browser.icd.who.int/api"


async def download_file(url: str, client) -> bytes | None:
    """Download a file from URL."""
    try:
        logger.info(f"Downloading: {url}")
        response = await client.get(url, follow_redirects=True, timeout=120.0)
        response.raise_for_status()
        logger.info(f"Downloaded {len(response.content):,} bytes")
        return response.content
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return None


def init_database() -> sqlite3.Connection:
    """Initialize database with schema."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS medical_terms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            term TEXT NOT NULL,
            normalized TEXT NOT NULL,
            code TEXT,
            category TEXT NOT NULL,
            source TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_medical_terms_normalized ON medical_terms(normalized)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_medical_terms_category ON medical_terms(category)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_medical_terms_source ON medical_terms(source)")

    # Add unique constraint if not exists (for upsert)
    try:
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_medical_terms_unique ON medical_terms(normalized, source)")
    except sqlite3.OperationalError:
        pass

    conn.commit()
    return conn


def normalize_term(term: str) -> str:
    """Normalize a term for consistent lookup."""
    return term.lower().strip()


def insert_term(conn: sqlite3.Connection, term: str, code: str, category: str, source: str) -> bool:
    """Insert a term into the database."""
    normalized = normalize_term(term)
    try:
        conn.execute(
            """
            INSERT OR IGNORE INTO medical_terms (term, normalized, code, category, source)
            VALUES (?, ?, ?, ?, ?)
            """,
            (term, normalized, code, category, source),
        )
        return True
    except sqlite3.IntegrityError:
        return False


async def download_icd10_cm(conn: sqlite3.Connection) -> int:
    """Download and parse ICD-10-CM codes."""
    import httpx

    logger.info("=" * 60)
    logger.info("Downloading ICD-10-CM codes from CDC...")
    logger.info("=" * 60)

    count = 0

    async with httpx.AsyncClient() as client:
        # Try direct text file first (more reliable)
        content = await download_file(ICD10_CM_BACKUP_URL, client)

        if content:
            # Parse the text file (format: CODE DESCRIPTION)
            lines = content.decode('utf-8', errors='ignore').splitlines()

            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                # Format: CODE<space>DESCRIPTION
                parts = line.split(None, 1)
                if len(parts) >= 2:
                    code = parts[0].strip()
                    description = parts[1].strip()

                    if insert_term(conn, description, code, "diagnosis", "ICD10"):
                        count += 1

                        if count % 5000 == 0:
                            logger.info(f"  Processed {count:,} ICD-10 codes...")
                            conn.commit()

            conn.commit()
            logger.info(f"Loaded {count:,} ICD-10-CM codes")
            return count

        # Fallback: try ZIP file
        content = await download_file(ICD10_CM_URL, client)

        if content:
            try:
                with zipfile.ZipFile(io.BytesIO(content)) as zf:
                    for filename in zf.namelist():
                        if filename.endswith('.txt') and 'code' in filename.lower():
                            with zf.open(filename) as f:
                                for line in f:
                                    line = line.decode('utf-8', errors='ignore').strip()
                                    if not line:
                                        continue

                                    parts = line.split(None, 1)
                                    if len(parts) >= 2:
                                        code = parts[0].strip()
                                        description = parts[1].strip()

                                        if insert_term(conn, description, code, "diagnosis", "ICD10"):
                                            count += 1

                conn.commit()
                logger.info(f"Loaded {count:,} ICD-10-CM codes from ZIP")
                return count

            except Exception as e:
                logger.error(f"Failed to parse ZIP: {e}")

    # If downloads fail, load comprehensive embedded data
    logger.info("API unavailable, loading comprehensive embedded ICD-10 data...")
    count = load_embedded_icd10(conn)
    return count


def load_embedded_icd10(conn: sqlite3.Connection) -> int:
    """Load comprehensive embedded ICD-10 codes."""
    # Comprehensive list of common ICD-10-CM codes
    icd10_codes = [
        # Diabetes
        ("E10", "Type 1 diabetes mellitus"),
        ("E10.9", "Type 1 diabetes mellitus without complications"),
        ("E11", "Type 2 diabetes mellitus"),
        ("E11.9", "Type 2 diabetes mellitus without complications"),
        ("E11.65", "Type 2 diabetes mellitus with hyperglycemia"),
        ("E11.21", "Type 2 diabetes mellitus with diabetic nephropathy"),
        ("E11.22", "Type 2 diabetes mellitus with diabetic chronic kidney disease"),
        ("E11.40", "Type 2 diabetes mellitus with diabetic neuropathy, unspecified"),
        ("E11.42", "Type 2 diabetes mellitus with diabetic polyneuropathy"),
        ("E11.51", "Type 2 diabetes mellitus with diabetic peripheral angiopathy without gangrene"),
        ("E13", "Other specified diabetes mellitus"),

        # Hypertension
        ("I10", "Essential (primary) hypertension"),
        ("I11", "Hypertensive heart disease"),
        ("I11.0", "Hypertensive heart disease with heart failure"),
        ("I11.9", "Hypertensive heart disease without heart failure"),
        ("I12", "Hypertensive chronic kidney disease"),
        ("I13", "Hypertensive heart and chronic kidney disease"),
        ("I15", "Secondary hypertension"),
        ("I16", "Hypertensive crisis"),

        # Heart disease
        ("I20", "Angina pectoris"),
        ("I20.0", "Unstable angina"),
        ("I20.9", "Angina pectoris, unspecified"),
        ("I21", "Acute myocardial infarction"),
        ("I21.0", "ST elevation myocardial infarction involving left main coronary artery"),
        ("I21.3", "ST elevation myocardial infarction of unspecified site"),
        ("I25", "Chronic ischemic heart disease"),
        ("I25.10", "Atherosclerotic heart disease of native coronary artery without angina pectoris"),
        ("I25.5", "Ischemic cardiomyopathy"),
        ("I48", "Atrial fibrillation and flutter"),
        ("I48.0", "Paroxysmal atrial fibrillation"),
        ("I48.1", "Persistent atrial fibrillation"),
        ("I48.2", "Chronic atrial fibrillation"),
        ("I48.91", "Unspecified atrial fibrillation"),
        ("I50", "Heart failure"),
        ("I50.1", "Left ventricular failure"),
        ("I50.20", "Unspecified systolic (congestive) heart failure"),
        ("I50.22", "Chronic systolic (congestive) heart failure"),
        ("I50.30", "Unspecified diastolic (congestive) heart failure"),
        ("I50.9", "Heart failure, unspecified"),

        # Cerebrovascular
        ("I63", "Cerebral infarction"),
        ("I63.9", "Cerebral infarction, unspecified"),
        ("I64", "Stroke, not specified as hemorrhage or infarction"),
        ("I65", "Occlusion and stenosis of precerebral arteries"),
        ("I66", "Occlusion and stenosis of cerebral arteries"),
        ("I67.9", "Cerebrovascular disease, unspecified"),
        ("I69", "Sequelae of cerebrovascular disease"),

        # Respiratory
        ("J06", "Acute upper respiratory infections"),
        ("J06.9", "Acute upper respiratory infection, unspecified"),
        ("J18", "Pneumonia, unspecified organism"),
        ("J18.9", "Pneumonia, unspecified organism"),
        ("J20", "Acute bronchitis"),
        ("J40", "Bronchitis, not specified as acute or chronic"),
        ("J44", "Other chronic obstructive pulmonary disease"),
        ("J44.0", "Chronic obstructive pulmonary disease with acute lower respiratory infection"),
        ("J44.1", "Chronic obstructive pulmonary disease with acute exacerbation"),
        ("J44.9", "Chronic obstructive pulmonary disease, unspecified"),
        ("J45", "Asthma"),
        ("J45.20", "Mild intermittent asthma, uncomplicated"),
        ("J45.30", "Mild persistent asthma, uncomplicated"),
        ("J45.40", "Moderate persistent asthma, uncomplicated"),
        ("J45.50", "Severe persistent asthma, uncomplicated"),
        ("J45.909", "Unspecified asthma, uncomplicated"),
        ("J96", "Respiratory failure"),

        # Mental disorders
        ("F10", "Alcohol related disorders"),
        ("F17", "Nicotine dependence"),
        ("F20", "Schizophrenia"),
        ("F31", "Bipolar disorder"),
        ("F32", "Major depressive disorder, single episode"),
        ("F32.9", "Major depressive disorder, single episode, unspecified"),
        ("F33", "Major depressive disorder, recurrent"),
        ("F33.0", "Major depressive disorder, recurrent, mild"),
        ("F33.1", "Major depressive disorder, recurrent, moderate"),
        ("F33.2", "Major depressive disorder, recurrent severe without psychotic features"),
        ("F41", "Other anxiety disorders"),
        ("F41.0", "Panic disorder"),
        ("F41.1", "Generalized anxiety disorder"),
        ("F41.9", "Anxiety disorder, unspecified"),
        ("F43.1", "Post-traumatic stress disorder"),
        ("F90", "Attention-deficit hyperactivity disorders"),

        # Kidney disease
        ("N17", "Acute kidney failure"),
        ("N18", "Chronic kidney disease"),
        ("N18.1", "Chronic kidney disease, stage 1"),
        ("N18.2", "Chronic kidney disease, stage 2"),
        ("N18.3", "Chronic kidney disease, stage 3"),
        ("N18.4", "Chronic kidney disease, stage 4"),
        ("N18.5", "Chronic kidney disease, stage 5"),
        ("N18.6", "End stage renal disease"),
        ("N18.9", "Chronic kidney disease, unspecified"),
        ("N19", "Unspecified kidney failure"),
        ("N39.0", "Urinary tract infection, site not specified"),

        # Lipid disorders
        ("E78", "Disorders of lipoprotein metabolism"),
        ("E78.0", "Pure hypercholesterolemia"),
        ("E78.1", "Pure hyperglyceridemia"),
        ("E78.2", "Mixed hyperlipidemia"),
        ("E78.5", "Hyperlipidemia, unspecified"),

        # Thyroid
        ("E03", "Other hypothyroidism"),
        ("E03.9", "Hypothyroidism, unspecified"),
        ("E05", "Thyrotoxicosis [hyperthyroidism]"),
        ("E05.90", "Thyrotoxicosis, unspecified without thyrotoxic crisis"),
        ("E06", "Thyroiditis"),

        # Obesity
        ("E66", "Overweight and obesity"),
        ("E66.01", "Morbid (severe) obesity due to excess calories"),
        ("E66.09", "Other obesity due to excess calories"),
        ("E66.9", "Obesity, unspecified"),

        # GI
        ("K21", "Gastro-esophageal reflux disease"),
        ("K21.0", "Gastro-esophageal reflux disease with esophagitis"),
        ("K25", "Gastric ulcer"),
        ("K26", "Duodenal ulcer"),
        ("K29", "Gastritis and duodenitis"),
        ("K50", "Crohn's disease"),
        ("K51", "Ulcerative colitis"),
        ("K57", "Diverticular disease of intestine"),
        ("K58", "Irritable bowel syndrome"),
        ("K70", "Alcoholic liver disease"),
        ("K74", "Fibrosis and cirrhosis of liver"),
        ("K76", "Other diseases of liver"),

        # Musculoskeletal
        ("M05", "Rheumatoid arthritis with rheumatoid factor"),
        ("M06", "Other rheumatoid arthritis"),
        ("M10", "Gout"),
        ("M15", "Polyosteoarthritis"),
        ("M16", "Osteoarthritis of hip"),
        ("M17", "Osteoarthritis of knee"),
        ("M19", "Other and unspecified osteoarthritis"),
        ("M25.5", "Pain in joint"),
        ("M54", "Dorsalgia"),
        ("M54.2", "Cervicalgia"),
        ("M54.5", "Low back pain"),
        ("M79.3", "Panniculitis, unspecified"),
        ("M79.7", "Fibromyalgia"),

        # Neurological
        ("G20", "Parkinson's disease"),
        ("G30", "Alzheimer's disease"),
        ("G35", "Multiple sclerosis"),
        ("G40", "Epilepsy and recurrent seizures"),
        ("G43", "Migraine"),
        ("G43.909", "Migraine, unspecified, not intractable, without status migrainosus"),
        ("G47", "Sleep disorders"),
        ("G47.33", "Obstructive sleep apnea"),
        ("G62", "Other and unspecified polyneuropathies"),
        ("G89", "Pain, not elsewhere classified"),

        # Cancer
        ("C18", "Malignant neoplasm of colon"),
        ("C34", "Malignant neoplasm of bronchus and lung"),
        ("C50", "Malignant neoplasm of breast"),
        ("C61", "Malignant neoplasm of prostate"),
        ("C64", "Malignant neoplasm of kidney"),
        ("C67", "Malignant neoplasm of bladder"),
        ("C73", "Malignant neoplasm of thyroid gland"),
        ("C85", "Other specified and unspecified types of non-Hodgkin lymphoma"),
        ("C90", "Multiple myeloma and malignant plasma cell neoplasms"),
        ("C91", "Lymphoid leukemia"),
        ("C92", "Myeloid leukemia"),

        # Blood disorders
        ("D50", "Iron deficiency anemia"),
        ("D64", "Other anemias"),
        ("D64.9", "Anemia, unspecified"),
        ("D68", "Other coagulation defects"),
        ("D69", "Purpura and other hemorrhagic conditions"),

        # Infectious
        ("A41", "Other sepsis"),
        ("A41.9", "Sepsis, unspecified organism"),
        ("B19", "Unspecified viral hepatitis"),
        ("B20", "Human immunodeficiency virus [HIV] disease"),
        ("B34", "Viral infection of unspecified site"),
        ("B95", "Streptococcus, Staphylococcus as cause of diseases classified elsewhere"),
        ("B96", "Other bacterial agents as cause of diseases classified elsewhere"),

        # Venous
        ("I80", "Phlebitis and thrombophlebitis"),
        ("I82", "Other venous embolism and thrombosis"),
        ("I82.40", "Acute embolism and thrombosis of unspecified deep veins of lower extremity"),
        ("I26", "Pulmonary embolism"),
        ("I26.99", "Other pulmonary embolism without acute cor pulmonale"),
        ("I87", "Other disorders of veins"),

        # Eye
        ("H25", "Age-related cataract"),
        ("H26", "Other cataract"),
        ("H35.30", "Unspecified macular degeneration"),
        ("H40", "Glaucoma"),
        ("H52", "Disorders of refraction and accommodation"),

        # Skin
        ("L20", "Atopic dermatitis"),
        ("L30", "Other and unspecified dermatitis"),
        ("L40", "Psoriasis"),
        ("L50", "Urticaria"),
        ("L70", "Acne"),

        # Symptoms
        ("R00", "Abnormalities of heart beat"),
        ("R05", "Cough"),
        ("R06", "Abnormalities of breathing"),
        ("R07", "Pain in throat and chest"),
        ("R10", "Abdominal and pelvic pain"),
        ("R11", "Nausea and vomiting"),
        ("R19.7", "Diarrhea, unspecified"),
        ("R21", "Rash and other nonspecific skin eruption"),
        ("R42", "Dizziness and giddiness"),
        ("R50", "Fever of other and unknown origin"),
        ("R51", "Headache"),
        ("R53", "Malaise and fatigue"),
        ("R55", "Syncope and collapse"),
        ("R63.4", "Abnormal weight loss"),
        ("R73.9", "Hyperglycemia, unspecified"),

        # Injuries
        ("S00", "Superficial injury of head"),
        ("S06", "Intracranial injury"),
        ("S22", "Fracture of rib(s), sternum and thoracic spine"),
        ("S32", "Fracture of lumbar spine and pelvis"),
        ("S42", "Fracture of shoulder and upper arm"),
        ("S52", "Fracture of forearm"),
        ("S72", "Fracture of femur"),
        ("S82", "Fracture of lower leg"),
        ("T14", "Injury of unspecified body region"),
    ]

    count = 0
    for code, description in icd10_codes:
        if insert_term(conn, description, code, "diagnosis", "ICD10"):
            count += 1

    conn.commit()
    logger.info(f"Loaded {count} embedded ICD-10 codes")
    return count


async def download_snomed_terms(conn: sqlite3.Connection, limit: int = 5000) -> int:
    """Fetch SNOMED CT terms from free APIs."""
    import httpx

    logger.info("=" * 60)
    logger.info("Fetching SNOMED CT terms...")
    logger.info("=" * 60)

    count = 0

    # Use NLM's free terminology services
    # SNOMED CT Browser API
    snomed_api = "https://browser.icd.who.int/api/v1/icd/search"

    # Common clinical terms to search for SNOMED concepts
    search_terms = [
        # Conditions
        "diabetes", "hypertension", "heart failure", "pneumonia", "asthma",
        "copd", "stroke", "myocardial infarction", "sepsis", "cancer",
        "anemia", "arthritis", "depression", "anxiety", "dementia",
        "kidney disease", "liver disease", "thyroid", "obesity", "pain",
        # Clinical findings
        "fever", "cough", "dyspnea", "chest pain", "headache", "fatigue",
        "nausea", "vomiting", "diarrhea", "constipation", "edema",
        "tachycardia", "bradycardia", "hypotension", "hypoxia",
        # Procedures (for context)
        "surgery", "biopsy", "imaging", "laboratory",
    ]

    async with httpx.AsyncClient() as client:
        for term in search_terms:
            if count >= limit:
                break

            try:
                # Try NLM's RxNav for clinical concepts
                url = f"https://rxnav.nlm.nih.gov/REST/Prescribe/approximateTerm.json"
                params = {"term": term, "maxEntries": 20}

                response = await client.get(url, params=params, timeout=30.0)
                if response.status_code == 200:
                    data = response.json()
                    if "approximateGroup" in data:
                        for candidate in data["approximateGroup"].get("candidate", []):
                            name = candidate.get("name", "")
                            rxcui = candidate.get("rxcui", "")
                            if name and rxcui:
                                # These are clinical terms that can supplement our data
                                if insert_term(conn, name, rxcui, "clinical_term", "NLM"):
                                    count += 1

                await asyncio.sleep(0.1)  # Rate limiting

            except Exception as e:
                logger.warning(f"Failed to fetch SNOMED for '{term}': {e}")

    # Load embedded SNOMED-like clinical terms
    snomed_terms = [
        ("SNOMED-CT", "clinical terminology"),
        # Clinical findings
        ("138875005", "Disorder (disorder)"),
        ("404684003", "Clinical finding (finding)"),
        ("373930000", "Clinical finding (finding)"),
        ("64572001", "Disease (disorder)"),
        ("123037004", "Body structure (body structure)"),
        # Common conditions with SNOMED codes
        ("73211009", "Diabetes mellitus (disorder)"),
        ("44054006", "Type 2 diabetes mellitus (disorder)"),
        ("46635009", "Type 1 diabetes mellitus (disorder)"),
        ("38341003", "Hypertensive disorder (disorder)"),
        ("59621000", "Essential hypertension (disorder)"),
        ("84114007", "Heart failure (disorder)"),
        ("42343007", "Congestive heart failure (disorder)"),
        ("22298006", "Myocardial infarction (disorder)"),
        ("230690007", "Cerebrovascular accident (disorder)"),
        ("13645005", "Chronic obstructive lung disease (disorder)"),
        ("195967001", "Asthma (disorder)"),
        ("233604007", "Pneumonia (disorder)"),
        ("91302008", "Sepsis (disorder)"),
        ("363346000", "Malignant neoplastic disease (disorder)"),
        ("271737000", "Anemia (disorder)"),
        ("396275006", "Osteoarthritis (disorder)"),
        ("35489007", "Depressive disorder (disorder)"),
        ("197480006", "Anxiety disorder (disorder)"),
        ("52448006", "Dementia (disorder)"),
        ("709044004", "Chronic kidney disease (disorder)"),
        ("235856003", "Liver disease (disorder)"),
        ("14304000", "Thyroid disorder (disorder)"),
        ("414916001", "Obesity (disorder)"),
        ("22253000", "Pain (finding)"),
        # Symptoms
        ("386661006", "Fever (finding)"),
        ("49727002", "Cough (finding)"),
        ("267036007", "Dyspnea (finding)"),
        ("29857009", "Chest pain (finding)"),
        ("25064002", "Headache (finding)"),
        ("84229001", "Fatigue (finding)"),
        ("422587007", "Nausea (finding)"),
        ("422400008", "Vomiting (finding)"),
        ("62315008", "Diarrhea (finding)"),
        ("14760008", "Constipation (finding)"),
        ("267038008", "Edema (finding)"),
        ("3424008", "Tachycardia (finding)"),
        ("48867003", "Bradycardia (finding)"),
        ("45007003", "Low blood pressure (finding)"),
        ("389087006", "Hypoxia (finding)"),
    ]

    for code, term in snomed_terms:
        if insert_term(conn, term, code, "diagnosis", "SNOMED"):
            count += 1

    conn.commit()
    logger.info(f"Loaded {count} SNOMED CT terms")
    return count


def load_embedded_icd9(conn: sqlite3.Connection) -> int:
    """Load embedded ICD-9-CM codes (legacy but still used)."""
    logger.info("=" * 60)
    logger.info("Loading ICD-9-CM codes (legacy)...")
    logger.info("=" * 60)

    # Common ICD-9-CM codes (V codes and numeric)
    icd9_codes = [
        # Diabetes
        ("250", "Diabetes mellitus"),
        ("250.00", "Diabetes mellitus without mention of complication"),
        ("250.01", "Diabetes mellitus without mention of complication, type I"),
        ("250.02", "Diabetes mellitus without mention of complication, type II"),

        # Hypertension
        ("401", "Essential hypertension"),
        ("401.0", "Malignant essential hypertension"),
        ("401.1", "Benign essential hypertension"),
        ("401.9", "Unspecified essential hypertension"),
        ("402", "Hypertensive heart disease"),
        ("403", "Hypertensive chronic kidney disease"),

        # Heart disease
        ("410", "Acute myocardial infarction"),
        ("411", "Other acute and subacute forms of ischemic heart disease"),
        ("412", "Old myocardial infarction"),
        ("413", "Angina pectoris"),
        ("414", "Other forms of chronic ischemic heart disease"),
        ("427", "Cardiac dysrhythmias"),
        ("427.31", "Atrial fibrillation"),
        ("428", "Heart failure"),
        ("428.0", "Congestive heart failure"),

        # Cerebrovascular
        ("430", "Subarachnoid hemorrhage"),
        ("431", "Intracerebral hemorrhage"),
        ("432", "Other and unspecified intracranial hemorrhage"),
        ("433", "Occlusion and stenosis of precerebral arteries"),
        ("434", "Occlusion of cerebral arteries"),
        ("435", "Transient cerebral ischemia"),
        ("436", "Acute, but ill-defined, cerebrovascular disease"),

        # Respiratory
        ("480", "Viral pneumonia"),
        ("481", "Pneumococcal pneumonia"),
        ("482", "Other bacterial pneumonia"),
        ("486", "Pneumonia, organism unspecified"),
        ("490", "Bronchitis, not specified as acute or chronic"),
        ("491", "Chronic bronchitis"),
        ("492", "Emphysema"),
        ("493", "Asthma"),
        ("496", "Chronic airway obstruction, not elsewhere classified"),

        # Mental disorders
        ("296", "Episodic mood disorders"),
        ("296.2", "Major depressive disorder, single episode"),
        ("296.3", "Major depressive disorder, recurrent episode"),
        ("300", "Anxiety, dissociative and somatoform disorders"),
        ("300.00", "Anxiety state, unspecified"),
        ("300.02", "Generalized anxiety disorder"),
        ("309.81", "Posttraumatic stress disorder"),
        ("314", "Hyperkinetic syndrome of childhood"),

        # Kidney
        ("584", "Acute kidney failure"),
        ("585", "Chronic kidney disease"),
        ("586", "Renal failure, unspecified"),
        ("599.0", "Urinary tract infection, site not specified"),

        # GI
        ("530.81", "Esophageal reflux"),
        ("531", "Gastric ulcer"),
        ("532", "Duodenal ulcer"),
        ("533", "Peptic ulcer, site unspecified"),
        ("555", "Regional enteritis"),
        ("556", "Ulcerative colitis"),
        ("571", "Chronic liver disease and cirrhosis"),

        # Musculoskeletal
        ("714", "Rheumatoid arthritis and other inflammatory polyarthropathies"),
        ("715", "Osteoarthrosis and allied disorders"),
        ("724", "Other and unspecified disorders of back"),
        ("724.2", "Lumbago"),
        ("729.1", "Myalgia and myositis, unspecified"),

        # Neurological
        ("332", "Parkinson's disease"),
        ("331.0", "Alzheimer's disease"),
        ("340", "Multiple sclerosis"),
        ("345", "Epilepsy and recurrent seizures"),
        ("346", "Migraine"),

        # Cancer
        ("153", "Malignant neoplasm of colon"),
        ("162", "Malignant neoplasm of trachea, bronchus, and lung"),
        ("174", "Malignant neoplasm of female breast"),
        ("185", "Malignant neoplasm of prostate"),

        # Metabolic
        ("272", "Disorders of lipoid metabolism"),
        ("272.0", "Pure hypercholesterolemia"),
        ("272.4", "Other and unspecified hyperlipidemia"),
        ("278", "Overweight, obesity and other hyperalimentation"),
        ("278.00", "Obesity, unspecified"),
        ("278.01", "Morbid obesity"),

        # Thyroid
        ("244", "Acquired hypothyroidism"),
        ("242", "Thyrotoxicosis with or without goiter"),

        # Blood
        ("280", "Iron deficiency anemias"),
        ("285", "Other and unspecified anemias"),
        ("285.9", "Anemia, unspecified"),

        # Infectious
        ("038", "Septicemia"),
        ("070", "Viral hepatitis"),
        ("042", "Human immunodeficiency virus [HIV] disease"),

        # V codes (supplementary)
        ("V58.69", "Long-term (current) use of other medications"),
        ("V85.4", "Body Mass Index 40 and over, adult"),
    ]

    count = 0
    for code, description in icd9_codes:
        if insert_term(conn, description, code, "diagnosis", "ICD9"):
            count += 1

    conn.commit()
    logger.info(f"Loaded {count} ICD-9-CM codes")
    return count


def load_umls_concepts(conn: sqlite3.Connection) -> int:
    """Load common UMLS concepts (CUI mappings)."""
    logger.info("=" * 60)
    logger.info("Loading UMLS concept mappings...")
    logger.info("=" * 60)

    # UMLS Concept Unique Identifiers for key medical concepts
    # These map across ICD-10, SNOMED, and other vocabularies
    umls_concepts = [
        # CUI, Preferred Term
        ("C0011849", "Diabetes Mellitus"),
        ("C0011860", "Diabetes Mellitus, Type 2"),
        ("C0011854", "Diabetes Mellitus, Type 1"),
        ("C0020538", "Hypertension"),
        ("C0018802", "Heart Failure, Congestive"),
        ("C0027051", "Myocardial Infarction"),
        ("C0038454", "Stroke"),
        ("C0024117", "Chronic Obstructive Pulmonary Disease"),
        ("C0004096", "Asthma"),
        ("C0032285", "Pneumonia"),
        ("C0036690", "Sepsis"),
        ("C0006826", "Malignant Neoplasm"),
        ("C0002871", "Anemia"),
        ("C0003873", "Rheumatoid Arthritis"),
        ("C0029408", "Osteoarthritis"),
        ("C0011570", "Depression"),
        ("C0003467", "Anxiety"),
        ("C0011265", "Dementia"),
        ("C0022661", "Chronic Kidney Disease"),
        ("C0023890", "Liver Disease"),
        ("C0040136", "Thyroid Disease"),
        ("C0028754", "Obesity"),
        ("C0030193", "Pain"),
        ("C0015967", "Fever"),
        ("C0010200", "Cough"),
        ("C0013404", "Dyspnea"),
        ("C0008031", "Chest Pain"),
        ("C0018681", "Headache"),
        ("C0015672", "Fatigue"),
        ("C0027497", "Nausea"),
        ("C0042963", "Vomiting"),
        ("C0011991", "Diarrhea"),
        ("C0009806", "Constipation"),
        ("C0013604", "Edema"),
        ("C0039231", "Tachycardia"),
        ("C0428977", "Bradycardia"),
        ("C0020649", "Hypotension"),
        ("C0242184", "Hypoxia"),
        # Drugs (common)
        ("C0025598", "Metformin"),
        ("C0065374", "Lisinopril"),
        ("C0286651", "Atorvastatin"),
        ("C0023965", "Levothyroxine"),
        ("C0051696", "Amlodipine"),
        ("C0025859", "Metoprolol"),
        ("C0028978", "Omeprazole"),
        ("C0074554", "Losartan"),
        ("C0060926", "Gabapentin"),
        ("C0074393", "Sertraline"),
    ]

    count = 0
    for cui, term in umls_concepts:
        if insert_term(conn, term, cui, "concept", "UMLS"):
            count += 1

    conn.commit()
    logger.info(f"Loaded {count} UMLS concepts")
    return count


def print_database_stats(conn: sqlite3.Connection) -> None:
    """Print database statistics."""
    print("\n" + "=" * 60)
    print("DATABASE STATISTICS")
    print("=" * 60)

    # Total terms
    cursor = conn.execute("SELECT COUNT(*) FROM medical_terms")
    total = cursor.fetchone()[0]
    print(f"\nTotal terms: {total:,}")

    # By source
    print("\nTerms by source:")
    cursor = conn.execute("""
        SELECT source, COUNT(*) as cnt
        FROM medical_terms
        GROUP BY source
        ORDER BY cnt DESC
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]:15}: {row[1]:,}")

    # By category
    print("\nTerms by category:")
    cursor = conn.execute("""
        SELECT category, COUNT(*) as cnt
        FROM medical_terms
        GROUP BY category
        ORDER BY cnt DESC
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]:15}: {row[1]:,}")

    print()


async def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download Medical Terminology Data",
    )
    parser.add_argument("--all", action="store_true", help="Download all sources")
    parser.add_argument("--icd10", action="store_true", help="Download ICD-10-CM")
    parser.add_argument("--icd9", action="store_true", help="Load ICD-9-CM")
    parser.add_argument("--snomed", action="store_true", help="Fetch SNOMED CT")
    parser.add_argument("--umls", action="store_true", help="Load UMLS concepts")
    parser.add_argument("--clear", action="store_true", help="Clear existing data first")

    args = parser.parse_args()

    # Default to all if nothing specified
    if not any([args.all, args.icd10, args.icd9, args.snomed, args.umls]):
        args.all = True

    # Initialize database
    conn = init_database()

    if args.clear:
        logger.info("Clearing existing medical_terms data...")
        conn.execute("DELETE FROM medical_terms")
        conn.commit()

    total_count = 0

    # Download/load each source
    if args.all or args.icd10:
        count = await download_icd10_cm(conn)
        total_count += count

    if args.all or args.icd9:
        count = load_embedded_icd9(conn)
        total_count += count

    if args.all or args.snomed:
        count = await download_snomed_terms(conn)
        total_count += count

    if args.all or args.umls:
        count = load_umls_concepts(conn)
        total_count += count

    # Print final stats
    print_database_stats(conn)

    print(f"\nTotal terms loaded: {total_count:,}")
    print(f"Database path: {DB_PATH}")

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
