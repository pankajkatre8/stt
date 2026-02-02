#!/usr/bin/env python3
"""
Generate test conversations for HSTTB benchmarking evaluation.

Creates 50+ doctor-patient conversation scenarios with ground truth
and transcribed text variants organized into folders by error type.

Usage:
    python scripts/generate_test_conversations.py

Output Structure:
    test_data/
    ├── 01_perfect_transcriptions/
    ├── 02_drug_name_errors/
    ├── 03_dosage_errors/
    ├── 04_negation_flips/
    ├── 05_medical_condition_errors/
    ├── 06_minor_variations/
    ├── 07_spelling_inconsistencies/
    ├── 08_multiple_errors/
    ├── 09_specialty_cardiology/
    ├── 10_specialty_endocrinology/
    ├── 11_specialty_neurology/
    ├── 12_specialty_pulmonology/
    └── 13_complex_scenarios/
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path


@dataclass
class TestCase:
    """A single test case with ground truth and transcription."""
    id: str
    name: str
    category: str
    description: str
    ground_truth: str
    transcribed: str
    error_types: list[str]
    severity: str  # "none", "low", "medium", "high", "critical"
    expected_ter_range: tuple[float, float]  # Expected TER range
    notes: str = ""


def create_test_cases() -> list[TestCase]:
    """Create all test cases."""
    cases = []

    # =========================================================================
    # Category 1: Perfect Transcriptions (5 cases)
    # =========================================================================

    cases.append(TestCase(
        id="perfect_001",
        name="Routine Checkup - Perfect",
        category="01_perfect_transcriptions",
        description="Standard routine checkup with no transcription errors",
        ground_truth="""Doctor: Good morning, Mrs. Johnson. How are you feeling today?
Patient: I've been having some headaches lately, mostly in the morning.
Doctor: I see. How long have these headaches been occurring?
Patient: About two weeks now. They usually go away after I take some ibuprofen.
Doctor: Any other symptoms? Nausea, sensitivity to light?
Patient: No, just the headache. Sometimes a bit of neck stiffness.
Doctor: Let me check your blood pressure. It's 128 over 82, which is slightly elevated.
Patient: Is that concerning?
Doctor: We should monitor it. I'd like you to keep a blood pressure log for the next two weeks.""",
        transcribed="""Doctor: Good morning, Mrs. Johnson. How are you feeling today?
Patient: I've been having some headaches lately, mostly in the morning.
Doctor: I see. How long have these headaches been occurring?
Patient: About two weeks now. They usually go away after I take some ibuprofen.
Doctor: Any other symptoms? Nausea, sensitivity to light?
Patient: No, just the headache. Sometimes a bit of neck stiffness.
Doctor: Let me check your blood pressure. It's 128 over 82, which is slightly elevated.
Patient: Is that concerning?
Doctor: We should monitor it. I'd like you to keep a blood pressure log for the next two weeks.""",
        error_types=[],
        severity="none",
        expected_ter_range=(0.0, 0.0),
        notes="Baseline perfect transcription for comparison"
    ))

    cases.append(TestCase(
        id="perfect_002",
        name="Diabetes Management - Perfect",
        category="01_perfect_transcriptions",
        description="Diabetes follow-up visit with medication review",
        ground_truth="""Doctor: Your HbA1c came back at 7.2 percent, which is an improvement from last time.
Patient: That's good news. I've been trying to watch my diet more carefully.
Doctor: Excellent. Are you still taking metformin 500 milligrams twice daily?
Patient: Yes, I take it with breakfast and dinner.
Doctor: Any side effects? Stomach upset or diarrhea?
Patient: No, I've been tolerating it well.
Doctor: Good. Let's continue with the current regimen. I also want to check your feet today.
Patient: My feet have been fine, no numbness or tingling.
Doctor: That's reassuring. Keep up the good work with your diet and exercise.""",
        transcribed="""Doctor: Your HbA1c came back at 7.2 percent, which is an improvement from last time.
Patient: That's good news. I've been trying to watch my diet more carefully.
Doctor: Excellent. Are you still taking metformin 500 milligrams twice daily?
Patient: Yes, I take it with breakfast and dinner.
Doctor: Any side effects? Stomach upset or diarrhea?
Patient: No, I've been tolerating it well.
Doctor: Good. Let's continue with the current regimen. I also want to check your feet today.
Patient: My feet have been fine, no numbness or tingling.
Doctor: That's reassuring. Keep up the good work with your diet and exercise.""",
        error_types=[],
        severity="none",
        expected_ter_range=(0.0, 0.0)
    ))

    cases.append(TestCase(
        id="perfect_003",
        name="Hypertension Follow-up - Perfect",
        category="01_perfect_transcriptions",
        description="Blood pressure management discussion",
        ground_truth="""Doctor: Mr. Thompson, your blood pressure today is 142 over 88.
Patient: That seems high. I've been taking my lisinopril every day.
Doctor: I see you're on lisinopril 10 milligrams once daily. Have you been checking your pressure at home?
Patient: Yes, it's usually around 135 over 85 in the mornings.
Doctor: That's still above our target. I'd like to increase your lisinopril to 20 milligrams.
Patient: Will that cause any problems?
Doctor: Some patients experience dizziness initially. Take it at bedtime if that happens.
Patient: Okay, I'll try that.
Doctor: Also, please reduce your sodium intake and try to exercise for 30 minutes daily.""",
        transcribed="""Doctor: Mr. Thompson, your blood pressure today is 142 over 88.
Patient: That seems high. I've been taking my lisinopril every day.
Doctor: I see you're on lisinopril 10 milligrams once daily. Have you been checking your pressure at home?
Patient: Yes, it's usually around 135 over 85 in the mornings.
Doctor: That's still above our target. I'd like to increase your lisinopril to 20 milligrams.
Patient: Will that cause any problems?
Doctor: Some patients experience dizziness initially. Take it at bedtime if that happens.
Patient: Okay, I'll try that.
Doctor: Also, please reduce your sodium intake and try to exercise for 30 minutes daily.""",
        error_types=[],
        severity="none",
        expected_ter_range=(0.0, 0.0)
    ))

    cases.append(TestCase(
        id="perfect_004",
        name="Chest Pain Evaluation - Perfect",
        category="01_perfect_transcriptions",
        description="Cardiac symptom assessment",
        ground_truth="""Patient: Doctor, I've been having chest pain for the past three days.
Doctor: Can you describe the pain? Is it sharp, dull, or pressure-like?
Patient: It feels like pressure, right in the center of my chest.
Doctor: Does it radiate anywhere? To your arm, jaw, or back?
Patient: Sometimes I feel it in my left arm.
Doctor: Does anything make it better or worse? Activity or rest?
Patient: It gets worse when I climb stairs. Resting helps.
Doctor: I'm going to order an EKG and some blood tests including troponin levels.
Patient: Do you think it's my heart?
Doctor: We need to rule out cardiac causes. The tests will help us determine that.""",
        transcribed="""Patient: Doctor, I've been having chest pain for the past three days.
Doctor: Can you describe the pain? Is it sharp, dull, or pressure-like?
Patient: It feels like pressure, right in the center of my chest.
Doctor: Does it radiate anywhere? To your arm, jaw, or back?
Patient: Sometimes I feel it in my left arm.
Doctor: Does anything make it better or worse? Activity or rest?
Patient: It gets worse when I climb stairs. Resting helps.
Doctor: I'm going to order an EKG and some blood tests including troponin levels.
Patient: Do you think it's my heart?
Doctor: We need to rule out cardiac causes. The tests will help us determine that.""",
        error_types=[],
        severity="none",
        expected_ter_range=(0.0, 0.0)
    ))

    cases.append(TestCase(
        id="perfect_005",
        name="Medication Reconciliation - Perfect",
        category="01_perfect_transcriptions",
        description="Complete medication list review",
        ground_truth="""Doctor: Let's review all your current medications.
Patient: I take aspirin 81 milligrams daily for my heart.
Doctor: Good. What else?
Patient: Atorvastatin 40 milligrams at night for cholesterol.
Doctor: And your blood pressure medications?
Patient: Amlodipine 5 milligrams in the morning and hydrochlorothiazide 25 milligrams.
Doctor: Any supplements or over-the-counter medications?
Patient: I take vitamin D 1000 units and fish oil daily.
Doctor: That's a comprehensive list. Are you experiencing any side effects?
Patient: No, everything seems to be working well.""",
        transcribed="""Doctor: Let's review all your current medications.
Patient: I take aspirin 81 milligrams daily for my heart.
Doctor: Good. What else?
Patient: Atorvastatin 40 milligrams at night for cholesterol.
Doctor: And your blood pressure medications?
Patient: Amlodipine 5 milligrams in the morning and hydrochlorothiazide 25 milligrams.
Doctor: Any supplements or over-the-counter medications?
Patient: I take vitamin D 1000 units and fish oil daily.
Doctor: That's a comprehensive list. Are you experiencing any side effects?
Patient: No, everything seems to be working well.""",
        error_types=[],
        severity="none",
        expected_ter_range=(0.0, 0.0)
    ))

    # =========================================================================
    # Category 2: Drug Name Errors (8 cases)
    # =========================================================================

    cases.append(TestCase(
        id="drug_001",
        name="Metformin to Methotrexate Substitution",
        category="02_drug_name_errors",
        description="Critical: diabetes drug confused with cancer/autoimmune drug",
        ground_truth="""Doctor: I see you're taking metformin for your diabetes.
Patient: Yes, metformin 500 milligrams twice daily.
Doctor: How is your blood sugar control with the metformin?
Patient: It's been stable. The metformin works well for me.""",
        transcribed="""Doctor: I see you're taking methotrexate for your diabetes.
Patient: Yes, methotrexate 500 milligrams twice daily.
Doctor: How is your blood sugar control with the methotrexate?
Patient: It's been stable. The methotrexate works well for me.""",
        error_types=["drug_substitution", "critical_error"],
        severity="critical",
        expected_ter_range=(0.15, 0.25),
        notes="CRITICAL: Methotrexate is a chemotherapy drug, not a diabetes medication"
    ))

    cases.append(TestCase(
        id="drug_002",
        name="Lisinopril to Lisinipril Misspelling",
        category="02_drug_name_errors",
        description="Common phonetic misspelling of ACE inhibitor",
        ground_truth="""Doctor: Your lisinopril seems to be working well.
Patient: I've been taking the lisinopril 20 milligrams every morning.
Doctor: Good. The lisinopril is helping control your blood pressure.""",
        transcribed="""Doctor: Your lisinipril seems to be working well.
Patient: I've been taking the lisinipril 20 milligrams every morning.
Doctor: Good. The lisinipril is helping control your blood pressure.""",
        error_types=["drug_misspelling"],
        severity="medium",
        expected_ter_range=(0.05, 0.15)
    ))

    cases.append(TestCase(
        id="drug_003",
        name="Omeprazole to Omeprezole Misspelling",
        category="02_drug_name_errors",
        description="Slight vowel change in proton pump inhibitor",
        ground_truth="""Patient: I take omeprazole for my acid reflux.
Doctor: How long have you been on omeprazole?
Patient: About six months. The omeprazole really helps with my heartburn.""",
        transcribed="""Patient: I take omeprezole for my acid reflux.
Doctor: How long have you been on omeprezole?
Patient: About six months. The omeprezole really helps with my heartburn.""",
        error_types=["drug_misspelling"],
        severity="low",
        expected_ter_range=(0.03, 0.10)
    ))

    cases.append(TestCase(
        id="drug_004",
        name="Atorvastatin to Rosuvastatin Substitution",
        category="02_drug_name_errors",
        description="Similar class drug substitution (both statins)",
        ground_truth="""Doctor: I'm prescribing atorvastatin 20 milligrams for your cholesterol.
Patient: Is atorvastatin a statin medication?
Doctor: Yes, atorvastatin is very effective for lowering LDL cholesterol.""",
        transcribed="""Doctor: I'm prescribing rosuvastatin 20 milligrams for your cholesterol.
Patient: Is rosuvastatin a statin medication?
Doctor: Yes, rosuvastatin is very effective for lowering LDL cholesterol.""",
        error_types=["drug_substitution", "same_class"],
        severity="high",
        expected_ter_range=(0.10, 0.20),
        notes="While both are statins, dosing and interactions differ"
    ))

    cases.append(TestCase(
        id="drug_005",
        name="Gabapentin to Gabapentine Misspelling",
        category="02_drug_name_errors",
        description="Added letter to anticonvulsant name",
        ground_truth="""Doctor: Let's start you on gabapentin for your nerve pain.
Patient: What is gabapentin used for?
Doctor: Gabapentin helps with neuropathic pain and can improve your sleep.""",
        transcribed="""Doctor: Let's start you on gabapentine for your nerve pain.
Patient: What is gabapentine used for?
Doctor: Gabapentine helps with neuropathic pain and can improve your sleep.""",
        error_types=["drug_misspelling"],
        severity="low",
        expected_ter_range=(0.03, 0.10)
    ))

    cases.append(TestCase(
        id="drug_006",
        name="Warfarin to Heparin Substitution",
        category="02_drug_name_errors",
        description="Different anticoagulant substitution",
        ground_truth="""Doctor: You need to continue your warfarin for blood clot prevention.
Patient: What's my target INR for warfarin?
Doctor: We aim for an INR between 2 and 3 with your warfarin therapy.""",
        transcribed="""Doctor: You need to continue your heparin for blood clot prevention.
Patient: What's my target INR for heparin?
Doctor: We aim for an INR between 2 and 3 with your heparin therapy.""",
        error_types=["drug_substitution", "critical_error"],
        severity="critical",
        expected_ter_range=(0.10, 0.20),
        notes="CRITICAL: Warfarin and heparin have completely different monitoring and dosing"
    ))

    cases.append(TestCase(
        id="drug_007",
        name="Sertraline to Citalopram Substitution",
        category="02_drug_name_errors",
        description="Different SSRI antidepressant substitution",
        ground_truth="""Doctor: How is the sertraline working for your anxiety?
Patient: The sertraline has really helped. I feel much better.
Doctor: Good. We'll continue the sertraline 100 milligrams daily.""",
        transcribed="""Doctor: How is the citalopram working for your anxiety?
Patient: The citalopram has really helped. I feel much better.
Doctor: Good. We'll continue the citalopram 100 milligrams daily.""",
        error_types=["drug_substitution", "same_class"],
        severity="high",
        expected_ter_range=(0.10, 0.20)
    ))

    cases.append(TestCase(
        id="drug_008",
        name="Insulin Glargine to Insulin Lispro",
        category="02_drug_name_errors",
        description="Different insulin type substitution",
        ground_truth="""Doctor: You're currently on insulin glargine 20 units at bedtime.
Patient: Yes, the insulin glargine works well as my basal insulin.
Doctor: Good. Insulin glargine provides steady coverage throughout the day.""",
        transcribed="""Doctor: You're currently on insulin lispro 20 units at bedtime.
Patient: Yes, the insulin lispro works well as my basal insulin.
Doctor: Good. Insulin lispro provides steady coverage throughout the day.""",
        error_types=["drug_substitution", "critical_error"],
        severity="critical",
        expected_ter_range=(0.08, 0.18),
        notes="CRITICAL: Glargine is long-acting, lispro is rapid-acting - completely different usage"
    ))

    # =========================================================================
    # Category 3: Dosage Errors (6 cases)
    # =========================================================================

    cases.append(TestCase(
        id="dosage_001",
        name="10mg to 100mg Error",
        category="03_dosage_errors",
        description="Tenfold dosage error - potentially lethal",
        ground_truth="""Doctor: I'm prescribing lisinopril 10 milligrams once daily.
Patient: Is 10 milligrams a normal starting dose?
Doctor: Yes, 10 milligrams is standard. We may increase if needed.""",
        transcribed="""Doctor: I'm prescribing lisinopril 100 milligrams once daily.
Patient: Is 100 milligrams a normal starting dose?
Doctor: Yes, 100 milligrams is standard. We may increase if needed.""",
        error_types=["dosage_error", "critical_error"],
        severity="critical",
        expected_ter_range=(0.05, 0.15),
        notes="CRITICAL: 100mg lisinopril would be a massive overdose"
    ))

    cases.append(TestCase(
        id="dosage_002",
        name="500mg to 50mg Error",
        category="03_dosage_errors",
        description="Tenfold under-dosing error",
        ground_truth="""Doctor: Continue your metformin 500 milligrams twice daily.
Patient: I'll take the 500 milligrams with meals as usual.
Doctor: Correct, 500 milligrams with breakfast and dinner.""",
        transcribed="""Doctor: Continue your metformin 50 milligrams twice daily.
Patient: I'll take the 50 milligrams with meals as usual.
Doctor: Correct, 50 milligrams with breakfast and dinner.""",
        error_types=["dosage_error"],
        severity="high",
        expected_ter_range=(0.05, 0.15),
        notes="50mg metformin would be ineffective for diabetes management"
    ))

    cases.append(TestCase(
        id="dosage_003",
        name="Frequency Error - BID to QID",
        category="03_dosage_errors",
        description="Doubled frequency error",
        ground_truth="""Doctor: Take the medication twice daily, once in the morning and once at night.
Patient: So that's twice daily, morning and evening?
Doctor: Correct, twice daily with meals.""",
        transcribed="""Doctor: Take the medication four times daily, once in the morning and once at night.
Patient: So that's four times daily, morning and evening?
Doctor: Correct, four times daily with meals.""",
        error_types=["frequency_error"],
        severity="high",
        expected_ter_range=(0.08, 0.18)
    ))

    cases.append(TestCase(
        id="dosage_004",
        name="Decimal Point Error",
        category="03_dosage_errors",
        description="0.5mg transcribed as 5mg",
        ground_truth="""Doctor: Start with alprazolam 0.5 milligrams as needed for anxiety.
Patient: Is 0.5 milligrams a low dose?
Doctor: Yes, 0.5 milligrams is the lowest effective dose.""",
        transcribed="""Doctor: Start with alprazolam 5 milligrams as needed for anxiety.
Patient: Is 5 milligrams a low dose?
Doctor: Yes, 5 milligrams is the lowest effective dose.""",
        error_types=["dosage_error", "critical_error"],
        severity="critical",
        expected_ter_range=(0.05, 0.12),
        notes="CRITICAL: 5mg alprazolam would be dangerous sedation"
    ))

    cases.append(TestCase(
        id="dosage_005",
        name="Unit Error - mg to mcg",
        category="03_dosage_errors",
        description="Milligrams confused with micrograms",
        ground_truth="""Doctor: Your levothyroxine dose is 75 micrograms daily.
Patient: I take the 75 micrograms every morning before breakfast.
Doctor: Good. Continue the 75 micrograms and we'll recheck your TSH.""",
        transcribed="""Doctor: Your levothyroxine dose is 75 milligrams daily.
Patient: I take the 75 milligrams every morning before breakfast.
Doctor: Good. Continue the 75 milligrams and we'll recheck your TSH.""",
        error_types=["unit_error", "critical_error"],
        severity="critical",
        expected_ter_range=(0.05, 0.12),
        notes="CRITICAL: 75mg levothyroxine would be 1000x the intended dose"
    ))

    cases.append(TestCase(
        id="dosage_006",
        name="Insulin Units Error",
        category="03_dosage_errors",
        description="Insulin dose transcription error",
        ground_truth="""Doctor: Inject 15 units of insulin before dinner.
Patient: So 15 units with my evening meal?
Doctor: Yes, 15 units about 15 minutes before eating.""",
        transcribed="""Doctor: Inject 50 units of insulin before dinner.
Patient: So 50 units with my evening meal?
Doctor: Yes, 50 units about 15 minutes before eating.""",
        error_types=["dosage_error", "critical_error"],
        severity="critical",
        expected_ter_range=(0.05, 0.12),
        notes="CRITICAL: 50 units instead of 15 could cause severe hypoglycemia"
    ))

    # =========================================================================
    # Category 4: Negation Flips (6 cases)
    # =========================================================================

    cases.append(TestCase(
        id="negation_001",
        name="Chest Pain Negation Flip",
        category="04_negation_flips",
        description="Patient denies chest pain but transcribed as having it",
        ground_truth="""Doctor: Are you experiencing any chest pain?
Patient: No, I don't have any chest pain.
Doctor: Good. Patient denies chest pain. No cardiac symptoms reported.""",
        transcribed="""Doctor: Are you experiencing any chest pain?
Patient: Yes, I have chest pain.
Doctor: Good. Patient has chest pain. Cardiac symptoms reported.""",
        error_types=["negation_flip", "critical_error"],
        severity="critical",
        expected_ter_range=(0.15, 0.25),
        notes="CRITICAL: Changes entire clinical picture and treatment plan"
    ))

    cases.append(TestCase(
        id="negation_002",
        name="Drug Allergy Negation",
        category="04_negation_flips",
        description="No allergy transcribed as allergy present",
        ground_truth="""Doctor: Do you have any drug allergies?
Patient: No, I don't have any known drug allergies.
Doctor: Patient has no known drug allergies. NKDA documented.""",
        transcribed="""Doctor: Do you have any drug allergies?
Patient: Yes, I have drug allergies.
Doctor: Patient has known drug allergies. Allergies documented.""",
        error_types=["negation_flip", "critical_error"],
        severity="critical",
        expected_ter_range=(0.15, 0.25)
    ))

    cases.append(TestCase(
        id="negation_003",
        name="Shortness of Breath Negation",
        category="04_negation_flips",
        description="Denies dyspnea transcribed as having dyspnea",
        ground_truth="""Doctor: Any shortness of breath?
Patient: No shortness of breath at rest or with activity.
Doctor: Patient denies dyspnea. Lungs are clear bilaterally.""",
        transcribed="""Doctor: Any shortness of breath?
Patient: Shortness of breath at rest and with activity.
Doctor: Patient reports dyspnea. Lungs are clear bilaterally.""",
        error_types=["negation_flip"],
        severity="high",
        expected_ter_range=(0.12, 0.22)
    ))

    cases.append(TestCase(
        id="negation_004",
        name="Fever Negation Flip",
        category="04_negation_flips",
        description="Afebrile patient transcribed as having fever",
        ground_truth="""Doctor: Have you had any fever?
Patient: No fever. I've been checking my temperature and it's been normal.
Doctor: Patient is afebrile. No signs of infection.""",
        transcribed="""Doctor: Have you had any fever?
Patient: Yes, fever. I've been checking my temperature and it's been elevated.
Doctor: Patient is febrile. Signs of infection present.""",
        error_types=["negation_flip"],
        severity="high",
        expected_ter_range=(0.15, 0.25)
    ))

    cases.append(TestCase(
        id="negation_005",
        name="Suicidal Ideation Negation",
        category="04_negation_flips",
        description="Critical: denies SI transcribed as having SI",
        ground_truth="""Doctor: Are you having any thoughts of harming yourself?
Patient: No, I have no thoughts of suicide or self-harm.
Doctor: Patient denies suicidal ideation. No safety concerns at this time.""",
        transcribed="""Doctor: Are you having any thoughts of harming yourself?
Patient: Yes, I have thoughts of suicide and self-harm.
Doctor: Patient reports suicidal ideation. Safety concerns present.""",
        error_types=["negation_flip", "critical_error"],
        severity="critical",
        expected_ter_range=(0.15, 0.25),
        notes="CRITICAL: Would trigger completely different psychiatric intervention"
    ))

    cases.append(TestCase(
        id="negation_006",
        name="Family History Negation",
        category="04_negation_flips",
        description="No family history transcribed as positive",
        ground_truth="""Doctor: Is there any family history of heart disease?
Patient: No, there's no family history of heart disease.
Doctor: Negative family history for cardiovascular disease.""",
        transcribed="""Doctor: Is there any family history of heart disease?
Patient: Yes, there's family history of heart disease.
Doctor: Positive family history for cardiovascular disease.""",
        error_types=["negation_flip"],
        severity="medium",
        expected_ter_range=(0.12, 0.20)
    ))

    # =========================================================================
    # Category 5: Medical Condition Errors (5 cases)
    # =========================================================================

    cases.append(TestCase(
        id="condition_001",
        name="Diabetes Type Confusion",
        category="05_medical_condition_errors",
        description="Type 2 diabetes transcribed as Type 1",
        ground_truth="""Doctor: Your type 2 diabetes is well controlled.
Patient: I was diagnosed with type 2 diabetes five years ago.
Doctor: Managing type 2 diabetes with diet and metformin has worked well for you.""",
        transcribed="""Doctor: Your type 1 diabetes is well controlled.
Patient: I was diagnosed with type 1 diabetes five years ago.
Doctor: Managing type 1 diabetes with diet and metformin has worked well for you.""",
        error_types=["condition_error"],
        severity="high",
        expected_ter_range=(0.08, 0.18),
        notes="Type 1 and Type 2 have completely different pathophysiology and treatment"
    ))

    cases.append(TestCase(
        id="condition_002",
        name="Hypertension to Hypotension",
        category="05_medical_condition_errors",
        description="High blood pressure confused with low blood pressure",
        ground_truth="""Doctor: Your hypertension needs better control.
Patient: My hypertension has been difficult to manage.
Doctor: We need to adjust your hypertension medications.""",
        transcribed="""Doctor: Your hypotension needs better control.
Patient: My hypotension has been difficult to manage.
Doctor: We need to adjust your hypotension medications.""",
        error_types=["condition_error", "critical_error"],
        severity="critical",
        expected_ter_range=(0.08, 0.15),
        notes="CRITICAL: Hypertension and hypotension require opposite treatments"
    ))

    cases.append(TestCase(
        id="condition_003",
        name="Hypothyroid to Hyperthyroid",
        category="05_medical_condition_errors",
        description="Underactive thyroid confused with overactive",
        ground_truth="""Doctor: Your hypothyroidism requires continued levothyroxine.
Patient: I've had hypothyroidism since my twenties.
Doctor: Hypothyroidism is well managed with your current medication.""",
        transcribed="""Doctor: Your hyperthyroidism requires continued levothyroxine.
Patient: I've had hyperthyroidism since my twenties.
Doctor: Hyperthyroidism is well managed with your current medication.""",
        error_types=["condition_error", "critical_error"],
        severity="critical",
        expected_ter_range=(0.08, 0.15),
        notes="CRITICAL: Levothyroxine would worsen hyperthyroidism"
    ))

    cases.append(TestCase(
        id="condition_004",
        name="GERD to GORD Regional Spelling",
        category="05_medical_condition_errors",
        description="Minor regional spelling variation",
        ground_truth="""Doctor: Your GERD symptoms seem improved.
Patient: The GERD medication has helped a lot.
Doctor: We'll continue managing your GERD with omeprazole.""",
        transcribed="""Doctor: Your GORD symptoms seem improved.
Patient: The GORD medication has helped a lot.
Doctor: We'll continue managing your GORD with omeprazole.""",
        error_types=["spelling_variation"],
        severity="low",
        expected_ter_range=(0.03, 0.08),
        notes="GERD (US) vs GORD (UK) - same condition, different spelling"
    ))

    cases.append(TestCase(
        id="condition_005",
        name="Angina to Asthma Substitution",
        category="05_medical_condition_errors",
        description="Cardiac condition confused with respiratory",
        ground_truth="""Doctor: Your angina is triggered by exertion.
Patient: The angina pain usually comes when I exercise.
Doctor: We'll prescribe nitroglycerin for your angina episodes.""",
        transcribed="""Doctor: Your asthma is triggered by exertion.
Patient: The asthma pain usually comes when I exercise.
Doctor: We'll prescribe nitroglycerin for your asthma episodes.""",
        error_types=["condition_error", "critical_error"],
        severity="critical",
        expected_ter_range=(0.10, 0.18),
        notes="CRITICAL: Nitroglycerin is not appropriate for asthma"
    ))

    # =========================================================================
    # Category 6: Minor Variations (5 cases)
    # =========================================================================

    cases.append(TestCase(
        id="minor_001",
        name="Synonym Substitutions",
        category="06_minor_variations",
        description="Medical synonyms used interchangeably",
        ground_truth="""Doctor: The patient presents with cephalgia and myalgia.
Patient: My headache and muscle pain started yesterday.
Doctor: I'll prescribe an analgesic for the pain.""",
        transcribed="""Doctor: The patient presents with headache and muscle pain.
Patient: My head pain and body aches started yesterday.
Doctor: I'll prescribe a pain reliever for the discomfort.""",
        error_types=["synonym_variation"],
        severity="low",
        expected_ter_range=(0.10, 0.20),
        notes="Medically equivalent but different terminology"
    ))

    cases.append(TestCase(
        id="minor_002",
        name="Word Order Variation",
        category="06_minor_variations",
        description="Slight reordering of words",
        ground_truth="""Doctor: Patient has a history of diabetes mellitus type 2.
Patient: I've had type 2 diabetes for about ten years now.
Doctor: Blood glucose levels are well controlled currently.""",
        transcribed="""Doctor: Patient has a history of type 2 diabetes mellitus.
Patient: I've had diabetes type 2 for about ten years now.
Doctor: Blood glucose levels are currently well controlled.""",
        error_types=["word_order"],
        severity="none",
        expected_ter_range=(0.02, 0.08)
    ))

    cases.append(TestCase(
        id="minor_003",
        name="Filler Word Additions",
        category="06_minor_variations",
        description="Extra filler words in transcription",
        ground_truth="""Doctor: Take the medication with food.
Patient: I take it every morning.
Doctor: Good. Continue the same dose.""",
        transcribed="""Doctor: So, take the medication with food, okay?
Patient: Um, I take it every morning, you know.
Doctor: Good, good. Continue the, uh, same dose.""",
        error_types=["filler_words"],
        severity="none",
        expected_ter_range=(0.08, 0.18),
        notes="Filler words don't change medical meaning"
    ))

    cases.append(TestCase(
        id="minor_004",
        name="Contraction Differences",
        category="06_minor_variations",
        description="Contracted vs expanded forms",
        ground_truth="""Doctor: You will need to take this medication daily.
Patient: I have not been sleeping well lately.
Doctor: We will discuss sleep hygiene strategies.""",
        transcribed="""Doctor: You'll need to take this medication daily.
Patient: I haven't been sleeping well lately.
Doctor: We'll discuss sleep hygiene strategies.""",
        error_types=["contraction_variation"],
        severity="none",
        expected_ter_range=(0.02, 0.06)
    ))

    cases.append(TestCase(
        id="minor_005",
        name="Number Format Variations",
        category="06_minor_variations",
        description="Numbers written differently",
        ground_truth="""Doctor: Take two tablets three times daily for seven days.
Patient: So six tablets per day for a week?
Doctor: Correct, twenty-one tablets total.""",
        transcribed="""Doctor: Take 2 tablets 3 times daily for 7 days.
Patient: So 6 tablets per day for a week?
Doctor: Correct, 21 tablets total.""",
        error_types=["number_format"],
        severity="none",
        expected_ter_range=(0.02, 0.08)
    ))

    # =========================================================================
    # Category 7: Spelling Inconsistencies (4 cases)
    # =========================================================================

    cases.append(TestCase(
        id="inconsistent_001",
        name="Same Drug Multiple Spellings",
        category="07_spelling_inconsistencies",
        description="Same drug spelled differently within conversation",
        ground_truth="""Doctor: Let's continue your metformin therapy.
Patient: The metformin has been helping my blood sugar.
Doctor: Good. Metformin is working well for you.
Patient: I'll keep taking the metformin as prescribed.""",
        transcribed="""Doctor: Let's continue your metformin therapy.
Patient: The metforman has been helping my blood sugar.
Doctor: Good. Metphormin is working well for you.
Patient: I'll keep taking the metformin as prescribed.""",
        error_types=["spelling_inconsistency"],
        severity="medium",
        expected_ter_range=(0.05, 0.12),
        notes="Same drug spelled three different ways - indicates transcription issues"
    ))

    cases.append(TestCase(
        id="inconsistent_002",
        name="Condition Name Inconsistency",
        category="07_spelling_inconsistencies",
        description="Medical condition spelled inconsistently",
        ground_truth="""Doctor: Your hypertension is a concern.
Patient: My hypertension runs in the family.
Doctor: We need to control the hypertension with medication.""",
        transcribed="""Doctor: Your hypertension is a concern.
Patient: My hypertention runs in the family.
Doctor: We need to control the hypertenshun with medication.""",
        error_types=["spelling_inconsistency"],
        severity="medium",
        expected_ter_range=(0.05, 0.12)
    ))

    cases.append(TestCase(
        id="inconsistent_003",
        name="Patient Name Inconsistency",
        category="07_spelling_inconsistencies",
        description="Patient name spelled differently",
        ground_truth="""Doctor: Mrs. Patterson, how are you feeling?
Patient: I'm feeling better, thank you.
Doctor: Mrs. Patterson, your test results look good.
Patient: That's a relief.""",
        transcribed="""Doctor: Mrs. Patterson, how are you feeling?
Patient: I'm feeling better, thank you.
Doctor: Mrs. Paterson, your test results look good.
Patient: That's a relief.""",
        error_types=["name_inconsistency"],
        severity="low",
        expected_ter_range=(0.02, 0.06)
    ))

    cases.append(TestCase(
        id="inconsistent_004",
        name="Abbreviation Inconsistency",
        category="07_spelling_inconsistencies",
        description="Mix of abbreviations and full terms",
        ground_truth="""Doctor: Your blood pressure is elevated.
Patient: My BP has always been high.
Doctor: We need to monitor blood pressure closely.
Patient: I'll check my BP daily.""",
        transcribed="""Doctor: Your BP is elevated.
Patient: My blood pressure has always been high.
Doctor: We need to monitor BP closely.
Patient: I'll check my blood pressure daily.""",
        error_types=["abbreviation_inconsistency"],
        severity="low",
        expected_ter_range=(0.05, 0.12)
    ))

    # =========================================================================
    # Category 8: Multiple Errors (5 cases)
    # =========================================================================

    cases.append(TestCase(
        id="multiple_001",
        name="Drug + Dosage Error Combo",
        category="08_multiple_errors",
        description="Both drug name and dosage incorrect",
        ground_truth="""Doctor: Take lisinopril 10 milligrams once daily.
Patient: Is lisinopril 10 milligrams a blood pressure medication?
Doctor: Yes, lisinopril 10 milligrams will help control your hypertension.""",
        transcribed="""Doctor: Take lisinipril 100 milligrams once daily.
Patient: Is lisinipril 100 milligrams a blood pressure medication?
Doctor: Yes, lisinipril 100 milligrams will help control your hypertension.""",
        error_types=["drug_misspelling", "dosage_error", "critical_error"],
        severity="critical",
        expected_ter_range=(0.12, 0.22)
    ))

    cases.append(TestCase(
        id="multiple_002",
        name="Negation + Condition Error",
        category="08_multiple_errors",
        description="Negation flip combined with condition error",
        ground_truth="""Doctor: You don't have diabetes at this time.
Patient: I was worried about diabetes.
Doctor: Your glucose levels are normal. No diabetes present.""",
        transcribed="""Doctor: You have diabetis at this time.
Patient: I was worried about diabetis.
Doctor: Your glucose levels are abnormal. Diabetes present.""",
        error_types=["negation_flip", "condition_misspelling"],
        severity="critical",
        expected_ter_range=(0.15, 0.28)
    ))

    cases.append(TestCase(
        id="multiple_003",
        name="Multiple Drug Errors",
        category="08_multiple_errors",
        description="Several medication errors in one conversation",
        ground_truth="""Doctor: Let's review your medications. You're on metformin for diabetes.
Patient: Yes, and lisinopril for blood pressure.
Doctor: You also take atorvastatin for cholesterol.
Patient: And aspirin for my heart.""",
        transcribed="""Doctor: Let's review your medications. You're on methotrexate for diabetes.
Patient: Yes, and lisinipril for blood pressure.
Doctor: You also take atorvastain for cholesterol.
Patient: And asprin for my heart.""",
        error_types=["drug_substitution", "drug_misspelling", "critical_error"],
        severity="critical",
        expected_ter_range=(0.12, 0.22)
    ))

    cases.append(TestCase(
        id="multiple_004",
        name="Dosage + Frequency Error",
        category="08_multiple_errors",
        description="Both dose and frequency incorrect",
        ground_truth="""Doctor: Take metformin 500 milligrams twice daily with meals.
Patient: So 500 milligrams in the morning and evening?
Doctor: Correct, twice daily, 1000 milligrams total per day.""",
        transcribed="""Doctor: Take metformin 250 milligrams four times daily with meals.
Patient: So 250 milligrams in the morning and evening?
Doctor: Correct, four times daily, 1000 milligrams total per day.""",
        error_types=["dosage_error", "frequency_error"],
        severity="high",
        expected_ter_range=(0.12, 0.22)
    ))

    cases.append(TestCase(
        id="multiple_005",
        name="Condition + Medication + Negation",
        category="08_multiple_errors",
        description="Triple error - condition, drug, and negation",
        ground_truth="""Doctor: Your hypertension is not controlled with amlodipine alone.
Patient: I don't want to add another medication.
Doctor: Uncontrolled hypertension can lead to serious problems.""",
        transcribed="""Doctor: Your hypotension is controlled with amlopidine alone.
Patient: I want to add another medication.
Doctor: Controlled hypotension can lead to serious problems.""",
        error_types=["condition_error", "drug_misspelling", "negation_flip", "critical_error"],
        severity="critical",
        expected_ter_range=(0.18, 0.30)
    ))

    # =========================================================================
    # Category 9: Specialty - Cardiology (4 cases)
    # =========================================================================

    cases.append(TestCase(
        id="cardio_001",
        name="Cardiac Catheterization Discussion",
        category="09_specialty_cardiology",
        description="Discussion of cardiac procedure findings",
        ground_truth="""Doctor: Your cardiac catheterization showed a 70 percent blockage in the LAD artery.
Patient: Is that serious?
Doctor: It's significant. We found stenosis in the left anterior descending artery.
Patient: What are my options?
Doctor: We can manage with medications or consider angioplasty with stent placement.
Patient: What medications would I need?
Doctor: Aspirin, clopidogrel for blood thinning, a statin for cholesterol, and a beta blocker.""",
        transcribed="""Doctor: Your cardiac catheterization showed a 70 percent blockage in the LAD artery.
Patient: Is that serious?
Doctor: It's significant. We found stenosis in the left anterior descending artery.
Patient: What are my options?
Doctor: We can manage with medications or consider angioplasty with stent placement.
Patient: What medications would I need?
Doctor: Aspirin, clopidogrel for blood thinning, a statin for cholesterol, and a beta blocker.""",
        error_types=[],
        severity="none",
        expected_ter_range=(0.0, 0.0)
    ))

    cases.append(TestCase(
        id="cardio_002",
        name="Heart Failure Management - Errors",
        category="09_specialty_cardiology",
        description="Heart failure discussion with medication errors",
        ground_truth="""Doctor: Your ejection fraction is 35 percent, indicating heart failure.
Patient: What does that mean?
Doctor: Your heart isn't pumping as efficiently. We'll start carvedilol and lisinopril.
Patient: Will I need a pacemaker?
Doctor: Possibly a defibrillator. Your BNP level is elevated at 450.""",
        transcribed="""Doctor: Your ejection fraction is 53 percent, indicating heart failure.
Patient: What does that mean?
Doctor: Your heart isn't pumping as efficiently. We'll start corvedilol and lisinipril.
Patient: Will I need a pacemaker?
Doctor: Possibly a defibrillator. Your BMP level is elevated at 450.""",
        error_types=["value_error", "drug_misspelling", "abbreviation_error"],
        severity="high",
        expected_ter_range=(0.08, 0.18),
        notes="EF of 53% would be normal, not indicating heart failure"
    ))

    cases.append(TestCase(
        id="cardio_003",
        name="Arrhythmia Medication",
        category="09_specialty_cardiology",
        description="Atrial fibrillation management discussion",
        ground_truth="""Doctor: Your ECG shows atrial fibrillation with rapid ventricular response.
Patient: Is that why my heart feels like it's racing?
Doctor: Yes. We need to control the rate and prevent blood clots.
Patient: What medications do I need?
Doctor: Metoprolol to slow your heart and apixaban to prevent strokes.""",
        transcribed="""Doctor: Your ECG shows atrial fibrillation with rapid ventricular response.
Patient: Is that why my heart feels like it's racing?
Doctor: Yes. We need to control the rate and prevent blood clots.
Patient: What medications do I need?
Doctor: Metoprolol to slow your heart and warfarin to prevent strokes.""",
        error_types=["drug_substitution"],
        severity="medium",
        expected_ter_range=(0.03, 0.10),
        notes="Apixaban and warfarin are both anticoagulants but have different monitoring needs"
    ))

    cases.append(TestCase(
        id="cardio_004",
        name="Chest Pain Emergency",
        category="09_specialty_cardiology",
        description="Acute coronary syndrome evaluation",
        ground_truth="""Patient: The chest pain started an hour ago. It feels like an elephant on my chest.
Doctor: Any radiation to your arm or jaw? Shortness of breath?
Patient: Yes, my left arm is numb and I'm sweating.
Doctor: This could be a heart attack. We're giving you aspirin 325 milligrams now.
Patient: What's happening?
Doctor: We're doing an EKG and checking your troponin levels. You may need the cath lab.""",
        transcribed="""Patient: The chest pain started an hour ago. It feels like an elephant on my chest.
Doctor: Any radiation to your arm or jaw? Shortness of breath?
Patient: No, my left arm is fine and I'm not sweating.
Doctor: This could be a heart attack. We're giving you aspirin 325 milligrams now.
Patient: What's happening?
Doctor: We're doing an EKG and checking your troponin levels. You may need the cath lab.""",
        error_types=["negation_flip", "critical_error"],
        severity="critical",
        expected_ter_range=(0.08, 0.18),
        notes="CRITICAL: Denying classic MI symptoms changes clinical urgency"
    ))

    # =========================================================================
    # Category 10: Specialty - Endocrinology (4 cases)
    # =========================================================================

    cases.append(TestCase(
        id="endo_001",
        name="Diabetes Insulin Adjustment",
        category="10_specialty_endocrinology",
        description="Complex insulin regimen discussion",
        ground_truth="""Doctor: Your HbA1c is 8.5 percent. We need to adjust your insulin.
Patient: I'm currently on insulin glargine 20 units at bedtime and insulin lispro with meals.
Doctor: How many units of lispro are you taking?
Patient: Usually 5 units before each meal, but more if I eat carbs.
Doctor: Let's increase your glargine to 24 units and keep the lispro the same.
Patient: Should I still check my blood sugar four times daily?
Doctor: Yes, fasting and before meals. Target is 80 to 130 before meals.""",
        transcribed="""Doctor: Your HbA1c is 8.5 percent. We need to adjust your insulin.
Patient: I'm currently on insulin glargine 20 units at bedtime and insulin lispro with meals.
Doctor: How many units of lispro are you taking?
Patient: Usually 5 units before each meal, but more if I eat carbs.
Doctor: Let's increase your glargine to 24 units and keep the lispro the same.
Patient: Should I still check my blood sugar four times daily?
Doctor: Yes, fasting and before meals. Target is 80 to 130 before meals.""",
        error_types=[],
        severity="none",
        expected_ter_range=(0.0, 0.0)
    ))

    cases.append(TestCase(
        id="endo_002",
        name="Thyroid Medication Error",
        category="10_specialty_endocrinology",
        description="Thyroid dose with unit error",
        ground_truth="""Doctor: Your TSH is elevated at 8.5. We need to increase your levothyroxine.
Patient: I'm currently taking 50 micrograms daily.
Doctor: Let's increase to 75 micrograms. Take it on an empty stomach.
Patient: How long until I feel better?
Doctor: Usually 4 to 6 weeks. We'll recheck your TSH in 6 weeks.""",
        transcribed="""Doctor: Your TSH is elevated at 8.5. We need to increase your levothyroxine.
Patient: I'm currently taking 50 milligrams daily.
Doctor: Let's increase to 75 milligrams. Take it on an empty stomach.
Patient: How long until I feel better?
Doctor: Usually 4 to 6 weeks. We'll recheck your TSH in 6 weeks.""",
        error_types=["unit_error", "critical_error"],
        severity="critical",
        expected_ter_range=(0.05, 0.12),
        notes="CRITICAL: milligrams instead of micrograms - 1000x dose error"
    ))

    cases.append(TestCase(
        id="endo_003",
        name="Diabetes Oral Medications",
        category="10_specialty_endocrinology",
        description="Multiple diabetes medications discussion",
        ground_truth="""Doctor: We're going to add a second medication to your metformin.
Patient: Is metformin not working anymore?
Doctor: It's helping, but we need better control. I'm adding sitagliptin 100 milligrams.
Patient: What does sitagliptin do?
Doctor: It helps your body release more insulin after meals. Take it once daily.
Patient: Any side effects?
Doctor: It's generally well tolerated. Some patients report joint pain.""",
        transcribed="""Doctor: We're going to add a second medication to your metformin.
Patient: Is metformin not working anymore?
Doctor: It's helping, but we need better control. I'm adding saxagliptin 100 milligrams.
Patient: What does saxagliptin do?
Doctor: It helps your body release more insulin after meals. Take it once daily.
Patient: Any side effects?
Doctor: It's generally well tolerated. Some patients report joint pain.""",
        error_types=["drug_substitution"],
        severity="medium",
        expected_ter_range=(0.05, 0.12),
        notes="Both are DPP-4 inhibitors but different drugs with different dosing"
    ))

    cases.append(TestCase(
        id="endo_004",
        name="Adrenal Insufficiency",
        category="10_specialty_endocrinology",
        description="Steroid replacement discussion",
        ground_truth="""Doctor: Your cortisol levels confirm adrenal insufficiency.
Patient: What does that mean for treatment?
Doctor: You'll need hydrocortisone replacement. Start with 15 milligrams in the morning and 5 milligrams in the afternoon.
Patient: Is this lifelong?
Doctor: Yes, but you can live a normal life. You'll need to increase the dose during illness or stress.
Patient: What if I get sick?
Doctor: Double your dose during illness and seek medical help if you can't take it orally.""",
        transcribed="""Doctor: Your cortisol levels confirm adrenal insufficiency.
Patient: What does that mean for treatment?
Doctor: You'll need prednisone replacement. Start with 15 milligrams in the morning and 5 milligrams in the afternoon.
Patient: Is this lifelong?
Doctor: Yes, but you can live a normal life. You'll need to decrease the dose during illness or stress.
Patient: What if I get sick?
Doctor: Halve your dose during illness and seek medical help if you can't take it orally.""",
        error_types=["drug_substitution", "negation_flip", "critical_error"],
        severity="critical",
        expected_ter_range=(0.10, 0.20),
        notes="CRITICAL: Wrong steroid and opposite sick day rules could be fatal"
    ))

    # =========================================================================
    # Category 11: Specialty - Neurology (4 cases)
    # =========================================================================

    cases.append(TestCase(
        id="neuro_001",
        name="Migraine Management",
        category="11_specialty_neurology",
        description="Migraine prevention and treatment discussion",
        ground_truth="""Doctor: You're having about 12 migraine days per month. That qualifies for prevention.
Patient: What are my options?
Doctor: We can try topiramate, propranolol, or amitriptyline. I'd recommend starting with topiramate 25 milligrams.
Patient: Any side effects?
Doctor: Some people experience tingling in hands and feet, difficulty finding words, or weight loss.
Patient: What about when I get a migraine?
Doctor: Continue using sumatriptan 50 milligrams for acute attacks.""",
        transcribed="""Doctor: You're having about 12 migraine days per month. That qualifies for prevention.
Patient: What are my options?
Doctor: We can try topiramate, propranolol, or amitriptyline. I'd recommend starting with topiramate 25 milligrams.
Patient: Any side effects?
Doctor: Some people experience tingling in hands and feet, difficulty finding words, or weight loss.
Patient: What about when I get a migraine?
Doctor: Continue using sumatriptan 50 milligrams for acute attacks.""",
        error_types=[],
        severity="none",
        expected_ter_range=(0.0, 0.0)
    ))

    cases.append(TestCase(
        id="neuro_002",
        name="Epilepsy Medication Change",
        category="11_specialty_neurology",
        description="Anticonvulsant adjustment discussion",
        ground_truth="""Doctor: Your levetiracetam level is subtherapeutic. You had a breakthrough seizure last week.
Patient: I've been taking it as prescribed, 500 milligrams twice daily.
Doctor: We need to increase to 750 milligrams twice daily.
Patient: Will that cause more side effects?
Doctor: Possibly more fatigue or irritability initially. These usually improve.
Patient: Should I still avoid driving?
Doctor: Yes, no driving until you're seizure-free for six months.""",
        transcribed="""Doctor: Your levetiracetam level is subtherapeutic. You had a breakthrough seizure last week.
Patient: I've been taking it as prescribed, 500 milligrams twice daily.
Doctor: We need to decrease to 250 milligrams twice daily.
Patient: Will that cause more side effects?
Doctor: Possibly more fatigue or irritability initially. These usually improve.
Patient: Should I still avoid driving?
Doctor: No, driving is fine since you're seizure-free for six months.""",
        error_types=["direction_error", "negation_flip", "critical_error"],
        severity="critical",
        expected_ter_range=(0.10, 0.20),
        notes="CRITICAL: Decreasing instead of increasing dose, and allowing driving after seizure"
    ))

    cases.append(TestCase(
        id="neuro_003",
        name="Parkinson's Disease Review",
        category="11_specialty_neurology",
        description="Parkinson's medication management",
        ground_truth="""Doctor: How is the carbidopa-levodopa working for your Parkinson's symptoms?
Patient: It helps for about 3 hours, then the tremor comes back.
Doctor: That's wearing-off. We can add a COMT inhibitor like entacapone.
Patient: Will that extend the effect?
Doctor: Yes, entacapone helps levodopa last longer. Take it with each carbidopa-levodopa dose.
Patient: Any concerns?
Doctor: Watch for dyskinesias or unusual movements. Also, your urine might turn orange.""",
        transcribed="""Doctor: How is the carbidopa-levodopa working for your Parkinson's symptoms?
Patient: It helps for about 3 hours, then the tremor comes back.
Doctor: That's wearing-off. We can add a COMT inhibitor like entacapone.
Patient: Will that extend the effect?
Doctor: Yes, entacopone helps levodopa last longer. Take it with each carbidopa-levodopa dose.
Patient: Any concerns?
Doctor: Watch for dyskinesias or unusual movements. Also, your urine might turn orange.""",
        error_types=["drug_misspelling"],
        severity="low",
        expected_ter_range=(0.02, 0.06)
    ))

    cases.append(TestCase(
        id="neuro_004",
        name="Stroke Prevention Discussion",
        category="11_specialty_neurology",
        description="TIA follow-up and secondary prevention",
        ground_truth="""Doctor: Your MRI confirms you had a transient ischemic attack, or mini-stroke.
Patient: Am I at risk for a bigger stroke?
Doctor: Yes, TIAs are warning signs. We need aggressive prevention.
Patient: What medications do I need?
Doctor: High-intensity statin, aspirin 81 milligrams daily, and blood pressure control.
Patient: My blood pressure is usually around 150 over 90.
Doctor: We need to get that below 130 over 80. I'm adding lisinopril 10 milligrams.""",
        transcribed="""Doctor: Your MRI confirms you had a transient ischemic attack, or mini-stroke.
Patient: Am I at risk for a bigger stroke?
Doctor: No, TIAs are not warning signs. We don't need aggressive prevention.
Patient: What medications do I need?
Doctor: Low-intensity statin, aspirin 81 milligrams daily, and blood pressure control.
Patient: My blood pressure is usually around 150 over 90.
Doctor: That's fine at 150 over 90. No need to add lisinopril.""",
        error_types=["negation_flip", "critical_error"],
        severity="critical",
        expected_ter_range=(0.12, 0.22),
        notes="CRITICAL: Denying stroke risk and not treating increases major stroke risk"
    ))

    # =========================================================================
    # Category 12: Specialty - Pulmonology (4 cases)
    # =========================================================================

    cases.append(TestCase(
        id="pulm_001",
        name="COPD Exacerbation",
        category="12_specialty_pulmonology",
        description="COPD flare treatment discussion",
        ground_truth="""Doctor: Your oxygen saturation is 88 percent on room air. This is a COPD exacerbation.
Patient: I've been using my albuterol inhaler more frequently.
Doctor: We need to add prednisone 40 milligrams daily for 5 days.
Patient: What about antibiotics?
Doctor: Yes, you have a productive cough with yellow sputum. I'll prescribe azithromycin.
Patient: Should I continue my maintenance inhalers?
Doctor: Yes, keep using your Symbicort twice daily.""",
        transcribed="""Doctor: Your oxygen saturation is 88 percent on room air. This is a COPD exacerbation.
Patient: I've been using my albuterol inhaler more frequently.
Doctor: We need to add prednisone 40 milligrams daily for 5 days.
Patient: What about antibiotics?
Doctor: Yes, you have a productive cough with yellow sputum. I'll prescribe azithromycin.
Patient: Should I continue my maintenance inhalers?
Doctor: Yes, keep using your Symbicort twice daily.""",
        error_types=[],
        severity="none",
        expected_ter_range=(0.0, 0.0)
    ))

    cases.append(TestCase(
        id="pulm_002",
        name="Asthma Control Assessment",
        category="12_specialty_pulmonology",
        description="Asthma step-up therapy discussion",
        ground_truth="""Doctor: Your peak flow is only 65 percent of your personal best.
Patient: I've been waking up at night with wheezing about twice a week.
Doctor: That indicates poorly controlled asthma. We need to step up your therapy.
Patient: I'm currently on fluticasone 110 micrograms twice daily.
Doctor: Let's increase to 220 micrograms twice daily and add a long-acting beta agonist.
Patient: Will that be another inhaler?
Doctor: We can use a combination inhaler like Advair for convenience.""",
        transcribed="""Doctor: Your peak flow is only 65 percent of your personal best.
Patient: I haven't been waking up at night with wheezing.
Doctor: That indicates well controlled asthma. We can step down your therapy.
Patient: I'm currently on fluticasone 110 micrograms twice daily.
Doctor: Let's decrease to 55 micrograms twice daily and stop the long-acting beta agonist.
Patient: Will that be another inhaler?
Doctor: We can use a combination inhaler like Advair for convenience.""",
        error_types=["negation_flip", "direction_error", "critical_error"],
        severity="critical",
        expected_ter_range=(0.12, 0.22),
        notes="CRITICAL: Opposite assessment and treatment direction for asthma"
    ))

    cases.append(TestCase(
        id="pulm_003",
        name="Sleep Apnea CPAP Follow-up",
        category="12_specialty_pulmonology",
        description="CPAP compliance and settings review",
        ground_truth="""Doctor: Your CPAP download shows you're using it about 4 hours per night.
Patient: I try to use it, but I take it off during the night.
Doctor: We need at least 4 hours on most nights for benefit. Your AHI with the mask is 3.
Patient: Is that good?
Doctor: Yes, that's well controlled. Without CPAP, your AHI was 45.
Patient: What pressure am I on?
Doctor: You're on 12 centimeters of water pressure. That seems to be working well.""",
        transcribed="""Doctor: Your CPAP download shows you're using it about 4 hours per night.
Patient: I try to use it, but I take it off during the night.
Doctor: We need at least 4 hours on most nights for benefit. Your AHI with the mask is 30.
Patient: Is that good?
Doctor: Yes, that's well controlled. Without CPAP, your AHI was 45.
Patient: What pressure am I on?
Doctor: You're on 12 centimeters of water pressure. That seems to be working well.""",
        error_types=["value_error"],
        severity="high",
        expected_ter_range=(0.02, 0.08),
        notes="AHI of 30 vs 3 - significant difference in apnea control assessment"
    ))

    cases.append(TestCase(
        id="pulm_004",
        name="Pulmonary Fibrosis Discussion",
        category="12_specialty_pulmonology",
        description="IPF diagnosis and treatment options",
        ground_truth="""Doctor: Your CT scan shows a pattern consistent with idiopathic pulmonary fibrosis.
Patient: What does that mean for my prognosis?
Doctor: IPF is a progressive condition, but we have medications that can slow it down.
Patient: What are the treatment options?
Doctor: We can start pirfenidone or nintedanib. Both have been shown to slow decline.
Patient: Any cure?
Doctor: Unfortunately no cure exists, but lung transplant may be an option in the future.""",
        transcribed="""Doctor: Your CT scan shows a pattern consistent with idiopathic pulmonary fibrosis.
Patient: What does that mean for my prognosis?
Doctor: IPF is a progressive condition, but we have medications that can reverse it.
Patient: What are the treatment options?
Doctor: We can start pirfenidone or nintedinab. Both have been shown to slow decline.
Patient: Any cure?
Doctor: Yes, a cure exists and lung transplant is not needed.""",
        error_types=["information_error", "drug_misspelling", "negation_flip"],
        severity="high",
        expected_ter_range=(0.10, 0.20),
        notes="Incorrect information about reversibility and cure availability"
    ))

    # =========================================================================
    # Category 13: Complex Scenarios (5 cases)
    # =========================================================================

    cases.append(TestCase(
        id="complex_001",
        name="Multi-System Disease Management",
        category="13_complex_scenarios",
        description="Patient with diabetes, hypertension, and kidney disease",
        ground_truth="""Doctor: Let's review your conditions. You have type 2 diabetes, hypertension, and stage 3 chronic kidney disease.
Patient: How do my kidneys affect my other medications?
Doctor: Good question. We need to be careful with metformin at your kidney function level. Your eGFR is 42.
Patient: Should I stop the metformin?
Doctor: No, but we reduced it to 500 milligrams once daily. We're also using lisinopril which protects your kidneys.
Patient: What about my blood pressure goal?
Doctor: With kidney disease and diabetes, we aim for less than 130 over 80.
Patient: And my diabetes target?
Doctor: HbA1c below 7 percent, but we can be more flexible given your kidney disease.""",
        transcribed="""Doctor: Let's review your conditions. You have type 2 diabetes, hypertension, and stage 3 chronic kidney disease.
Patient: How do my kidneys affect my other medications?
Doctor: Good question. We need to be careful with metformin at your kidney function level. Your eGFR is 42.
Patient: Should I stop the metformin?
Doctor: No, but we reduced it to 500 milligrams once daily. We're also using lisinopril which protects your kidneys.
Patient: What about my blood pressure goal?
Doctor: With kidney disease and diabetes, we aim for less than 130 over 80.
Patient: And my diabetes target?
Doctor: HbA1c below 7 percent, but we can be more flexible given your kidney disease.""",
        error_types=[],
        severity="none",
        expected_ter_range=(0.0, 0.0)
    ))

    cases.append(TestCase(
        id="complex_002",
        name="Post-Hospitalization Reconciliation",
        category="13_complex_scenarios",
        description="Complex medication reconciliation after hospital stay",
        ground_truth="""Doctor: I see you were recently hospitalized for heart failure. Let's review your new medications.
Patient: They changed a lot of my medicines in the hospital.
Doctor: Yes, I see they started carvedilol 6.25 milligrams twice daily, increased your lisinopril to 20 milligrams, and added spironolactone 25 milligrams.
Patient: I'm also still taking my furosemide.
Doctor: Correct, furosemide 40 milligrams twice daily. They also continued your atorvastatin and aspirin.
Patient: That's a lot of pills.
Doctor: It is, but each medication serves an important purpose for your heart. We should also check your potassium regularly with the spironolactone.""",
        transcribed="""Doctor: I see you were recently hospitalized for heart failure. Let's review your new medications.
Patient: They changed a lot of my medicines in the hospital.
Doctor: Yes, I see they started carvedilol 6.25 milligrams twice daily, decreased your lisinopril to 10 milligrams, and added spirolactone 25 milligrams.
Patient: I'm also still taking my furosemide.
Doctor: Correct, furosemide 40 milligrams once daily. They also continued your atorvastatin and aspirin.
Patient: That's a lot of pills.
Doctor: It is, but each medication serves an important purpose for your heart. We should also check your potassium regularly with the spirolactone.""",
        error_types=["direction_error", "drug_misspelling", "frequency_error"],
        severity="high",
        expected_ter_range=(0.08, 0.18)
    ))

    cases.append(TestCase(
        id="complex_003",
        name="Oncology Treatment Discussion",
        category="13_complex_scenarios",
        description="Cancer treatment and supportive care",
        ground_truth="""Doctor: Your pathology shows stage 2B breast cancer, estrogen receptor positive.
Patient: What does that mean for treatment?
Doctor: We'll do surgery first, then chemotherapy with doxorubicin and cyclophosphamide, followed by hormonal therapy.
Patient: What are the side effects?
Doctor: Chemotherapy can cause nausea, hair loss, and fatigue. We'll give you ondansetron for nausea.
Patient: Will I need radiation?
Doctor: Yes, after chemotherapy. Then you'll take tamoxifen for 5 to 10 years.
Patient: What are my chances?
Doctor: Stage 2B has a good prognosis. Five-year survival is about 80 to 85 percent.""",
        transcribed="""Doctor: Your pathology shows stage 4B breast cancer, estrogen receptor positive.
Patient: What does that mean for treatment?
Doctor: We'll do surgery first, then chemotherapy with doxorubicin and cyclophosphamide, followed by hormonal therapy.
Patient: What are the side effects?
Doctor: Chemotherapy can cause nausea, hair loss, and fatigue. We'll give you ondansetron for nausea.
Patient: Will I need radiation?
Doctor: No, after chemotherapy. Then you'll take tamoxifen for 5 to 10 years.
Patient: What are my chances?
Doctor: Stage 4B has a good prognosis. Five-year survival is about 80 to 85 percent.""",
        error_types=["staging_error", "negation_flip", "critical_error"],
        severity="critical",
        expected_ter_range=(0.08, 0.18),
        notes="CRITICAL: Stage 4B vs 2B has drastically different prognosis and treatment approach"
    ))

    cases.append(TestCase(
        id="complex_004",
        name="Geriatric Polypharmacy Review",
        category="13_complex_scenarios",
        description="Elderly patient medication deprescribing",
        ground_truth="""Doctor: Mrs. Williams, you're on 14 medications. Let's see if we can simplify.
Patient: I get confused about which pills to take when.
Doctor: I understand. Let's start by stopping the diphenhydramine for sleep. It's not safe for older adults.
Patient: What about my ranitidine?
Doctor: Ranitidine has been recalled. Let's switch to famotidine 20 milligrams at bedtime.
Patient: I've been taking the baby aspirin for years.
Doctor: At 85 with no heart disease, the bleeding risk outweighs the benefit. We can stop that.
Patient: What about my vitamin D?
Doctor: Keep that. You're deficient and it's important for bone health.""",
        transcribed="""Doctor: Mrs. Williams, you're on 14 medications. Let's see if we can simplify.
Patient: I get confused about which pills to take when.
Doctor: I understand. Let's start by continuing the diphenhydramine for sleep. It's safe for older adults.
Patient: What about my ranitidine?
Doctor: Ranitidine is fine. Continue ranitidine 20 milligrams at bedtime.
Patient: I've been taking the baby aspirin for years.
Doctor: At 85 with no heart disease, the bleeding risk is low. We should continue that.
Patient: What about my vitamin D?
Doctor: Stop that. You don't need it for bone health.""",
        error_types=["medication_safety_error", "negation_flip", "critical_error"],
        severity="critical",
        expected_ter_range=(0.15, 0.25),
        notes="CRITICAL: Opposite recommendations for medication safety in elderly"
    ))

    cases.append(TestCase(
        id="complex_005",
        name="Pregnancy Medication Counseling",
        category="13_complex_scenarios",
        description="Medication safety during pregnancy",
        ground_truth="""Patient: I just found out I'm pregnant. I'm worried about my medications.
Doctor: Congratulations! Let's review each medication for pregnancy safety.
Patient: I take lisinopril for blood pressure.
Doctor: We need to stop the lisinopril immediately. It can harm the developing baby.
Patient: What will I take instead?
Doctor: Labetalol is safe during pregnancy. We'll switch you to that.
Patient: I also take atorvastatin for cholesterol.
Doctor: Stop that too. Statins are contraindicated in pregnancy. Cholesterol naturally rises during pregnancy.
Patient: What about my prenatal vitamins?
Doctor: Continue those. Make sure you're getting enough folic acid.""",
        transcribed="""Patient: I just found out I'm pregnant. I'm worried about my medications.
Doctor: Congratulations! Let's review each medication for pregnancy safety.
Patient: I take lisinopril for blood pressure.
Doctor: We should continue the lisinopril. It's safe for the developing baby.
Patient: What will I take instead?
Doctor: Lisinopril is safe during pregnancy. We'll keep you on that.
Patient: I also take atorvastatin for cholesterol.
Doctor: Continue that too. Statins are safe in pregnancy. Cholesterol naturally rises during pregnancy.
Patient: What about my prenatal vitamins?
Doctor: Stop those. You don't need folic acid.""",
        error_types=["medication_safety_error", "critical_error"],
        severity="critical",
        expected_ter_range=(0.15, 0.25),
        notes="CRITICAL: ACE inhibitors and statins are contraindicated in pregnancy - stopping folic acid increases neural tube defect risk"
    ))

    return cases


def write_test_cases(cases: list[TestCase], output_dir: Path) -> dict:
    """Write test cases to organized folder structure."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group cases by category
    categories: dict[str, list[TestCase]] = {}
    for case in cases:
        if case.category not in categories:
            categories[case.category] = []
        categories[case.category].append(case)

    stats = {
        "total_cases": len(cases),
        "categories": {},
        "severity_counts": {"none": 0, "low": 0, "medium": 0, "high": 0, "critical": 0},
        "generated_at": datetime.now().isoformat(),
    }

    # Write each category
    for category, category_cases in sorted(categories.items()):
        category_dir = output_dir / category
        category_dir.mkdir(exist_ok=True)

        stats["categories"][category] = {
            "count": len(category_cases),
            "cases": []
        }

        for case in category_cases:
            # Create case directory
            case_dir = category_dir / case.id
            case_dir.mkdir(exist_ok=True)

            # Write ground truth
            gt_file = case_dir / "ground_truth.txt"
            gt_file.write_text(case.ground_truth)

            # Write transcribed
            trans_file = case_dir / "transcribed.txt"
            trans_file.write_text(case.transcribed)

            # Write metadata
            metadata = {
                "id": case.id,
                "name": case.name,
                "category": case.category,
                "description": case.description,
                "error_types": case.error_types,
                "severity": case.severity,
                "expected_ter_range": list(case.expected_ter_range),
                "notes": case.notes,
            }
            metadata_file = case_dir / "metadata.json"
            metadata_file.write_text(json.dumps(metadata, indent=2))

            stats["categories"][category]["cases"].append(case.id)
            stats["severity_counts"][case.severity] += 1

    # Write summary index
    index_file = output_dir / "index.json"
    index_file.write_text(json.dumps(stats, indent=2))

    # Write README
    readme = f"""# HSTTB Test Conversations

Generated: {stats['generated_at']}
Total Test Cases: {stats['total_cases']}

## Categories

| Category | Count | Description |
|----------|-------|-------------|
| 01_perfect_transcriptions | {len(categories.get('01_perfect_transcriptions', []))} | Baseline perfect transcriptions |
| 02_drug_name_errors | {len(categories.get('02_drug_name_errors', []))} | Drug name substitutions and misspellings |
| 03_dosage_errors | {len(categories.get('03_dosage_errors', []))} | Dosage and unit errors |
| 04_negation_flips | {len(categories.get('04_negation_flips', []))} | Negation errors changing clinical meaning |
| 05_medical_condition_errors | {len(categories.get('05_medical_condition_errors', []))} | Condition name errors |
| 06_minor_variations | {len(categories.get('06_minor_variations', []))} | Acceptable variations (synonyms, etc.) |
| 07_spelling_inconsistencies | {len(categories.get('07_spelling_inconsistencies', []))} | Same term spelled differently |
| 08_multiple_errors | {len(categories.get('08_multiple_errors', []))} | Combinations of multiple error types |
| 09_specialty_cardiology | {len(categories.get('09_specialty_cardiology', []))} | Cardiology-specific scenarios |
| 10_specialty_endocrinology | {len(categories.get('10_specialty_endocrinology', []))} | Endocrinology-specific scenarios |
| 11_specialty_neurology | {len(categories.get('11_specialty_neurology', []))} | Neurology-specific scenarios |
| 12_specialty_pulmonology | {len(categories.get('12_specialty_pulmonology', []))} | Pulmonology-specific scenarios |
| 13_complex_scenarios | {len(categories.get('13_complex_scenarios', []))} | Complex multi-system scenarios |

## Severity Distribution

| Severity | Count | Description |
|----------|-------|-------------|
| None | {stats['severity_counts']['none']} | No errors (perfect transcription) |
| Low | {stats['severity_counts']['low']} | Minor spelling variations |
| Medium | {stats['severity_counts']['medium']} | Errors requiring review |
| High | {stats['severity_counts']['high']} | Significant clinical errors |
| Critical | {stats['severity_counts']['critical']} | Potentially dangerous errors |

## Directory Structure

Each test case contains:
- `ground_truth.txt` - The reference (correct) transcription
- `transcribed.txt` - The simulated STT output
- `metadata.json` - Test case metadata including expected TER range

## Usage

```python
from pathlib import Path
import json

test_dir = Path("test_data")

# Load a specific test case
case_dir = test_dir / "02_drug_name_errors" / "drug_001"
ground_truth = (case_dir / "ground_truth.txt").read_text()
transcribed = (case_dir / "transcribed.txt").read_text()
metadata = json.loads((case_dir / "metadata.json").read_text())

# Run evaluation
from hsttb.metrics.quality import QualityEngine
engine = QualityEngine()
result = engine.compute(transcribed)
```
"""
    readme_file = output_dir / "README.md"
    readme_file.write_text(readme)

    return stats


def main():
    """Generate test conversations."""
    print("Generating test conversations...")

    # Get output directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    output_dir = project_root / "test_data" / "conversations"

    # Generate test cases
    cases = create_test_cases()
    print(f"Created {len(cases)} test cases")

    # Write to disk
    stats = write_test_cases(cases, output_dir)

    print(f"\nTest data written to: {output_dir}")
    print(f"\nSummary:")
    print(f"  Total cases: {stats['total_cases']}")
    print(f"  Categories: {len(stats['categories'])}")
    print(f"\nSeverity distribution:")
    for severity, count in stats['severity_counts'].items():
        print(f"  {severity}: {count}")

    print(f"\nCategory breakdown:")
    for cat, info in sorted(stats['categories'].items()):
        print(f"  {cat}: {info['count']} cases")


if __name__ == "__main__":
    main()
