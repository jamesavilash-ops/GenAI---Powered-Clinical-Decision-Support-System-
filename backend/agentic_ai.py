import json
import re
from typing import Dict, Any, List


class ClinicalAgent:

    def __init__(self, rag_pipeline):
        self.rag_pipeline = rag_pipeline

    def parse_llm_response(self, text: str):
        """Try to extract JSON from LLM response"""

        try:
            json_match = re.search(r'\{.*\}', text, re.DOTALL)

            if json_match:
                return json.loads(json_match.group())

        except:
            pass

        return None

    def symptom_match_score(self, patient_symptoms, disease_symptoms):
        """Calculate explainable symptom match score"""

        matches = []
        for s in disease_symptoms:
            for ps in patient_symptoms:
                if s.lower() in ps.lower() or ps.lower() in s.lower():
                    matches.append(s)

        score = len(matches) / max(len(disease_symptoms), 1)

        return score, matches

    def fallback_from_symptoms(self, patient_data):

        patient_symptoms = patient_data["symptoms"]

        diagnoses = []

        # Huntington's Disease
        huntington_symptoms = ["involuntary movements", "cognitive decline", "mood changes"]
        score, matches = self.symptom_match_score(patient_symptoms, huntington_symptoms)

        if score > 0:
            diagnoses.append({
                "disease_name": "Huntington's Disease",
                "probability": round(score, 2),
                "confidence": round(score, 2),
                "supporting_symptoms": matches,
                "conflicting_factors": []
            })

        # ALS
        als_symptoms = ["muscle weakness", "muscle atrophy", "speech difficulty"]
        score, matches = self.symptom_match_score(patient_symptoms, als_symptoms)

        if score > 0:
            diagnoses.append({
                "disease_name": "Amyotrophic Lateral Sclerosis (ALS)",
                "probability": round(score, 2),
                "confidence": round(score, 2),
                "supporting_symptoms": matches,
                "conflicting_factors": []
            })

        # Retinoblastoma
        retina_symptoms = ["vision problems", "white pupil reflex", "eye redness"]
        score, matches = self.symptom_match_score(patient_symptoms, retina_symptoms)

        if score > 0:
            diagnoses.append({
                "disease_name": "Retinoblastoma",
                "probability": round(score, 2),
                "confidence": round(score, 2),
                "supporting_symptoms": matches,
                "conflicting_factors": []
            })

        # Tay-Sachs
        tay_symptoms = ["developmental regression", "seizures", "blindness"]
        score, matches = self.symptom_match_score(patient_symptoms, tay_symptoms)

        if score > 0:
            diagnoses.append({
                "disease_name": "Tay-Sachs Disease",
                "probability": round(score, 2),
                "confidence": round(score, 2),
                "supporting_symptoms": matches,
                "conflicting_factors": []
            })

        if not diagnoses:
            diagnoses.append({
                "disease_name": "Unknown Rare Neurological Disorder",
                "probability": 0.5,
                "confidence": 0.5,
                "supporting_symptoms": patient_symptoms,
                "conflicting_factors": []
            })

        return diagnoses

    def build_reasoning_chain(self, patient_data, diagnoses):
        """Create structured AI reasoning steps"""

        symptoms = ", ".join(patient_data["symptoms"])

        reasoning = f"""
Step 1: Symptom Extraction
Detected symptoms from patient input:
{symptoms}

Step 2: Evidence Retrieval (RAG)
Relevant clinical knowledge retrieved from the rare disease knowledge base.

Step 3: Differential Diagnosis
AI compared patient symptoms against known symptom patterns of supported diseases.

Step 4: Symptom Matching
Each disease received a score based on overlap between patient symptoms and known disease symptoms.

Step 5: Clinical Recommendation
Top diseases are suggested along with recommended diagnostic tests for confirmation.
"""

        return reasoning.strip()

    def recommend_tests(self, diagnoses):

        tests = set()

        for d in diagnoses:
            name = d["disease_name"]

            if "Huntington" in name:
                tests.update([
                    "Genetic testing for HTT mutation",
                    "Neurological examination",
                    "MRI brain scan"
                ])

            if "ALS" in name:
                tests.update([
                    "Electromyography (EMG)",
                    "Nerve conduction study",
                    "MRI scan"
                ])

            if "Retinoblastoma" in name:
                tests.update([
                    "Eye examination",
                    "MRI eye scan",
                    "RB1 genetic testing"
                ])

            if "Tay-Sachs" in name:
                tests.update([
                    "Hexosaminidase A enzyme test",
                    "Genetic screening",
                    "Neurological evaluation"
                ])

        return list(tests)

    def treatment_plan(self, diagnoses):

        treatments = []

        for d in diagnoses:
            name = d["disease_name"]

            if "Huntington" in name:
                treatments.append({
                    "treatment": "Symptomatic treatment with tetrabenazine and psychiatric support",
                    "confidence": 0.8,
                    "source": "Clinical neurology guidelines"
                })

            if "ALS" in name:
                treatments.append({
                    "treatment": "Riluzole therapy and respiratory support",
                    "confidence": 0.78,
                    "source": "ALS treatment protocols"
                })

            if "Retinoblastoma" in name:
                treatments.append({
                    "treatment": "Chemotherapy, laser therapy, or surgery",
                    "confidence": 0.82,
                    "source": "Pediatric oncology guidelines"
                })

            if "Tay-Sachs" in name:
                treatments.append({
                    "treatment": "Supportive care and genetic counseling",
                    "confidence": 0.75,
                    "source": "Rare disease treatment guidelines"
                })

        return treatments

    def process_patient_case(self, patient_data: Dict):

        rag_result = self.rag_pipeline.generate_clinical_response(patient_data)

        llm_text = rag_result["llm_text"]
        evidence = rag_result["evidence"]

        parsed = self.parse_llm_response(llm_text)

        if parsed is None:

            diagnoses = self.fallback_from_symptoms(patient_data)

            parsed = {
                "possible_diagnoses": diagnoses,
                "recommended_tests": self.recommend_tests(diagnoses),
                "treatment_suggestions": self.treatment_plan(diagnoses),
                "reasoning_chain": self.build_reasoning_chain(patient_data, diagnoses),
                "overall_confidence": max(d["confidence"] for d in diagnoses)
            }

        parsed["retrieved_evidence"] = evidence

        return parsed