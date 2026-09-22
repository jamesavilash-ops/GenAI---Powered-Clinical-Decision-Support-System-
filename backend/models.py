from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime

class PatientData(BaseModel):
    patient_id: str
    symptoms: str
    lab_results: Optional[str] = ""
    medical_history: Optional[str] = ""
    age: int
    gender: str

class DiagnosisRequest(BaseModel):
    patient_data: PatientData

class DiseaseDiagnosis(BaseModel):
    disease_name: str
    probability: float
    confidence: float
    evidence: List[str]
    supporting_symptoms: List[str]
    conflicting_factors: List[str]

class TreatmentRecommendation(BaseModel):
    treatment: str
    evidence: str
    confidence: float
    source: str

class DiagnosisResponse(BaseModel):
    patient_id: str
    possible_diagnoses: List[DiseaseDiagnosis]
    recommended_tests: List[str]
    treatment_suggestions: List[TreatmentRecommendation]
    overall_confidence: float
    reasoning_chain: str
    metrics: Dict[str, float]

class MetricsResponse(BaseModel):
    precision: float
    recall: float
    f1_score: float
    bleu_score: float
    rouge_score: float
    mrr: float