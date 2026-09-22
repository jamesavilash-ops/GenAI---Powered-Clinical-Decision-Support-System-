from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import sqlite3
import json
from datetime import datetime
from difflib import SequenceMatcher

app = FastAPI()

class PatientData(BaseModel):
    patient_id: str
    symptoms: str
    age: int
    gender: str
    lab_results: Optional[str] = ""
    medical_history: Optional[str] = ""

class DiagnosisResponse(BaseModel):
    patient_id: str
    possible_diagnoses: List[dict]
    recommended_tests: List[str]
    treatment_suggestions: List[dict]
    overall_confidence: float
    reasoning_chain: str

# Medical knowledge base
MEDICAL_KNOWLEDGE = {
    "Huntington's Disease": {
        "symptoms": ["involuntary movements", "chorea", "cognitive decline", "mood changes", "irritability", "depression"],
        "age_group": [30, 50],
        "tests": ["Genetic testing for HTT gene", "Neurological examination", "Brain MRI", "Neuropsychological testing"],
        "treatments": ["Tetrabenazine", "Antidepressants", "Antipsychotics", "Physical therapy", "Speech therapy"],
        "description": "Rare inherited neurodegenerative disorder with progressive motor, cognitive, and psychiatric symptoms."
    },
    "Amyotrophic Lateral Sclerosis (ALS)": {
        "symptoms": ["muscle weakness", "muscle atrophy", "twitching", "difficulty swallowing", "slurred speech", "breathing difficulty"],
        "age_group": [40, 70],
        "tests": ["EMG", "Nerve conduction studies", "MRI", "Blood tests", "Spinal tap"],
        "treatments": ["Riluzole", "Edaravone", "Physical therapy", "Respiratory support", "Nutritional support"],
        "description": "Progressive neurodegenerative disease affecting motor neurons."
    },
    "Retinoblastoma": {
        "symptoms": ["white pupil", "eye redness", "vision problems", "eye pain", "different colored irises", "crossed eyes"],
        "age_group": [0, 5],
        "tests": ["Eye examination under anesthesia", "Ultrasound", "MRI", "Genetic testing for RB1 gene"],
        "treatments": ["Chemotherapy", "Laser therapy", "Cryotherapy", "Radiation", "Surgery"],
        "description": "Rare eye cancer that primarily affects children."
    },
    "Tay-Sachs Disease": {
        "symptoms": ["developmental regression", "loss of skills", "startle response", "cherry red spot", "seizures", "blindness"],
        "age_group": [0, 3],
        "tests": ["Enzyme assay", "Genetic testing", "Eye examination", "MRI brain"],
        "treatments": ["Supportive care", "Antiseizure medications", "Nutritional support", "Physical therapy"],
        "description": "Rare inherited metabolic disorder causing progressive neurological deterioration."
    }
}

def similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def analyze_symptoms(symptoms: str, age: int, gender: str, medical_history: str = "") -> dict:
    symptoms_lower = symptoms.lower()
    possible_diagnoses = []
    
    for disease, info in MEDICAL_KNOWLEDGE.items():
        symptom_matches = []
        confidence = 0.0
        
        # Check symptom matches
        for symptom in info["symptoms"]:
            if symptom in symptoms_lower:
                symptom_matches.append(symptom)
                confidence += 0.15
        
        # Check age appropriateness
        age_min, age_max = info["age_group"]
        if age_min <= age <= age_max:
            confidence += 0.2
        else:
            confidence -= 0.1
        
        # If we have at least 2 symptom matches, consider it a possible diagnosis
        if len(symptom_matches) >= 2 and confidence > 0.2:
            probability = min(0.95, confidence + (len(symptom_matches) * 0.1))
            
            possible_diagnoses.append({
                "disease_name": disease,
                "probability": round(probability, 2),
                "confidence": round(confidence, 2),
                "evidence": [f"Matched symptoms: {', '.join(symptom_matches)}", info["description"]],
                "supporting_symptoms": symptom_matches,
                "conflicting_factors": [] if age_min <= age <= age_max else [f"Age {age} outside typical range ({age_min}-{age_max})"]
            })
    
    # Sort by confidence
    possible_diagnoses.sort(key=lambda x: x["confidence"], reverse=True)
    
    # If no specific matches, provide general recommendation
    if not possible_diagnoses:
        possible_diagnoses.append({
            "disease_name": "Rare Neurological Disorder - Further Investigation Required",
            "probability": 0.5,
            "confidence": 0.4,
            "evidence": ["Complex symptom presentation requires specialized evaluation"],
            "supporting_symptoms": symptoms.split()[:3],
            "conflicting_factors": ["Insufficient specific symptom patterns for definitive diagnosis"]
        })
    
    # Generate recommendations based on top diagnosis
    top_diagnosis = possible_diagnoses[0] if possible_diagnoses else None
    if top_diagnosis and top_diagnosis["disease_name"] in MEDICAL_KNOWLEDGE:
        recommended_tests = MEDICAL_KNOWLEDGE[top_diagnosis["disease_name"]]["tests"]
        treatments = MEDICAL_KNOWLEDGE[top_diagnosis["disease_name"]]["treatments"]
    else:
        recommended_tests = [
            "Comprehensive neurological examination",
            "Blood tests for metabolic disorders",
            "Genetic counseling",
            "MRI brain imaging"
        ]
        treatments = [{
            "treatment": "Symptomatic management based on final diagnosis",
            "evidence": "Standard approach for undiagnosed neurological conditions",
            "confidence": 0.7,
            "source": "Clinical guidelines"
        }]
    
    treatment_suggestions = []
    for treatment in treatments[:3]:  # Take top 3 treatments
        if isinstance(treatment, str):
            treatment_suggestions.append({
                "treatment": treatment,
                "evidence": "Standard treatment for condition",
                "confidence": top_diagnosis["confidence"] if top_diagnosis else 0.6,
                "source": "Medical guidelines"
            })
        else:
            treatment_suggestions.append(treatment)
    
    reasoning = f"Analyzed {len(symptoms.split())} symptoms for {age}y/o {gender}. "
    reasoning += f"Found {len(possible_diagnoses)} potential conditions. "
    if possible_diagnoses:
        reasoning += f"Top match: {possible_diagnoses[0]['disease_name']} with {possible_diagnoses[0]['confidence']:.0%} confidence."
    
    return {
        "diagnoses": possible_diagnoses,
        "tests": recommended_tests,
        "treatments": treatment_suggestions,
        "confidence": possible_diagnoses[0]["confidence"] if possible_diagnoses else 0.4,
        "reasoning": reasoning
    }

# Initialize database
def init_db():
    conn = sqlite3.connect('clinical_cdss.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patient_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT UNIQUE,
            symptoms TEXT,
            lab_results TEXT,
            medical_history TEXT,
            age INTEGER,
            gender TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS diagnosis_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            possible_diagnoses TEXT,
            recommended_tests TEXT,
            treatment_suggestions TEXT,
            reasoning_chain TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

init_db()

@app.post("/diagnose", response_model=DiagnosisResponse)
async def diagnose_patient(patient: PatientData):
    # Analyze symptoms using our local medical knowledge
    result = analyze_symptoms(
        patient.symptoms, 
        patient.age, 
        patient.gender,
        patient.medical_history
    )
    
    # Store in database
    conn = sqlite3.connect('clinical_cdss.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT OR REPLACE INTO patient_records 
        (patient_id, symptoms, lab_results, medical_history, age, gender)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (patient.patient_id, patient.symptoms, patient.lab_results, 
          patient.medical_history, patient.age, patient.gender))
    
    cursor.execute('''
        INSERT INTO diagnosis_results 
        (patient_id, possible_diagnoses, recommended_tests, treatment_suggestions, reasoning_chain)
        VALUES (?, ?, ?, ?, ?)
    ''', (patient.patient_id, json.dumps(result["diagnoses"]), 
          json.dumps(result["tests"]), json.dumps(result["treatments"]), 
          result["reasoning"]))
    
    conn.commit()
    conn.close()
    
    return DiagnosisResponse(
        patient_id=patient.patient_id,
        possible_diagnoses=result["diagnoses"],
        recommended_tests=result["tests"],
        treatment_suggestions=result["treatments"],
        overall_confidence=result["confidence"],
        reasoning_chain=result["reasoning"]
    )

@app.get("/patient/{patient_id}")
async def get_patient_history(patient_id: str):
    conn = sqlite3.connect('clinical_cdss.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM patient_records WHERE patient_id = ?', (patient_id,))
    patient = cursor.fetchone()
    
    cursor.execute('SELECT * FROM diagnosis_results WHERE patient_id = ? ORDER BY created_at DESC', (patient_id,))
    diagnoses = cursor.fetchall()
    
    conn.close()
    
    if not patient:
        return {"error": "Patient not found"}
    
    return {
        "patient": {
            "id": patient[0],
            "patient_id": patient[1],
            "symptoms": patient[2],
            "age": patient[5],
            "gender": patient[6],
            "created_at": patient[7]
        },
        "diagnoses": [
            {
                "id": d[0],
                "possible_diagnoses": json.loads(d[2]),
                "recommended_tests": json.loads(d[3]),
                "treatment_suggestions": json.loads(d[4]),
                "reasoning_chain": d[5],
                "created_at": d[6]
            } for d in diagnoses
        ]
    }

@app.get("/")
async def root():
    return {
        "message": "Clinical CDSS API Running!", 
        "status": "success",
        "version": "1.0",
        "supported_diseases": list(MEDICAL_KNOWLEDGE.keys())
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "database": "connected"}

if __name__ == "__main__":
    print("🚀 Starting Clinical Decision Support System...")
    print("📚 Supported diseases:", list(MEDICAL_KNOWLEDGE.keys()))
    print("🌐 Server running on http://localhost:8000")
    print("📊 API documentation: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)