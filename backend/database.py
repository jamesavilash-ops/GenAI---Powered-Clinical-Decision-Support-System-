from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import json

Base = declarative_base()

class PatientRecord(Base):
    __tablename__ = 'patient_records'
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, unique=True, index=True)
    symptoms = Column(Text)
    lab_results = Column(Text)
    medical_history = Column(Text)
    age = Column(Integer)
    gender = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class DiagnosisResult(Base):
    __tablename__ = 'diagnosis_results'
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, index=True)
    possible_diagnoses = Column(JSON)  # List of diagnoses with probabilities
    recommended_tests = Column(JSON)
    treatment_suggestions = Column(JSON)
    retrieved_evidence = Column(JSON)
    confidence_scores = Column(JSON)
    reasoning_chain = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

# SQLite database
SQLALCHEMY_DATABASE_URL = "sqlite:///./clinical_cdss.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()