import json
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document
from typing import List


class ClinicalRAGPipeline:

    def __init__(self, openai_api_key: str):

        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model="gpt-3.5-turbo",
            temperature=0.2
        )

        self.vector_store = None
        self.retriever = None

        self.initialize_knowledge_base()

    def initialize_knowledge_base(self):

        clinical_data = [
            {
                "disease": "Huntington's Disease",
                "source": "Orphanet Rare Disease Database",
                "content": """
Huntington's Disease is a rare inherited neurodegenerative disorder.

Symptoms:
chorea (involuntary movements), cognitive decline, psychiatric symptoms,
impaired coordination, speech and swallowing difficulties.

Diagnosis:
Genetic testing for HTT gene mutation and neurological examination.

Treatment:
Tetrabenazine for chorea, antidepressants, antipsychotics,
physical therapy and speech therapy.
"""
            },
            {
                "disease": "Amyotrophic Lateral Sclerosis (ALS)",
                "source": "PubMed Clinical Literature",
                "content": """
Amyotrophic Lateral Sclerosis is a progressive neurodegenerative disease affecting motor neurons.

Symptoms:
muscle weakness, muscle atrophy, fasciculations,
difficulty speaking, swallowing and breathing.

Diagnosis:
Electromyography (EMG), nerve conduction studies,
MRI scans and blood tests to rule out other disorders.

Treatment:
Riluzole, Edaravone, respiratory therapy and supportive care.
"""
            },
            {
                "disease": "Retinoblastoma",
                "source": "NIH Pediatric Oncology Resources",
                "content": """
Retinoblastoma is a rare eye cancer that affects young children.

Symptoms:
white pupil reflex (leukocoria), crossed eyes (strabismus),
red painful eye and poor vision.

Diagnosis:
Eye examination, ultrasound imaging, MRI scan
and RB1 genetic mutation testing.

Treatment:
Chemotherapy, laser therapy, cryotherapy or surgery depending on stage.
"""
            },
            {
                "disease": "Tay-Sachs Disease",
                "source": "Orphanet Genetic Disorder Database",
                "content": """
Tay-Sachs disease is a rare inherited metabolic disorder.

Symptoms:
developmental regression, seizures,
cherry-red spot on retina, blindness and paralysis.

Diagnosis:
Hexosaminidase A enzyme assay and HEXA genetic testing.

Treatment:
Supportive care, seizure management and nutritional support.
"""
            }
        ]

        documents = []

        for data in clinical_data:

            doc = Document(
                page_content=data["content"],
                metadata={
                    "disease": data["disease"],
                    "source": data["source"]
                }
            )

            documents.append(doc)

        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )

        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

    def retrieve_relevant_evidence(self, query: str) -> List[Document]:

        return self.retriever.invoke(query)

    def generate_clinical_response(self, patient_data: dict):

        symptoms_text = " ".join(patient_data["symptoms"])

        evidence_docs = self.retrieve_relevant_evidence(symptoms_text)

        context = "\n\n".join([doc.page_content for doc in evidence_docs])

        evidence_output = [
            {
                "disease": doc.metadata.get("disease"),
                "source": doc.metadata.get("source"),
                "content": doc.page_content
            }
            for doc in evidence_docs
        ]

        prompt = f"""
You are an AI-powered Clinical Decision Support System for rare diseases.

PATIENT INFORMATION
Age: {patient_data['age']}
Gender: {patient_data['gender']}
Symptoms: {symptoms_text}
Lab Results: {patient_data['lab_results']}
Medical History: {patient_data['medical_history']}

CLINICAL KNOWLEDGE RETRIEVED FROM RAG
{context}

TASK

1 Identify possible rare diseases that match the patient symptoms
2 Provide confidence scores between 0 and 1
3 Suggest diagnostic tests
4 Suggest treatment approaches
5 Explain reasoning briefly

Return STRICT JSON format like this:

{{
"possible_diagnoses":[
{{"disease_name":"Disease","probability":0.0,"confidence":0.0,"supporting_symptoms":[],"conflicting_factors":[]}}
],
"recommended_tests":[],
"treatment_suggestions":[],
"reasoning_chain":"AI reasoning",
"overall_confidence":0.0
}}
"""

        response = self.llm.invoke(prompt)

        return {
            "llm_text": response.content,
            "evidence": evidence_output
        }