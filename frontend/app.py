import streamlit as st
import requests
import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="GenAI Clinical Decision Support System",
    page_icon="🏥",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }

    .diagnosis-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #2E86AB;
    }

    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }

    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }

    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# MAIN FRONTEND CLASS
# ============================================================

class ClinicalCDSSFrontend:

    def __init__(self):

        # ----------------------------------------------------
        # RENDER BACKEND
        # ----------------------------------------------------
        #
        # This is your deployed FastAPI backend on Render.
        #
        # If you later want to use Streamlit Secrets, you can
        # replace this with:
        #
        # self.backend_url = st.secrets["BACKEND_URL"].rstrip("/")
        #
        # For now, we keep your working Render URL directly.
        # ----------------------------------------------------

        self.backend_url = (
            "https://genai-powered-clinical-decision-support.onrender.com"
        )

        self.initialize_session_state()


    # ========================================================
    # SESSION STATE
    # ========================================================

    def initialize_session_state(self):

        if "diagnosis_history" not in st.session_state:
            st.session_state.diagnosis_history = []

        if "current_patient" not in st.session_state:
            st.session_state.current_patient = None


    # ========================================================
    # HEADER
    # ========================================================

    def render_header(self):

        col1, col2 = st.columns([1, 4])

        with col1:
            st.markdown(
                "<h1 style='text-align: center; font-size: 80px;'>🏥</h1>",
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(
                '<h1 class="main-header">'
                'GenAI Clinical Decision Support System'
                '</h1>',
                unsafe_allow_html=True
            )

            st.markdown(
                "### AI-Powered Rare Disease Diagnosis with RAG and Agentic Reasoning"
            )


    # ========================================================
    # SIDEBAR
    # ========================================================

    def render_sidebar(self):

        with st.sidebar:

            st.header("System Information")

            st.info("""
            **Supported Rare Diseases:**

            - Huntington's Disease
            - Amyotrophic Lateral Sclerosis (ALS)
            - Retinoblastoma
            - Tay-Sachs Disease

            **Features:**

            ✅ RAG-powered evidence retrieval  
            ✅ Agentic AI reasoning  
            ✅ Confidence scoring  
            ✅ Treatment recommendations
            """)

            st.header("Performance Metrics")

            if st.session_state.current_patient:

                try:

                    response = requests.get(
                        f"{self.backend_url}/metrics/"
                        f"{st.session_state.current_patient}",
                        timeout=20
                    )

                    if response.status_code == 200:

                        metrics = response.json()

                        self.render_metrics_chart(metrics)

                    else:

                        st.warning(
                            f"Metrics unavailable "
                            f"(HTTP {response.status_code})"
                        )

                except requests.exceptions.RequestException:

                    st.info(
                        "Performance metrics are currently unavailable."
                    )


    # ========================================================
    # METRICS CHART
    # ========================================================

    def render_metrics_chart(self, metrics):

        fig = make_subplots(
            rows=1,
            cols=2,
            specs=[[{"type": "domain"}, {"type": "xy"}]]
        )

        # Pie chart
        labels = [
            "Precision",
            "Recall",
            "F1-Score"
        ]

        values = [
            metrics.get("precision", 0),
            metrics.get("recall", 0),
            metrics.get("f1_score", 0)
        ]

        fig.add_trace(
            go.Pie(
                labels=labels,
                values=values,
                name="Accuracy Metrics"
            ),
            1,
            1
        )

        # Bar chart
        text_metrics = [
            "BLEU",
            "ROUGE",
            "MRR"
        ]

        text_values = [
            metrics.get("bleu_score", 0),
            metrics.get("rouge_score", 0),
            metrics.get("mrr", 0)
        ]

        fig.add_trace(
            go.Bar(
                x=text_metrics,
                y=text_values,
                name="Text Quality"
            ),
            1,
            2
        )

        fig.update_layout(
            height=300,
            showlegend=False,
            title_text="System Performance Metrics"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # ========================================================
    # PATIENT INPUT FORM
    # ========================================================

    def render_patient_input_form(self):

        st.header("📋 Patient Information")

        with st.form("patient_form"):

            col1, col2 = st.columns(2)

            # ------------------------------------------------
            # Patient information
            # ------------------------------------------------

            with col1:

                patient_id = st.text_input(
                    "Patient ID",
                    value=(
                        f"PAT_"
                        f"{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
                    )
                )

                age = st.number_input(
                    "Age",
                    min_value=0,
                    max_value=120,
                    value=45
                )

                gender = st.selectbox(
                    "Gender",
                    [
                        "Male",
                        "Female",
                        "Other"
                    ]
                )

            # ------------------------------------------------
            # Clinical presentation
            # ------------------------------------------------

            with col2:

                st.subheader("Clinical Presentation")

                symptoms = st.text_area(
                    "Symptoms",
                    placeholder=(
                        "Describe patient symptoms in detail...\n"
                        "Example: involuntary movements, "
                        "cognitive decline, mood changes"
                    ),
                    height=100
                )

                lab_results = st.text_area(
                    "Lab Results",
                    placeholder="Relevant laboratory findings...",
                    height=80
                )

                medical_history = st.text_area(
                    "Medical History",
                    placeholder="Family history, previous conditions...",
                    height=80
                )

            submitted = st.form_submit_button(
                "🚀 Generate AI Diagnosis"
            )

            if submitted:

                if not symptoms.strip():

                    st.warning(
                        "Please enter at least one symptom."
                    )

                    return None

                patient_data = {

                    "patient_id": patient_id,

                    "age": age,

                    "gender": gender,

                    "symptoms": [
                        s.strip()
                        for s in symptoms.split(",")
                        if s.strip()
                    ],

                    "lab_results": lab_results,

                    "medical_history": medical_history
                }

                return patient_data

        return None


    # ========================================================
    # DIAGNOSIS RESULTS
    # ========================================================

    def render_diagnosis_results(self, diagnosis_data):

        st.header("🔍 AI Diagnosis Results")

        overall_conf = diagnosis_data.get(
            "overall_confidence",
            diagnosis_data.get(
                "confidence_scores",
                {}
            ).get(
                "overall",
                0
            )
        )

        confidence_color = (
            "confidence-high"
            if overall_conf > 0.7
            else
            "confidence-medium"
            if overall_conf > 0.5
            else
            "confidence-low"
        )

        st.markdown(
            f"**Overall Confidence:** "
            f"<span class='{confidence_color}'>"
            f"{overall_conf:.1%}"
            f"</span>",
            unsafe_allow_html=True
        )

        # ----------------------------------------------------
        # Possible diagnoses
        # ----------------------------------------------------

        st.subheader("Possible Diagnoses")

        diagnoses = diagnosis_data.get(
            "possible_diagnoses",
            []
        )

        if not diagnoses:

            st.info(
                "No possible diagnoses were returned."
            )

        for i, diagnosis in enumerate(
            diagnoses,
            1
        ):

            with st.container():

                col1, col2 = st.columns([3, 1])

                with col1:

                    disease_name = diagnosis.get(
                        "disease",
                        diagnosis.get(
                            "disease_name",
                            "Unknown"
                        )
                    )

                    probability = diagnosis.get(
                        "probability",
                        diagnosis.get(
                            "confidence_score",
                            0
                        )
                    )

                    confidence = diagnosis.get(
                        "confidence",
                        probability
                    )

                    st.markdown(
                        f"**{i}. {disease_name}**"
                    )

                    st.write(
                        f"**Probability:** "
                        f"{probability:.1%}"
                    )

                    st.write(
                        f"**Confidence:** "
                        f"{confidence:.1%}"
                    )

                    supporting_symptoms = diagnosis.get(
                        "supporting_symptoms",
                        diagnosis.get(
                            "matched_symptoms",
                            []
                        )
                    )

                    if supporting_symptoms:

                        st.write(
                            "**Supporting Symptoms:**"
                        )

                        for symptom in supporting_symptoms:

                            st.write(
                                f"  ✅ {symptom}"
                            )

                    conflicting_factors = diagnosis.get(
                        "conflicting_factors",
                        []
                    )

                    if conflicting_factors:

                        st.write(
                            "**Conflicting Factors:**"
                        )

                        for conflict in conflicting_factors:

                            st.write(
                                f"  ⚠️ {conflict}"
                            )

                # ------------------------------------------------
                # Confidence gauge
                # ------------------------------------------------

                with col2:

                    fig = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=confidence * 100,
                            domain={
                                "x": [0, 1],
                                "y": [0, 1]
                            },
                            gauge={
                                "axis": {
                                    "range": [None, 100]
                                },
                                "bar": {
                                    "color": "darkblue"
                                },
                                "steps": [
                                    {
                                        "range": [0, 50],
                                        "color": "lightgray"
                                    },
                                    {
                                        "range": [50, 80],
                                        "color": "yellow"
                                    },
                                    {
                                        "range": [80, 100],
                                        "color": "lightgreen"
                                    }
                                ]
                            }
                        )
                    )

                    fig.update_layout(
                        height=150,
                        margin=dict(
                            l=10,
                            r=10,
                            t=10,
                            b=10
                        )
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

        # ----------------------------------------------------
        # Recommended tests
        # ----------------------------------------------------

        st.subheader(
            "🩺 Recommended Diagnostic Tests"
        )

        tests = diagnosis_data.get(
            "recommended_tests",
            diagnosis_data.get(
                "recommendations",
                []
            )
        )

        if tests:

            for test in tests[:5]:

                st.write(
                    f"• {test}"
                )

        else:

            st.info(
                "No diagnostic tests were returned."
            )

        # ----------------------------------------------------
        # Treatment recommendations
        # ----------------------------------------------------

        st.subheader(
            "💊 Treatment Recommendations"
        )

        treatments = diagnosis_data.get(
            "treatment_suggestions",
            []
        )

        if treatments:

            for treatment in treatments:

                if isinstance(
                    treatment,
                    dict
                ):

                    treatment_name = treatment.get(
                        "treatment",
                        "Treatment"
                    )

                    treatment_confidence = treatment.get(
                        "confidence",
                        0
                    )

                    with st.expander(
                        f"{treatment_name} "
                        f"(Confidence: "
                        f"{treatment_confidence:.1%})"
                    ):

                        st.write(
                            "**Evidence:** "
                            f"{treatment.get('evidence', 'No evidence provided')}"
                        )

                        st.write(
                            "**Source:** "
                            f"{treatment.get('source', 'Unknown source')}"
                        )

                else:

                    st.write(
                        f"• {treatment}"
                    )

        else:

            st.info(
                "No treatment suggestions were returned."
            )

        # ----------------------------------------------------
        # Reasoning
        # ----------------------------------------------------

        st.subheader(
            "🤖 AI Reasoning Process"
        )

        reasoning = diagnosis_data.get(
            "reasoning_chain",
            "No reasoning chain available"
        )

        st.text_area(
            "Reasoning Chain",
            reasoning,
            height=200,
            key="reasoning_display",
            disabled=True
        )


    # ========================================================
    # EVIDENCE
    # ========================================================

    def render_evidence_retrieval(
        self,
        diagnosis_data
    ):

        st.header(
            "📚 Retrieved Clinical Evidence"
        )

        evidence = diagnosis_data.get(
            "retrieved_evidence",
            diagnosis_data.get(
                "evidence",
                []
            )
        )

        if not evidence:

            st.info(
                "No clinical evidence retrieved for this case."
            )

            return

        for i, doc in enumerate(
            evidence,
            1
        ):

            with st.expander(
                f"Evidence Source {i}"
            ):

                st.write(
                    f"**Disease:** "
                    f"{doc.get('disease', 'Unknown')}"
                )

                st.write(
                    f"**Source:** "
                    f"{doc.get('source', 'Medical Knowledge Base')}"
                )

                st.write(
                    "**Content:**"
                )

                content = doc.get(
                    "content",
                    doc.get(
                        "description",
                        "No content available"
                    )
                )

                st.write(
                    content
                )


    # ========================================================
    # BACKEND HEALTH CHECK
    # ========================================================

    def check_backend(self):

        try:

            health_url = (
                f"{self.backend_url}/health"
            )

            response = requests.get(
                health_url,
                timeout=20
            )

            if response.status_code == 200:

                return True, response.json()

            return False, (
                f"HTTP {response.status_code}: "
                f"{response.text}"
            )

        except requests.exceptions.Timeout:

            return False, (
                "Render backend timed out after "
                "20 seconds. The service may still "
                "be waking up."
            )

        except requests.exceptions.ConnectionError as e:

            return False, (
                f"Connection error: {str(e)}"
            )

        except requests.exceptions.RequestException as e:

            return False, (
                f"Request error: {str(e)}"
            )

        except Exception as e:

            return False, (
                f"Unexpected error: {str(e)}"
            )


    # ========================================================
    # DIAGNOSIS REQUEST
    # ========================================================

    def request_diagnosis(
        self,
        patient_data
    ):

        try:

            diagnosis_url = (
                f"{self.backend_url}/diagnose"
            )

            payload = {
                "patient_data": patient_data
            }

            response = requests.post(
                diagnosis_url,
                json=payload,
                timeout=120
            )

            if response.status_code == 200:

                return True, response.json()

            return False, (
                f"Backend returned HTTP "
                f"{response.status_code}: "
                f"{response.text}"
            )

        except requests.exceptions.Timeout:

            return False, (
                "Diagnosis request timed out after "
                "120 seconds."
            )

        except requests.exceptions.ConnectionError as e:

            return False, (
                f"Could not connect to Render: "
                f"{str(e)}"
            )

        except requests.exceptions.RequestException as e:

            return False, (
                f"Request error: {str(e)}"
            )

        except Exception as e:

            return False, (
                f"Unexpected error: {str(e)}"
            )


    # ========================================================
    # MAIN
    # ========================================================

    def main(self):

        self.render_header()

        self.render_sidebar()

        patient_data = (
            self.render_patient_input_form()
        )

        if patient_data:

            with st.spinner(
                "🔄 Connecting to Render backend..."
            ):

                backend_available, health_result = (
                    self.check_backend()
                )

            # ------------------------------------------------
            # Backend available
            # ------------------------------------------------

            if backend_available:

                st.success(
                    "✅ Backend connected successfully."
                )

                with st.spinner(
                    "🤖 AI is analyzing patient data "
                    "and retrieving clinical evidence..."
                ):

                    success, result = (
                        self.request_diagnosis(
                            patient_data
                        )
                    )

                if success:

                    diagnosis_data = result

                    diagnosis_data[
                        "patient_id"
                    ] = patient_data[
                        "patient_id"
                    ]

                else:

                    st.error(
                        f"❌ Diagnosis request failed:\n\n"
                        f"{result}"
                    )

                    st.info(
                        "The backend is reachable, "
                        "but the diagnosis request "
                        "returned an error."
                    )

                    return

            # ------------------------------------------------
            # Backend unavailable
            # ------------------------------------------------

            else:

                st.error(
                    "❌ Backend server could not be reached."
                )

                st.code(
                    str(health_result)
                )

                st.warning(
                    "Demo mode is being used so the "
                    "interface can still be demonstrated."
                )

                diagnosis_data = (
                    self.get_demo_diagnosis(
                        patient_data
                    )
                )

            # ------------------------------------------------
            # Store history
            # ------------------------------------------------

            st.session_state.current_patient = (
                patient_data["patient_id"]
            )

            st.session_state.diagnosis_history.append(
                diagnosis_data
            )

            # ------------------------------------------------
            # Display results
            # ------------------------------------------------

            tab1, tab2, tab3 = st.tabs(
                [
                    "Diagnosis Results",
                    "Clinical Evidence",
                    "Export Report"
                ]
            )

            with tab1:

                self.render_diagnosis_results(
                    diagnosis_data
                )

            with tab2:

                self.render_evidence_retrieval(
                    diagnosis_data
                )

            with tab3:

                self.render_export_section(
                    diagnosis_data
                )


    # ========================================================
    # DEMO DIAGNOSIS
    # ========================================================

    def get_demo_diagnosis(
        self,
        patient_data
    ):

        symptoms = patient_data[
            "symptoms"
        ]

        symptom_text = (
            " ".join(symptoms)
            .lower()
        )

        demo_diagnoses = []

        # ----------------------------------------------------
        # Huntington's Disease
        # ----------------------------------------------------

        if any(
            word in symptom_text
            for word in [
                "movement",
                "involuntary",
                "chorea"
            ]
        ):

            demo_diagnoses.append(
                {
                    "disease": "Huntington's Disease",
                    "probability": 0.85,
                    "confidence": 0.82,
                    "supporting_symptoms": [
                        "Involuntary movements",
                        "Cognitive decline"
                    ],
                    "conflicting_factors": [
                        "No family history reported"
                    ]
                }
            )

        # ----------------------------------------------------
        # ALS
        # ----------------------------------------------------

        if any(
            word in symptom_text
            for word in [
                "weakness",
                "muscle",
                "atrophy"
            ]
        ):

            demo_diagnoses.append(
                {
                    "disease": (
                        "Amyotrophic Lateral "
                        "Sclerosis (ALS)"
                    ),
                    "probability": 0.65,
                    "confidence": 0.60,
                    "supporting_symptoms": [
                        "Muscle weakness",
                        "Progressive symptoms"
                    ],
                    "conflicting_factors": [
                        "No EMG confirmation"
                    ]
                }
            )

        # ----------------------------------------------------
        # No matching disease
        # ----------------------------------------------------

        if not demo_diagnoses:

            demo_diagnoses.append(
                {
                    "disease": (
                        "Further Investigation Required"
                    ),
                    "probability": 0.30,
                    "confidence": 0.25,
                    "supporting_symptoms": [
                        "Multiple non-specific symptoms"
                    ],
                    "conflicting_factors": [
                        "Symptoms do not match typical "
                        "rare disease patterns"
                    ]
                }
            )

        return {

            "patient_id": patient_data[
                "patient_id"
            ],

            "overall_confidence": max(
                [
                    d["confidence"]
                    for d in demo_diagnoses
                ]
            ),

            "possible_diagnoses": (
                demo_diagnoses
            ),

            "recommendations": [

                "Consult with neurology specialist",

                "Consider genetic testing "
                "if family history is positive",

                "MRI brain for structural abnormalities",

                "Complete blood work "
                "and metabolic panel"
            ],

            "evidence": [

                {
                    "disease": (
                        "Huntington's Disease"
                    ),
                    "source": (
                        "Orphanet Database"
                    ),
                    "content": (
                        "Neurodegenerative genetic "
                        "disorder characterized by "
                        "chorea, cognitive decline, "
                        "and psychiatric symptoms. "
                        "Onset typically occurs in "
                        "middle age."
                    )
                },

                {
                    "disease": (
                        "Amyotrophic Lateral Sclerosis"
                    ),
                    "source": (
                        "PubMed Clinical Guidelines"
                    ),
                    "content": (
                        "Progressive neurodegenerative "
                        "disease affecting motor neurons. "
                        "Diagnosis requires appropriate "
                        "clinical assessment."
                    )
                }
            ],

            "reasoning_chain": (
                f"Patient presented with symptoms: "
                f"{', '.join(symptoms)}. "
                "The system analyzed symptom patterns "
                "against the rare disease knowledge "
                "base. Possible differentials were "
                "identified based on symptom overlap "
                "and clinical presentation patterns."
            )
        }


    # ========================================================
    # EXPORT SECTION
    # ========================================================

    def render_export_section(
        self,
        diagnosis_data
    ):

        st.header(
            "📄 Export Diagnosis Report"
        )

        report_json = json.dumps(
            diagnosis_data,
            indent=2
        )

        col1, col2, col3 = st.columns(3)

        # ----------------------------------------------------
        # JSON
        # ----------------------------------------------------

        with col1:

            st.download_button(
                label="📥 Download JSON Report",
                data=report_json,
                file_name=(
                    f"diagnosis_report_"
                    f"{diagnosis_data.get('patient_id', 'unknown')}.json"
                ),
                mime="application/json"
            )

        # ----------------------------------------------------
        # CSV
        # ----------------------------------------------------

        with col2:

            st.download_button(
                label="📥 Download CSV Summary",
                data=self.generate_csv_summary(
                    diagnosis_data
                ),
                file_name=(
                    f"diagnosis_summary_"
                    f"{diagnosis_data.get('patient_id', 'unknown')}.csv"
                ),
                mime="text/csv"
            )

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        with col3:

            if st.button(
                "🖨️ Print Report"
            ):

                st.success(
                    "Report ready for printing. "
                    "Use your browser's print function."
                )

        st.subheader(
            "Report Preview"
        )

        st.json(
            diagnosis_data
        )


    # ========================================================
    # CSV GENERATION
    # ========================================================

    def generate_csv_summary(
        self,
        diagnosis_data
    ):

        import io
        import csv

        output = io.StringIO()

        writer = csv.writer(
            output
        )

        writer.writerow(
            [
                "Category",
                "Details"
            ]
        )

        writer.writerow(
            [
                "Patient ID",
                diagnosis_data.get(
                    "patient_id",
                    "Unknown"
                )
            ]
        )

        writer.writerow(
            [
                "Overall Confidence",
                f"{diagnosis_data.get('overall_confidence', 0):.1%}"
            ]
        )

        writer.writerow([])

        writer.writerow(
            [
                "Possible Diagnoses",
                "Probability",
                "Confidence"
            ]
        )

        for diagnosis in diagnosis_data.get(
            "possible_diagnoses",
            []
        ):

            writer.writerow(
                [
                    diagnosis.get(
                        "disease",
                        diagnosis.get(
                            "disease_name",
                            "Unknown"
                        )
                    ),

                    f"{diagnosis.get(
                        'probability',
                        diagnosis.get(
                            'confidence_score',
                            0
                        )
                    ):.1%}",

                    f"{diagnosis.get(
                        'confidence',
                        diagnosis.get(
                            'confidence_score',
                            0
                        )
                    ):.1%}"
                ]
            )

        return output.getvalue()


# ============================================================
# APPLICATION START
# ============================================================

if __name__ == "__main__":

    frontend = ClinicalCDSSFrontend()

    frontend.main()
```
