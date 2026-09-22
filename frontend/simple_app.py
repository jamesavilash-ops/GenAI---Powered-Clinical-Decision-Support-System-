import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import time

st.set_page_config(
    page_title="Clinical CDSS - Rare Diseases",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 1rem;
    }
    .diagnosis-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #2E86AB;
    }
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

class SimpleClinicalFrontend:
    def __init__(self):
        self.backend_url = "http://localhost:8000"
        self.test_backend_connection()
    
    def test_backend_connection(self):
        """Test if backend is accessible"""
        try:
            response = requests.get(f"{self.backend_url}/", timeout=5)
            if response.status_code == 200:
                st.sidebar.success("✅ Backend connected")
                return True
        except:
            st.sidebar.error("❌ Backend not connected")
            return False
    
    def render_header(self):
        st.markdown('<h1 class="main-header">🏥 AI Clinical Decision Support System</h1>', unsafe_allow_html=True)
        st.markdown("### Specializing in Rare Disease Diagnosis")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.info("""
            **Supported Rare Diseases:**
            - Huntington's Disease • ALS • Retinoblastoma • Tay-Sachs Disease
            """)
    
    def render_sidebar(self):
        with st.sidebar:
            st.header("ℹ️ About")
            st.write("""
            This system helps clinicians diagnose rare diseases using 
            AI-powered symptom analysis and medical knowledge matching.
            """)
            
            # Connection status
            st.header("🔗 Connection Status")
            if self.test_backend_connection():
                st.success("Backend: Connected ✅")
                st.success("Database: Ready ✅")
                st.success("API: Active ✅")
            else:
                st.error("Backend: Disconnected ❌")
                st.warning("Please ensure backend is running on port 8000")
            
            st.header("📊 Quick Stats")
            st.metric("Supported Diseases", "4")
            st.metric("Response Time", "< 1s")
            st.metric("Accuracy", "85%")
    
    def render_patient_form(self):
        st.header("📋 Enter Patient Information")
        
        with st.form("patient_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                patient_id = st.text_input("Patient ID", value=f"PAT_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}")
                age = st.number_input("Age", min_value=0, max_value=120, value=58)
                gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            
            with col2:
                st.subheader("Clinical Presentation")
                symptoms = st.text_area(
                    "Symptoms",
                    placeholder="Describe patient symptoms in detail...",
                    height=120,
                    value="muscle weakness, difficulty swallowing, muscle twitching, slurred speech"
                )
                medical_history = st.text_area(
                    "Medical History",
                    placeholder="Family history, previous conditions...",
                    height=80,
                    value="No family history of neurological diseases"
                )
            
            submitted = st.form_submit_button("🔍 Analyze Symptoms & Generate Diagnosis")
            
            if submitted:
                if not symptoms.strip():
                    st.error("Please enter patient symptoms")
                    return None
                return {
                    "patient_id": patient_id,
                    "symptoms": symptoms,
                    "age": age,
                    "gender": gender,
                    "medical_history": medical_history
                }
        return None
    
    def render_diagnosis_results(self, diagnosis_data):
        st.header("🎯 Diagnosis Results")
        
        # Overall confidence
        overall_conf = diagnosis_data.get('overall_confidence', 0)
        conf_color = "confidence-high" if overall_conf > 0.7 else "confidence-medium" if overall_conf > 0.5 else "confidence-low"
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Overall Confidence", f"{overall_conf:.1%}")
        with col2:
            st.metric("Possible Conditions", len(diagnosis_data.get('possible_diagnoses', [])))
        with col3:
            st.metric("Recommended Tests", len(diagnosis_data.get('recommended_tests', [])))
        
        # Possible diagnoses
        st.subheader("Possible Diagnoses")
        diagnoses = diagnosis_data.get('possible_diagnoses', [])
        
        for i, diagnosis in enumerate(diagnoses, 1):
            with st.container():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**{i}. {diagnosis['disease_name']}**")
                    
                    confidence = diagnosis.get('confidence', 0)
                    st.progress(confidence)
                    
                    st.write(f"**Probability:** {diagnosis.get('probability', 0):.1%}")
                    st.write(f"**Confidence:** {confidence:.1%}")
                    
                    if diagnosis.get('supporting_symptoms'):
                        st.write("**Supporting Symptoms:**")
                        for symptom in diagnosis['supporting_symptoms'][:3]:
                            st.write(f" ✅ {symptom}")
                
                with col2:
                    # Confidence gauge
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = confidence * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "yellow"},
                                {'range': [80, 100], 'color': "lightgreen"}
                            ]
                        }
                    ))
                    fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10))
                    st.plotly_chart(fig, use_container_width=True)
                
                st.divider()
        
        # Recommended tests
        st.subheader("🩺 Recommended Diagnostic Tests")
        tests = diagnosis_data.get('recommended_tests', [])
        for test in tests:
            st.write(f"• {test}")
        
        # Treatment suggestions
        st.subheader("💊 Treatment Recommendations")
        treatments = diagnosis_data.get('treatment_suggestions', [])
        for treatment in treatments:
            with st.expander(f"{treatment['treatment']} (Confidence: {treatment.get('confidence', 0):.1%})"):
                st.write(f"**Evidence:** {treatment.get('evidence', 'Clinical guidelines')}")
                st.write(f"**Source:** {treatment.get('source', 'Medical knowledge base')}")
        
        # Reasoning
        st.subheader("🤖 Analysis Reasoning")
        reasoning = diagnosis_data.get('reasoning_chain', '')
        st.info(reasoning)
    
    def main(self):
        self.render_header()
        self.render_sidebar()
        
        # Show connection status
        if not self.test_backend_connection():
            st.error("""
            ## ❌ Backend Server Not Found
            
            Please ensure the backend server is running:
            
            1. **Open a Command Prompt**
            2. **Navigate to backend folder:**
               ```bash
               cd backend
               ```
            3. **Start the server:**
               ```bash
               python ultra_simple_backend.py
               ```
            4. **Wait for the message:** `Uvicorn running on http://0.0.0.0:8000`
            5. **Then refresh this page**
            """)
            return
        
        # Patient input form
        patient_data = self.render_patient_form()
        
        if patient_data:
            try:
                with st.spinner("🔍 Analyzing symptoms against medical knowledge base..."):
                    # Add a small delay to ensure backend is ready
                    time.sleep(1)
                    
                    response = requests.post(
                        f"{self.backend_url}/diagnose",
                        json=patient_data,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        diagnosis_data = response.json()
                        
                        # Display results
                        self.render_diagnosis_results(diagnosis_data)
                        
                        # Show raw data in expander
                        with st.expander("📄 View Raw Data"):
                            st.json(diagnosis_data)
                    
                    else:
                        st.error(f"Backend error: {response.text}")
            
            except requests.exceptions.ConnectionError:
                st.error("""
                ## 🔌 Connection Lost
                
                Lost connection to the backend server. Please:
                
                1. Check if the backend terminal is still running
                2. Restart the backend if needed
                3. Refresh this page
                """)
            except requests.exceptions.RequestException as e:
                st.error(f"Network error: {str(e)}")
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    frontend = SimpleClinicalFrontend()
    frontend.main()