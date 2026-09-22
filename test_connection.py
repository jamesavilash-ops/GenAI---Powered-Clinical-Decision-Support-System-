import requests
import json

def test_backend():
    try:
        # Test basic connection
        response = requests.get("http://localhost:8000/", timeout=5)
        print(f"✅ Backend is running: {response.status_code}")
        print(f"Response: {response.json()}")
        
        # Test diagnosis endpoint
        test_patient = {
            "patient_id": "TEST_001",
            "symptoms": "muscle weakness, difficulty swallowing",
            "age": 58,
            "gender": "Female",
            "medical_history": "No family history"
        }
        
        response = requests.post("http://localhost:8000/diagnose", json=test_patient, timeout=10)
        print(f"✅ Diagnosis endpoint working: {response.status_code}")
        print("Diagnosis result:")
        print(json.dumps(response.json(), indent=2))
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to backend. Make sure it's running on port 8000")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_backend()