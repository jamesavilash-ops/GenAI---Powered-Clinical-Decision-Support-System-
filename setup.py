import os
import subprocess
import sys

def run_command(command):
    """Run a shell command and check return code"""
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {command}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {command}")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("🚀 Setting up Clinical CDSS...")
    
    # Create directories
    directories = ['backend', 'frontend', 'data', 'chroma_db']
    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)
        print(f"✅ Created directory: {dir_name}")
    
    print("\n📦 Installing backend dependencies...")
    run_command("cd backend && pip install -r requirements.txt")
    
    print("\n📦 Installing frontend dependencies...") 
    run_command("cd frontend && pip install -r requirements.txt")
    
    print("\n🎉 Setup complete!")
    print("\n📋 Next steps:")
    print("1. Add your OpenAI API key to backend/app.py")
    print("2. Start backend: cd backend && python app.py")
    print("3. Start frontend: cd frontend && streamlit run app.py")

if __name__ == "__main__":
    main()