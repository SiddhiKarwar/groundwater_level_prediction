"""
Test RAG Chatbot Setup
Quick verification script
"""

import sys
import os

print("🔍 Testing RAG Chatbot Setup...\n")

# Test 1: Check Python version
print("1️⃣ Python Version:")
print(f"   {sys.version}")
if sys.version_info < (3, 10):
    print("   ⚠️  Python 3.10+ recommended")
else:
    print("   ✅ Python version OK\n")

# Test 2: Check .env file
print("2️⃣ Environment Variables:")
try:
    from dotenv import load_dotenv
    load_dotenv()
    
    groq_key = os.getenv("GROQ_API_TOKEN")
    if groq_key:
        print(f"   ✅ GROQ_API_TOKEN found (length: {len(groq_key)})")
    else:
        print("   ❌ GROQ_API_TOKEN not found in .env")
    print()
except ImportError:
    print("   ❌ python-dotenv not installed")
    print("   Install: pip install python-dotenv\n")

# Test 3: Check required packages
print("3️⃣ Required Packages:")
required_packages = [
    "langchain",
    "langchain_groq",
    "langchain_chroma",
    "langchain_community",
    "pypdf",
    "chromadb",
    "dotenv"
]

for package in required_packages:
    try:
        __import__(package)
        print(f"   ✅ {package}")
    except ImportError:
        print(f"   ❌ {package} - Not installed")

print()

# Test 4: Check Ollama
print("4️⃣ Embeddings (Local - Sentence Transformers):")
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": "cpu"})
    # Try to create a test embedding
    test_result = embeddings.embed_query("test")
    if test_result and isinstance(test_result, list) and len(test_result) > 0:
        print("   ✅ HuggingFace embeddings working")
        print(f"   ✅ Model: {model_name}")
    else:
        print("   ⚠️  Embedding vector is empty")
except Exception as e:
    print(f"   ❌ Embeddings failed: {str(e)}")
    print("   Reinstall: pip install sentence-transformers langchain-community")

print()

# Test 5: Check Groq API
print("5️⃣ Groq API Connection:")
try:
    from langchain_groq import ChatGroq
    groq_key = os.getenv("GROQ_API_TOKEN")
    if groq_key:
        llm = ChatGroq(
            groq_api_key=groq_key,
            model_name="llama-3.1-8b-instant",  # Updated to new supported model
            temperature=0.7
        )
        print("   ✅ Groq LLM initialized")
        
        # Try a simple query
        try:
            response = llm.invoke("Say 'Hello'")
            print("   ✅ Groq API connection successful")
            print(f"   Response: {response.content}")
        except Exception as e:
            print(f"   ⚠️  Groq API call failed: {str(e)}")
    else:
        print("   ❌ GROQ_API_TOKEN not found")
except Exception as e:
    print(f"   ❌ Groq setup failed: {str(e)}")

print("\n" + "="*50)
print("✨ Setup Check Complete!")
print("="*50)
