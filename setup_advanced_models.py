#!/usr/bin/env python3
"""
Setup script for advanced embedding models
Downloads and configures all required models for neural processing
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_gpu_availability():
    """Check if GPU is available for PyTorch"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("💻 Using CPU for neural processing")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed yet")
        return False

def install_pytorch():
    """Install PyTorch with appropriate configuration"""
    gpu_available = check_gpu_availability()
    
    if gpu_available:
        # Install GPU version
        command = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
        description = "Installing PyTorch with CUDA support"
    else:
        # Install CPU version
        command = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
        description = "Installing PyTorch (CPU version)"
    
    return run_command(command, description)

def install_transformers():
    """Install transformers and related packages"""
    commands = [
        ("pip install transformers==4.30.0", "Installing Transformers library"),
        ("pip install sentence-transformers==2.2.2", "Installing Sentence Transformers"),
        ("pip install datasets==2.12.0", "Installing Datasets library"),
        ("pip install tokenizers==0.13.3", "Installing Tokenizers"),
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            return False
    return True

def install_ml_packages():
    """Install machine learning packages"""
    commands = [
        ("pip install scikit-learn==1.3.0", "Installing scikit-learn"),
        ("pip install umap-learn==0.5.3", "Installing UMAP"),
        ("pip install hdbscan==0.8.29", "Installing HDBSCAN"),
        ("pip install faiss-cpu==1.7.4", "Installing FAISS (CPU)"),
        ("pip install networkx==3.1", "Installing NetworkX"),
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            return False
    return True

def install_nlp_packages():
    """Install NLP packages"""
    commands = [
        ("pip install spacy==3.6.0", "Installing spaCy"),
        ("pip install textstat==0.7.3", "Installing TextStat"),
        ("pip install nltk==3.8.1", "Installing NLTK"),
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            return False
    
    # Download spaCy model
    return run_command("python -m spacy download en_core_web_sm", "Downloading spaCy English model")

def download_pretrained_models():
    """Download and cache pretrained models"""
    print("🤖 Pre-downloading neural models...")
    
    try:
        from sentence_transformers import SentenceTransformer
        from transformers import AutoTokenizer, AutoModel
        
        # Download sentence transformer model
        print("📥 Downloading sentence transformer model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Sentence transformer model cached")
        
        # Download transformer model
        print("📥 Downloading transformer model...")
        tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
        model = AutoModel.from_pretrained('microsoft/DialoGPT-medium')
        print("✅ Transformer model cached")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading models: {e}")
        return False

def create_model_config():
    """Create configuration file for models"""
    config = {
        "neural_models": {
            "sentence_transformer": "all-MiniLM-L6-v2",
            "transformer_model": "microsoft/DialoGPT-medium",
            "embedding_model": "text-embedding-3-small",
            "query_expansion_model": "gpt2"
        },
        "processing_config": {
            "enable_neural_chunking": True,
            "enable_hybrid_embeddings": True,
            "enable_semantic_clustering": True,
            "enable_adaptive_learning": True,
            "batch_size": 32,
            "max_chunk_size": 1000
        },
        "hardware": {
            "gpu_available": check_gpu_availability(),
            "device": "cuda" if check_gpu_availability() else "cpu"
        }
    }
    
    try:
        import json
        with open('neural_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        print("✅ Created neural_config.json")
        return True
    except Exception as e:
        print(f"❌ Error creating config: {e}")
        return False

def verify_installation():
    """Verify all components are installed correctly"""
    print("🔍 Verifying installation...")
    
    try:
        # Test PyTorch
        import torch
        print(f"✅ PyTorch {torch.__version__} - GPU: {torch.cuda.is_available()}")
        
        # Test Transformers
        import transformers
        print(f"✅ Transformers {transformers.__version__}")
        
        # Test Sentence Transformers
        import sentence_transformers
        print(f"✅ Sentence Transformers {sentence_transformers.__version__}")
        
        # Test FAISS
        import faiss
        print(f"✅ FAISS available")
        
        # Test spaCy
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print(f"✅ spaCy {spacy.__version__} with English model")
        
        # Test other packages
        import umap
        import hdbscan
        import networkx
        import textstat
        print("✅ All ML packages available")
        
        return True
        
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Setting up Advanced Neural Embedding System")
    print("=" * 60)
    
    steps = [
        ("Installing PyTorch", install_pytorch),
        ("Installing Transformers", install_transformers),
        ("Installing ML packages", install_ml_packages),
        ("Installing NLP packages", install_nlp_packages),
        ("Downloading pretrained models", download_pretrained_models),
        ("Creating configuration", create_model_config),
        ("Verifying installation", verify_installation)
    ]
    
    failed_steps = []
    
    for step_name, step_function in steps:
        print(f"\n📋 {step_name}...")
        if not step_function():
            failed_steps.append(step_name)
            print(f"⚠️  {step_name} failed but continuing...")
    
    print("\n" + "=" * 60)
    if not failed_steps:
        print("🎉 Setup completed successfully!")
        print("\n🚀 Your RAG system now supports:")
        print("   • Neural network-based chunking")
        print("   • Hybrid multi-model embeddings")
        print("   • Advanced semantic clustering")
        print("   • FAISS vector indexing")
        print("   • Adaptive learning algorithms")
        print("   • Real-time quality prediction")
        print("\n💡 Run your RAG processor to see the improvements!")
    else:
        print(f"⚠️  Setup completed with {len(failed_steps)} issues:")
        for step in failed_steps:
            print(f"   • {step}")
        print("\n💡 The system will fall back to available components.")
    
    print("\n📖 Next steps:")
    print("   1. Run: python web_server.py")
    print("   2. Upload documents to see neural processing in action")
    print("   3. Check the enhanced results and performance metrics")

if __name__ == "__main__":
    main()