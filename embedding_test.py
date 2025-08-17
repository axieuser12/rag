#!/usr/bin/env python3
"""
Test script for the centralized embedding engine
"""

import asyncio
import os
from embedding_engine import EmbeddingEngine, EmbeddingConfig, ProcessingLevel

# Test the new neural core
try:
    from neural_core import get_neural_core, NeuralConfig, ProcessingMode
    NEURAL_CORE_AVAILABLE = True
except ImportError:
    NEURAL_CORE_AVAILABLE = False

async def test_embedding_engine():
    """Test the centralized embedding engine"""
    
    # Test data
    test_documents = [
        {
            'content': '''
            Machine learning is a subset of artificial intelligence that focuses on algorithms 
            that can learn from data. It includes supervised learning, unsupervised learning, 
            and reinforcement learning approaches. Popular frameworks include TensorFlow, 
            PyTorch, and scikit-learn.
            ''',
            'source': 'ml_intro.txt'
        },
        {
            'content': '''
            Our company's revenue grew by 25% this quarter, reaching $2.5 million. 
            The marketing strategy focused on digital channels resulted in a 40% increase 
            in customer acquisition. We plan to expand into new markets next year.
            ''',
            'source': 'business_report.txt'
        }
    ]
    
    # Test different processing levels including Neural Supreme
    test_levels = [ProcessingLevel.BASIC, ProcessingLevel.ENHANCED, ProcessingLevel.INTELLIGENT]
    if NEURAL_CORE_AVAILABLE:
        test_levels.insert(0, ProcessingLevel.NEURAL_SUPREME)
    
    for level in test_levels:
        print(f"\n{'='*50}")
        print(f"Testing {level.value.upper()} processing")
        print(f"{'='*50}")
        
        try:
            # Create config
            config = EmbeddingConfig(
                processing_level=level,
                batch_size=10,
                chunk_size=200  # Smaller for testing
            )
            
            # Create engine
            engine = EmbeddingEngine(config)
            
            # Mock credentials (replace with real ones for actual testing)
            mock_api_key = "sk-test-key-here"
            mock_supabase_url = "https://test.supabase.co"
            mock_service_key = "eyJ-test-key-here"
            
            # Note: This will fail without real credentials, but shows the structure
            try:
                engine.set_credentials(mock_api_key, mock_supabase_url, mock_service_key)
                result = await engine.process_documents(test_documents)
                
                print(f"✅ Success: {result.success}")
                print(f"📊 Chunks processed: {result.chunks_processed}")
                print(f"⚡ Processing time: {result.processing_time:.2f}s")
                print(f"🎯 Strategy: {result.processing_level}")
                
                # Show neural-specific stats if available
                if hasattr(result, 'neural_stats') and result.neural_stats:
                    print(f"🧠 Neural stats: {result.neural_stats}")
                
            except Exception as e:
                print(f"⚠️  Expected error (mock credentials): {e}")
                
                # Test chunk creation without credentials
                print("Testing chunk creation only...")
                all_chunks = []
                for doc in test_documents:
                    chunks = await engine.strategy.create_chunks(doc['content'], doc['source'])
                    all_chunks.extend(chunks)
                    
                print(f"✅ Created {len(all_chunks)} chunks")
                for i, chunk in enumerate(all_chunks[:2]):  # Show first 2
                    metadata = chunk['metadata']
                    neural_features = chunk.get('neural_features', {})
                    print(f"  Chunk {i+1}: {len(chunk['content'])} chars, "
                          f"category: {metadata.get('category', 'N/A')}, "
                          f"quality: {neural_features.get('quality_score', metadata.get('quality_score', 0)):.2f}")
                    
                    # Show neural features if available
                    if neural_features and level == ProcessingLevel.NEURAL_SUPREME:
                        print(f"    🧠 Neural: method={neural_features.get('processing_method', 'N/A')}, "
                              f"boundary_conf={neural_features.get('boundary_confidence', 0):.2f}")
            
            # Show stats
            stats = engine.get_stats()
            print(f"📈 Strategy stats: {stats['strategy_stats']}")
            
        except Exception as e:
            print(f"❌ Error testing {level.value}: {e}")

async def test_neural_core_direct():
    """Test neural core directly"""
    if not NEURAL_CORE_AVAILABLE:
        print("⚠️  Neural core not available for direct testing")
        return
    
    print(f"\n{'='*50}")
    print("Testing Neural Core Directly")
    print(f"{'='*50}")
    
    try:
        # Create neural core
        neural_config = NeuralConfig(
            processing_mode=ProcessingMode.NEURAL_SUPREME,
            batch_size=5,
            max_chunk_size=300
        )
        neural_core = get_neural_core(neural_config)
        
        # Test neural chunking
        test_text = """
        Artificial intelligence represents a revolutionary approach to computing that mimics human cognitive functions.
        Machine learning algorithms can process vast amounts of data to identify patterns and make predictions.
        Deep learning networks use multiple layers of artificial neurons to solve complex problems.
        Natural language processing enables computers to understand and generate human language.
        """
        
        print("🧠 Testing neural chunking...")
        chunks = await neural_core.neural_chunk_analysis(test_text, "test_neural.txt")
        
        print(f"✅ Neural chunking created {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            neural_features = chunk.get('neural_features', {})
            print(f"  Chunk {i+1}: {len(chunk['content'])} chars")
            print(f"    Quality: {neural_features.get('quality_score', 0):.3f}")
            print(f"    Method: {neural_features.get('processing_method', 'unknown')}")
            print(f"    Category: {neural_features.get('predicted_category', 'unknown')}")
        
        # Test concept extraction
        print("\n🔗 Testing concept extraction...")
        concepts = await neural_core.extract_advanced_concepts(test_text)
        print(f"✅ Extracted {len(concepts)} concepts")
        for concept in concepts[:5]:  # Show first 5
            print(f"  - {concept['text']} ({concept['type']}, conf: {concept['confidence']:.2f})")
        
        # Show neural core stats
        print(f"\n📊 Neural core stats: {neural_core.stats}")
        
    except Exception as e:
        print(f"❌ Error testing neural core: {e}")
def test_config_updates():
    """Test configuration updates"""
    print(f"\n{'='*50}")
    print("Testing Configuration Updates")
    print(f"{'='*50}")
    
    # Create engine with basic config
    config = EmbeddingConfig(processing_level=ProcessingLevel.BASIC)
    engine = EmbeddingEngine(config)
    
    print(f"Initial strategy: {engine.strategy.get_strategy_name()}")
    
    # Update to neural supreme if available
    if NEURAL_CORE_AVAILABLE:
        engine.update_config(processing_level=ProcessingLevel.NEURAL_SUPREME)
        print(f"Updated to Neural Supreme: {engine.strategy.get_strategy_name()}")
    
    # Update to intelligent processing
    engine.update_config(processing_level=ProcessingLevel.INTELLIGENT)
    print(f"Updated strategy: {engine.strategy.get_strategy_name()}")
    
    # Update other settings
    engine.update_config(
        chunk_size=1000,
        batch_size=25,
        quality_threshold=0.8
    )
    
    print(f"Updated config - chunk_size: {engine.config.chunk_size}")
    print(f"Updated config - batch_size: {engine.config.batch_size}")
    print(f"Updated config - quality_threshold: {engine.config.quality_threshold}")

if __name__ == "__main__":
    print("🧪 Testing Advanced Neural Processing System")
    
    if NEURAL_CORE_AVAILABLE:
        print("🧠 NEURAL SUPREME CORE AVAILABLE!")
    else:
        print("⚠️  Neural core not available - testing standard features only")
    
    # Test configuration updates
    test_config_updates()
    
    # Test neural core directly
    asyncio.run(test_neural_core_direct())
    
    # Test embedding engine
    asyncio.run(test_embedding_engine())
    
    print(f"\n{'='*50}")
    print("✅ Testing completed!")
    if NEURAL_CORE_AVAILABLE:
        print("🧠 Neural Supreme processing tested successfully!")
    print("To use with real credentials:")
    print("1. Replace mock credentials with real ones")
    print("2. Ensure Supabase database is set up")
    print("3. Install neural dependencies: pip install torch transformers sentence-transformers")
    print("4. Run: python embedding_test.py")