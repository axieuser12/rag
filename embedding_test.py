#!/usr/bin/env python3
"""
Test script for the centralized embedding engine
"""

import asyncio
import os
from embedding_engine import EmbeddingEngine, EmbeddingConfig, ProcessingLevel

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
    
    # Test different processing levels
    for level in [ProcessingLevel.BASIC, ProcessingLevel.ENHANCED, ProcessingLevel.INTELLIGENT]:
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
                    print(f"  Chunk {i+1}: {len(chunk['content'])} chars, "
                          f"category: {metadata.get('category', 'N/A')}, "
                          f"quality: {metadata.get('quality_score', 0):.2f}")
            
            # Show stats
            stats = engine.get_stats()
            print(f"📈 Strategy stats: {stats['strategy_stats']}")
            
        except Exception as e:
            print(f"❌ Error testing {level.value}: {e}")

def test_config_updates():
    """Test configuration updates"""
    print(f"\n{'='*50}")
    print("Testing Configuration Updates")
    print(f"{'='*50}")
    
    # Create engine with basic config
    config = EmbeddingConfig(processing_level=ProcessingLevel.BASIC)
    engine = EmbeddingEngine(config)
    
    print(f"Initial strategy: {engine.strategy.get_strategy_name()}")
    
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
    print("🧪 Testing Centralized Embedding Engine")
    
    # Test configuration updates
    test_config_updates()
    
    # Test embedding engine
    asyncio.run(test_embedding_engine())
    
    print(f"\n{'='*50}")
    print("✅ Testing completed!")
    print("To use with real credentials:")
    print("1. Replace mock credentials with real ones")
    print("2. Ensure Supabase database is set up")
    print("3. Run: python embedding_test.py")