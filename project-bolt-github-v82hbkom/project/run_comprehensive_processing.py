#!/usr/bin/env python3
"""
Comprehensive RAG Processing Script
This script processes all txt files and uploads everything to Supabase
"""

import sys
import os
from improved_chunk_processor import ComprehensiveChunkProcessor

def main():
    print("🚀 Starting Comprehensive RAG Processing")
    print("=" * 50)
    
    # Initialize processor
    processor = ComprehensiveChunkProcessor()
    
    # Check if txt files exist
    txt_files = processor.get_all_txt_files()
    if not txt_files:
        print("❌ No txt files found in 'Txt File' folder!")
        print("Please make sure your txt files are in the 'Txt File' directory.")
        return
    
    print(f"📁 Found {len(txt_files)} txt files:")
    for file in txt_files:
        print(f"   - {os.path.basename(file)}")
    
    # Process all files
    print("\n📝 Processing files...")
    chunks = processor.process_all_files()
    
    if not chunks:
        print("❌ No chunks were created!")
        return
    
    # Show statistics
    chunk_types = {}
    sources = {}
    for chunk in chunks:
        chunk_type = chunk.get("chunk_type", "unknown")
        source = chunk.get("source", "unknown")
        chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
        sources[source] = sources.get(source, 0) + 1
    
    print(f"\n📊 Processing Statistics:")
    print(f"   Total chunks: {len(chunks)}")
    print(f"   Chunk types:")
    for chunk_type, count in chunk_types.items():
        print(f"     - {chunk_type}: {count}")
    print(f"   Sources:")
    for source, count in sources.items():
        print(f"     - {source}: {count}")
    
    # Save to file for review
    import json
    with open("comprehensive_chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Saved chunks to comprehensive_chunks.json for review")
    
    # Upload to Supabase
    print(f"\n🚀 Uploading to Supabase...")
    processor.upload_to_supabase(chunks)
    
    print(f"\n✅ Process completed successfully!")
    print(f"   All {len(chunks)} chunks have been uploaded to Supabase")
    print(f"   Your RAG system now has comprehensive coverage of all txt file content")
    
    # Test the system
    print(f"\n🔍 Testing the system...")
    try:
        from enhanced_query_system import EnhancedQuerySystem
        query_system = EnhancedQuerySystem()
        
        test_queries = [
            "What is Axie Studio?",
            "What are the prices?",
            "How can I contact you?"
        ]
        
        for query in test_queries:
            results = query_system.search_documents(query, limit=3)
            print(f"   Query: '{query}' -> {len(results)} results found")
        
        print(f"✅ System test completed successfully!")
        
    except Exception as e:
        print(f"⚠️  System test failed: {e}")
        print(f"   But the data upload was successful!")

if __name__ == "__main__":
    main()