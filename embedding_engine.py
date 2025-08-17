#!/usr/bin/env python3
"""
Centralized Embedding Engine
Unified system for all embedding processing with pluggable strategies
"""

import asyncio
import time
import json
import hashlib
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from enum import Enum
import tiktoken
from openai import OpenAI
from supabase import create_client

# Import the new neural core
try:
    from neural_core import get_neural_core, NeuralConfig, ProcessingMode, run_neural_supreme_processing
    NEURAL_CORE_AVAILABLE = True
except ImportError:
    NEURAL_CORE_AVAILABLE = False
    print("⚠️  Neural core not available - using standard processing")

class ProcessingLevel(Enum):
    """Available processing levels"""
    NEURAL_SUPREME = "neural_supreme"  # NEW: Ultimate neural processing
    NEURAL = "neural"
    ADAPTIVE = "adaptive"
    INTELLIGENT = "intelligent"
    ENHANCED = "enhanced"
    BASIC = "basic"

@dataclass
class EmbeddingConfig:
    """Unified configuration for embedding processing"""
    # Model settings
    openai_model: str = "text-embedding-3-small"
    sentence_transformer_model: str = "all-MiniLM-L6-v2"
    
    # Processing settings
    processing_level: ProcessingLevel = ProcessingLevel.ENHANCED
    batch_size: int = 50
    max_tokens: int = 8191
    chunk_size: int = 800
    chunk_overlap: int = 100
    
    # Quality settings
    quality_threshold: float = 0.7
    semantic_similarity_threshold: float = 0.85
    
    # Feature flags
    enable_hybrid_embeddings: bool = False
    enable_neural_chunking: bool = False
    enable_semantic_clustering: bool = False
    enable_adaptive_learning: bool = False
    enable_concept_graphs: bool = False
    enable_cross_references: bool = False
    
    # Performance settings
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit_delay: float = 0.1

@dataclass
class ChunkMetadata:
    """Standardized chunk metadata"""
    chunk_index: int
    source: str
    token_count: int
    quality_score: float = 0.7
    content_type: str = "text"
    category: str = "general"
    category_confidence: float = 0.5
    key_concepts: List[str] = None
    processing_method: str = "basic"
    created_at: float = None
    
    def __post_init__(self):
        if self.key_concepts is None:
            self.key_concepts = []
        if self.created_at is None:
            self.created_at = time.time()

@dataclass
class ProcessingResult:
    """Standardized processing result"""
    success: bool
    chunks_processed: int
    successful_embeddings: int
    failed_embeddings: int
    processing_time: float
    processing_level: str
    upload_stats: Dict[str, Any]
    performance_stats: Dict[str, Any]
    error_message: Optional[str] = None

class EmbeddingStrategy(ABC):
    """Abstract base class for embedding strategies"""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.stats = {
            'chunks_processed': 0,
            'embeddings_generated': 0,
            'processing_time': 0,
            'quality_scores': []
        }
    
    @abstractmethod
    async def create_chunks(self, text: str, source: str) -> List[Dict[str, Any]]:
        """Create chunks from text"""
        pass
    
    @abstractmethod
    async def generate_embeddings(self, chunks: List[Dict[str, Any]], openai_client: OpenAI) -> List[Tuple[Dict[str, Any], Optional[List[float]]]]:
        """Generate embeddings for chunks"""
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get strategy name"""
        pass

class BasicEmbeddingStrategy(EmbeddingStrategy):
    """Basic embedding strategy - simple chunking and OpenAI embeddings"""
    
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    
    def get_strategy_name(self) -> str:
        return "basic"
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        try:
            return len(self.encoding.encode(text))
        except Exception:
            return len(text) // 4
    
    def _simple_text_split(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Simple text splitting"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        separators = ["\n\n", "\n", ". ", " "]
        
        for separator in separators:
            if separator in text:
                parts = text.split(separator)
                current_chunk = ""
                
                for part in parts:
                    potential_chunk = current_chunk + (separator if current_chunk else '') + part
                    
                    if len(potential_chunk) <= chunk_size or not current_chunk:
                        current_chunk = potential_chunk
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                            # Add overlap
                            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                            current_chunk = overlap_text + separator + part
                        else:
                            current_chunk = part
                
                if current_chunk:
                    chunks.append(current_chunk)
                return chunks
        
        # Fallback: character-based splitting
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    async def create_chunks(self, text: str, source: str) -> List[Dict[str, Any]]:
        """Create basic chunks"""
        if not text.strip():
            return []
        
        chunks = self._simple_text_split(text, self.config.chunk_size, self.config.chunk_overlap)
        
        chunk_objects = []
        for i, chunk_text in enumerate(chunks):
            if len(chunk_text.strip()) < 50:
                continue
            
            metadata = ChunkMetadata(
                chunk_index=i,
                source=source,
                token_count=self._count_tokens(chunk_text),
                processing_method=self.get_strategy_name()
            )
            
            chunk_obj = {
                'content': chunk_text.strip(),
                'metadata': asdict(metadata)
            }
            chunk_objects.append(chunk_obj)
        
        self.stats['chunks_processed'] += len(chunk_objects)
        return chunk_objects
    
    async def generate_embeddings(self, chunks: List[Dict[str, Any]], openai_client: OpenAI) -> List[Tuple[Dict[str, Any], Optional[List[float]]]]:
        """Generate basic embeddings"""
        results = []
        batch_size = self.config.batch_size
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [chunk['content'] for chunk in batch]
            
            try:
                response = openai_client.embeddings.create(
                    input=texts,
                    model=self.config.openai_model
                )
                
                for j, chunk in enumerate(batch):
                    if j < len(response.data):
                        embedding = response.data[j].embedding
                        results.append((chunk, embedding))
                        self.stats['embeddings_generated'] += 1
                    else:
                        results.append((chunk, None))
                        
            except Exception as e:
                print(f"Error processing batch: {e}")
                for chunk in batch:
                    results.append((chunk, None))
            
            # Rate limiting
            if i + batch_size < len(chunks):
                await asyncio.sleep(self.config.rate_limit_delay)
        
        return results

class EnhancedEmbeddingStrategy(BasicEmbeddingStrategy):
    """Enhanced strategy with better chunking and retry logic"""
    
    def get_strategy_name(self) -> str:
        return "enhanced"
    
    def _semantic_text_split(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Enhanced semantic text splitting"""
        separators = [
            ('\n\n\n', 'major_section'),
            ('\n\n', 'paragraph'),
            ('.\n', 'sentence_break'),
            ('. ', 'sentence'),
            ('; ', 'clause'),
            (', ', 'comma'),
            (' ', 'word')
        ]
        
        chunks = []
        current_chunk = ""
        
        for separator, sep_type in separators:
            if separator in text:
                parts = text.split(separator)
                current_chunk = ""
                
                for part in parts:
                    potential_chunk = current_chunk + (separator if current_chunk else '') + part
                    
                    if len(potential_chunk) <= chunk_size or not current_chunk:
                        current_chunk = potential_chunk
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                            # Create overlap
                            words = current_chunk.split()
                            overlap_words = words[-overlap//4:] if len(words) > overlap//4 else words
                            current_chunk = ' '.join(overlap_words) + separator + part
                        else:
                            current_chunk = part
                
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                return chunks
        
        return self._simple_text_split(text, chunk_size, overlap)
    
    async def create_chunks(self, text: str, source: str) -> List[Dict[str, Any]]:
        """Create enhanced chunks with better splitting"""
        if not text.strip():
            return []
        
        chunks = self._semantic_text_split(text, self.config.chunk_size, self.config.chunk_overlap)
        
        chunk_objects = []
        for i, chunk_text in enumerate(chunks):
            if len(chunk_text.strip()) < 50:
                continue
            
            # Calculate quality score based on length and content
            quality_score = min(1.0, len(chunk_text) / 500) * 0.5 + 0.5
            
            metadata = ChunkMetadata(
                chunk_index=i,
                source=source,
                token_count=self._count_tokens(chunk_text),
                quality_score=quality_score,
                processing_method=self.get_strategy_name()
            )
            
            chunk_obj = {
                'content': chunk_text.strip(),
                'metadata': asdict(metadata)
            }
            chunk_objects.append(chunk_obj)
        
        self.stats['chunks_processed'] += len(chunk_objects)
        return chunk_objects
    
    async def generate_embeddings(self, chunks: List[Dict[str, Any]], openai_client: OpenAI) -> List[Tuple[Dict[str, Any], Optional[List[float]]]]:
        """Generate embeddings with retry logic"""
        results = []
        batch_size = self.config.batch_size
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            # Retry logic
            for attempt in range(self.config.max_retries):
                try:
                    texts = [chunk['content'] for chunk in batch]
                    
                    response = openai_client.embeddings.create(
                        input=texts,
                        model=self.config.openai_model
                    )
                    
                    for j, chunk in enumerate(batch):
                        if j < len(response.data):
                            embedding = response.data[j].embedding
                            results.append((chunk, embedding))
                            self.stats['embeddings_generated'] += 1
                        else:
                            results.append((chunk, None))
                    
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    print(f"Batch processing attempt {attempt + 1} failed: {e}")
                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
                    else:
                        # Final failure
                        for chunk in batch:
                            results.append((chunk, None))
            
            # Rate limiting
            if i + batch_size < len(chunks):
                await asyncio.sleep(self.config.rate_limit_delay)
        
        return results

class IntelligentEmbeddingStrategy(EnhancedEmbeddingStrategy):
    """Intelligent strategy with content categorization"""
    
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        self.categories = {
            'technical': {
                'keywords': ['api', 'function', 'class', 'method', 'algorithm', 'code'],
                'patterns': [r'\b(def|class|function)\b', r'\b[A-Z][a-zA-Z]*\(\)'],
                'weight': 1.2
            },
            'business': {
                'keywords': ['revenue', 'market', 'customer', 'profit', 'strategy'],
                'patterns': [r'\$[\d,]+', r'\b\d+%\b'],
                'weight': 1.1
            },
            'general': {
                'keywords': ['information', 'description', 'overview'],
                'patterns': [],
                'weight': 1.0
            }
        }
    
    def get_strategy_name(self) -> str:
        return "intelligent"
    
    def _categorize_content(self, text: str) -> Tuple[str, float]:
        """Categorize content and return category with confidence"""
        scores = {}
        
        for category_id, category in self.categories.items():
            score = 0.0
            
            # Keyword matching
            keyword_matches = sum(1 for keyword in category['keywords'] 
                                if keyword.lower() in text.lower())
            if category['keywords']:
                score += (keyword_matches / len(category['keywords'])) * 0.6
            
            # Pattern matching
            if category['patterns']:
                import re
                pattern_matches = sum(1 for pattern in category['patterns'] 
                                    if re.search(pattern, text, re.IGNORECASE))
                score += (pattern_matches / len(category['patterns'])) * 0.4
            
            scores[category_id] = score
        
        best_category = max(scores.keys(), key=lambda k: scores[k])
        confidence = scores[best_category]
        
        if confidence < 0.1:
            return 'general', 0.5
        
        return best_category, min(1.0, confidence)
    
    def _extract_key_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text"""
        import re
        concepts = []
        
        # Technical terms
        tech_terms = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', text)
        concepts.extend(tech_terms[:5])
        
        # Important nouns
        important_nouns = re.findall(r'(?<!^)(?<!\. )\b[A-Z][a-z]+\b', text)
        concepts.extend(important_nouns[:5])
        
        return list(set(concepts))[:10]
    
    async def create_chunks(self, text: str, source: str) -> List[Dict[str, Any]]:
        """Create intelligent chunks with categorization"""
        if not text.strip():
            return []
        
        # Analyze overall content
        category, category_confidence = self._categorize_content(text)
        
        # Adjust chunk size based on category
        category_weight = self.categories[category]['weight']
        adjusted_chunk_size = int(self.config.chunk_size * category_weight)
        
        chunks = self._semantic_text_split(text, adjusted_chunk_size, self.config.chunk_overlap)
        
        chunk_objects = []
        for i, chunk_text in enumerate(chunks):
            if len(chunk_text.strip()) < 50:
                continue
            
            # Analyze individual chunk
            chunk_category, chunk_confidence = self._categorize_content(chunk_text)
            key_concepts = self._extract_key_concepts(chunk_text)
            
            # Calculate quality score
            quality_score = min(1.0, len(chunk_text) / 500) * 0.4 + chunk_confidence * 0.6
            
            metadata = ChunkMetadata(
                chunk_index=i,
                source=source,
                token_count=self._count_tokens(chunk_text),
                quality_score=quality_score,
                category=chunk_category,
                category_confidence=chunk_confidence,
                key_concepts=key_concepts,
                processing_method=self.get_strategy_name()
            )
            
            chunk_obj = {
                'content': chunk_text.strip(),
                'metadata': asdict(metadata)
            }
            chunk_objects.append(chunk_obj)
        
        self.stats['chunks_processed'] += len(chunk_objects)
        return chunk_objects

class EmbeddingEngine:
    """Centralized embedding engine"""
    
    def __init__(self, config: EmbeddingConfig = None):
        self.config = config or EmbeddingConfig()
        self.strategy = self._create_strategy()
        self.openai_client = None
        self.supabase_client = None
        
        # Global stats
        self.global_stats = {
            'total_files_processed': 0,
            'total_chunks_created': 0,
            'total_embeddings_generated': 0,
            'total_processing_time': 0,
            'strategy_usage': {}
        }
    
    def _create_strategy(self) -> EmbeddingStrategy:
        """Create embedding strategy based on configuration"""
        if self.config.processing_level == ProcessingLevel.NEURAL_SUPREME:
            if NEURAL_CORE_AVAILABLE:
                # Use neural core for supreme processing
                return NeuralSupremeStrategy(self.config)
            else:
                print("⚠️  Neural Supreme not available, falling back to Intelligent")
                return IntelligentEmbeddingStrategy(self.config)
        if self.config.processing_level == ProcessingLevel.INTELLIGENT:
            return IntelligentEmbeddingStrategy(self.config)
        elif self.config.processing_level == ProcessingLevel.ENHANCED:
            return EnhancedEmbeddingStrategy(self.config)
        else:
            return BasicEmbeddingStrategy(self.config)
    
class NeuralSupremeStrategy(EmbeddingStrategy):
    """Neural Supreme strategy using the advanced neural core"""
    
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        self.neural_config = NeuralConfig(
            processing_mode=ProcessingMode.NEURAL_SUPREME,
            batch_size=config.batch_size,
            max_chunk_size=config.chunk_size,
            overlap_ratio=config.chunk_overlap / config.chunk_size,
            quality_threshold=config.quality_threshold
        )
        self.neural_core = get_neural_core(self.neural_config)
    
    def get_strategy_name(self) -> str:
        return "neural_supreme"
    
    async def create_chunks(self, text: str, source: str) -> List[Dict[str, Any]]:
        """Create chunks using neural core"""
        chunks = await self.neural_core.neural_chunk_analysis(text, source)
        self.stats['chunks_processed'] += len(chunks)
        return chunks
    
    async def generate_embeddings(self, chunks: List[Dict[str, Any]], openai_client: OpenAI) -> List[Tuple[Dict[str, Any], Optional[List[float]]]]:
        """Generate embeddings using neural core"""
        # Set credentials in neural core
        if hasattr(self.neural_core, 'openai_client'):
            self.neural_core.openai_client = openai_client
        
        # Generate embeddings with caching and fusion
        results = []
        for chunk in chunks:
            try:
                embeddings_dict = await self.neural_core.generate_hybrid_embeddings(chunk['content'])
                if embeddings_dict:
                    fused_embedding, quality = self.neural_core.embedding_fusion.fuse_embeddings(embeddings_dict)
                    # Update chunk with quality info
                    chunk['neural_features']['fusion_quality'] = quality
                    results.append((chunk, fused_embedding.tolist()))
                    self.stats['embeddings_generated'] += 1
                else:
                    results.append((chunk, None))
            except Exception as e:
                print(f"Neural embedding failed: {e}")
                results.append((chunk, None))
        
        return results
    def set_credentials(self, openai_api_key: str, supabase_url: str, supabase_service_key: str):
        """Set user credentials"""
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.supabase_client = create_client(supabase_url, supabase_service_key)
    
    def update_config(self, **kwargs):
        """Update configuration and recreate strategy if needed"""
        old_level = self.config.processing_level
        
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Recreate strategy if processing level changed
        if self.config.processing_level != old_level:
            self.strategy = self._create_strategy()
    
    async def process_documents(self, documents: List[Dict[str, str]]) -> ProcessingResult:
        """Process multiple documents
        
        Args:
            documents: List of dicts with 'content' and 'source' keys
        """
        start_time = time.time()
        
        if not self.openai_client or not self.supabase_client:
            raise ValueError("Credentials not set. Call set_credentials() first.")
        
        print(f"Processing {len(documents)} documents with {self.strategy.get_strategy_name()} strategy")
        
        all_chunks = []
        
        # Create chunks for all documents
        for doc in documents:
            chunks = await self.strategy.create_chunks(doc['content'], doc['source'])
            all_chunks.extend(chunks)
        
        print(f"Created {len(all_chunks)} chunks total")
        
        # Generate embeddings
        chunk_embedding_pairs = await self.strategy.generate_embeddings(all_chunks, self.openai_client)
        
        # Upload to Supabase
        upload_stats = await self._upload_to_supabase(chunk_embedding_pairs)
        
        processing_time = time.time() - start_time
        
        # Update global stats
        self.global_stats['total_files_processed'] += len(documents)
        self.global_stats['total_chunks_created'] += len(all_chunks)
        self.global_stats['total_embeddings_generated'] += upload_stats['successful_uploads']
        self.global_stats['total_processing_time'] += processing_time
        
        strategy_name = self.strategy.get_strategy_name()
        self.global_stats['strategy_usage'][strategy_name] = self.global_stats['strategy_usage'].get(strategy_name, 0) + 1
        
        return ProcessingResult(
            success=upload_stats['successful_uploads'] > 0,
            chunks_processed=len(all_chunks),
            successful_embeddings=upload_stats['successful_uploads'],
            failed_embeddings=upload_stats['failed_uploads'],
            processing_time=processing_time,
            processing_level=strategy_name,
            upload_stats=upload_stats,
            performance_stats={
                'chunks_per_second': len(all_chunks) / processing_time if processing_time > 0 else 0,
                'embeddings_per_second': upload_stats['successful_uploads'] / processing_time if processing_time > 0 else 0,
                'strategy_stats': self.strategy.stats
            }
        )
    
    async def _upload_to_supabase(self, chunk_embedding_pairs: List[Tuple[Dict[str, Any], Optional[List[float]]]]) -> Dict[str, Any]:
        """Upload embeddings to Supabase"""
        successful_uploads = 0
        failed_uploads = 0
        validation_failures = 0
        
        batch_data = []
        
        for chunk, embedding in chunk_embedding_pairs:
            if embedding is None:
                failed_uploads += 1
                continue
            
            # Validate embedding
            if not self._validate_embedding(embedding):
                validation_failures += 1
                continue
            
            # Prepare data for Supabase
            data = {
                'content': chunk['content'],
                'embedding': embedding,
                'source': chunk['metadata']['source'],
                'metadata': chunk['metadata']
            }
            batch_data.append(data)
        
        # Upload in batches
        supabase_batch_size = 25
        for i in range(0, len(batch_data), supabase_batch_size):
            batch = batch_data[i:i + supabase_batch_size]
            
            try:
                response = self.supabase_client.table('documents').insert(batch).execute()
                
                if hasattr(response, 'error') and response.error:
                    print(f"Supabase batch upload error: {response.error}")
                    failed_uploads += len(batch)
                else:
                    successful_uploads += len(batch)
                    
            except Exception as e:
                print(f"Exception during Supabase upload: {e}")
                failed_uploads += len(batch)
        
        return {
            'successful_uploads': successful_uploads,
            'failed_uploads': failed_uploads,
            'validation_failures': validation_failures,
            'total_chunks': len(chunk_embedding_pairs)
        }
    
    def _validate_embedding(self, embedding: List[float]) -> bool:
        """Validate embedding quality"""
        if not embedding or len(embedding) != 1536:
            return False
        
        arr = np.array(embedding)
        if np.all(arr == 0) or np.any(np.isnan(arr)):
            return False
        
        magnitude = np.linalg.norm(arr)
        if magnitude < 0.5 or magnitude > 2.0:
            return False
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        return {
            'global_stats': self.global_stats,
            'current_strategy': self.strategy.get_strategy_name(),
            'strategy_stats': self.strategy.stats,
            'config': asdict(self.config)
        }

# Convenience functions for backward compatibility
async def run_neural_supreme_processing_compat(chunks: List[Dict[str, Any]], openai_api_key: str, 
                                             supabase_url: str, supabase_service_key: str) -> Dict[str, Any]:
    """Run neural supreme processing (compatibility wrapper)"""
    if NEURAL_CORE_AVAILABLE:
        return await run_neural_supreme_processing(chunks, openai_api_key, supabase_url, supabase_service_key)
    else:
        # Fallback to intelligent processing
        return await run_intelligent_processing(chunks, openai_api_key, supabase_url, supabase_service_key)

async def run_basic_processing(chunks: List[Dict[str, Any]], openai_api_key: str, 
                             supabase_url: str, supabase_service_key: str) -> Dict[str, Any]:
    """Run basic processing"""
    config = EmbeddingConfig(processing_level=ProcessingLevel.BASIC)
    engine = EmbeddingEngine(config)
    engine.set_credentials(openai_api_key, supabase_url, supabase_service_key)
    
    documents = [{'content': chunk['content'], 'source': chunk['source']} for chunk in chunks]
    result = await engine.process_documents(documents)
    
    return {
        'success': result.success,
        'chunks_processed': result.chunks_processed,
        'successful_embeddings': result.successful_embeddings,
        'failed_embeddings': result.failed_embeddings,
        'upload_stats': result.upload_stats,
        'performance_stats': result.performance_stats
    }

async def run_enhanced_processing(chunks: List[Dict[str, Any]], openai_api_key: str, 
                                supabase_url: str, supabase_service_key: str) -> Dict[str, Any]:
    """Run enhanced processing"""
    config = EmbeddingConfig(processing_level=ProcessingLevel.ENHANCED)
    engine = EmbeddingEngine(config)
    engine.set_credentials(openai_api_key, supabase_url, supabase_service_key)
    
    documents = [{'content': chunk['content'], 'source': chunk['source']} for chunk in chunks]
    result = await engine.process_documents(documents)
    
    return {
        'success': result.success,
        'chunks_processed': result.chunks_processed,
        'successful_embeddings': result.successful_embeddings,
        'failed_embeddings': result.failed_embeddings,
        'upload_stats': result.upload_stats,
        'performance_stats': result.performance_stats
    }

async def run_intelligent_processing(chunks: List[Dict[str, Any]], openai_api_key: str, 
                                   supabase_url: str, supabase_service_key: str) -> Dict[str, Any]:
    """Run intelligent processing"""
    config = EmbeddingConfig(processing_level=ProcessingLevel.INTELLIGENT)
    engine = EmbeddingEngine(config)
    engine.set_credentials(openai_api_key, supabase_url, supabase_service_key)
    
    documents = [{'content': chunk['content'], 'source': chunk['source']} for chunk in chunks]
    result = await engine.process_documents(documents)
    
    return {
        'success': result.success,
        'chunks_processed': result.chunks_processed,
        'successful_embeddings': result.successful_embeddings,
        'failed_embeddings': result.failed_embeddings,
        'upload_stats': result.upload_stats,
        'performance_stats': result.performance_stats,
        'intelligent_chunks_created': result.chunks_processed,
        'high_quality_chunks': sum(1 for _, emb in result.upload_stats.items() if emb),
        'semantic_clusters': 0  # Placeholder for compatibility
    }

if __name__ == "__main__":
    print("🚀 Centralized Embedding Engine")
    if NEURAL_CORE_AVAILABLE:
        print("🧠 NEURAL SUPREME MODE AVAILABLE!")
    print("Available processing levels:")
    for level in ProcessingLevel:
        print(f"  - {level.value}")
    print("\nUsage:")
    print("  from embedding_engine import EmbeddingEngine, EmbeddingConfig, ProcessingLevel")
    if NEURAL_CORE_AVAILABLE:
        print("  config = EmbeddingConfig(processing_level=ProcessingLevel.NEURAL_SUPREME)")
    print("  config = EmbeddingConfig(processing_level=ProcessingLevel.INTELLIGENT)")
    print("  engine = EmbeddingEngine(config)")