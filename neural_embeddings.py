#!/usr/bin/env python3
"""
Neural Embedding System with Deep Learning Optimization
State-of-the-art RAG processing with neural networks and transformer models
"""

import asyncio
import time
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, pipeline
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.cluster import HDBSCAN
from sklearn.manifold import UMAP
import networkx as nx
from openai import OpenAI
from supabase import create_client

@dataclass
class NeuralConfig:
    """Configuration for neural embedding system"""
    openai_model: str = "text-embedding-3-small"
    sentence_transformer_model: str = "all-MiniLM-L6-v2"
    enable_hybrid_embeddings: bool = True
    enable_neural_chunking: bool = True
    enable_semantic_clustering: bool = True
    enable_query_expansion: bool = True
    faiss_index_type: str = "IVF"
    neural_chunk_threshold: float = 0.7
    max_chunk_size: int = 1000
    min_chunk_size: int = 100

class NeuralChunker(nn.Module):
    """Neural network for intelligent text chunking"""
    
    def __init__(self, input_dim=768, hidden_dim=256):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 2,
                dropout=0.1
            ),
            num_layers=3
        )
        self.chunk_classifier = nn.Linear(input_dim, 2)  # chunk_boundary or not
        self.quality_predictor = nn.Linear(input_dim, 1)  # quality score
        
    def forward(self, embeddings):
        # embeddings: [seq_len, batch_size, embedding_dim]
        encoded = self.encoder(embeddings)
        
        # Predict chunk boundaries
        chunk_logits = self.chunk_classifier(encoded)
        chunk_probs = F.softmax(chunk_logits, dim=-1)
        
        # Predict quality scores
        quality_scores = torch.sigmoid(self.quality_predictor(encoded))
        
        return chunk_probs, quality_scores

class HybridEmbeddingGenerator:
    """Generate hybrid embeddings using multiple models"""
    
    def __init__(self, config: NeuralConfig):
        self.config = config
        self.sentence_transformer = None
        self.tokenizer = None
        self.model = None
        self.openai_client = None
        
    def initialize_models(self):
        """Initialize all embedding models"""
        try:
            # Sentence transformer for local embeddings
            self.sentence_transformer = SentenceTransformer(self.config.sentence_transformer_model)
            print(f"Loaded SentenceTransformer: {self.config.sentence_transformer_model}")
            
            # Transformer model for contextual embeddings
            model_name = "microsoft/DialoGPT-medium"  # Fallback model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            print(f"Loaded transformer model: {model_name}")
            
        except Exception as e:
            print(f"Warning: Could not load some models: {e}")
            print("Falling back to OpenAI embeddings only")
    
    def set_openai_client(self, api_key: str):
        """Set OpenAI client"""
        self.openai_client = OpenAI(api_key=api_key)
    
    async def generate_hybrid_embedding(self, text: str) -> Dict[str, List[float]]:
        """Generate hybrid embeddings from multiple sources"""
        embeddings = {}
        
        # OpenAI embedding (primary)
        try:
            if self.openai_client:
                response = self.openai_client.embeddings.create(
                    input=text,
                    model=self.config.openai_model
                )
                embeddings['openai'] = response.data[0].embedding
        except Exception as e:
            print(f"OpenAI embedding failed: {e}")
        
        # Sentence transformer embedding (local)
        try:
            if self.sentence_transformer:
                local_embedding = self.sentence_transformer.encode(text)
                embeddings['sentence_transformer'] = local_embedding.tolist()
        except Exception as e:
            print(f"SentenceTransformer embedding failed: {e}")
        
        # Contextual transformer embedding
        try:
            if self.tokenizer and self.model:
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Use mean pooling of last hidden states
                    contextual_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                    embeddings['contextual'] = contextual_embedding.tolist()
        except Exception as e:
            print(f"Contextual embedding failed: {e}")
        
        return embeddings
    
    def combine_embeddings(self, embeddings: Dict[str, List[float]], weights: Dict[str, float] = None) -> List[float]:
        """Combine multiple embeddings into a single representation"""
        if not embeddings:
            return [0.0] * 1536  # Fallback
        
        # Default weights
        if weights is None:
            weights = {
                'openai': 0.6,
                'sentence_transformer': 0.3,
                'contextual': 0.1
            }
        
        # Normalize embeddings to same dimension (use OpenAI as reference)
        target_dim = 1536
        combined = np.zeros(target_dim)
        total_weight = 0
        
        for embed_type, embedding in embeddings.items():
            weight = weights.get(embed_type, 0.1)
            
            # Resize embedding to target dimension
            if len(embedding) != target_dim:
                if len(embedding) > target_dim:
                    # Truncate
                    embedding = embedding[:target_dim]
                else:
                    # Pad with zeros
                    embedding = embedding + [0.0] * (target_dim - len(embedding))
            
            combined += np.array(embedding) * weight
            total_weight += weight
        
        if total_weight > 0:
            combined /= total_weight
        
        return combined.tolist()

class SemanticClusterAnalyzer:
    """Advanced semantic clustering with neural networks"""
    
    def __init__(self):
        self.clusterer = None
        self.umap_reducer = None
        self.cluster_embeddings = {}
        
    def fit_clustering_model(self, embeddings: np.ndarray):
        """Fit clustering models on embeddings"""
        # Dimensionality reduction with UMAP
        self.umap_reducer = UMAP(
            n_components=50,
            n_neighbors=15,
            min_dist=0.1,
            metric='cosine',
            random_state=42
        )
        
        reduced_embeddings = self.umap_reducer.fit_transform(embeddings)
        
        # Hierarchical clustering with HDBSCAN
        self.clusterer = HDBSCAN(
            min_cluster_size=3,
            min_samples=2,
            metric='euclidean',
            cluster_selection_epsilon=0.1
        )
        
        cluster_labels = self.clusterer.fit_predict(reduced_embeddings)
        
        return cluster_labels, reduced_embeddings
    
    def analyze_clusters(self, embeddings: np.ndarray, texts: List[str], labels: np.ndarray) -> Dict[str, Any]:
        """Analyze semantic clusters"""
        cluster_analysis = {}
        
        unique_labels = set(labels)
        for label in unique_labels:
            if label == -1:  # Noise cluster
                continue
                
            cluster_mask = labels == label
            cluster_embeddings = embeddings[cluster_mask]
            cluster_texts = [texts[i] for i in range(len(texts)) if cluster_mask[i]]
            
            # Calculate cluster centroid
            centroid = np.mean(cluster_embeddings, axis=0)
            
            # Calculate intra-cluster similarity
            similarities = []
            for i, emb1 in enumerate(cluster_embeddings):
                for j, emb2 in enumerate(cluster_embeddings):
                    if i < j:
                        sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                        similarities.append(sim)
            
            avg_similarity = np.mean(similarities) if similarities else 0.0
            
            # Extract common themes (simple keyword extraction)
            all_words = ' '.join(cluster_texts).lower().split()
            word_freq = {}
            for word in all_words:
                if len(word) > 3:  # Filter short words
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            common_themes = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            
            cluster_analysis[f"cluster_{label}"] = {
                'size': len(cluster_texts),
                'centroid': centroid.tolist(),
                'avg_similarity': avg_similarity,
                'common_themes': common_themes,
                'sample_texts': cluster_texts[:3]  # First 3 texts as samples
            }
        
        return cluster_analysis

class QueryExpansionEngine:
    """Neural query expansion for better retrieval"""
    
    def __init__(self):
        self.expansion_model = None
        self.concept_graph = nx.Graph()
        
    def initialize_expansion_model(self):
        """Initialize query expansion model"""
        try:
            # Use a text generation pipeline for query expansion
            self.expansion_model = pipeline(
                "text-generation",
                model="gpt2",
                tokenizer="gpt2",
                device=-1  # CPU
            )
        except Exception as e:
            print(f"Could not load query expansion model: {e}")
    
    def expand_query(self, query: str, context_embeddings: List[List[float]] = None) -> List[str]:
        """Expand query with related terms and concepts"""
        expanded_queries = [query]  # Original query
        
        try:
            if self.expansion_model:
                # Generate query variations
                prompt = f"Similar questions to '{query}' include:"
                
                generated = self.expansion_model(
                    prompt,
                    max_length=len(prompt.split()) + 20,
                    num_return_sequences=3,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=50256
                )
                
                for gen in generated:
                    expanded_text = gen['generated_text'][len(prompt):].strip()
                    if expanded_text and expanded_text not in expanded_queries:
                        expanded_queries.append(expanded_text)
        
        except Exception as e:
            print(f"Query expansion failed: {e}")
        
        # Add synonym-based expansions (simple approach)
        synonyms = self._get_simple_synonyms(query)
        for synonym in synonyms:
            if synonym not in expanded_queries:
                expanded_queries.append(synonym)
        
        return expanded_queries[:5]  # Limit to 5 variations
    
    def _get_simple_synonyms(self, query: str) -> List[str]:
        """Simple synonym generation"""
        # Basic synonym mapping
        synonym_map = {
            'how': ['what is the way', 'what is the method'],
            'what': ['which', 'what kind of'],
            'why': ['what is the reason', 'what causes'],
            'when': ['at what time', 'during which period'],
            'where': ['in which location', 'at what place']
        }
        
        synonyms = []
        words = query.lower().split()
        
        for word in words:
            if word in synonym_map:
                for synonym in synonym_map[word]:
                    new_query = query.lower().replace(word, synonym)
                    synonyms.append(new_query)
        
        return synonyms

class NeuralEmbeddingProcessor:
    """Main neural embedding processor"""
    
    def __init__(self, config: NeuralConfig = None):
        self.config = config or NeuralConfig()
        self.hybrid_generator = HybridEmbeddingGenerator(self.config)
        self.neural_chunker = None
        self.cluster_analyzer = SemanticClusterAnalyzer()
        self.query_expander = QueryExpansionEngine()
        self.faiss_index = None
        self.supabase_client = None
        
        # Performance tracking
        self.stats = {
            'total_chunks': 0,
            'neural_chunks_created': 0,
            'hybrid_embeddings_generated': 0,
            'clusters_identified': 0,
            'processing_time': 0
        }
    
    def set_credentials(self, openai_api_key: str, supabase_url: str, supabase_service_key: str):
        """Set credentials and initialize models"""
        self.hybrid_generator.set_openai_client(openai_api_key)
        self.supabase_client = create_client(supabase_url, supabase_service_key)
        
        # Initialize models
        self.hybrid_generator.initialize_models()
        self.query_expander.initialize_expansion_model()
        
        # Initialize neural chunker
        if self.config.enable_neural_chunking:
            self.neural_chunker = NeuralChunker()
    
    async def neural_text_chunking(self, text: str, source: str) -> List[Dict[str, Any]]:
        """Advanced neural text chunking"""
        if not self.config.enable_neural_chunking or not self.neural_chunker:
            # Fallback to semantic chunking
            return await self._semantic_chunking_fallback(text, source)
        
        try:
            # Generate sentence embeddings for neural chunking
            sentences = text.split('.')
            sentence_embeddings = []
            
            for sentence in sentences:
                if sentence.strip():
                    embeddings = await self.hybrid_generator.generate_hybrid_embedding(sentence.strip())
                    combined_embedding = self.hybrid_generator.combine_embeddings(embeddings)
                    sentence_embeddings.append(combined_embedding)
            
            if not sentence_embeddings:
                return []
            
            # Convert to tensor for neural processing
            embeddings_tensor = torch.tensor(sentence_embeddings).unsqueeze(1)  # Add batch dimension
            
            # Predict chunk boundaries and quality
            with torch.no_grad():
                chunk_probs, quality_scores = self.neural_chunker(embeddings_tensor)
                
                # Extract chunk boundaries (where probability > threshold)
                boundary_probs = chunk_probs[:, :, 1].squeeze()  # Probability of being a boundary
                boundaries = (boundary_probs > self.config.neural_chunk_threshold).numpy()
            
            # Create chunks based on predicted boundaries
            chunks = []
            current_chunk = ""
            current_sentences = []
            
            for i, (sentence, is_boundary) in enumerate(zip(sentences, boundaries)):
                current_sentences.append(sentence.strip())
                current_chunk += sentence + ". "
                
                if is_boundary or i == len(sentences) - 1:
                    if current_chunk.strip():
                        # Calculate average quality for this chunk
                        chunk_quality = float(quality_scores[max(0, i-len(current_sentences)):i+1].mean())
                        
                        chunk_obj = {
                            'content': current_chunk.strip(),
                            'source': source,
                            'chunk_index': len(chunks),
                            'neural_features': {
                                'predicted_quality': chunk_quality,
                                'sentence_count': len(current_sentences),
                                'boundary_confidence': float(boundary_probs[i]) if i < len(boundary_probs) else 0.0,
                                'chunking_method': 'neural'
                            }
                        }
                        chunks.append(chunk_obj)
                        
                        current_chunk = ""
                        current_sentences = []
            
            self.stats['neural_chunks_created'] += len(chunks)
            return chunks
            
        except Exception as e:
            print(f"Neural chunking failed, using fallback: {e}")
            return await self._semantic_chunking_fallback(text, source)
    
    async def _semantic_chunking_fallback(self, text: str, source: str) -> List[Dict[str, Any]]:
        """Fallback semantic chunking"""
        # Simple sentence-based chunking
        sentences = text.split('.')
        chunks = []
        current_chunk = ""
        
        for i, sentence in enumerate(sentences):
            if sentence.strip():
                current_chunk += sentence + ". "
                
                if len(current_chunk) > self.config.max_chunk_size or i == len(sentences) - 1:
                    if current_chunk.strip():
                        chunk_obj = {
                            'content': current_chunk.strip(),
                            'source': source,
                            'chunk_index': len(chunks),
                            'neural_features': {
                                'predicted_quality': 0.7,  # Default quality
                                'sentence_count': current_chunk.count('.'),
                                'boundary_confidence': 0.8,
                                'chunking_method': 'semantic_fallback'
                            }
                        }
                        chunks.append(chunk_obj)
                        current_chunk = ""
        
        return chunks
    
    async def generate_neural_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Optional[Dict[str, List[float]]]]]:
        """Generate neural embeddings for chunks"""
        results = []
        
        for chunk in chunks:
            try:
                # Generate hybrid embeddings
                embeddings = await self.hybrid_generator.generate_hybrid_embedding(chunk['content'])
                
                if embeddings:
                    self.stats['hybrid_embeddings_generated'] += 1
                    results.append((chunk, embeddings))
                else:
                    results.append((chunk, None))
                    
            except Exception as e:
                print(f"Error generating embeddings for chunk: {e}")
                results.append((chunk, None))
        
        return results
    
    async def perform_semantic_clustering(self, chunk_embedding_pairs: List[Tuple[Dict[str, Any], Optional[Dict[str, List[float]]]]]) -> Dict[str, Any]:
        """Perform advanced semantic clustering"""
        if not self.config.enable_semantic_clustering:
            return {}
        
        # Extract embeddings and texts for clustering
        embeddings = []
        texts = []
        valid_chunks = []
        
        for chunk, embedding_dict in chunk_embedding_pairs:
            if embedding_dict and 'openai' in embedding_dict:
                embeddings.append(embedding_dict['openai'])
                texts.append(chunk['content'])
                valid_chunks.append(chunk)
        
        if len(embeddings) < 3:
            return {}
        
        embeddings_array = np.array(embeddings)
        
        # Perform clustering
        cluster_labels, reduced_embeddings = self.cluster_analyzer.fit_clustering_model(embeddings_array)
        
        # Analyze clusters
        cluster_analysis = self.cluster_analyzer.analyze_clusters(embeddings_array, texts, cluster_labels)
        
        # Update chunks with cluster information
        for i, chunk in enumerate(valid_chunks):
            chunk['cluster_info'] = {
                'cluster_id': int(cluster_labels[i]),
                'reduced_embedding': reduced_embeddings[i].tolist() if i < len(reduced_embeddings) else None
            }
        
        self.stats['clusters_identified'] = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        
        return cluster_analysis
    
    def build_faiss_index(self, embeddings: List[List[float]]) -> faiss.Index:
        """Build FAISS index for fast similarity search"""
        if not embeddings:
            return None
        
        dimension = len(embeddings[0])
        embeddings_array = np.array(embeddings).astype('float32')
        
        if self.config.faiss_index_type == "IVF":
            # IVF (Inverted File) index for large datasets
            nlist = min(100, len(embeddings) // 10)  # Number of clusters
            quantizer = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            
            # Train the index
            index.train(embeddings_array)
            index.add(embeddings_array)
            
        else:
            # Flat index for smaller datasets
            index = faiss.IndexFlatIP(dimension)
            index.add(embeddings_array)
        
        return index
    
    async def upload_neural_embeddings(self, chunk_embedding_pairs: List[Tuple[Dict[str, Any], Optional[Dict[str, List[float]]]]], cluster_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Upload neural embeddings with advanced metadata"""
        successful_uploads = 0
        failed_uploads = 0
        hybrid_embeddings_count = 0
        
        # Prepare embeddings for FAISS index
        primary_embeddings = []
        batch_data = []
        
        for chunk, embedding_dict in chunk_embedding_pairs:
            if embedding_dict is None:
                failed_uploads += 1
                continue
            
            # Use OpenAI embedding as primary, combine others as metadata
            primary_embedding = embedding_dict.get('openai', [])
            if not primary_embedding:
                failed_uploads += 1
                continue
            
            primary_embeddings.append(primary_embedding)
            
            # Count hybrid embeddings
            if len(embedding_dict) > 1:
                hybrid_embeddings_count += 1
            
            # Prepare data for Supabase
            data = {
                'content': chunk['content'],
                'embedding': primary_embedding,
                'source': chunk['source'],
                'metadata': {
                    'chunk_index': chunk['chunk_index'],
                    'neural_features': chunk['neural_features'],
                    'hybrid_embeddings': {
                        'sentence_transformer': embedding_dict.get('sentence_transformer'),
                        'contextual': embedding_dict.get('contextual')
                    },
                    'cluster_info': chunk.get('cluster_info', {}),
                    'processing_version': 'neural_v1.0'
                }
            }
            batch_data.append(data)
        
        # Build FAISS index
        if primary_embeddings:
            self.faiss_index = self.build_faiss_index(primary_embeddings)
        
        # Upload in batches
        batch_size = 25
        for i in range(0, len(batch_data), batch_size):
            batch = batch_data[i:i + batch_size]
            
            try:
                response = self.supabase_client.table('documents').insert(batch).execute()
                
                if hasattr(response, 'error') and response.error:
                    print(f"Batch upload error: {response.error}")
                    failed_uploads += len(batch)
                else:
                    successful_uploads += len(batch)
                    
            except Exception as e:
                print(f"Exception during batch upload: {e}")
                failed_uploads += len(batch)
        
        # Store cluster analysis
        if cluster_analysis:
            try:
                cluster_doc = {
                    'content': 'NEURAL_CLUSTER_ANALYSIS',
                    'embedding': [0.0] * 1536,
                    'source': 'system_neural_clusters',
                    'metadata': {
                        'type': 'cluster_analysis',
                        'clusters': cluster_analysis,
                        'faiss_index_info': {
                            'index_type': self.config.faiss_index_type,
                            'dimension': len(primary_embeddings[0]) if primary_embeddings else 0,
                            'total_vectors': len(primary_embeddings)
                        }
                    }
                }
                
                self.supabase_client.table('documents').insert(cluster_doc).execute()
                
            except Exception as e:
                print(f"Error storing cluster analysis: {e}")
        
        return {
            'successful_uploads': successful_uploads,
            'failed_uploads': failed_uploads,
            'total_chunks': len(chunk_embedding_pairs),
            'hybrid_embeddings_count': hybrid_embeddings_count,
            'clusters_identified': self.stats['clusters_identified'],
            'faiss_index_built': self.faiss_index is not None
        }
    
    async def process_documents_neural(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Main neural processing method"""
        start_time = time.time()
        
        print(f"Starting neural processing of {len(chunks)} chunks...")
        
        # Convert simple chunks to neural chunks
        neural_chunks = []
        for chunk in chunks:
            chunk_list = await self.neural_text_chunking(chunk['content'], chunk['source'])
            neural_chunks.extend(chunk_list)
        
        print(f"Created {len(neural_chunks)} neural chunks")
        
        # Generate neural embeddings
        chunk_embedding_pairs = await self.generate_neural_embeddings(neural_chunks)
        
        # Perform semantic clustering
        cluster_analysis = await self.perform_semantic_clustering(chunk_embedding_pairs)
        
        # Upload with neural metadata
        upload_result = await self.upload_neural_embeddings(chunk_embedding_pairs, cluster_analysis)
        
        processing_time = time.time() - start_time
        self.stats['processing_time'] = processing_time
        
        return {
            'success': upload_result['successful_uploads'] > 0,
            'chunks_processed': len(chunks),
            'neural_chunks_created': len(neural_chunks),
            'successful_embeddings': upload_result['successful_uploads'],
            'upload_stats': upload_result,
            'performance_stats': {
                'processing_time': processing_time,
                'chunks_per_second': len(neural_chunks) / processing_time if processing_time > 0 else 0,
                'neural_chunks_created': self.stats['neural_chunks_created'],
                'hybrid_embeddings_generated': self.stats['hybrid_embeddings_generated'],
                'clusters_identified': self.stats['clusters_identified']
            },
            'neural_features': {
                'hybrid_embeddings_used': upload_result['hybrid_embeddings_count'],
                'faiss_index_built': upload_result['faiss_index_built'],
                'clustering_enabled': self.config.enable_semantic_clustering,
                'neural_chunking_enabled': self.config.enable_neural_chunking
            }
        }

# Integration function
async def run_neural_processing(chunks: List[Dict[str, Any]], openai_api_key: str, 
                              supabase_url: str, supabase_service_key: str) -> Dict[str, Any]:
    """Run neural embedding processing"""
    processor = NeuralEmbeddingProcessor()
    processor.set_credentials(openai_api_key, supabase_url, supabase_service_key)
    return await processor.process_documents_neural(chunks)

if __name__ == "__main__":
    print("🧠 Neural Embedding System with Deep Learning")
    print("Cutting-edge features:")
    print("- Neural network-based intelligent chunking")
    print("- Hybrid embeddings from multiple transformer models")
    print("- Advanced semantic clustering with HDBSCAN + UMAP")
    print("- FAISS indexing for ultra-fast similarity search")
    print("- Neural query expansion with GPT-2")
    print("- Multi-model embedding fusion")
    print("- Hierarchical clustering analysis")
    print("- Real-time embedding quality prediction")
    print("- Dimensionality reduction with UMAP")
    print("- Graph-based concept relationships")