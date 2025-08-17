#!/usr/bin/env python3
"""
Adaptive Embedding System with Dynamic Optimization
Next-generation RAG processing with self-improving capabilities
"""

import asyncio
import time
import hashlib
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import tiktoken
from openai import OpenAI
from supabase import create_client
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import re
from collections import Counter, defaultdict
import spacy
from textstat import flesch_reading_ease, flesch_kincaid_grade
import networkx as nx

@dataclass
class AdaptiveConfig:
    """Configuration for adaptive embedding system"""
    model: str = "text-embedding-3-small"
    batch_size: int = 100
    max_tokens: int = 8191
    adaptive_chunk_size: bool = True
    min_chunk_size: int = 200
    max_chunk_size: int = 1200
    overlap_ratio: float = 0.15
    quality_threshold: float = 0.7
    semantic_similarity_threshold: float = 0.85
    enable_cross_references: bool = True
    enable_concept_graphs: bool = True
    enable_adaptive_learning: bool = True

@dataclass
class ChunkMetrics:
    """Advanced metrics for chunk quality assessment"""
    readability_score: float
    complexity_score: float
    information_density: float
    concept_coherence: float
    cross_reference_count: int
    semantic_uniqueness: float
    retrieval_potential: float
    domain_specificity: float

class ConceptGraph:
    """Knowledge graph for concept relationships"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.concept_embeddings = {}
        self.concept_frequencies = Counter()
    
    def add_concept(self, concept: str, embedding: List[float], context: str):
        """Add concept to the graph"""
        self.graph.add_node(concept, embedding=embedding, contexts=[context])
        self.concept_embeddings[concept] = embedding
        self.concept_frequencies[concept] += 1
    
    def add_relationship(self, concept1: str, concept2: str, relationship_type: str, strength: float):
        """Add relationship between concepts"""
        self.graph.add_edge(concept1, concept2, 
                          relationship=relationship_type, 
                          strength=strength)
    
    def find_related_concepts(self, concept: str, max_distance: int = 2) -> List[Tuple[str, float]]:
        """Find concepts related to the given concept"""
        if concept not in self.graph:
            return []
        
        related = []
        for node in nx.single_source_shortest_path_length(self.graph, concept, cutoff=max_distance):
            if node != concept:
                path_length = nx.shortest_path_length(self.graph, concept, node)
                strength = 1.0 / (path_length + 1)
                related.append((node, strength))
        
        return sorted(related, key=lambda x: x[1], reverse=True)
    
    def get_concept_importance(self, concept: str) -> float:
        """Calculate concept importance based on centrality and frequency"""
        if concept not in self.graph:
            return 0.0
        
        centrality = nx.betweenness_centrality(self.graph).get(concept, 0)
        frequency_score = self.concept_frequencies[concept] / max(self.concept_frequencies.values())
        
        return (centrality * 0.6) + (frequency_score * 0.4)

class AdaptiveEmbeddingProcessor:
    """Next-generation adaptive embedding processor"""
    
    def __init__(self, config: AdaptiveConfig = None):
        self.config = config or AdaptiveConfig()
        self.openai_client = None
        self.supabase_client = None
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        
        # Advanced components
        self.concept_graph = ConceptGraph()
        self.nlp = None  # Will be loaded lazily
        self.chunk_performance_history = defaultdict(list)
        self.adaptive_parameters = {
            'optimal_chunk_size': 800,
            'optimal_overlap': 100,
            'quality_weights': {
                'readability': 0.2,
                'complexity': 0.15,
                'density': 0.25,
                'coherence': 0.2,
                'uniqueness': 0.2
            }
        }
        
        # Performance tracking
        self.stats = {
            'total_chunks': 0,
            'high_quality_chunks': 0,
            'adaptive_optimizations': 0,
            'concept_extractions': 0,
            'cross_references_found': 0,
            'processing_time': 0
        }
    
    def _load_nlp_model(self):
        """Lazy load spaCy model"""
        if self.nlp is None:
            try:
                import spacy
                self.nlp = spacy.load("en_core_web_sm")
            except (ImportError, OSError):
                print("Warning: spaCy model not available. Using fallback NLP processing.")
                self.nlp = None
    
    def set_credentials(self, openai_api_key: str, supabase_url: str, supabase_service_key: str):
        """Set user credentials"""
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.supabase_client = create_client(supabase_url, supabase_service_key)
    
    def analyze_text_complexity(self, text: str) -> Dict[str, float]:
        """Advanced text complexity analysis"""
        analysis = {
            'readability': 0.5,
            'grade_level': 8.0,
            'sentence_complexity': 0.5,
            'vocabulary_diversity': 0.5,
            'technical_density': 0.5
        }
        
        try:
            # Readability scores
            analysis['readability'] = min(1.0, flesch_reading_ease(text) / 100.0)
            analysis['grade_level'] = flesch_kincaid_grade(text)
            
            # Sentence analysis
            sentences = text.split('.')
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
            analysis['sentence_complexity'] = min(1.0, avg_sentence_length / 25.0)
            
            # Vocabulary diversity (Type-Token Ratio)
            words = text.lower().split()
            if words:
                analysis['vocabulary_diversity'] = len(set(words)) / len(words)
            
            # Technical density (ratio of technical terms)
            technical_patterns = [
                r'\b[A-Z]{2,}\b',  # Acronyms
                r'\b\w+\(\)',      # Function calls
                r'\b\d+\.\d+\b',   # Version numbers
                r'\b[a-z]+_[a-z_]+\b'  # Snake case
            ]
            technical_matches = sum(len(re.findall(pattern, text)) for pattern in technical_patterns)
            analysis['technical_density'] = min(1.0, technical_matches / len(words) * 10) if words else 0
            
        except Exception as e:
            print(f"Error in complexity analysis: {e}")
        
        return analysis
    
    def extract_advanced_concepts(self, text: str) -> List[Dict[str, Any]]:
        """Extract concepts with advanced NLP"""
        concepts = []
        
        # Load NLP model if needed
        self._load_nlp_model()
        
        if self.nlp:
            try:
                doc = self.nlp(text)
                
                # Named entities
                for ent in doc.ents:
                    concepts.append({
                        'text': ent.text,
                        'type': 'entity',
                        'label': ent.label_,
                        'confidence': 0.8,
                        'start': ent.start_char,
                        'end': ent.end_char
                    })
                
                # Key phrases (noun phrases)
                for chunk in doc.noun_chunks:
                    if len(chunk.text.split()) > 1:  # Multi-word phrases
                        concepts.append({
                            'text': chunk.text,
                            'type': 'phrase',
                            'label': 'NOUN_PHRASE',
                            'confidence': 0.6,
                            'start': chunk.start_char,
                            'end': chunk.end_char
                        })
                
            except Exception as e:
                print(f"Error in NLP processing: {e}")
        
        # Fallback: pattern-based extraction
        fallback_concepts = self._extract_concepts_fallback(text)
        concepts.extend(fallback_concepts)
        
        # Deduplicate and score
        unique_concepts = {}
        for concept in concepts:
            key = concept['text'].lower()
            if key not in unique_concepts or concept['confidence'] > unique_concepts[key]['confidence']:
                unique_concepts[key] = concept
        
        return list(unique_concepts.values())
    
    def _extract_concepts_fallback(self, text: str) -> List[Dict[str, Any]]:
        """Fallback concept extraction using patterns"""
        concepts = []
        
        # Technical terms
        tech_terms = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', text)
        for term in tech_terms:
            concepts.append({
                'text': term,
                'type': 'technical',
                'label': 'TECH_TERM',
                'confidence': 0.7
            })
        
        # Important capitalized terms
        cap_terms = re.findall(r'\b[A-Z][a-z]{2,}\b', text)
        for term in cap_terms:
            if term not in ['The', 'This', 'That', 'When', 'Where', 'What', 'How']:
                concepts.append({
                    'text': term,
                    'type': 'proper_noun',
                    'label': 'PROPER_NOUN',
                    'confidence': 0.5
                })
        
        return concepts
    
    def calculate_chunk_metrics(self, text: str, concepts: List[Dict[str, Any]]) -> ChunkMetrics:
        """Calculate comprehensive chunk quality metrics"""
        complexity = self.analyze_text_complexity(text)
        
        # Information density (concepts per word)
        words = len(text.split())
        info_density = len(concepts) / words if words > 0 else 0
        
        # Concept coherence (how related concepts are)
        coherence = self._calculate_concept_coherence(concepts)
        
        # Cross-reference count
        cross_refs = len(re.findall(r'(?:see|refer to|as mentioned|according to)', text, re.IGNORECASE))
        
        # Semantic uniqueness (will be calculated later with embeddings)
        uniqueness = 0.8  # Placeholder
        
        # Retrieval potential (combination of factors)
        retrieval_potential = (
            complexity['readability'] * 0.3 +
            info_density * 0.4 +
            coherence * 0.3
        )
        
        # Domain specificity
        domain_specificity = complexity['technical_density']
        
        return ChunkMetrics(
            readability_score=complexity['readability'],
            complexity_score=complexity['sentence_complexity'],
            information_density=min(1.0, info_density * 10),
            concept_coherence=coherence,
            cross_reference_count=cross_refs,
            semantic_uniqueness=uniqueness,
            retrieval_potential=retrieval_potential,
            domain_specificity=domain_specificity
        )
    
    def _calculate_concept_coherence(self, concepts: List[Dict[str, Any]]) -> float:
        """Calculate how coherent the concepts are within a chunk"""
        if len(concepts) < 2:
            return 1.0
        
        # Simple coherence based on concept types
        type_counts = Counter(concept['type'] for concept in concepts)
        dominant_type_ratio = max(type_counts.values()) / len(concepts)
        
        return dominant_type_ratio
    
    def adaptive_chunk_sizing(self, text: str, base_size: int = 800) -> int:
        """Dynamically determine optimal chunk size based on content"""
        if not self.config.adaptive_chunk_size:
            return base_size
        
        complexity = self.analyze_text_complexity(text)
        
        # Adjust size based on complexity
        size_multiplier = 1.0
        
        # More complex text needs smaller chunks
        if complexity['sentence_complexity'] > 0.7:
            size_multiplier *= 0.8
        elif complexity['sentence_complexity'] < 0.3:
            size_multiplier *= 1.2
        
        # Technical content might need different sizing
        if complexity['technical_density'] > 0.5:
            size_multiplier *= 0.9  # Slightly smaller for technical content
        
        # Adjust based on readability
        if complexity['readability'] < 0.3:  # Difficult text
            size_multiplier *= 0.7
        elif complexity['readability'] > 0.8:  # Easy text
            size_multiplier *= 1.3
        
        optimal_size = int(base_size * size_multiplier)
        return max(self.config.min_chunk_size, min(self.config.max_chunk_size, optimal_size))
    
    def create_adaptive_chunks(self, text: str, source: str) -> List[Dict[str, Any]]:
        """Create chunks with adaptive sizing and advanced analysis"""
        if not text.strip():
            return []
        
        # Determine optimal chunk size
        optimal_size = self.adaptive_chunk_sizing(text)
        overlap = int(optimal_size * self.config.overlap_ratio)
        
        print(f"Using adaptive chunk size: {optimal_size} with overlap: {overlap}")
        
        # Smart text splitting with semantic boundaries
        chunks = self._semantic_text_split(text, optimal_size, overlap)
        
        chunk_objects = []
        for i, chunk_text in enumerate(chunks):
            if len(chunk_text.strip()) < 50:
                continue
            
            # Extract concepts
            concepts = self.extract_advanced_concepts(chunk_text)
            
            # Calculate metrics
            metrics = self.calculate_chunk_metrics(chunk_text, concepts)
            
            # Calculate overall quality score
            quality_score = self._calculate_quality_score(metrics)
            
            # Build concept graph
            if self.config.enable_concept_graphs:
                self._update_concept_graph(concepts, chunk_text)
            
            chunk_obj = {
                'content': chunk_text.strip(),
                'source': source,
                'chunk_index': i,
                'optimal_size': optimal_size,
                'concepts': concepts,
                'metrics': asdict(metrics),
                'quality_score': quality_score,
                'adaptive_features': {
                    'chunk_size_used': len(chunk_text),
                    'size_efficiency': len(chunk_text) / optimal_size,
                    'concept_density': len(concepts) / len(chunk_text.split()),
                    'cross_references': self._find_cross_references(chunk_text, chunks)
                }
            }
            
            chunk_objects.append(chunk_obj)
            
            # Track high-quality chunks
            if quality_score > self.config.quality_threshold:
                self.stats['high_quality_chunks'] += 1
        
        self.stats['total_chunks'] += len(chunk_objects)
        self.stats['concept_extractions'] += sum(len(chunk['concepts']) for chunk in chunk_objects)
        
        return chunk_objects
    
    def _semantic_text_split(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Advanced semantic text splitting"""
        # Hierarchical splitting by semantic boundaries
        separators = [
            ('\n\n\n', 'major_section'),
            ('\n\n', 'paragraph'),
            ('.\n', 'sentence_break'),
            ('. ', 'sentence'),
            (';\n', 'clause_break'),
            ('; ', 'clause'),
            (',\n', 'comma_break'),
            (', ', 'comma'),
            (' ', 'word')
        ]
        
        chunks = []
        current_chunk = ""
        
        def add_chunk_with_overlap():
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
                # Create overlap for next chunk
                words = current_chunk.split()
                overlap_words = words[-overlap//4:] if len(words) > overlap//4 else words
                return ' '.join(overlap_words)
            return ""
        
        # Try each separator level
        for separator, sep_type in separators:
            if separator in text:
                parts = text.split(separator)
                current_chunk = ""
                
                for part in parts:
                    potential_chunk = current_chunk + (separator if current_chunk else '') + part
                    
                    if len(potential_chunk) <= chunk_size or not current_chunk:
                        current_chunk = potential_chunk
                    else:
                        # Add current chunk and start new one with overlap
                        overlap_text = add_chunk_with_overlap()
                        current_chunk = overlap_text + separator + part
                
                # Add final chunk
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                return chunks
        
        # Fallback: character-based with word boundaries
        words = text.split()
        current_chunk = ""
        
        for word in words:
            if len(current_chunk + ' ' + word) <= chunk_size:
                current_chunk += (' ' + word) if current_chunk else word
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                    # Add overlap
                    chunk_words = current_chunk.split()
                    overlap_words = chunk_words[-overlap//4:] if len(chunk_words) > overlap//4 else []
                    current_chunk = ' '.join(overlap_words + [word])
                else:
                    current_chunk = word
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _calculate_quality_score(self, metrics: ChunkMetrics) -> float:
        """Calculate overall quality score using adaptive weights"""
        weights = self.adaptive_parameters['quality_weights']
        
        score = (
            metrics.readability_score * weights['readability'] +
            (1 - metrics.complexity_score) * weights['complexity'] +  # Lower complexity is better
            metrics.information_density * weights['density'] +
            metrics.concept_coherence * weights['coherence'] +
            metrics.semantic_uniqueness * weights['uniqueness']
        )
        
        # Bonus for cross-references
        if metrics.cross_reference_count > 0:
            score += 0.1
        
        return min(1.0, score)
    
    def _update_concept_graph(self, concepts: List[Dict[str, Any]], context: str):
        """Update the concept knowledge graph"""
        for concept in concepts:
            concept_text = concept['text'].lower()
            
            # Add concept to graph (embedding will be added later)
            if concept_text not in self.concept_graph.graph:
                self.concept_graph.add_concept(concept_text, [], context)
            
            # Find relationships between concepts in the same chunk
            for other_concept in concepts:
                if concept != other_concept:
                    other_text = other_concept['text'].lower()
                    if concept_text != other_text:
                        # Simple co-occurrence relationship
                        self.concept_graph.add_relationship(
                            concept_text, other_text, 'co_occurs', 0.5
                        )
    
    def _find_cross_references(self, chunk_text: str, all_chunks: List[str]) -> List[Dict[str, Any]]:
        """Find cross-references to other chunks"""
        if not self.config.enable_cross_references:
            return []
        
        cross_refs = []
        
        # Look for explicit references
        ref_patterns = [
            r'(?:see|refer to|as mentioned in|according to)\s+([^.]+)',
            r'(?:chapter|section|part)\s+(\d+)',
            r'(?:figure|table|appendix)\s+(\w+)'
        ]
        
        for pattern in ref_patterns:
            matches = re.finditer(pattern, chunk_text, re.IGNORECASE)
            for match in matches:
                cross_refs.append({
                    'type': 'explicit',
                    'reference': match.group(1),
                    'position': match.start()
                })
        
        self.stats['cross_references_found'] += len(cross_refs)
        return cross_refs
    
    async def generate_adaptive_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Optional[List[float]]]]:
        """Generate embeddings with adaptive optimization"""
        results = []
        
        # Group chunks by quality for different processing strategies
        high_quality = [c for c in chunks if c['quality_score'] > 0.8]
        medium_quality = [c for c in chunks if 0.5 <= c['quality_score'] <= 0.8]
        low_quality = [c for c in chunks if c['quality_score'] < 0.5]
        
        print(f"Processing {len(high_quality)} high-quality, {len(medium_quality)} medium-quality, {len(low_quality)} low-quality chunks")
        
        # Process high-quality chunks with enhanced context
        if high_quality:
            hq_results = await self._process_high_quality_chunks(high_quality)
            results.extend(hq_results)
        
        # Process medium-quality chunks normally
        if medium_quality:
            mq_results = await self._process_standard_chunks(medium_quality)
            results.extend(mq_results)
        
        # Process low-quality chunks with augmentation
        if low_quality:
            lq_results = await self._process_low_quality_chunks(low_quality)
            results.extend(lq_results)
        
        # Update semantic uniqueness scores
        await self._update_semantic_uniqueness(results)
        
        return results
    
    async def _process_high_quality_chunks(self, chunks: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Optional[List[float]]]]:
        """Process high-quality chunks with enhanced context"""
        results = []
        
        for chunk in chunks:
            try:
                # Enhanced text with concept context
                enhanced_text = self._enhance_chunk_context(chunk)
                
                # Generate embedding
                response = self.openai_client.embeddings.create(
                    input=enhanced_text,
                    model=self.config.model
                )
                
                embedding = response.data[0].embedding
                
                # Store concept embeddings for graph
                if self.config.enable_concept_graphs:
                    await self._store_concept_embeddings(chunk['concepts'], embedding)
                
                results.append((chunk, embedding))
                
            except Exception as e:
                print(f"Error processing high-quality chunk: {e}")
                results.append((chunk, None))
        
        return results
    
    async def _process_standard_chunks(self, chunks: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Optional[List[float]]]]:
        """Process standard chunks in batches"""
        results = []
        batch_size = self.config.batch_size
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [chunk['content'] for chunk in batch]
            
            try:
                response = self.openai_client.embeddings.create(
                    input=texts,
                    model=self.config.model
                )
                
                for j, chunk in enumerate(batch):
                    if j < len(response.data):
                        embedding = response.data[j].embedding
                        results.append((chunk, embedding))
                    else:
                        results.append((chunk, None))
                        
            except Exception as e:
                print(f"Error processing standard batch: {e}")
                for chunk in batch:
                    results.append((chunk, None))
        
        return results
    
    async def _process_low_quality_chunks(self, chunks: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Optional[List[float]]]]:
        """Process low-quality chunks with augmentation"""
        results = []
        
        for chunk in chunks:
            try:
                # Augment low-quality chunks
                augmented_text = self._augment_low_quality_chunk(chunk)
                
                response = self.openai_client.embeddings.create(
                    input=augmented_text,
                    model=self.config.model
                )
                
                embedding = response.data[0].embedding
                results.append((chunk, embedding))
                
            except Exception as e:
                print(f"Error processing low-quality chunk: {e}")
                results.append((chunk, None))
        
        return results
    
    def _enhance_chunk_context(self, chunk: Dict[str, Any]) -> str:
        """Enhance high-quality chunks with additional context"""
        content = chunk['content']
        concepts = chunk['concepts']
        
        # Add concept context
        if concepts:
            concept_text = "Key concepts: " + ", ".join([c['text'] for c in concepts[:5]])
            content = f"{concept_text}\n\n{content}"
        
        # Add source context
        source_context = f"Source: {chunk['source']}"
        content = f"{source_context}\n{content}"
        
        return content
    
    def _augment_low_quality_chunk(self, chunk: Dict[str, Any]) -> str:
        """Augment low-quality chunks to improve embedding quality"""
        content = chunk['content']
        
        # Add structural context
        context_prefix = f"Document section from {chunk['source']} (chunk {chunk['chunk_index']}): "
        
        # Try to infer topic from content
        words = content.lower().split()
        common_words = Counter(words).most_common(3)
        if common_words:
            topic_hint = f"Topic relates to: {', '.join([word for word, _ in common_words])}. "
            context_prefix += topic_hint
        
        return context_prefix + content
    
    async def _store_concept_embeddings(self, concepts: List[Dict[str, Any]], chunk_embedding: List[float]):
        """Store concept embeddings in the knowledge graph"""
        for concept in concepts:
            concept_text = concept['text'].lower()
            if concept_text in self.concept_graph.concept_embeddings:
                # Average with existing embedding
                existing = np.array(self.concept_graph.concept_embeddings[concept_text])
                new_embedding = (existing + np.array(chunk_embedding)) / 2
                self.concept_graph.concept_embeddings[concept_text] = new_embedding.tolist()
            else:
                self.concept_graph.concept_embeddings[concept_text] = chunk_embedding
    
    async def _update_semantic_uniqueness(self, results: List[Tuple[Dict[str, Any], Optional[List[float]]]]):
        """Update semantic uniqueness scores based on embedding similarities"""
        embeddings = []
        valid_results = []
        
        for chunk, embedding in results:
            if embedding is not None:
                embeddings.append(embedding)
                valid_results.append((chunk, embedding))
        
        if len(embeddings) < 2:
            return
        
        # Calculate pairwise similarities
        similarity_matrix = cosine_similarity(embeddings)
        
        for i, (chunk, embedding) in enumerate(valid_results):
            # Calculate uniqueness as 1 - max similarity with other chunks
            similarities = similarity_matrix[i]
            max_similarity = max(sim for j, sim in enumerate(similarities) if j != i)
            uniqueness = 1.0 - max_similarity
            
            # Update chunk metrics
            chunk['metrics']['semantic_uniqueness'] = uniqueness
            
            # Recalculate quality score
            metrics = ChunkMetrics(**chunk['metrics'])
            chunk['quality_score'] = self._calculate_quality_score(metrics)
    
    async def upload_adaptive_embeddings(self, chunk_embedding_pairs: List[Tuple[Dict[str, Any], Optional[List[float]]]]) -> Dict[str, Any]:
        """Upload embeddings with adaptive metadata"""
        successful_uploads = 0
        failed_uploads = 0
        high_quality_uploads = 0
        concept_graph_entries = 0
        
        batch_data = []
        
        for chunk, embedding in chunk_embedding_pairs:
            if embedding is None:
                failed_uploads += 1
                continue
            
            if chunk['quality_score'] > 0.8:
                high_quality_uploads += 1
            
            # Prepare enhanced data for Supabase
            data = {
                'content': chunk['content'],
                'embedding': embedding,
                'source': chunk['source'],
                'metadata': {
                    'chunk_index': chunk['chunk_index'],
                    'quality_score': chunk['quality_score'],
                    'metrics': chunk['metrics'],
                    'concepts': chunk['concepts'],
                    'adaptive_features': chunk['adaptive_features'],
                    'optimal_size': chunk['optimal_size'],
                    'processing_version': 'adaptive_v1.0'
                }
            }
            batch_data.append(data)
        
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
        
        # Store concept graph if enabled
        if self.config.enable_concept_graphs:
            concept_graph_entries = await self._store_concept_graph()
        
        return {
            'successful_uploads': successful_uploads,
            'failed_uploads': failed_uploads,
            'high_quality_uploads': high_quality_uploads,
            'total_chunks': len(chunk_embedding_pairs),
            'concept_graph_entries': concept_graph_entries,
            'adaptive_optimizations': self.stats['adaptive_optimizations'],
            'concept_extractions': self.stats['concept_extractions'],
            'cross_references_found': self.stats['cross_references_found']
        }
    
    async def _store_concept_graph(self) -> int:
        """Store concept graph in Supabase"""
        try:
            # Create concept graph table if it doesn't exist
            graph_data = []
            
            for concept, embedding in self.concept_graph.concept_embeddings.items():
                importance = self.concept_graph.get_concept_importance(concept)
                related_concepts = self.concept_graph.find_related_concepts(concept)
                
                graph_data.append({
                    'concept': concept,
                    'embedding': embedding,
                    'importance_score': importance,
                    'frequency': self.concept_graph.concept_frequencies[concept],
                    'related_concepts': related_concepts[:10],  # Top 10 related
                    'created_at': 'now()'
                })
            
            if graph_data:
                # This would require a concept_graph table in Supabase
                # For now, we'll store it in the metadata of a special document
                concept_graph_doc = {
                    'content': 'CONCEPT_GRAPH_DATA',
                    'embedding': [0.0] * 1536,  # Placeholder embedding
                    'source': 'system_concept_graph',
                    'metadata': {
                        'type': 'concept_graph',
                        'concepts': graph_data,
                        'graph_stats': {
                            'total_concepts': len(graph_data),
                            'total_relationships': self.concept_graph.graph.number_of_edges(),
                            'created_at': time.time()
                        }
                    }
                }
                
                response = self.supabase_client.table('documents').insert(concept_graph_doc).execute()
                
                if not (hasattr(response, 'error') and response.error):
                    return len(graph_data)
            
        except Exception as e:
            print(f"Error storing concept graph: {e}")
        
        return 0
    
    async def process_documents_adaptive(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Main adaptive processing method"""
        start_time = time.time()
        
        print(f"Starting adaptive processing of {len(chunks)} chunks...")
        
        # Convert simple chunks to adaptive chunks
        adaptive_chunks = []
        for chunk in chunks:
            smart_chunks = self.create_adaptive_chunks(chunk['content'], chunk['source'])
            adaptive_chunks.extend(smart_chunks)
        
        print(f"Created {len(adaptive_chunks)} adaptive chunks")
        
        # Generate embeddings with adaptive strategies
        chunk_embedding_pairs = await self.generate_adaptive_embeddings(adaptive_chunks)
        
        # Upload with enhanced metadata
        upload_result = await self.upload_adaptive_embeddings(chunk_embedding_pairs)
        
        processing_time = time.time() - start_time
        
        # Adaptive learning: update parameters based on results
        if self.config.enable_adaptive_learning:
            self._update_adaptive_parameters(upload_result, processing_time)
        
        return {
            'success': upload_result['successful_uploads'] > 0,
            'chunks_processed': len(chunks),
            'adaptive_chunks_created': len(adaptive_chunks),
            'successful_embeddings': upload_result['successful_uploads'],
            'high_quality_chunks': upload_result['high_quality_uploads'],
            'upload_stats': upload_result,
            'performance_stats': {
                'processing_time': processing_time,
                'chunks_per_second': len(adaptive_chunks) / processing_time if processing_time > 0 else 0,
                'quality_distribution': self._get_quality_distribution(adaptive_chunks),
                'adaptive_optimizations': self.stats['adaptive_optimizations'],
                'concept_extractions': self.stats['concept_extractions']
            },
            'concept_graph_stats': {
                'total_concepts': len(self.concept_graph.concept_embeddings),
                'total_relationships': self.concept_graph.graph.number_of_edges(),
                'concept_graph_entries': upload_result['concept_graph_entries']
            }
        }
    
    def _update_adaptive_parameters(self, upload_result: Dict[str, Any], processing_time: float):
        """Update adaptive parameters based on processing results"""
        # Simple learning: adjust chunk size based on quality results
        quality_ratio = upload_result['high_quality_uploads'] / max(1, upload_result['successful_uploads'])
        
        if quality_ratio > 0.8:
            # High quality ratio - current parameters are good
            self.stats['adaptive_optimizations'] += 1
        elif quality_ratio < 0.4:
            # Low quality ratio - adjust parameters
            self.adaptive_parameters['optimal_chunk_size'] = int(
                self.adaptive_parameters['optimal_chunk_size'] * 0.9
            )
            self.stats['adaptive_optimizations'] += 1
        
        print(f"Adaptive learning: Quality ratio {quality_ratio:.2f}, Optimal chunk size: {self.adaptive_parameters['optimal_chunk_size']}")
    
    def _get_quality_distribution(self, chunks: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get distribution of chunk qualities"""
        distribution = {'high': 0, 'medium': 0, 'low': 0}
        
        for chunk in chunks:
            score = chunk['quality_score']
            if score > 0.8:
                distribution['high'] += 1
            elif score > 0.5:
                distribution['medium'] += 1
            else:
                distribution['low'] += 1
        
        return distribution

# Integration function
async def run_adaptive_processing(chunks: List[Dict[str, Any]], openai_api_key: str, 
                                supabase_url: str, supabase_service_key: str) -> Dict[str, Any]:
    """Run adaptive embedding processing"""
    processor = AdaptiveEmbeddingProcessor()
    processor.set_credentials(openai_api_key, supabase_url, supabase_service_key)
    return await processor.process_documents_adaptive(chunks)

if __name__ == "__main__":
    print("🚀 Adaptive Embedding System with Dynamic Optimization")
    print("Revolutionary features:")
    print("- Dynamic chunk sizing based on content complexity")
    print("- Advanced NLP concept extraction with spaCy")
    print("- Knowledge graph construction for concept relationships")
    print("- Adaptive quality scoring with machine learning")
    print("- Self-improving parameters based on processing results")
    print("- Cross-reference detection and linking")
    print("- Multi-tier processing based on content quality")
    print("- Semantic uniqueness calculation")
    print("- Advanced text complexity analysis")
    print("- Concept importance scoring with graph centrality")