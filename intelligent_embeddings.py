#!/usr/bin/env python3
"""
Intelligent Embedding System with Categorization and Smart Retrieval
Implements semantic understanding, content categorization, and optimized vector search
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
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import re

@dataclass
class ContentCategory:
    """Content category with semantic understanding"""
    name: str
    keywords: List[str]
    patterns: List[str]
    weight: float = 1.0
    embedding_strategy: str = "default"  # default, detailed, summary

@dataclass
class EmbeddingMetadata:
    """Rich metadata for embeddings"""
    content_type: str
    category: str
    confidence: float
    semantic_density: float
    key_concepts: List[str]
    relationships: List[str]
    chunk_quality: float
    retrieval_hints: List[str]

class IntelligentEmbeddingProcessor:
    """Advanced embedding processor with semantic understanding"""
    
    def __init__(self):
        self.openai_client = None
        self.supabase_client = None
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        
        # Content categories for intelligent processing
        self.categories = self._initialize_categories()
        
        # Performance and quality tracking
        self.stats = {
            'total_chunks': 0,
            'categorized_chunks': 0,
            'high_quality_chunks': 0,
            'embedding_cache_hits': 0,
            'semantic_clusters': 0,
            'processing_time': 0
        }
        
        # Embedding cache for deduplication
        self.embedding_cache = {}
        
        # Semantic clusters for related content
        self.semantic_clusters = {}
    
    def _initialize_categories(self) -> Dict[str, ContentCategory]:
        """Initialize content categories for intelligent processing"""
        return {
            'technical': ContentCategory(
                name='Technical Documentation',
                keywords=['api', 'function', 'class', 'method', 'algorithm', 'implementation', 'code', 'syntax'],
                patterns=[r'\b(def|class|function|method)\b', r'\b[A-Z][a-zA-Z]*\(\)', r'```.*```'],
                weight=1.2,
                embedding_strategy='detailed'
            ),
            'business': ContentCategory(
                name='Business Content',
                keywords=['strategy', 'revenue', 'market', 'customer', 'business', 'profit', 'growth', 'analysis'],
                patterns=[r'\$[\d,]+', r'\b\d+%\b', r'\b(Q[1-4]|quarter|fiscal)\b'],
                weight=1.1,
                embedding_strategy='summary'
            ),
            'legal': ContentCategory(
                name='Legal Documents',
                keywords=['contract', 'agreement', 'terms', 'conditions', 'liability', 'compliance', 'regulation'],
                patterns=[r'\b(shall|hereby|whereas|therefore)\b', r'Section \d+', r'\b[A-Z]{2,}\b'],
                weight=1.3,
                embedding_strategy='detailed'
            ),
            'research': ContentCategory(
                name='Research & Academic',
                keywords=['study', 'research', 'analysis', 'methodology', 'results', 'conclusion', 'hypothesis'],
                patterns=[r'\b(Figure|Table|Appendix)\s+\d+', r'\[\d+\]', r'\b(et al\.|PhD|Dr\.)\b'],
                weight=1.2,
                embedding_strategy='detailed'
            ),
            'general': ContentCategory(
                name='General Content',
                keywords=['information', 'description', 'overview', 'summary', 'details'],
                patterns=[],
                weight=1.0,
                embedding_strategy='default'
            )
        }
    
    def set_credentials(self, openai_api_key: str, supabase_url: str, supabase_service_key: str):
        """Set user credentials"""
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.supabase_client = create_client(supabase_url, supabase_service_key)
    
    def analyze_content_semantics(self, text: str) -> Dict[str, Any]:
        """Analyze content for semantic understanding"""
        analysis = {
            'key_concepts': [],
            'semantic_density': 0.0,
            'content_complexity': 0.0,
            'relationships': [],
            'retrieval_hints': []
        }
        
        # Extract key concepts using NLP patterns
        concepts = self._extract_key_concepts(text)
        analysis['key_concepts'] = concepts[:10]  # Top 10 concepts
        
        # Calculate semantic density (information richness)
        analysis['semantic_density'] = self._calculate_semantic_density(text)
        
        # Analyze content complexity
        analysis['content_complexity'] = self._analyze_complexity(text)
        
        # Find relationships and references
        analysis['relationships'] = self._find_relationships(text)
        
        # Generate retrieval hints
        analysis['retrieval_hints'] = self._generate_retrieval_hints(text, concepts)
        
        return analysis
    
    def _extract_key_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text"""
        # Simple but effective concept extraction
        concepts = []
        
        # Technical terms (CamelCase, snake_case)
        tech_terms = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b|\b[a-z]+_[a-z_]+\b', text)
        concepts.extend(tech_terms)
        
        # Important nouns (capitalized words that aren't sentence starters)
        important_nouns = re.findall(r'(?<!^)(?<!\. )\b[A-Z][a-z]+\b', text)
        concepts.extend(important_nouns)
        
        # Numbers and measurements
        numbers = re.findall(r'\b\d+(?:\.\d+)?(?:%|px|em|rem|kg|lb|ft|m|km|mb|gb|tb)?\b', text)
        concepts.extend(numbers)
        
        # Remove duplicates and return most frequent
        concept_counts = {}
        for concept in concepts:
            concept_counts[concept] = concept_counts.get(concept, 0) + 1
        
        return sorted(concept_counts.keys(), key=lambda x: concept_counts[x], reverse=True)
    
    def _calculate_semantic_density(self, text: str) -> float:
        """Calculate semantic density of content"""
        words = text.split()
        if not words:
            return 0.0
        
        # Factors that increase semantic density
        unique_words = len(set(words))
        avg_word_length = sum(len(word) for word in words) / len(words)
        technical_terms = len(re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', text))
        numbers = len(re.findall(r'\b\d+', text))
        
        # Normalize to 0-1 scale
        density = min(1.0, (unique_words / len(words)) * 0.5 + 
                     (avg_word_length / 10) * 0.2 + 
                     (technical_terms / len(words)) * 0.2 + 
                     (numbers / len(words)) * 0.1)
        
        return round(density, 3)
    
    def _analyze_complexity(self, text: str) -> float:
        """Analyze content complexity"""
        sentences = text.split('.')
        if not sentences:
            return 0.0
        
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        nested_structures = len(re.findall(r'[({[].*[)}\]]', text))
        
        complexity = min(1.0, (avg_sentence_length / 20) * 0.7 + (nested_structures / len(sentences)) * 0.3)
        return round(complexity, 3)
    
    def _find_relationships(self, text: str) -> List[str]:
        """Find relationships and references in text"""
        relationships = []
        
        # Cross-references
        refs = re.findall(r'(?:see|refer to|as mentioned in|according to)\s+([^.]+)', text, re.IGNORECASE)
        relationships.extend(refs[:5])
        
        # Dependencies
        deps = re.findall(r'(?:depends on|requires|needs|uses)\s+([^.]+)', text, re.IGNORECASE)
        relationships.extend(deps[:5])
        
        return relationships
    
    def _generate_retrieval_hints(self, text: str, concepts: List[str]) -> List[str]:
        """Generate hints for better retrieval"""
        hints = []
        
        # Add top concepts as hints
        hints.extend(concepts[:5])
        
        # Add context hints
        if 'how to' in text.lower():
            hints.append('tutorial')
        if any(word in text.lower() for word in ['error', 'problem', 'issue', 'bug']):
            hints.append('troubleshooting')
        if any(word in text.lower() for word in ['example', 'sample', 'demo']):
            hints.append('example')
        
        return hints
    
    def categorize_content(self, text: str) -> Tuple[str, float]:
        """Categorize content and return category with confidence"""
        scores = {}
        
        for category_id, category in self.categories.items():
            score = 0.0
            
            # Keyword matching
            keyword_matches = sum(1 for keyword in category.keywords 
                                if keyword.lower() in text.lower())
            score += (keyword_matches / len(category.keywords)) * 0.6
            
            # Pattern matching
            if category.patterns:
                pattern_matches = sum(1 for pattern in category.patterns 
                                    if re.search(pattern, text, re.IGNORECASE))
                score += (pattern_matches / len(category.patterns)) * 0.4
            
            scores[category_id] = score
        
        # Get best category
        best_category = max(scores.keys(), key=lambda k: scores[k])
        confidence = scores[best_category]
        
        # If confidence is too low, use general category
        if confidence < 0.1:
            return 'general', 0.5
        
        return best_category, min(1.0, confidence)
    
    def create_intelligent_chunks(self, text: str, source: str) -> List[Dict[str, Any]]:
        """Create semantically aware chunks with rich metadata"""
        if not text.strip():
            return []
        
        # Analyze overall content
        content_analysis = self.analyze_content_semantics(text)
        category, category_confidence = self.categorize_content(text)
        
        # Get category-specific chunking strategy
        category_obj = self.categories[category]
        
        # Adjust chunk size based on category
        if category_obj.embedding_strategy == 'detailed':
            chunk_size = 600  # Smaller chunks for detailed content
            overlap = 150
        elif category_obj.embedding_strategy == 'summary':
            chunk_size = 1000  # Larger chunks for summary content
            overlap = 100
        else:
            chunk_size = 800  # Default
            overlap = 100
        
        # Smart text splitting with semantic awareness
        chunks = self._smart_text_split(text, chunk_size, overlap)
        
        # Create chunk objects with rich metadata
        chunk_objects = []
        for i, chunk_text in enumerate(chunks):
            if len(chunk_text.strip()) < 50:  # Skip very short chunks
                continue
            
            # Analyze individual chunk
            chunk_analysis = self.analyze_content_semantics(chunk_text)
            chunk_category, chunk_confidence = self.categorize_content(chunk_text)
            
            # Calculate chunk quality score
            quality_score = self._calculate_chunk_quality(chunk_text, chunk_analysis)
            
            # Create rich metadata
            metadata = EmbeddingMetadata(
                content_type=self._detect_content_type(chunk_text),
                category=chunk_category,
                confidence=chunk_confidence,
                semantic_density=chunk_analysis['semantic_density'],
                key_concepts=chunk_analysis['key_concepts'],
                relationships=chunk_analysis['relationships'],
                chunk_quality=quality_score,
                retrieval_hints=chunk_analysis['retrieval_hints']
            )
            
            chunk_obj = {
                'content': chunk_text.strip(),
                'source': source,
                'chunk_index': i,
                'category': chunk_category,
                'category_confidence': chunk_confidence,
                'quality_score': quality_score,
                'semantic_metadata': asdict(metadata),
                'embedding_strategy': category_obj.embedding_strategy,
                'weight_multiplier': category_obj.weight
            }
            
            chunk_objects.append(chunk_obj)
        
        print(f"Created {len(chunk_objects)} intelligent chunks from {source}")
        print(f"Primary category: {category} (confidence: {category_confidence:.2f})")
        
        return chunk_objects
    
    def _smart_text_split(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Smart text splitting with semantic boundary awareness"""
        # Split by semantic boundaries in order of preference
        separators = [
            '\n\n\n',  # Major sections
            '\n\n',    # Paragraphs
            '\n',      # Lines
            '. ',      # Sentences
            ', ',      # Clauses
            ' '        # Words
        ]
        
        chunks = []
        current_chunk = ""
        
        # Try each separator level
        for separator in separators:
            if separator in text:
                parts = text.split(separator)
                
                for part in parts:
                    potential_chunk = current_chunk + (separator if current_chunk else '') + part
                    
                    if len(potential_chunk) <= chunk_size or not current_chunk:
                        current_chunk = potential_chunk
                    else:
                        # Save current chunk with overlap
                        if current_chunk:
                            chunks.append(current_chunk)
                            # Create overlap from end of current chunk
                            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                            current_chunk = overlap_text + separator + part
                        else:
                            current_chunk = part
                
                # Add final chunk
                if current_chunk:
                    chunks.append(current_chunk)
                
                return chunks
        
        # Fallback: character-based splitting
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def _detect_content_type(self, text: str) -> str:
        """Detect specific content type"""
        if re.search(r'```|def |class |function', text):
            return 'code'
        elif re.search(r'^\s*[-*+]\s+', text, re.MULTILINE):
            return 'list'
        elif re.search(r'^\s*\d+\.\s+', text, re.MULTILINE):
            return 'numbered_list'
        elif re.search(r'\|.*\|', text):
            return 'table'
        elif len(text.split('.')) > 5:
            return 'prose'
        else:
            return 'fragment'
    
    def _calculate_chunk_quality(self, text: str, analysis: Dict[str, Any]) -> float:
        """Calculate chunk quality score"""
        factors = {
            'length': min(1.0, len(text) / 500),  # Optimal around 500 chars
            'semantic_density': analysis['semantic_density'],
            'concept_richness': min(1.0, len(analysis['key_concepts']) / 5),
            'completeness': 1.0 if text.strip().endswith(('.', '!', '?', ':')) else 0.7
        }
        
        # Weighted average
        weights = {'length': 0.2, 'semantic_density': 0.4, 'concept_richness': 0.3, 'completeness': 0.1}
        quality = sum(factors[k] * weights[k] for k in factors)
        
        return round(quality, 3)
    
    async def generate_smart_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Optional[List[float]]]]:
        """Generate embeddings with intelligent strategies"""
        results = []
        batch_size = 50  # Smaller batches for better control
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_results = await self._process_intelligent_batch(batch)
            results.extend(batch_results)
            
            # Progress update
            print(f"Processed {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")
            
            # Rate limiting
            if i + batch_size < len(chunks):
                await asyncio.sleep(0.1)
        
        return results
    
    async def _process_intelligent_batch(self, batch: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Optional[List[float]]]]:
        """Process batch with intelligent embedding strategies"""
        try:
            # Group by embedding strategy
            strategy_groups = {}
            for chunk in batch:
                strategy = chunk.get('embedding_strategy', 'default')
                if strategy not in strategy_groups:
                    strategy_groups[strategy] = []
                strategy_groups[strategy].append(chunk)
            
            results = []
            
            # Process each strategy group
            for strategy, chunks_group in strategy_groups.items():
                strategy_results = await self._process_strategy_group(chunks_group, strategy)
                results.extend(strategy_results)
            
            return results
            
        except Exception as e:
            print(f"Error in intelligent batch processing: {e}")
            return [(chunk, None) for chunk in batch]
    
    async def _process_strategy_group(self, chunks: List[Dict[str, Any]], strategy: str) -> List[Tuple[Dict[str, Any], Optional[List[float]]]]:
        """Process chunks with specific embedding strategy"""
        texts = []
        
        for chunk in chunks:
            content = chunk['content']
            
            # Apply strategy-specific preprocessing
            if strategy == 'detailed':
                # For detailed content, include more context
                processed_text = f"Content: {content}"
                if chunk['semantic_metadata']['key_concepts']:
                    concepts = ', '.join(chunk['semantic_metadata']['key_concepts'][:5])
                    processed_text += f" Key concepts: {concepts}"
            
            elif strategy == 'summary':
                # For summary content, focus on main points
                sentences = content.split('.')
                if len(sentences) > 3:
                    # Take first and last sentences for context
                    processed_text = f"{sentences[0]}. {sentences[-1]}."
                else:
                    processed_text = content
            
            else:  # default
                processed_text = content
            
            # Check cache first
            content_hash = hashlib.md5(processed_text.encode()).hexdigest()
            if content_hash in self.embedding_cache:
                self.stats['embedding_cache_hits'] += 1
                texts.append(None)  # Placeholder for cached embedding
            else:
                texts.append(processed_text)
        
        # Generate embeddings for non-cached texts
        embeddings = []
        if any(text is not None for text in texts):
            non_cached_texts = [text for text in texts if text is not None]
            
            response = self.openai_client.embeddings.create(
                input=non_cached_texts,
                model="text-embedding-3-small"
            )
            
            embedding_iter = iter(response.data)
            for text in texts:
                if text is None:
                    # Get from cache
                    content_hash = hashlib.md5(chunks[texts.index(text)]['content'].encode()).hexdigest()
                    embeddings.append(self.embedding_cache[content_hash])
                else:
                    # Get from API response
                    embedding = next(embedding_iter).embedding
                    # Cache it
                    content_hash = hashlib.md5(text.encode()).hexdigest()
                    self.embedding_cache[content_hash] = embedding
                    embeddings.append(embedding)
        
        # Pair chunks with embeddings
        results = []
        for i, chunk in enumerate(chunks):
            if i < len(embeddings):
                results.append((chunk, embeddings[i]))
            else:
                results.append((chunk, None))
        
        return results
    
    def create_semantic_clusters(self, embeddings: List[List[float]], chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create semantic clusters for related content"""
        if len(embeddings) < 3:
            return {'clusters': [], 'cluster_info': {}}
        
        # Convert to numpy array
        embedding_matrix = np.array(embeddings)
        
        # Determine optimal number of clusters
        n_clusters = min(max(2, len(embeddings) // 10), 20)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embedding_matrix)
        
        # Organize clusters
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append({
                'chunk_index': i,
                'chunk': chunks[i],
                'embedding': embeddings[i]
            })
        
        # Analyze clusters
        cluster_info = {}
        for cluster_id, cluster_items in clusters.items():
            # Find cluster centroid
            cluster_embeddings = [item['embedding'] for item in cluster_items]
            centroid = np.mean(cluster_embeddings, axis=0)
            
            # Find most representative chunk (closest to centroid)
            similarities = [cosine_similarity([centroid], [emb])[0][0] for emb in cluster_embeddings]
            representative_idx = np.argmax(similarities)
            
            # Extract common concepts
            all_concepts = []
            for item in cluster_items:
                all_concepts.extend(item['chunk']['semantic_metadata']['key_concepts'])
            
            concept_counts = {}
            for concept in all_concepts:
                concept_counts[concept] = concept_counts.get(concept, 0) + 1
            
            common_concepts = sorted(concept_counts.keys(), 
                                   key=lambda x: concept_counts[x], reverse=True)[:5]
            
            cluster_info[cluster_id] = {
                'size': len(cluster_items),
                'representative_chunk': cluster_items[representative_idx]['chunk']['content'][:200] + '...',
                'common_concepts': common_concepts,
                'avg_quality': np.mean([item['chunk']['quality_score'] for item in cluster_items]),
                'categories': list(set(item['chunk']['category'] for item in cluster_items))
            }
        
        self.stats['semantic_clusters'] = len(clusters)
        
        return {
            'clusters': clusters,
            'cluster_info': cluster_info,
            'cluster_labels': cluster_labels.tolist()
        }
    
    async def upload_intelligent_embeddings(self, chunk_embedding_pairs: List[Tuple[Dict[str, Any], Optional[List[float]]]]) -> Dict[str, Any]:
        """Upload embeddings with intelligent metadata to Supabase"""
        successful_uploads = 0
        failed_uploads = 0
        high_quality_uploads = 0
        
        # Extract valid embeddings for clustering
        valid_pairs = [(chunk, emb) for chunk, emb in chunk_embedding_pairs if emb is not None]
        
        if valid_pairs:
            embeddings = [pair[1] for pair in valid_pairs]
            chunks = [pair[0] for pair in valid_pairs]
            
            # Create semantic clusters
            cluster_info = self.create_semantic_clusters(embeddings, chunks)
            
            # Add cluster information to chunks
            for i, (chunk, embedding) in enumerate(valid_pairs):
                if i < len(cluster_info['cluster_labels']):
                    chunk['cluster_id'] = int(cluster_info['cluster_labels'][i])
        
        # Upload in batches
        batch_size = 25
        for i in range(0, len(chunk_embedding_pairs), batch_size):
            batch = chunk_embedding_pairs[i:i + batch_size]
            
            batch_data = []
            for chunk, embedding in batch:
                if embedding is None:
                    failed_uploads += 1
                    continue
                
                # Track high quality chunks
                if chunk['quality_score'] > 0.7:
                    high_quality_uploads += 1
                
                # Prepare enhanced data for Supabase
                data = {
                    'content': chunk['content'],
                    'embedding': embedding,
                    'source': chunk['source'],
                    'metadata': {
                        'chunk_index': chunk['chunk_index'],
                        'category': chunk['category'],
                        'category_confidence': chunk['category_confidence'],
                        'quality_score': chunk['quality_score'],
                        'semantic_metadata': chunk['semantic_metadata'],
                        'cluster_id': chunk.get('cluster_id'),
                        'weight_multiplier': chunk['weight_multiplier']
                    }
                }
                batch_data.append(data)
            
            # Upload batch
            if batch_data:
                try:
                    response = self.supabase_client.table('documents').insert(batch_data).execute()
                    
                    if hasattr(response, 'error') and response.error:
                        print(f"Batch upload error: {response.error}")
                        failed_uploads += len(batch_data)
                    else:
                        successful_uploads += len(batch_data)
                        
                except Exception as e:
                    print(f"Exception during batch upload: {e}")
                    failed_uploads += len(batch_data)
        
        return {
            'successful_uploads': successful_uploads,
            'failed_uploads': failed_uploads,
            'high_quality_uploads': high_quality_uploads,
            'total_chunks': len(chunk_embedding_pairs),
            'cache_hits': self.stats['embedding_cache_hits'],
            'semantic_clusters': self.stats['semantic_clusters']
        }

# Smart retrieval functions for RAG queries
class SmartRetriever:
    """Intelligent retrieval system for RAG queries"""
    
    def __init__(self, supabase_client, openai_client):
        self.supabase = supabase_client
        self.openai = openai_client
    
    async def smart_search(self, query: str, limit: int = 10, category_filter: str = None) -> List[Dict[str, Any]]:
        """Intelligent search with category awareness and semantic ranking"""
        
        # Generate query embedding
        query_embedding = await self._get_query_embedding(query)
        
        # Analyze query for category hints
        query_category = self._analyze_query_category(query)
        
        # Build search parameters
        search_params = {
            'query_embedding': query_embedding,
            'match_threshold': 0.7,
            'match_count': limit * 2  # Get more results for re-ranking
        }
        
        # Add category filter if specified or detected
        category_to_use = category_filter or query_category
        
        # Execute semantic search
        if category_to_use and category_to_use != 'general':
            # Category-aware search
            results = await self._category_aware_search(search_params, category_to_use)
        else:
            # General semantic search
            results = await self._general_semantic_search(search_params)
        
        # Re-rank results with intelligent scoring
        ranked_results = self._intelligent_rerank(query, results, limit)
        
        return ranked_results
    
    async def _get_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for search query"""
        response = self.openai.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    
    def _analyze_query_category(self, query: str) -> str:
        """Analyze query to determine likely content category"""
        query_lower = query.lower()
        
        # Technical queries
        if any(word in query_lower for word in ['how to', 'function', 'api', 'code', 'implement', 'algorithm']):
            return 'technical'
        
        # Business queries
        if any(word in query_lower for word in ['revenue', 'profit', 'market', 'business', 'strategy', 'growth']):
            return 'business'
        
        # Research queries
        if any(word in query_lower for word in ['study', 'research', 'analysis', 'methodology', 'results']):
            return 'research'
        
        # Legal queries
        if any(word in query_lower for word in ['contract', 'legal', 'terms', 'compliance', 'regulation']):
            return 'legal'
        
        return 'general'
    
    async def _category_aware_search(self, params: Dict[str, Any], category: str) -> List[Dict[str, Any]]:
        """Search with category awareness"""
        # This would use a custom Supabase function for category-aware search
        # For now, we'll simulate with a regular search and post-filter
        results = await self._general_semantic_search(params)
        
        # Filter and boost results from the target category
        category_results = []
        other_results = []
        
        for result in results:
            metadata = result.get('metadata', {})
            result_category = metadata.get('category', 'general')
            
            if result_category == category:
                # Boost similarity score for category matches
                result['similarity'] = min(1.0, result['similarity'] * 1.2)
                category_results.append(result)
            else:
                other_results.append(result)
        
        # Combine with category results first
        return category_results + other_results
    
    async def _general_semantic_search(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """General semantic search using Supabase vector similarity"""
        try:
            # Use the search function we created in the database
            response = self.supabase.rpc('search_documents', {
                'query_embedding': params['query_embedding'],
                'match_threshold': params['match_threshold'],
                'match_count': params['match_count']
            }).execute()
            
            return response.data if response.data else []
            
        except Exception as e:
            print(f"Error in semantic search: {e}")
            return []
    
    def _intelligent_rerank(self, query: str, results: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        """Re-rank results using intelligent scoring"""
        if not results:
            return []
        
        query_lower = query.lower()
        
        for result in results:
            base_score = result.get('similarity', 0.0)
            metadata = result.get('metadata', {})
            
            # Quality boost
            quality_score = metadata.get('quality_score', 0.5)
            quality_boost = quality_score * 0.1
            
            # Category relevance boost
            category_confidence = metadata.get('category_confidence', 0.5)
            category_boost = category_confidence * 0.05
            
            # Concept matching boost
            key_concepts = metadata.get('semantic_metadata', {}).get('key_concepts', [])
            concept_matches = sum(1 for concept in key_concepts if concept.lower() in query_lower)
            concept_boost = min(0.1, concept_matches * 0.02)
            
            # Retrieval hints boost
            retrieval_hints = metadata.get('semantic_metadata', {}).get('retrieval_hints', [])
            hint_matches = sum(1 for hint in retrieval_hints if hint.lower() in query_lower)
            hint_boost = min(0.05, hint_matches * 0.01)
            
            # Calculate final score
            final_score = base_score + quality_boost + category_boost + concept_boost + hint_boost
            result['final_score'] = min(1.0, final_score)
        
        # Sort by final score and return top results
        ranked_results = sorted(results, key=lambda x: x['final_score'], reverse=True)
        return ranked_results[:limit]

# Integration function for existing system
async def run_intelligent_processing(chunks: List[Dict[str, Any]], openai_api_key: str, 
                                   supabase_url: str, supabase_service_key: str) -> Dict[str, Any]:
    """Run intelligent embedding processing"""
    start_time = time.time()
    
    processor = IntelligentEmbeddingProcessor()
    processor.set_credentials(openai_api_key, supabase_url, supabase_service_key)
    
    print(f"Starting intelligent processing of {len(chunks)} chunks...")
    
    # Convert simple chunks to intelligent chunks
    intelligent_chunks = []
    for chunk in chunks:
        # Create intelligent chunks from the content
        smart_chunks = processor.create_intelligent_chunks(chunk['content'], chunk['source'])
        intelligent_chunks.extend(smart_chunks)
    
    # Generate embeddings with intelligent strategies
    chunk_embedding_pairs = await processor.generate_smart_embeddings(intelligent_chunks)
    
    # Upload with intelligent metadata
    upload_result = await processor.upload_intelligent_embeddings(chunk_embedding_pairs)
    
    processing_time = time.time() - start_time
    
    return {
        'success': upload_result['successful_uploads'] > 0,
        'chunks_processed': len(chunks),
        'intelligent_chunks_created': len(intelligent_chunks),
        'successful_embeddings': upload_result['successful_uploads'],
        'high_quality_chunks': upload_result['high_quality_uploads'],
        'upload_stats': upload_result,
        'performance_stats': {
            'processing_time': processing_time,
            'chunks_per_second': len(intelligent_chunks) / processing_time if processing_time > 0 else 0,
            'cache_hits': upload_result['cache_hits'],
            'semantic_clusters': upload_result['semantic_clusters']
        }
    }

if __name__ == "__main__":
    print("Intelligent Embedding System with Categorization and Smart Retrieval")
    print("Key features:")
    print("- Semantic content categorization")
    print("- Intelligent chunking strategies")
    print("- Rich metadata extraction")
    print("- Embedding caching and deduplication")
    print("- Semantic clustering")
    print("- Smart retrieval with re-ranking")