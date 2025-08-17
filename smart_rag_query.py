#!/usr/bin/env python3
"""
Smart RAG Query System
Provides intelligent querying capabilities for the RAG system
"""

import asyncio
import json
from typing import List, Dict, Any, Optional
from openai import OpenAI
from supabase import create_client
from intelligent_embeddings import SmartRetriever

class SmartRAGQuery:
    """Smart RAG query system with context-aware retrieval"""
    
    def __init__(self, openai_api_key: str, supabase_url: str, supabase_service_key: str):
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.supabase_client = create_client(supabase_url, supabase_service_key)
        self.retriever = SmartRetriever(self.supabase_client, self.openai_client)
    
    async def query(self, question: str, context_limit: int = 5, category_filter: str = None) -> Dict[str, Any]:
        """Execute a smart RAG query"""
        
        print(f"Processing query: {question}")
        
        # Step 1: Intelligent retrieval
        relevant_docs = await self.retriever.smart_search(
            query=question,
            limit=context_limit,
            category_filter=category_filter
        )
        
        if not relevant_docs:
            return {
                'answer': "I couldn't find any relevant information in the knowledge base.",
                'sources': [],
                'confidence': 0.0,
                'query_analysis': self._analyze_query(question)
            }
        
        # Step 2: Context preparation
        context = self._prepare_context(relevant_docs)
        
        # Step 3: Generate answer with GPT
        answer_response = await self._generate_answer(question, context)
        
        # Step 4: Extract sources and metadata
        sources = self._extract_sources(relevant_docs)
        
        return {
            'answer': answer_response['answer'],
            'confidence': answer_response['confidence'],
            'sources': sources,
            'context_used': len(relevant_docs),
            'query_analysis': self._analyze_query(question),
            'retrieval_stats': {
                'total_matches': len(relevant_docs),
                'avg_similarity': sum(doc.get('final_score', 0) for doc in relevant_docs) / len(relevant_docs),
                'categories_found': list(set(doc.get('metadata', {}).get('category', 'unknown') for doc in relevant_docs))
            }
        }
    
    def _analyze_query(self, question: str) -> Dict[str, Any]:
        """Analyze the query to understand intent and type"""
        question_lower = question.lower()
        
        analysis = {
            'type': 'general',
            'intent': 'information',
            'complexity': 'simple',
            'keywords': []
        }
        
        # Determine query type
        if any(word in question_lower for word in ['how to', 'how do', 'how can']):
            analysis['type'] = 'how_to'
            analysis['intent'] = 'instruction'
        elif any(word in question_lower for word in ['what is', 'what are', 'define']):
            analysis['type'] = 'definition'
            analysis['intent'] = 'explanation'
        elif any(word in question_lower for word in ['why', 'because', 'reason']):
            analysis['type'] = 'explanation'
            analysis['intent'] = 'reasoning'
        elif any(word in question_lower for word in ['compare', 'difference', 'versus', 'vs']):
            analysis['type'] = 'comparison'
            analysis['intent'] = 'analysis'
        elif question_lower.endswith('?'):
            analysis['type'] = 'question'
        
        # Determine complexity
        if len(question.split()) > 15 or any(word in question_lower for word in ['complex', 'detailed', 'comprehensive']):
            analysis['complexity'] = 'complex'
        elif len(question.split()) > 8:
            analysis['complexity'] = 'medium'
        
        # Extract keywords (simple approach)
        import re
        words = re.findall(r'\b[a-zA-Z]{3,}\b', question)
        analysis['keywords'] = [word.lower() for word in words if word.lower() not in 
                               ['the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'how', 'what', 'why']]
        
        return analysis
    
    def _prepare_context(self, docs: List[Dict[str, Any]]) -> str:
        """Prepare context from retrieved documents"""
        context_parts = []
        
        for i, doc in enumerate(docs, 1):
            content = doc.get('content', '')
            source = doc.get('source', 'Unknown')
            metadata = doc.get('metadata', {})
            
            # Add source and quality information
            quality_score = metadata.get('quality_score', 0.5)
            category = metadata.get('category', 'general')
            
            context_part = f"[Source {i}: {source} (Quality: {quality_score:.2f}, Category: {category})]\n{content}\n"
            context_parts.append(context_part)
        
        return "\n---\n".join(context_parts)
    
    async def _generate_answer(self, question: str, context: str) -> Dict[str, Any]:
        """Generate answer using GPT with the retrieved context"""
        
        system_prompt = """You are a helpful AI assistant that answers questions based on the provided context. 

Instructions:
1. Use ONLY the information provided in the context to answer the question
2. If the context doesn't contain enough information, say so clearly
3. Cite specific sources when possible (e.g., "According to Source 1...")
4. Be concise but comprehensive
5. If you're uncertain about something, express that uncertainty
6. Provide a confidence score (0-1) for your answer

Format your response as:
ANSWER: [Your detailed answer here]
CONFIDENCE: [0.0-1.0]
"""
        
        user_prompt = f"""Context:
{context}

Question: {question}

Please provide a detailed answer based on the context above."""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Low temperature for factual responses
                max_tokens=1000
            )
            
            response_text = response.choices[0].message.content
            
            # Parse response
            if "CONFIDENCE:" in response_text:
                parts = response_text.split("CONFIDENCE:")
                answer = parts[0].replace("ANSWER:", "").strip()
                try:
                    confidence = float(parts[1].strip())
                except:
                    confidence = 0.7  # Default confidence
            else:
                answer = response_text.replace("ANSWER:", "").strip()
                confidence = 0.7
            
            return {
                'answer': answer,
                'confidence': min(1.0, max(0.0, confidence))
            }
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            return {
                'answer': f"I encountered an error while generating the answer: {str(e)}",
                'confidence': 0.0
            }
    
    def _extract_sources(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract source information from retrieved documents"""
        sources = []
        
        for doc in docs:
            source_info = {
                'source': doc.get('source', 'Unknown'),
                'similarity_score': doc.get('final_score', doc.get('similarity', 0.0)),
                'content_preview': doc.get('content', '')[:200] + '...',
                'metadata': {
                    'category': doc.get('metadata', {}).get('category', 'unknown'),
                    'quality_score': doc.get('metadata', {}).get('quality_score', 0.5),
                    'key_concepts': doc.get('metadata', {}).get('semantic_metadata', {}).get('key_concepts', [])[:3]
                }
            }
            sources.append(source_info)
        
        return sources
    
    async def batch_query(self, questions: List[str], category_filter: str = None) -> List[Dict[str, Any]]:
        """Process multiple queries efficiently"""
        results = []
        
        for question in questions:
            result = await self.query(question, category_filter=category_filter)
            results.append({
                'question': question,
                'result': result
            })
        
        return results
    
    async def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base"""
        try:
            # Get total document count
            total_response = self.supabase_client.table('documents').select('id', count='exact').execute()
            total_docs = total_response.count if hasattr(total_response, 'count') else 0
            
            # Get category distribution
            category_response = self.supabase_client.table('documents').select('metadata').execute()
            
            categories = {}
            quality_scores = []
            
            for doc in category_response.data:
                metadata = doc.get('metadata', {})
                category = metadata.get('category', 'unknown')
                categories[category] = categories.get(category, 0) + 1
                
                quality_score = metadata.get('quality_score', 0.5)
                quality_scores.append(quality_score)
            
            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
            
            return {
                'total_documents': total_docs,
                'category_distribution': categories,
                'average_quality_score': round(avg_quality, 3),
                'high_quality_documents': sum(1 for score in quality_scores if score > 0.7),
                'knowledge_base_health': 'Good' if avg_quality > 0.6 else 'Fair' if avg_quality > 0.4 else 'Poor'
            }
            
        except Exception as e:
            print(f"Error getting knowledge base stats: {e}")
            return {'error': str(e)}

# Synchronous wrapper for integration
def run_smart_query(question: str, openai_api_key: str, supabase_url: str, 
                   supabase_service_key: str, category_filter: str = None) -> Dict[str, Any]:
    """Synchronous wrapper for smart RAG query"""
    
    async def _async_query():
        rag_system = SmartRAGQuery(openai_api_key, supabase_url, supabase_service_key)
        return await rag_system.query(question, category_filter=category_filter)
    
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(_async_query())

if __name__ == "__main__":
    print("Smart RAG Query System")
    print("Usage: from smart_rag_query import run_smart_query")
    print("Example: result = run_smart_query('What is machine learning?', api_key, supabase_url, service_key)")