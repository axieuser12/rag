import React, { useState } from 'react';
import { Search, Brain, Loader2, MessageCircle, BookOpen, Target, TrendingUp } from 'lucide-react';
import { useLanguage } from '../hooks/useLanguage';

interface QueryResult {
  success: boolean;
  question: string;
  answer: string;
  confidence: number;
  sources: Array<{
    source: string;
    similarity_score: number;
    content_preview: string;
    metadata: {
      category: string;
      quality_score: number;
      key_concepts: string[];
    };
  }>;
  context_used: number;
  query_analysis: {
    type: string;
    intent: string;
    complexity: string;
    keywords: string[];
  };
  retrieval_stats: {
    total_matches: number;
    avg_similarity: number;
    categories_found: string[];
  };
}

interface Credentials {
  openai_api_key: string;
  supabase_url: string;
  supabase_service_key: string;
}

interface QueryInterfaceProps {
  credentials: Credentials | null;
}

const QueryInterface: React.FC<QueryInterfaceProps> = ({ credentials }) => {
  const { t } = useLanguage();
  const [question, setQuestion] = useState('');
  const [isQuerying, setIsQuerying] = useState(false);
  const [result, setResult] = useState<QueryResult | null>(null);
  const [categoryFilter, setCategoryFilter] = useState<string>('');

  const categories = [
    { value: '', label: 'All Categories' },
    { value: 'technical', label: 'Technical' },
    { value: 'business', label: 'Business' },
    { value: 'legal', label: 'Legal' },
    { value: 'research', label: 'Research' },
    { value: 'general', label: 'General' }
  ];

  const handleQuery = async () => {
    if (!question.trim() || !credentials) return;

    setIsQuerying(true);
    setResult(null);

    try {
      const response = await fetch('/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: question.trim(),
          category_filter: categoryFilter || undefined,
          ...credentials
        }),
      });

      const data = await response.json();
      setResult(data);
    } catch (error) {
      setResult({
        success: false,
        question: question,
        answer: `Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
        confidence: 0,
        sources: [],
        context_used: 0,
        query_analysis: { type: 'error', intent: 'error', complexity: 'simple', keywords: [] },
        retrieval_stats: { total_matches: 0, avg_similarity: 0, categories_found: [] }
      });
    } finally {
      setIsQuerying(false);
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-400';
    if (confidence >= 0.6) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getConfidenceLabel = (confidence: number) => {
    if (confidence >= 0.8) return 'High';
    if (confidence >= 0.6) return 'Medium';
    return 'Low';
  };

  if (!credentials) {
    return (
      <div className="glass-effect-dark rounded-2xl p-8">
        <div className="text-center">
          <MessageCircle className="w-12 h-12 text-white/50 mx-auto mb-4" />
          <h3 className="text-xl font-semibold text-white mb-2">Query Your Knowledge Base</h3>
          <p className="text-white/70">Configure your credentials first to start querying your documents</p>
        </div>
      </div>
    );
  }

  return (
    <div className="glass-effect-dark rounded-2xl p-8">
      <div className="flex items-center mb-6">
        <Brain className="w-6 h-6 text-white mr-3" />
        <h2 className="text-2xl font-semibold text-white">Smart RAG Query</h2>
      </div>

      {/* Query Input */}
      <div className="space-y-4 mb-6">
        <div>
          <label className="block text-white font-medium mb-2">Ask a Question</label>
          <div className="relative">
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="What would you like to know about your documents?"
              className="w-full p-4 pr-12 bg-white/10 border border-white/20 rounded-lg text-white placeholder-white/50 focus:outline-none focus:ring-2 focus:ring-white/30 resize-none"
              rows={3}
            />
            <Search className="absolute right-4 top-4 w-5 h-5 text-white/50" />
          </div>
        </div>

        <div className="flex gap-4">
          <div className="flex-1">
            <label className="block text-white font-medium mb-2">Category Filter (Optional)</label>
            <select
              value={categoryFilter}
              onChange={(e) => setCategoryFilter(e.target.value)}
              className="w-full p-3 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-white/30"
            >
              {categories.map((cat) => (
                <option key={cat.value} value={cat.value} className="bg-gray-800">
                  {cat.label}
                </option>
              ))}
            </select>
          </div>
          
          <div className="flex items-end">
            <button
              onClick={handleQuery}
              disabled={isQuerying || !question.trim()}
              className="flex items-center px-6 py-3 btn-primary rounded-lg font-semibold transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isQuerying ? (
                <>
                  <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                  Searching...
                </>
              ) : (
                <>
                  <Search className="w-5 h-5 mr-2" />
                  Query
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Results */}
      {result && (
        <div className="space-y-6">
          {/* Answer */}
          <div className="bg-black/20 border border-white/10 rounded-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-white">Answer</h3>
              <div className="flex items-center">
                <span className="text-white/70 text-sm mr-2">Confidence:</span>
                <span className={`font-bold ${getConfidenceColor(result.confidence)}`}>
                  {getConfidenceLabel(result.confidence)} ({Math.round(result.confidence * 100)}%)
                </span>
              </div>
            </div>
            
            <div className="text-white/90 whitespace-pre-wrap leading-relaxed">
              {result.answer}
            </div>
          </div>

          {/* Query Analysis */}
          {result.query_analysis && (
            <div className="bg-black/20 border border-white/10 rounded-lg p-4">
              <h4 className="text-white font-medium mb-3">Query Analysis</h4>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <span className="text-white/70">Type:</span>
                  <span className="text-white ml-2 capitalize">{result.query_analysis.type}</span>
                </div>
                <div>
                  <span className="text-white/70">Intent:</span>
                  <span className="text-white ml-2 capitalize">{result.query_analysis.intent}</span>
                </div>
                <div>
                  <span className="text-white/70">Complexity:</span>
                  <span className="text-white ml-2 capitalize">{result.query_analysis.complexity}</span>
                </div>
                <div>
                  <span className="text-white/70">Keywords:</span>
                  <span className="text-white ml-2">{result.query_analysis.keywords.slice(0, 3).join(', ')}</span>
                </div>
              </div>
            </div>
          )}

          {/* Retrieval Stats */}
          {result.retrieval_stats && (
            <div className="bg-black/20 border border-white/10 rounded-lg p-4">
              <h4 className="text-white font-medium mb-3">Retrieval Statistics</h4>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="text-center">
                  <Target className="w-6 h-6 text-white mx-auto mb-2" />
                  <div className="text-2xl font-bold text-white">{result.retrieval_stats.total_matches}</div>
                  <div className="text-white/70 text-sm">Documents Found</div>
                </div>
                <div className="text-center">
                  <TrendingUp className="w-6 h-6 text-white mx-auto mb-2" />
                  <div className="text-2xl font-bold text-white">
                    {Math.round(result.retrieval_stats.avg_similarity * 100)}%
                  </div>
                  <div className="text-white/70 text-sm">Avg Similarity</div>
                </div>
                <div className="text-center">
                  <BookOpen className="w-6 h-6 text-white mx-auto mb-2" />
                  <div className="text-2xl font-bold text-white">{result.context_used}</div>
                  <div className="text-white/70 text-sm">Context Used</div>
                </div>
              </div>
              
              {result.retrieval_stats.categories_found.length > 0 && (
                <div className="mt-4">
                  <span className="text-white/70 text-sm">Categories Found: </span>
                  <div className="flex flex-wrap gap-2 mt-2">
                    {result.retrieval_stats.categories_found.map((category, index) => (
                      <span
                        key={index}
                        className="px-2 py-1 bg-white/10 text-white text-xs rounded-full capitalize"
                      >
                        {category}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Sources */}
          {result.sources && result.sources.length > 0 && (
            <div className="bg-black/20 border border-white/10 rounded-lg p-4">
              <h4 className="text-white font-medium mb-4">Sources ({result.sources.length})</h4>
              <div className="space-y-4">
                {result.sources.map((source, index) => (
                  <div key={index} className="bg-black/20 border border-white/10 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <h5 className="text-white font-medium truncate">{source.source}</h5>
                      <div className="flex items-center gap-4 text-sm">
                        <span className="text-white/70">
                          Quality: <span className="text-white">{Math.round(source.metadata.quality_score * 100)}%</span>
                        </span>
                        <span className="text-white/70">
                          Match: <span className="text-white">{Math.round(source.similarity_score * 100)}%</span>
                        </span>
                      </div>
                    </div>
                    
                    <div className="text-white/80 text-sm mb-3">
                      {source.content_preview}
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <span className="px-2 py-1 bg-white/10 text-white text-xs rounded-full capitalize">
                        {source.metadata.category}
                      </span>
                      
                      {source.metadata.key_concepts.length > 0 && (
                        <div className="flex gap-1">
                          {source.metadata.key_concepts.map((concept, i) => (
                            <span
                              key={i}
                              className="px-2 py-1 bg-white/5 text-white/70 text-xs rounded"
                            >
                              {concept}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default QueryInterface;