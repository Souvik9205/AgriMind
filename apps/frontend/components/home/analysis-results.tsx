"use client";
import { CheckCircle, AlertTriangle, Leaf, Book, TrendingUp } from "lucide-react";
import { CombinedAnalysisResponse } from "@/lib/api";

interface AnalysisResultsProps {
  results: CombinedAnalysisResponse;
  onClose: () => void;
}

export default function AnalysisResults({ results, onClose }: AnalysisResultsProps) {
  const { disease_detection, rag_response, combined_insights, confidence_score } = results;

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return "text-green-600 bg-green-100 border-green-300";
    if (confidence >= 0.6) return "text-yellow-600 bg-yellow-100 border-yellow-300";
    return "text-red-600 bg-red-100 border-red-300";
  };

  const getConfidenceIcon = (confidence: number) => {
    if (confidence >= 0.7) return <CheckCircle className="w-5 h-5" />;
    return <AlertTriangle className="w-5 h-5" />;
  };

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="text-center space-y-2">
        <h2 className="text-3xl font-bold text-emerald-800">Crop Analysis Results</h2>
        <div
          className={`inline-flex items-center gap-2 px-4 py-2 rounded-full border ${getConfidenceColor(confidence_score)}`}
        >
          {getConfidenceIcon(confidence_score)}
          <span className="font-medium">
            Overall Confidence: {(confidence_score * 100).toFixed(1)}%
          </span>
        </div>
      </div>

      {/* Combined Insights */}
      <div className="bg-linear-to-r from-emerald-50 to-green-50 rounded-xl p-6 border border-emerald-200">
        <div className="flex items-start gap-3">
          <div className="p-2 bg-emerald-100 rounded-lg">
            <TrendingUp className="w-6 h-6 text-emerald-600" />
          </div>
          <div>
            <h3 className="text-xl font-semibold text-emerald-800 mb-2">Key Insights</h3>
            <p className="text-gray-700 leading-relaxed">{combined_insights}</p>
          </div>
        </div>
      </div>

      {/* Disease Detection Results */}
      <div className="bg-white rounded-xl p-6 border border-gray-200 shadow-sm">
        <div className="flex items-start gap-3 mb-4">
          <div className="p-2 bg-blue-100 rounded-lg">
            <Leaf className="w-6 h-6 text-blue-600" />
          </div>
          <div>
            <h3 className="text-xl font-semibold text-gray-800">Disease Detection</h3>
            <p className="text-gray-600">AI-powered image analysis results</p>
          </div>
        </div>

        <div className="space-y-4">
          <div className="flex justify-between items-center p-4 bg-gray-50 rounded-lg">
            <div>
              <span className="text-sm text-gray-600">Detected Condition:</span>
              <p className="font-semibold text-lg text-gray-800">{disease_detection.disease}</p>
            </div>
            <div
              className={`px-3 py-1 rounded-full border ${getConfidenceColor(disease_detection.confidence)}`}
            >
              {(disease_detection.confidence * 100).toFixed(1)}%
            </div>
          </div>

          {disease_detection.recommendations.length > 0 && (
            <div>
              <h4 className="font-semibold text-gray-800 mb-3">Recommended Actions:</h4>
              <ul className="space-y-2">
                {disease_detection.recommendations.map((rec, index) => (
                  <li key={index} className="flex items-start gap-2">
                    <CheckCircle className="w-4 h-4 text-green-500 mt-1 shrink-0" />
                    <span className="text-gray-700">{rec}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>

      {/* RAG Response */}
      <div className="bg-white rounded-xl p-6 border border-gray-200 shadow-sm">
        <div className="flex items-start gap-3 mb-4">
          <div className="p-2 bg-purple-100 rounded-lg">
            <Book className="w-6 h-6 text-purple-600" />
          </div>
          <div>
            <h3 className="text-xl font-semibold text-gray-800">Knowledge Base Insights</h3>
            <p className="text-gray-600">Expert information related to your query</p>
          </div>
        </div>

        <div className="space-y-4">
          <div className="p-4 bg-gray-50 rounded-lg">
            <span className="text-sm text-gray-600">Your Question:</span>
            <p className="font-medium text-gray-800">{rag_response.query}</p>
          </div>

          <div>
            <span className="text-sm text-gray-600">Expert Response:</span>
            <p className="text-gray-700 leading-relaxed mt-1">{rag_response.answer}</p>
          </div>

          {rag_response.sources.length > 0 && (
            <div>
              <h4 className="font-semibold text-gray-800 mb-2">Information Sources:</h4>
              <div className="flex flex-wrap gap-2">
                {rag_response.sources.map((source, index) => (
                  <span
                    key={index}
                    className="px-3 py-1 bg-purple-100 text-purple-700 rounded-full text-sm"
                  >
                    {source}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex justify-center gap-4 pt-6">
        <button
          onClick={onClose}
          className="px-6 py-3 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors font-medium"
        >
          Analyze Another Image
        </button>
        <button
          onClick={() => window.print()}
          className="px-6 py-3 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700 transition-colors font-medium"
        >
          Save Results
        </button>
      </div>
    </div>
  );
}
