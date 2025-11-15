"use client";
import { Loader2, Camera, Brain, Leaf } from "lucide-react";

interface AnalysisLoadingProps {
  stage: "uploading" | "processing" | "analyzing" | "complete";
}

export default function AnalysisLoading({ stage }: AnalysisLoadingProps) {
  const stages = [
    {
      key: "uploading",
      label: "Uploading Image",
      icon: <Camera className="w-6 h-6" />,
      description: "Securely uploading your crop image...",
    },
    {
      key: "processing",
      label: "Processing Image",
      icon: <Brain className="w-6 h-6" />,
      description: "AI is analyzing the image for diseases and issues...",
    },
    {
      key: "analyzing",
      label: "Generating Insights",
      icon: <Leaf className="w-6 h-6" />,
      description: "Combining image analysis with knowledge base...",
    },
    {
      key: "complete",
      label: "Analysis Complete",
      icon: <Leaf className="w-6 h-6" />,
      description: "Preparing your personalized crop report...",
    },
  ];

  const currentStageIndex = stages.findIndex((s) => s.key === stage);

  return (
    <div className="flex flex-col items-center justify-center min-h-[400px] p-8">
      <div className="text-center space-y-6 max-w-md">
        {/* Main Loading Animation */}
        <div className="relative">
          <div className="w-24 h-24 bg-emerald-100 rounded-full flex items-center justify-center mx-auto">
            <Loader2 className="w-12 h-12 text-emerald-600 animate-spin" />
          </div>
          <div className="absolute inset-0 w-24 h-24 border-4 border-emerald-200 border-t-emerald-600 rounded-full animate-spin mx-auto"></div>
        </div>

        {/* Current Stage */}
        <div className="space-y-2">
          <h3 className="text-xl font-semibold text-emerald-800">
            {stages[currentStageIndex]?.label}
          </h3>
          <p className="text-gray-600 text-sm">{stages[currentStageIndex]?.description}</p>
        </div>

        {/* Progress Indicators */}
        <div className="flex justify-center space-x-4">
          {stages.slice(0, -1).map((stageItem, index) => (
            <div key={stageItem.key} className="flex flex-col items-center space-y-2">
              <div
                className={`
                w-10 h-10 rounded-full flex items-center justify-center border-2 transition-all duration-300
                ${
                  index <= currentStageIndex
                    ? "bg-emerald-600 border-emerald-600 text-white"
                    : "bg-gray-100 border-gray-300 text-gray-400"
                }
                ${index === currentStageIndex ? "animate-pulse" : ""}
              `}
              >
                {stageItem.icon}
              </div>
              <span
                className={`text-xs font-medium ${
                  index <= currentStageIndex ? "text-emerald-600" : "text-gray-400"
                }`}
              >
                {stageItem.label}
              </span>
            </div>
          ))}
        </div>

        {/* Progress Bar */}
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div
            className="bg-emerald-600 h-2 rounded-full transition-all duration-1000 ease-out"
            style={{ width: `${((currentStageIndex + 1) / stages.length) * 100}%` }}
          ></div>
        </div>

        {/* Additional Info */}
        <div className="bg-emerald-50 rounded-lg p-4 border border-emerald-200">
          <div className="flex items-center gap-2 justify-center">
            <Leaf className="w-4 h-4 text-emerald-600" />
            <span className="text-sm text-emerald-700 font-medium">AgriMind AI is working...</span>
          </div>
          <p className="text-xs text-emerald-600 mt-1">
            Our advanced AI models are analyzing your crop for the most accurate results.
          </p>
        </div>

        {/* Estimated Time */}
        <div className="text-center">
          <p className="text-xs text-gray-500">Estimated time: 10-30 seconds</p>
        </div>
      </div>
    </div>
  );
}
