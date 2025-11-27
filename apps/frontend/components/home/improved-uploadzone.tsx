"use client";
import { Camera, CloudUpload, Mic, MicOff, Upload, CheckCircle, Clock, Zap } from "lucide-react";
import {
  Sheet,
  SheetClose,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "../ui/sheet";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import React, { useEffect, useRef, useState } from "react";
import {
  fastImageAnalysis,
  processQueryWithDetection,
  dataURLtoFile,
  ChatMessage,
} from "@/lib/api";
import { ChatInterface } from "./ChatInterface";
import { TypewriterText } from "../ui/TypewriterText";

declare global {
  interface Window {
    SpeechRecognition: typeof SpeechRecognition;
    webkitSpeechRecognition: typeof SpeechRecognition;
  }

  interface SpeechRecognition extends EventTarget {
    new (): SpeechRecognition;
    continuous: boolean;
    interimResults: boolean;
    lang: string;
    start: () => void;
    stop: () => void;
    onresult: (event: any) => void;
    onerror: (event: any) => void;
    onend: () => void;
  }

  var SpeechRecognition: {
    prototype: SpeechRecognition;
    new (): SpeechRecognition;
  };
}

interface QuickAnalysisResult {
  plant: string;
  status: "healthy" | "diseased";
  disease?: string;
  confidence: number;
  session_id: string;
  user_query?: string;
}

const ImprovedUploadZone = () => {
  const [image, setImage] = useState<string | null>(null);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [prompt, setPrompt] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [showPrompt, setShowPrompt] = useState(false);

  const [isQuickAnalyzing, setIsQuickAnalyzing] = useState(false);
  const [quickResults, setQuickResults] = useState<QuickAnalysisResult | null>(null);
  const [showChat, setShowChat] = useState(false);
  const [isDetailedAnalyzing, setIsDetailedAnalyzing] = useState(false);
  const [detailedResponse, setDetailedResponse] = useState<string>("");
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [sessionId, setSessionId] = useState<string>("");

  const [error, setError] = useState<string | null>(null);
  const [analysisProgress, setAnalysisProgress] = useState<number>(0);

  const [isListening, setIsListening] = useState(false);
  const [isSupported, setIsSupported] = useState(false);
  const recognitionRef = useRef<SpeechRecognition | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setImageFile(file);
      const reader = new FileReader();
      reader.onload = (event) => {
        setImage(event.target?.result as string);
        setShowPrompt(true);
      };
      reader.readAsDataURL(file);
    }
  };

  const startCamera = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({ video: true });
      setStream(mediaStream);
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }
    } catch (err) {
      console.error("Error accessing camera:", err);
    }
  };

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      setStream(null);
    }
  };

  const captureImage = () => {
    if (videoRef.current) {
      const canvas = document.createElement("canvas");
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      const ctx = canvas.getContext("2d");
      if (ctx) {
        ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
        const imageUrl = canvas.toDataURL("image/png");
        setImage(imageUrl);

        const file = dataURLtoFile(imageUrl, "captured-image.png");
        setImageFile(file);

        setShowPrompt(true);
        stopCamera();
      }
    }
  };

  useEffect(() => {
    if ("webkitSpeechRecognition" in window || "SpeechRecognition" in window) {
      setIsSupported(true);
      const SpeechRecognition =
        (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
      recognitionRef.current = new SpeechRecognition() as SpeechRecognition;

      recognitionRef.current.continuous = true;
      recognitionRef.current.interimResults = true;
      recognitionRef.current.lang = "bn-BD";

      recognitionRef.current!.onresult = (event: any) => {
        let finalTranscript = "";
        for (let i = event.resultIndex; i < event.results.length; i++) {
          if (event.results[i].isFinal) {
            finalTranscript += event.results[i][0].transcript + " ";
          }
        }
        if (finalTranscript) {
          setPrompt((prev) => prev + finalTranscript);
        }
      };

      recognitionRef.current!.onerror = (event: any) => {
        console.error("Speech recognition error:", event.error);
        setIsListening(false);
      };

      recognitionRef.current.onend = () => {
        setIsListening(false);
      };
    }

    return () => {
      if (recognitionRef.current) {
        recognitionRef.current?.stop();
      }
    };
  }, []);

  const toggleListening = () => {
    if (!recognitionRef.current) return;
    if (isListening) {
      recognitionRef.current.stop();
    } else {
      recognitionRef.current.start();
    }
    setIsListening(!isListening);
  };

  // New improved flow: Detection → Query Processing (if query exists)
  const handleSubmit = async () => {
    if (!imageFile) {
      setError("Please provide an image for analysis.");
      return;
    }

    setError(null);
    setAnalysisProgress(0);

    try {
      // Phase 1: Plant Detection (2-3 seconds)
      setIsQuickAnalyzing(true);
      setAnalysisProgress(25);

      const detectionResult = await fastImageAnalysis(imageFile, prompt.trim() || undefined);

      setQuickResults(detectionResult);
      setSessionId(detectionResult.session_id);
      setIsQuickAnalyzing(false);
      setAnalysisProgress(50);

      // Show detection results immediately
      setShowChat(true);

      // Create detection result message
      const timestamp = new Date().toISOString();
      let detectionMessage = "";

      if (detectionResult.status === "healthy") {
        detectionMessage = `🌱 **Healthy ${detectionResult.plant} Detected**\nConfidence: ${(detectionResult.confidence * 100).toFixed(0)}%\n\nYour plant looks healthy! ${prompt.trim() ? "Let me analyze your question..." : "Feel free to ask any questions about plant care."}`;
      } else {
        detectionMessage = `🎯 **${detectionResult.disease} Detected**\nPlant: ${detectionResult.plant}\nConfidence: ${(detectionResult.confidence * 100).toFixed(0)}%\n\n${prompt.trim() ? "Let me provide detailed guidance for your question..." : "Ask me about treatment, prevention, or management."}`;
      }

      const initialMessage: ChatMessage = {
        role: "assistant",
        content: detectionMessage,
        timestamp: timestamp,
      };
      setChatHistory([initialMessage]);

      // Phase 2: Process user query if provided (15-20 seconds)
      if (prompt.trim()) {
        setTimeout(async () => {
          try {
            setIsDetailedAnalyzing(true);
            setAnalysisProgress(75);

            const queryResult = await processQueryWithDetection(detectionResult);

            // Add detailed response to chat
            const detailedMessage: ChatMessage = {
              role: "assistant",
              content: queryResult.response,
              timestamp: new Date().toISOString(),
            };

            setChatHistory((prev) => [...prev, detailedMessage]);
            setIsDetailedAnalyzing(false);
            setAnalysisProgress(100);
          } catch (detailError) {
            console.error("Query processing failed:", detailError);
            setIsDetailedAnalyzing(false);
            const errorMessage: ChatMessage = {
              role: "assistant",
              content:
                "I detected the plant condition, but couldn't process your specific question. Feel free to ask me anything about this plant!",
              timestamp: new Date().toISOString(),
            };
            setChatHistory((prev) => [...prev, errorMessage]);
          }
        }, 1000);
      } else {
        setAnalysisProgress(100); // No query to process
      }
    } catch (error) {
      setError(error instanceof Error ? error.message : "Analysis failed. Please try again.");
      setIsQuickAnalyzing(false);
      setIsDetailedAnalyzing(false);
      console.error("Analysis error:", error);
    }
  };

  const resetForm = () => {
    setImage(null);
    setImageFile(null);
    setPrompt("");
    setShowPrompt(false);
    setIsQuickAnalyzing(false);
    setQuickResults(null);
    setShowChat(false);
    setIsDetailedAnalyzing(false);
    setDetailedResponse("");
    setChatHistory([]);
    setSessionId("");
    setError(null);
    setAnalysisProgress(0);
  };

  return (
    <Sheet onOpenChange={(open) => !open && stopCamera()}>
      <SheetTrigger asChild>
        <div className="relative rounded-2xl overflow-hidden bg-white/20 hover:bg-white/30 cursor-pointer">
          <div className="border-2 flex items-center justify-center flex-col border-dashed border-emerald-500/80 rounded-[15px] w-full aspect-video hover:bg-emerald-50/50 transition-colors">
            <CloudUpload strokeWidth={1.6} className="w-14 h-14 text-emerald-500" />
            <p className="text-[16px] text-emerald-500">Upload Crop Photo</p>
          </div>
          <div className="absolute bottom-3 left-3 bg-white/90 backdrop-blur-md px-3 py-2 rounded-xl text-[12px] font-bold text-emerald-900 border border-emerald-500/20">
            <Zap className="w-3 h-3 inline-block mr-1" />
            Lightning Fast Analysis
          </div>
        </div>
      </SheetTrigger>
      <SheetContent side="bottom" className="rounded-t-[20px] max-h-[90vh] overflow-y-auto">
        <SheetHeader className="text-center mt-6">
          <SheetTitle className="text-3xl text-emerald-500 font-bold">
            🌾 AgriMind Analysis
          </SheetTitle>
        </SheetHeader>

        <div className="space-y-6 p-6">
          {/* Error Display */}
          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
              <p className="text-red-700 text-sm">{error}</p>
            </div>
          )}

          {analysisProgress !== 100 &&
            (isQuickAnalyzing || isDetailedAnalyzing || analysisProgress > 0) && (
              <div className="bg-gradient-to-r from-emerald-50 to-blue-50 border border-emerald-200 rounded-lg p-4">
                {/* Header */}
                <div className="flex items-center justify-between mb-2">
                  <span className="text-emerald-800 font-medium">
                    {isQuickAnalyzing && "⚡ Quick Analysis..."}
                    {quickResults &&
                      isDetailedAnalyzing &&
                      "🔍 Getting Detailed Recommendations..."}
                    {!isQuickAnalyzing &&
                      !isDetailedAnalyzing &&
                      analysisProgress === 100 &&
                      "✅ Analysis Complete"}
                  </span>
                  <span className="text-emerald-600 text-sm">{analysisProgress}%</span>
                </div>

                {/* Progress bar */}
                <div className="w-full bg-emerald-100 rounded-full h-2">
                  <div
                    className="bg-gradient-to-r from-emerald-500 to-blue-500 h-2 rounded-full transition-all duration-500 ease-out"
                    style={{ width: `${analysisProgress}%` }}
                  ></div>
                </div>

                {/* Status text */}
                {isQuickAnalyzing && (
                  <p className="text-emerald-700 text-xs mt-2">
                    <Zap className="w-3 h-3 inline mr-1" />
                    Analyzing image... (~2–3 seconds)
                  </p>
                )}

                {isDetailedAnalyzing && (
                  <p className="text-blue-700 text-xs mt-2">
                    <Clock className="w-3 h-3 inline mr-1" />
                    Fetching detailed recommendations... (~15–20 seconds)
                  </p>
                )}
              </div>
            )}

          {/* Quick Results Display */}
          {quickResults && !showChat && (
            <div className="bg-gradient-to-r from-emerald-50 to-green-50 border border-emerald-200 rounded-lg p-6">
              <div className="flex items-start space-x-4">
                <CheckCircle className="w-8 h-8 text-emerald-600 shrink-0 mt-1" />
                <div className="flex-1">
                  <h3 className="text-lg font-semibold text-emerald-800 mb-2">
                    {quickResults.status === "healthy" ? "🌱" : "🎯"} {quickResults.plant} -{" "}
                    {quickResults.status === "healthy" ? "Healthy" : quickResults.disease}
                  </h3>
                  <p className="text-emerald-700 mb-3">
                    Confidence: {(quickResults.confidence * 100).toFixed(0)}%
                  </p>
                  <div className="mt-4 p-3 bg-emerald-100 rounded-lg">
                    <p className="text-emerald-800 text-sm">
                      💡 <strong>Opening chat interface...</strong>
                      {quickResults.user_query
                        ? " Processing your question..."
                        : " Ask me anything about this plant!"}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Fallback for Phase 1 failure */}
          {error && !quickResults && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
              <h3 className="text-red-800 font-medium mb-2">⚠️ Quick Analysis Failed</h3>
              <p className="text-red-700 text-sm mb-3">{error}</p>
              <Button
                onClick={handleSubmit}
                variant="outline"
                className="text-red-700 border-red-300"
              >
                Try Again
              </Button>
            </div>
          )}

          {/* Fallback for Phase 2 failure */}
          {quickResults && showChat && error && (
            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3 mb-4">
              <p className="text-yellow-800 text-sm">
                ⚠️ Detailed analysis failed, but you can continue chatting with the quick results.
              </p>
            </div>
          )}

          {/* Chat Interface */}
          {showChat && sessionId && (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-xl font-semibold text-emerald-800">💬 Chat with AgriMind</h3>
                {isDetailedAnalyzing && (
                  <div className="flex items-center space-x-2 text-blue-600">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
                    <span className="text-sm">Loading detailed analysis...</span>
                  </div>
                )}
              </div>

              <ChatInterface
                initialResponse=""
                sessionId={sessionId}
                initialHistory={chatHistory}
              />
            </div>
          )}

          {/* Upload Interface */}
          {!showPrompt && !isQuickAnalyzing && !quickResults ? (
            <div className="space-y-4">
              <div className="flex flex-col sm:flex-row gap-4">
                <div
                  onClick={() => fileInputRef.current?.click()}
                  className="flex-1 border-2 border-dashed rounded-lg p-6 flex flex-col items-center justify-center cursor-pointer hover:bg-gray-50 transition-colors"
                >
                  <Upload className="w-8 h-8 text-gray-400 mb-2" />
                  <p className="text-gray-600 text-center">Choose Photo</p>
                </div>

                <div
                  onClick={stream ? stopCamera : startCamera}
                  className="flex-1 border-2 border-dashed rounded-lg p-6 flex flex-col items-center justify-center cursor-pointer hover:bg-gray-50 transition-colors"
                >
                  <Camera className="w-8 h-8 text-gray-400 mb-2" />
                  <p className="text-gray-600 text-center">
                    {stream ? "Stop Camera" : "Use Camera"}
                  </p>
                </div>
              </div>

              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                accept="image/*"
                className="hidden"
              />

              {stream && (
                <div className="relative">
                  <video ref={videoRef} autoPlay playsInline className="w-full rounded-lg" />
                  <Button
                    onClick={captureImage}
                    className="absolute bottom-4 left-1/2 transform -translate-x-1/2 bg-white text-gray-800 hover:bg-gray-100"
                  >
                    Capture
                  </Button>
                </div>
              )}
            </div>
          ) : showPrompt && image && !isQuickAnalyzing && !quickResults ? (
            <div className="space-y-4">
              <div className="relative">
                <img
                  src={image}
                  alt="Uploaded crop"
                  className="w-full max-h-64 object-contain rounded-lg border"
                />
              </div>

              <div className="space-y-3">
                <label className="block text-sm font-medium text-gray-700">
                  Describe the issue with your crop:
                </label>
                <div className="flex gap-2">
                  <Input
                    type="text"
                    placeholder="e.g., What are these spots on my wheat leaves?"
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    className="flex-1"
                    onKeyPress={(e) => e.key === "Enter" && handleSubmit()}
                  />
                  {isSupported && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={toggleListening}
                      className={isListening ? "bg-red-100" : ""}
                    >
                      {isListening ? <MicOff className="w-4 h-4" /> : <Mic className="w-4 h-4" />}
                    </Button>
                  )}
                </div>
              </div>

              <div className="flex gap-3">
                <Button
                  onClick={handleSubmit}
                  className="flex-1 bg-emerald-600 hover:bg-emerald-700"
                >
                  <Zap className="w-4 h-4 mr-2" />
                  Start Analysis
                </Button>
                <Button variant="outline" onClick={resetForm}>
                  Reset
                </Button>
              </div>
            </div>
          ) : null}
        </div>
      </SheetContent>
    </Sheet>
  );
};

export default ImprovedUploadZone;
