"use client";
import { Camera, CloudUpload, Mic, MicOff, Upload } from "lucide-react";
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
    new(): SpeechRecognition;
  };
}

const UploadZone = () => {
  const [image, setImage] = useState<string | null>(null);
  const [prompt, setPrompt] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [showPrompt, setShowPrompt] = useState(false);

  const [isListening, setIsListening] = useState(false);
  const [isSupported, setIsSupported] = useState(false);
  const recognitionRef = useRef<SpeechRecognition | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
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
        setShowPrompt(true);
        stopCamera();
      }
    }
  };

  useEffect(() => {
    if ("webkitSpeechRecognition" in window || "SpeechRecognition" in window) {
      setIsSupported(true);
      const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
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

  const handleSubmit = () => {
    console.log({ image, prompt });
    setImage(null);
    setPrompt("");
    setShowPrompt(false);
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
            AI-Powered Analysis
          </div>
        </div>
      </SheetTrigger>
      <SheetContent side="bottom" className="rounded-t-[20px] max-h-[90vh] overflow-y-auto">
        <SheetHeader className="text-center mt-6">
          <SheetTitle className="text-3xl text-emerald-500 font-bold">
            Crop Health Analysis
          </SheetTitle>
        </SheetHeader>

        <div className="space-y-6 p-6">
          {!showPrompt ? (
            <div className="space-y-4">
              <div className="flex flex-col sm:flex-row gap-4">
                <div
                  onClick={() => fileInputRef.current?.click()}
                  className="flex-1 border-2 border-dashed rounded-lg p-6 flex flex-col items-center justify-center cursor-pointer hover:bg-gray-50 transition-colors"
                >
                  <Upload className="w-8 h-8 text-emerald-500 mb-2" />
                  <p className="text-center">Upload Crop Photo</p>
                  <p className="text-sm text-gray-500 text-center">
                    Take a clear photo of affected crop leaves
                  </p>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    className="hidden"
                    onChange={handleFileChange}
                  />
                </div>

                <div
                  onClick={stream ? stopCamera : startCamera}
                  className="flex-1 border-2 border-dashed rounded-lg p-6 flex flex-col items-center justify-center cursor-pointer hover:bg-gray-50 transition-colors"
                >
                  <Camera className="w-8 h-8 text-emerald-500 mb-2" />
                  <p className="text-center">{stream ? "Stop Camera" : "Use Camera"}</p>
                  <p className="text-sm text-gray-500 text-center">
                    Capture clear image of crop leaves
                  </p>
                </div>
              </div>

              {stream && (
                <div className="mt-4 space-y-2">
                  <div className="relative aspect-video bg-black rounded-lg overflow-hidden">
                    <video
                      ref={videoRef}
                      autoPlay
                      playsInline
                      className="w-full h-full object-cover"
                    />
                  </div>
                  <Button
                    onClick={captureImage}
                    className="w-full bg-emerald-600 hover:bg-emerald-700"
                  >
                    Capture Image
                  </Button>
                </div>
              )}
            </div>
          ) : (
            <div className="space-y-4 flex gap-6">
              {image && (
                <div className="relative bg-gray-100 rounded-lg overflow-hidden h-[500px] flex-1 ">
                  <img src={image} alt="Preview" className="h-full mx-auto object-contain" />
                </div>
              )}
              <div className="min-h-screen bg-gradient-to-br from-emerald-50 to-green-100 p-8">
                <div className="max-w-2xl mx-auto bg-white rounded-xl  p-6">
                  <div className="flex flex-col space-y-4">
                    <label className="text-lg font-medium text-gray-800">
                      Describe the issue or ask about your crop
                    </label>

                    <div className="relative">
                      <textarea
                        placeholder="E.g., Why are my leaves turning yellow? Is this a disease?"
                        value={prompt}
                        onChange={(e) => setPrompt(e.target.value)}
                        className="w-full p-3 pr-12 border border-gray-300 rounded-lg resize-none focus:border-emerald-500 focus:ring-2 focus:ring-emerald-500 focus:outline-none transition-all"
                        rows={4}
                      />

                      {/* Microphone Button */}
                      {isSupported && (
                        <button
                          onClick={toggleListening}
                          className={`absolute right-3 bottom-3 p-2.5 rounded-full transition-all duration-200 ${
                            isListening
                              ? "bg-red-500 hover:bg-red-600 animate-pulse"
                              : "bg-emerald-500 hover:bg-emerald-600"
                          } text-white `}
                          title={isListening ? "Stop listening" : "Start voice input"}
                        >
                          {isListening ? (
                            <MicOff className="w-5 h-5" />
                          ) : (
                            <Mic className="w-5 h-5" />
                          )}
                        </button>
                      )}
                    </div>

                    {/* Status Indicator */}
                    {isListening && (
                      <div className="flex items-center gap-2 text-sm text-red-600 animate-pulse">
                        <div className="w-2 h-2 bg-red-600 rounded-full"></div>
                        <span>Listening... Speak now</span>
                      </div>
                    )}

                    {!isSupported && (
                      <div className="text-sm text-amber-600 bg-amber-50 p-3 rounded-lg border border-amber-200">
                        ⚠️ Speech recognition is not supported in your browser. Please use Chrome,
                        Edge, or Safari.
                      </div>
                    )}
                  </div>

                  <div className="flex justify-between pt-6">
                    <button
                      className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors font-medium"
                      onClick={() => {
                        setPrompt("");
                      }}
                    >
                      Clear
                    </button>
                    <button
                      onClick={handleSubmit}
                      disabled={!prompt.trim()}
                      className={`px-6 py-2 bg-emerald-600 text-white rounded-lg font-medium transition-all ${
                        !prompt.trim()
                          ? "opacity-50 cursor-not-allowed"
                          : "hover:bg-emerald-700 "
                      }`}
                    >
                      Analyze Crop
                    </button>
                  </div>

                  {/* Example prompts */}
                  <div className="mt-6 pt-6 border-t border-gray-200">
                    <p className="text-sm font-medium text-gray-600 mb-3">Quick examples:</p>
                    <div className="flex flex-wrap gap-2">
                      {[
                        "Why are my leaves turning yellow?",
                        "Is this a disease?",
                        "What nutrients does my crop need?",
                      ].map((example, idx) => (
                        <button
                          key={idx}
                          onClick={() => setPrompt(example)}
                          className="text-xs px-3 py-1.5 bg-emerald-50 text-emerald-700 rounded-full hover:bg-emerald-100 transition-colors border border-emerald-200"
                        >
                          {example}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </SheetContent>
    </Sheet>
  );
};

export default UploadZone;
