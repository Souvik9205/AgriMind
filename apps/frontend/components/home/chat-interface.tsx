"use client";

import React, { useState, useRef, useEffect } from "react";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Send, Mic, MicOff, ArrowLeft } from "lucide-react";
import { ChatMessage, chatFollowUp } from "@/lib/api";

interface ChatInterfaceProps {
  initialResponse: string;
  chatHistory: ChatMessage[];
  sessionId: string;
  onBack: () => void;
}

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

const ChatInterface: React.FC<ChatInterfaceProps> = ({
  initialResponse,
  chatHistory: initialHistory,
  sessionId,
  onBack,
}) => {
  const [message, setMessage] = useState("");
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>(initialHistory);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isListening, setIsListening] = useState(false);
  const [isSupported, setIsSupported] = useState(false);
  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const MAX_MESSAGES = 4; // initial + 3 more

  useEffect(() => {
    // Setup speech recognition
    if ("webkitSpeechRecognition" in window || "SpeechRecognition" in window) {
      setIsSupported(true);
      const SpeechRecognition =
        (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
      recognitionRef.current = new SpeechRecognition() as SpeechRecognition;

      recognitionRef.current.continuous = true;
      recognitionRef.current.interimResults = true;
      recognitionRef.current.lang = "bn-BD"; // Bengali

      recognitionRef.current.onresult = (event: any) => {
        let finalTranscript = "";
        for (let i = event.resultIndex; i < event.results.length; i++) {
          if (event.results[i].isFinal) {
            finalTranscript += event.results[i][0].transcript + " ";
          }
        }
        if (finalTranscript) {
          setMessage((prev) => prev + finalTranscript);
        }
      };

      recognitionRef.current.onerror = (event: any) => {
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

  useEffect(() => {
    // Scroll to bottom when new messages are added
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatHistory]);

  const toggleListening = () => {
    if (!recognitionRef.current) return;

    if (isListening) {
      recognitionRef.current.stop();
    } else {
      recognitionRef.current.start();
    }
    setIsListening(!isListening);
  };

  const canSendMore = () => {
    return chatHistory.length < MAX_MESSAGES;
  };

  const handleSendMessage = async () => {
    if (!message.trim() || isLoading || !canSendMore()) return;

    setError(null);
    setIsLoading(true);

    try {
      const response = await chatFollowUp(message, chatHistory);
      setChatHistory(response.chat_history);
      setMessage("");

      // Focus back on input
      setTimeout(() => {
        inputRef.current?.focus();
      }, 100);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to send message");
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const formatMessage = (content: string) => {
    // Simple markdown-like formatting
    return content
      .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
      .replace(/\*(.*?)\*/g, "<em>$1</em>")
      .replace(/🔍|🌾|⚠️|💡|🚨/g, (match) => `<span class="text-lg">${match}</span>`);
  };

  return (
    <div className="flex flex-col h-[600px] bg-white rounded-xl shadow-lg overflow-hidden">
      {/* Header */}
      <div className="bg-emerald-600 text-white p-4 flex items-center gap-3">
        <Button
          variant="ghost"
          size="sm"
          onClick={onBack}
          className="text-white hover:bg-emerald-700 p-2"
        >
          <ArrowLeft className="h-4 w-4" />
        </Button>
        <div>
          <h3 className="font-semibold">AgriMind Chat</h3>
          <p className="text-sm text-emerald-100">
            {MAX_MESSAGES - chatHistory.length} messages remaining
          </p>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {chatHistory.map((msg, index) => (
          <div
            key={index}
            className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-[80%] p-3 rounded-lg ${
                msg.role === "user" ? "bg-emerald-600 text-white" : "bg-gray-100 text-gray-800"
              }`}
            >
              <div
                dangerouslySetInnerHTML={{
                  __html: formatMessage(msg.content),
                }}
                className="text-sm leading-relaxed"
              />
              <div className="text-xs opacity-70 mt-1">
                {new Date(msg.timestamp).toLocaleTimeString()}
              </div>
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-gray-100 p-3 rounded-lg">
              <div className="flex items-center gap-2 text-gray-600">
                <div className="animate-spin rounded-full h-4 w-4 border-2 border-emerald-600 border-t-transparent"></div>
                <span className="text-sm">AgriMind is thinking...</span>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Error Message */}
      {error && (
        <div className="px-4 py-2 bg-red-50 border-t border-red-200">
          <p className="text-sm text-red-600">{error}</p>
        </div>
      )}

      {/* Input Area */}
      <div className="p-4 border-t bg-gray-50">
        {canSendMore() ? (
          <div className="flex items-center gap-2">
            <div className="flex-1 relative">
              <Input
                ref={inputRef}
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask a follow-up question..."
                className="pr-10"
                disabled={isLoading}
              />
              {isSupported && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={toggleListening}
                  className={`absolute right-2 top-1/2 transform -translate-y-1/2 p-1 h-7 w-7 ${
                    isListening ? "text-red-500" : "text-gray-500"
                  }`}
                  disabled={isLoading}
                >
                  {isListening ? <MicOff className="h-4 w-4" /> : <Mic className="h-4 w-4" />}
                </Button>
              )}
            </div>
            <Button
              onClick={handleSendMessage}
              disabled={!message.trim() || isLoading}
              size="sm"
              className="bg-emerald-600 hover:bg-emerald-700"
            >
              <Send className="h-4 w-4" />
            </Button>
          </div>
        ) : (
          <div className="text-center py-3">
            <p className="text-sm text-gray-600 mb-2">
              Maximum chat limit reached. Start a new analysis to continue.
            </p>
            <Button
              onClick={onBack}
              variant="outline"
              size="sm"
              className="border-emerald-600 text-emerald-600 hover:bg-emerald-50"
            >
              New Analysis
            </Button>
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatInterface;
