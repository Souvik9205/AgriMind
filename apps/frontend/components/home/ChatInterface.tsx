"use client";

import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { TypewriterText } from "@/components/ui/TypewriterText";
import { Loader2, Send, User, Bot } from "lucide-react";
import { chatFollowUp, ChatMessage } from "@/lib/api";

interface ChatInterfaceProps {
  initialResponse: string;
  sessionId: string;
  initialHistory: ChatMessage[];
  fastMode?: boolean;
}

export function ChatInterface({
  initialResponse,
  sessionId,
  initialHistory,
  fastMode = true,
}: ChatInterfaceProps) {
  const [messages, setMessages] = useState<ChatMessage[]>(initialHistory);
  const [inputMessage, setInputMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [showTypewriter, setShowTypewriter] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Update messages when initialHistory changes
  useEffect(() => {
    setMessages(initialHistory);
    setShowTypewriter(true); // Reset typewriter for new messages
  }, [initialHistory]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

    const userMessage = inputMessage.trim();
    setInputMessage("");
    setIsLoading(true);

    try {
      const response = await chatFollowUp(userMessage, messages);

      setMessages(response.chat_history);
      setShowTypewriter(true);
    } catch (error) {
      console.error("Chat error:", error);
      // Add error message to chat
      const errorMessage: ChatMessage = {
        role: "assistant",
        content: "Sorry, I encountered an error. Please try again.",
        timestamp: new Date().toISOString(),
      };
      setMessages((prev) => [
        ...prev,
        {
          role: "user",
          content: userMessage,
          timestamp: new Date().toISOString(),
        },
        errorMessage,
      ]);
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
    return content
      .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
      .replace(/• (.*?)(?=\n|$)/g, '• <span class="ml-2">$1</span>')
      .replace(/\n/g, "<br />");
  };

  const isLastAssistantMessage = (index: number) => {
    return index === messages.length - 1 && messages[index].role === "assistant";
  };

  return (
    <div className="max-w-4xl mx-auto p-4">
      <div className="h-[600px] flex flex-col border rounded-lg bg-white shadow-sm">
        {/* Chat Header */}
        <div className="p-4 border-b bg-green-50">
          <div className="flex items-center space-x-2">
            <Bot className="h-6 w-6 text-green-600" />
            <span className="font-semibold text-green-800">AgriMind Assistant</span>
            <span className="text-xs text-green-600 bg-green-100 px-2 py-1 rounded-full">
              Session Active
            </span>
          </div>
        </div>

        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.map((message, index) => (
            <div
              key={index}
              className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-[80%] rounded-lg p-3 ${
                  message.role === "user" ? "bg-blue-600 text-white" : "bg-gray-100 text-gray-800"
                }`}
              >
                <div className="flex items-start space-x-2">
                  {message.role === "assistant" && (
                    <Bot className="h-4 w-4 mt-1 shrink-0 text-green-600" />
                  )}
                  {message.role === "user" && <User className="h-4 w-4 mt-1 shrink-0 text-white" />}
                  <div className="flex-1">
                    {message.role === "assistant" &&
                    isLastAssistantMessage(index) &&
                    showTypewriter ? (
                      <TypewriterText
                        text={message.content}
                        speed={20}
                        fastMode={fastMode}
                        onComplete={() => setShowTypewriter(false)}
                        className="prose prose-sm max-w-none"
                      />
                    ) : (
                      <div
                        className="prose prose-sm max-w-none"
                        dangerouslySetInnerHTML={{
                          __html: formatMessage(message.content),
                        }}
                      />
                    )}
                  </div>
                </div>
                <div className="text-xs opacity-70 mt-2">
                  {new Date(message.timestamp).toLocaleTimeString()}
                </div>
              </div>
            </div>
          ))}

          {isLoading && (
            <div className="flex justify-start">
              <div className="bg-gray-100 rounded-lg p-3">
                <div className="flex items-center space-x-2">
                  <Bot className="h-4 w-4 text-green-600" />
                  <Loader2 className="h-4 w-4 animate-spin text-gray-600" />
                  <span className="text-sm text-gray-600">AgriMind is thinking...</span>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="p-4 border-t bg-gray-50">
          <div className="flex space-x-2">
            <Input
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask a follow-up question about your crop..."
              disabled={isLoading}
              className="flex-1"
            />
            <Button
              onClick={handleSendMessage}
              disabled={!inputMessage.trim() || isLoading}
              size="icon"
              className="bg-green-600 hover:bg-green-700"
            >
              {isLoading ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Send className="h-4 w-4" />
              )}
            </Button>
          </div>
          <div className="text-xs text-gray-500 mt-2">
            You have {Math.max(0, 4 - Math.floor(messages.length / 2))} follow-up questions
            remaining
          </div>
        </div>
      </div>
    </div>
  );
}
