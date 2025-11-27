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

  useEffect(() => {
    setMessages(initialHistory);
    setShowTypewriter(true);
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

    // Add user message immediately
    const newUserMessage: ChatMessage = {
      role: "user",
      content: userMessage,
      timestamp: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, newUserMessage]);
    setIsLoading(true);

    try {
      const response = await chatFollowUp(userMessage, messages);
      setMessages(response.chat_history);
      setShowTypewriter(true);
    } catch (error) {
      console.error("Chat error:", error);
      const errorMessage: ChatMessage = {
        role: "assistant",
        content: "Sorry, I encountered an error. Please try again.",
        timestamp: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, errorMessage]);
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
    return content.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>").replace(/\n/g, "<br />");
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
            <Bot className="h-5 w-5 text-green-600" />
            <span className="font-semibold text-green-800">AgriMind Assistant</span>
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
                className={`max-w-[75%] rounded-lg p-3 ${
                  message.role === "user" ? "bg-blue-600 text-white" : "bg-gray-100 text-gray-800"
                }`}
              >
                {message.role === "assistant" && isLastAssistantMessage(index) && showTypewriter ? (
                  <TypewriterText
                    text={message.content}
                    speed={20}
                    fastMode={fastMode}
                    onComplete={() => setShowTypewriter(false)}
                  />
                ) : (
                  <div
                    dangerouslySetInnerHTML={{
                      __html: formatMessage(message.content),
                    }}
                  />
                )}
              </div>
            </div>
          ))}

          {isLoading && (
            <div className="flex justify-start">
              <div className="bg-gray-100 rounded-lg p-3">
                <div className="flex items-center space-x-2">
                  <Bot className="h-4 w-4 text-green-600" />
                  <Loader2 className="h-4 w-4 animate-spin text-gray-600" />
                  <span className="text-sm text-gray-600">Thinking...</span>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="p-4 border-t">
          <div className="flex space-x-2">
            <Input
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask a follow-up question..."
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
          <div className="text-xs text-gray-500 mt-2 text-center">
            {Math.max(0, 4 - Math.floor(messages.length / 2))} questions remaining
          </div>
        </div>
      </div>
    </div>
  );
}
