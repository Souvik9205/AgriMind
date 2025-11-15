"use client";

import { useState, useEffect } from "react";

interface TypewriterTextProps {
  text: string;
  speed?: number;
  onComplete?: () => void;
  className?: string;
  fastMode?: boolean; // New prop for faster typing
}

export function TypewriterText({
  text,
  speed = 30,
  onComplete,
  className = "",
  fastMode = false,
}: TypewriterTextProps) {
  const [displayedText, setDisplayedText] = useState("");
  const [currentIndex, setCurrentIndex] = useState(0);

  useEffect(() => {
    if (currentIndex < text.length) {
      const actualSpeed = fastMode ? Math.max(speed * 0.3, 10) : speed; // 3x faster in fast mode
      const timer = setTimeout(() => {
        setDisplayedText((prev) => prev + text[currentIndex]);
        setCurrentIndex((prev) => prev + 1);
      }, actualSpeed);

      return () => clearTimeout(timer);
    } else if (onComplete) {
      onComplete();
    }
  }, [currentIndex, text, speed, onComplete, fastMode]);

  // Reset when text changes
  useEffect(() => {
    setDisplayedText("");
    setCurrentIndex(0);
  }, [text]);

  return (
    <div className={className}>
      <div
        dangerouslySetInnerHTML={{
          __html: displayedText.replace(/\n/g, "<br />"),
        }}
      />
      {currentIndex < text.length && <span className="animate-pulse">|</span>}
    </div>
  );
}
