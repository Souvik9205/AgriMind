"use client";
import { translations } from "@/constants";
import { useLanguage } from "@/hooks/languageProvider";
import { LanguageToggle } from "../global/language-toggle";
import Link from "next/link";
import Image from "next/image";
export default function Header() {
  const { language } = useLanguage();
  const t = translations[language as keyof typeof translations];
  return (
    <header className="w-full sticky top-0 z-50 bg-white/80 backdrop-blur-xl border-b border-emerald-500/10 py-5">
      <div className="w-full max-w-[1200px] mx-auto px-4 flex items-center justify-between">
        {/* Brand */}
        <a href="#" className="flex items-center no-underline">
          <Image src="/logo.png" alt="logo" width={50} height={50} />
          <span className="text-[20px] font-extrabold tracking-wide text-emerald-900">
            AgriMine
          </span>
        </a>

        {/* Desktop Nav */}
        <nav className="hidden md:flex items-center gap-6">
          <LanguageToggle />

          <Link href="/login">
            <button className="bg-emerald-500 text-white px-4 py-2 rounded-[10px] font-bold shadow-lg shadow-emerald-500/35 transition hover:-translate-y-0.5">
              {t.startTrial}
            </button>
          </Link>
        </nav>

        {/* Mobile Menu Button */}
        <button className="md:hidden inline-flex items-center justify-center rounded-lg p-2 text-emerald-700 hover:bg-emerald-50 transition">
          <div className="w-[22px] h-[2px] bg-emerald-500 rounded relative">
            <div className="absolute -top-[7px] w-[22px] h-[2px] bg-emerald-500 rounded"></div>
            <div className="absolute top-0 w-[22px] h-[2px] bg-emerald-500 rounded"></div>
            <div className="absolute top-[7px] w-[22px] h-[2px] bg-emerald-500 rounded"></div>
          </div>
        </button>
      </div>
    </header>
  );
}
