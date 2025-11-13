import { useLanguage } from "@/hooks/languageProvider";
import { Globe } from "lucide-react";

export const LanguageToggle = () => {
  const { language, toggleLanguage } = useLanguage();

  return (
    <button
      onClick={toggleLanguage}
      className="flex items-center gap-2 px-3 py-2 rounded-lg bg-emerald-50 hover:bg-emerald-100 border border-emerald-200 transition-all"
      aria-label="Toggle language"
    >
      <Globe className="w-4 h-4 text-emerald-700" />
      <span className="text-sm font-semibold text-emerald-700">
        {language === 'en' ? 'বাংলা' : 'English'}
      </span>
    </button>
  );
};