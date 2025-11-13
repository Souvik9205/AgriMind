'use client'
import { Poppins } from "next/font/google";
import UploadZone from "./uploadzone";
import Image from "next/image";
import { useLanguage } from "@/hooks/languageProvider";
import { translations } from "@/constants";

const poppins = Poppins({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700", "800", "900"],
});

export default function Hero() {
  const { language } = useLanguage();
  const t = translations[language as keyof typeof translations];
  return (
    <section className="w-full p-3 ">
      <div className="relative rounded-xl overflow-hidden">
        <Image
          src={"/background.png"}
          alt="background"
          width={3000}
          height={2000}
          className="absolute -z-10 top-0 left-0 w-full h-full object-cove blur-xs"
        />
        <div className="w-full max-w-[1200px] mx-auto px-4 pt-14 pb-10">
          <div className="grid md:grid-cols-[1.1fr_0.9fr] gap-10 items-center">
            <div className="py-2">
              <span className="inline-block bg-white/30 text-emerald-800 px-3 py-1 rounded-full text-[12px] font-bold tracking-wider uppercase mb-3">
                {t.badge}
              </span>
              <h1
                className={`max-w-3xl text-[58px] ${poppins.className} leading-[1.05] font-extrabold text-emerald-800 mb-4`}
              >
                {t.headline}
              </h1>
              <p className="text-[18px] text-slate-100 mb-6">
                {t.description}
              </p>
              <div className="flex gap-3 flex-wrap items-center">
                <button className="bg-emerald-700 text-white px-5 py-3 rounded-xl font-extrabold transition hover:-translate-y-0.5">
                  {t.analyzeCrop}
                </button>

                <button className="bg-white text-emerald-900 px-4 py-3 rounded-xl font-bold border border-emerald-500/25 transition hover:-translate-y-0.5 hover:shadow-md">
                  {t.howItWorks}
                </button>
              </div>

              <div className="flex items-center gap-4 mt-6 flex-wrap">
                <div className="flex gap-1">
                  {[1, 2, 3, 4, 5].map((i) => (
                    <span key={i} className="text-[20px] text-emerald-700">
                      ★
                    </span>
                  ))}
                </div>
                <span className="text-slate-100 text-[14px]">{t.lovedBy}</span>
              </div>
            </div>
            <UploadZone />
          </div>
        </div>
        <div className="w-full max-w-[1200px] mx-auto px-4 pt-2 pb-9">
          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-white/70 rounded-[14px] p-5 shadow-lg shadow-black/5 border border-emerald-500/10">
              <p className="text-[13px] text-slate-500 font-bold uppercase tracking-wide mb-1.5">
                {t.monthlyGrowth}
              </p>
              <h3 className="text-[28px] font-extrabold text-emerald-900">+34%</h3>
            </div>

            <div className="bg-white/70 rounded-[14px] p-5 shadow-lg shadow-black/5 border border-emerald-500/10">
              <p className="text-[13px] text-slate-500 font-bold uppercase tracking-wide mb-1.5">
                {t.churnRate}
              </p>
              <h3 className="text-[28px] font-extrabold text-emerald-900">2.1%</h3>
            </div>

            <div className="bg-white/70 rounded-[14px] p-5 shadow-lg shadow-black/5 border border-emerald-500/10">
              <p className="text-[13px] text-slate-500 font-bold uppercase tracking-wide mb-1.5">
                {t.activeWorkspaces}
              </p>
              <h3 className="text-[28px] font-extrabold text-emerald-900">12,480</h3>
            </div>
          </div>
        </div>{" "}
      </div>
    </section>
  );
}
