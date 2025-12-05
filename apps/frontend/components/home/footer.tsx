import Image from "next/image";

export function Footer() {
  return (
    <footer className="w-full bg-white border-t border-emerald-500/10 py-6">
      {/* Top Grid */}
      <div className="max-w-[1200px] mx-auto px-4 grid grid-cols-1 md:grid-cols-[1.2fr_1fr_1fr_1fr] gap-4">
        {/* Brand */}
        <div>
          <div className="flex items-center mb-2.5">
          <Image src="/logo.png" alt="logo" width={50} height={50} />
            <span className="text-[18px] font-extrabold text-emerald-900">AgriMind</span>
          </div>

          <p className="text-[14px] text-slate-600 mb-2">
            AI-powered agricultural assistance for modern farming.
          </p>

          <div className="flex gap-3">
            <a href="#" className="text-emerald-700 hover:text-emerald-600">
              GitHub
            </a>
          </div>
        </div>

        {/* Features */}
        <div>
          <h4 className="text-[14px] font-extrabold text-emerald-900 mb-2">Features</h4>
          <ul className="grid gap-1.5">
            <a href="#" className="text-slate-700 hover:text-emerald-600">
              Disease Detection
            </a>
            <a href="#" className="text-slate-700 hover:text-emerald-600">
              Knowledge Assistant
            </a>
            <a href="#" className="text-slate-700 hover:text-emerald-600">
              Market Intelligence
            </a>
            <a href="#" className="text-slate-700 hover:text-emerald-600">
              Crop-Specific Insights
            </a>
          </ul>
        </div>

        {/* Resources */}
        <div>
          <h4 className="text-[14px] font-extrabold text-emerald-900 mb-2">Resources</h4>
          <ul className="grid gap-1.5">
            <a href="#" className="text-slate-700 hover:text-emerald-600">
              Documentation
            </a>
            <a href="#" className="text-slate-700 hover:text-emerald-600">
              API Docs
            </a>
            <a href="#" className="text-slate-700 hover:text-emerald-600">
              System Architecture
            </a>
          </ul>
        </div>

        {/* Legal */}
        <div>
          <h4 className="text-[14px] font-extrabold text-emerald-900 mb-2">Legal</h4>
          <ul className="grid gap-1.5">
            <a href="#" className="text-slate-700 hover:text-emerald-600">
              Privacy
            </a>
            <a href="#" className="text-slate-700 hover:text-emerald-600">
              Terms
            </a>
            <a href="#" className="text-slate-700 hover:text-emerald-600">
              Security
            </a>
          </ul>
        </div>
      </div>

      {/* Bottom Bar */}
      <div className="max-w-[1200px] mx-auto mt-3 px-4 flex justify-between items-center flex-wrap gap-2">
        <p className="text-[12px] text-slate-500">© 2025 AgriMind. All rights reserved.</p>

        <a href="#" className="text-emerald-700 hover:text-emerald-600 text-[12px] font-bold">
          Back to top ↑
        </a>
      </div>
    </footer>
  );
}
