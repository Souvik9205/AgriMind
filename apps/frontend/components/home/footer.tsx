export function Footer() {
  return (
    <footer className="w-full bg-white border-t border-emerald-500/10 py-6">
      {/* Top Grid */}
      <div className="max-w-[1200px] mx-auto px-4 grid grid-cols-1 md:grid-cols-[1.2fr_1fr_1fr_1fr] gap-4">

        {/* Brand */}
        <div>
          <div className="flex items-center gap-2.5 mb-2.5">
            <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-emerald-500 to-emerald-700 shadow-lg shadow-emerald-500/35"></div>
            <span className="text-[18px] font-extrabold text-emerald-900">GreenSaaS</span>
          </div>

          <p className="text-[14px] text-slate-600 mb-2">Build. Measure. Grow.</p>

          <div className="flex gap-3">
            <a href="#" className="text-emerald-700 hover:text-emerald-600">X</a>
            <a href="#" className="text-emerald-700 hover:text-emerald-600">GitHub</a>
            <a href="#" className="text-emerald-700 hover:text-emerald-600">LinkedIn</a>
          </div>
        </div>

        {/* Product */}
        <div>
          <h4 className="text-[14px] font-extrabold text-emerald-900 mb-2">Product</h4>
          <ul className="grid gap-1.5">
            <a href="#" className="text-slate-700 hover:text-emerald-600">Features</a>
            <a href="#" className="text-slate-700 hover:text-emerald-600">Integrations</a>
            <a href="#" className="text-slate-700 hover:text-emerald-600">API</a>
            <a href="#" className="text-slate-700 hover:text-emerald-600">Changelog</a>
          </ul>
        </div>

        {/* Company */}
        <div>
          <h4 className="text-[14px] font-extrabold text-emerald-900 mb-2">Company</h4>
          <ul className="grid gap-1.5">
            <a href="#" className="text-slate-700 hover:text-emerald-600">About</a>
            <a href="#" className="text-slate-700 hover:text-emerald-600">Careers</a>
            <a href="#" className="text-slate-700 hover:text-emerald-600">Blog</a>
            <a href="#" className="text-slate-700 hover:text-emerald-600">Press</a>
          </ul>
        </div>

        {/* Legal */}
        <div>
          <h4 className="text-[14px] font-extrabold text-emerald-900 mb-2">Legal</h4>
          <ul className="grid gap-1.5">
            <a href="#" className="text-slate-700 hover:text-emerald-600">Privacy</a>
            <a href="#" className="text-slate-700 hover:text-emerald-600">Terms</a>
            <a href="#" className="text-slate-700 hover:text-emerald-600">Security</a>
          </ul>
        </div>

      </div>

      {/* Bottom Bar */}
      <div className="max-w-[1200px] mx-auto mt-3 px-4 flex justify-between items-center flex-wrap gap-2">
        <p className="text-[12px] text-slate-500">© 2025 GreenSaaS. All rights reserved.</p>

        <a href="#" className="text-emerald-700 hover:text-emerald-600 text-[12px] font-bold">
          Back to top ↑
        </a>
      </div>
    </footer>
  );
}
