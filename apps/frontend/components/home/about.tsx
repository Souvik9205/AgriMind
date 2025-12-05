import Image from "next/image";

export function About() {
  return (
    <section className="w-full py-20">
      <div className="max-w-[1200px] mx-auto px-4 grid md:grid-cols-2 gap-12 items-center">
        <div className="relative">
          <Image
            src="/farmer-illustration.png"
            alt="Farmer tending to crops"
            width={500}
            height={500}
            className="rounded-2xl shadow-lg"
          />
        </div>
        <div>
          <h2 className="text-4xl font-extrabold text-emerald-900 mb-4">About AgriMind</h2>
          <p className="text-slate-600 mb-6">
            AgriMind is an AI-powered platform designed to empower farmers and agricultural
            professionals. We combine cutting-edge plant disease detection with an intelligent
            knowledge assistant to provide comprehensive support for modern farming.
          </p>
          <p className="text-slate-600 mb-6">
            Our platform offers a range of features, including:
          </p>
          <ul className="list-disc list-inside mb-6 pl-6">
            <li>Automated disease detection to help identify and manage crop health issues</li>
            <li>
              Access to a vast knowledge base to provide reliable and accurate answers to farming
              queries
            </li>
            <li>
              Real-time market data and insights to help farmers make informed decisions and
              optimize their business
            </li>
          </ul>
          <button className="bg-emerald-700 text-white px-5 py-3 rounded-xl font-extrabold transition hover:-translate-y-0.5">
            Learn More
          </button>
        </div>
      </div>
    </section>
  );
}
