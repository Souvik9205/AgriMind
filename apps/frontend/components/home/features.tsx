import { Leaf, Brain, LineChart } from "lucide-react";

const features = [
  {
    icon: <Leaf size={32} className="text-emerald-600" />,
    title: "Plant Disease Detection",
    description:
      "Identify plant diseases with high accuracy using our advanced Vision Transformer models. Just upload an image to get started.",
  },
  {
    icon: <Brain size={32} className="text-emerald-600" />,
    title: "Agricultural Knowledge Assistant",
    description:
      "Get instant, intelligent answers to your farming questions with our RAG-powered Q&A system, trained on a vast agricultural knowledge base.",
  },
  {
    icon: <LineChart size={32} className="text-emerald-600" />,
    title: "Market Intelligence",
    description:
      "Access real-time market data and price information to make informed decisions about your crops and maximize profitability.",
  },
];

export function Features() {
  return (
    <section className="w-full py-20 bg-slate-50">
      <div className="max-w-[1200px] mx-auto px-4">
        <div className="text-center mb-12">
          <h2 className="text-4xl font-extrabold text-emerald-900 mb-3">Our Core Services</h2>
          <p className="text-slate-600 max-w-2xl mx-auto">
            We provide a suite of AI-powered tools to help you every step of the way, from planting
            to market.
          </p>
        </div>
        <div className="grid md:grid-cols-3 gap-8">
          {features.map((feature, index) => (
            <div
              key={index}
              className="bg-white p-8 rounded-2xl shadow-lg shadow-black/5 border border-emerald-500/10 text-center"
            >
              <div className="inline-block bg-emerald-100 p-4 rounded-full mb-4">
                {feature.icon}
              </div>
              <h3 className="text-xl font-bold text-emerald-900 mb-2">{feature.title}</h3>
              <p className="text-slate-600 text-sm">{feature.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
