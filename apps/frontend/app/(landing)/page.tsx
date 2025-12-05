import React from "react";
import Header from "@/components/home/header";
import Hero from "@/components/home/hero";
import { About } from "@/components/home/about";
import { Features } from "@/components/home/features";
import { Footer } from "@/components/home/footer";

const page = () => {
  return (
    <div className="flex flex-col w-full">
      <Header />
      <Hero />
      <About />
      <Features />
      <Footer />
    </div>
  );
};

export default page;
