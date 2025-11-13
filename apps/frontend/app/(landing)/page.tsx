import React from "react";
import Header from "@/components/home/header";
import Hero from "@/components/home/hero";
import { Footer } from "@/components/home/footer";

const page = () => {
  return (
    <div className="flex flex-col w-full">
      <Header />
      <Hero />
      <Footer />
    </div>
  );
};

export default page;
