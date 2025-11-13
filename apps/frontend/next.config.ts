import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  images: {
    remotePatterns: [
      {
        protocol: "https",
        hostname: "stock.adobe.com",
      },
    ],
  },
  
};

export default nextConfig;
