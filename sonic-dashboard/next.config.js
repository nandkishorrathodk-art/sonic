/** @type {import('next').NextConfig} */
const nextConfig = {
  transpilePackages: ["three"],
  experimental: {
    allowedDevOrigins: ["localhost:12001", "127.0.0.1:12001", "localhost:12000", "127.0.0.1:12000"],
  },
};

module.exports = nextConfig;
