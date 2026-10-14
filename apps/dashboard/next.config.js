/** @type {import('next').NextConfig} */
const nextConfig = {
  transpilePackages: ["three"],
  async redirects() {
    return [{ source: "/landing", destination: "/", permanent: true }];
  },
};

module.exports = nextConfig;
