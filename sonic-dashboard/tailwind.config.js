/** @type {import('tailwindcss').Config} */
const colors = require("tailwindcss/colors");

module.exports = {
  darkMode: ["class"],
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        // ── Surface scale (quiet graphite layers) ───────────────────
        ink: {
          950: "#090c11",
          900: "#0d1218",
          850: "#111820",
          800: "#161e27",
          750: "#202a35",
          700: "#293541",
          600: "#354452",
          500: "#495968",
        },
        // ── Text / muted ────────────────────────────────────────────
        slate: colors.slate,
        muted: {
          DEFAULT: "#96a2ad",
          dim: "#687681",
          bright: "#d0d8dd",
        },
        // ── Brand accent (measured mineral teal) ────────────────────
        primary: {
          DEFAULT: "#8adfd2",
          400: "#a8eee2",
          500: "#8adfd2",
          600: "#63c5b9",
          700: "#419b92",
        },
        secondary: {
          DEFAULT: "#a9b8c2",
          400: "#c3d0d8",
          500: "#a9b8c2",
          600: "#8397a4",
          700: "#61747f",
        },
        accent: {
          DEFAULT: "#91a8bc",
          400: "#b0c1cf",
          500: "#91a8bc",
          600: "#70899e",
        },
        success: "#22E0A1",
        warning: "#FBBF24",
        danger: "#FF5C5C",
      },
      fontFamily: {
        sans: ["var(--font-sans)", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "ui-monospace", "monospace"],
      },
      boxShadow: {
        glow: "0 0 20px rgba(138, 223, 210, 0.12)",
        "glow-cyan": "0 0 20px rgba(138, 223, 210, 0.12)",
        "glow-violet": "0 0 20px rgba(145, 168, 188, 0.12)",
        "inner-glow": "inset 0 1px 0 0 rgba(255,255,255,0.035)",
      },
      backgroundImage: {
        "grid-glow":
          "radial-gradient(circle at 55% 0%, rgba(138,223,210,0.045), transparent 38%), linear-gradient(180deg, #090c11 0%, #0b0f14 100%)",
      },
      keyframes: {
        shimmer: {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
        "fade-in-up": {
          "0%": { opacity: "0", transform: "translateY(8px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        "pulse-ring": {
          "0%": { transform: "scale(0.8)", opacity: "0.6" },
          "100%": { transform: "scale(2.2)", opacity: "0" },
        },
      },
      animation: {
        shimmer: "shimmer 2s linear infinite",
        "fade-in-up": "fade-in-up 0.4s ease-out both",
        "pulse-ring": "pulse-ring 1s ease-out",
      },
    },
  },
  plugins: [],
};
