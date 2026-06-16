/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Dark glassmorphism palette. Surfaces sit over a deep slate base.
        base: {
          900: "#070a12",
          800: "#0b0f1a",
          700: "#111726",
        },
        ink: {
          DEFAULT: "#e6ebf5",
          muted: "#8a93a6",
          faint: "#5b6373",
        },
        accent: {
          DEFAULT: "#5eead4", // teal
          glow: "#5eead4",
          violet: "#a78bfa",
          amber: "#fbbf24",
          rose: "#fb7185",
        },
        line: "rgba(255,255,255,0.08)",
      },
      fontFamily: {
        sans: [
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "Roboto",
          "sans-serif",
        ],
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      backdropBlur: {
        xs: "2px",
      },
      boxShadow: {
        glass: "0 8px 32px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.05)",
        glow: "0 0 24px rgba(94,234,212,0.25)",
      },
      keyframes: {
        "fade-in": {
          from: { opacity: "0", transform: "translateY(6px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
        "pulse-dot": {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.35" },
        },
        shimmer: {
          "100%": { transform: "translateX(100%)" },
        },
      },
      animation: {
        "fade-in": "fade-in 0.4s ease-out both",
        "pulse-dot": "pulse-dot 1.6s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};
