/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        display: ["'Source Serif 4'", "Cambria", "serif"],
        mono: ["'JetBrains Mono'", "monospace"],
      },
      colors: {
        ink: { DEFAULT: "#1B1F3B", soft: "#475569", mute: "#64748B" },
        brand: { DEFAULT: "#4F46E5", 50: "#EEF2FF", 100: "#E0E7FF", 600: "#4F46E5", 700: "#4338CA", 900: "#312E81" },
        cyan: { DEFAULT: "#0891B2" },
        amber: { DEFAULT: "#F59E0B" },
        good: "#059669",
        warn: "#B45309",
        line: "#E2E8F0",
        surface: "#F7F8FC",
      },
      boxShadow: { card: "0 1px 2px rgba(27,31,59,.06), 0 8px 24px -12px rgba(27,31,59,.18)" },
    },
  },
  plugins: [],
};
