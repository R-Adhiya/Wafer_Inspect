/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        wafer: {
          dark: '#0f172a',
          card: '#1e293b',
          border: '#334155',
          accent: '#38bdf8',
          subtle: '#94a3b8'
        }
      }
    },
  },
  plugins: [],
}
