/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: "#1e1b4b", // deep indigo
        secondary: "#312e81",
        accent: "#6366f1",
        dark: "#0f172a"
      },
    },
  },
  plugins: [],
}
