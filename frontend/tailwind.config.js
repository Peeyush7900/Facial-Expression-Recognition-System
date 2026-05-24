/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        dark: {
          950: '#05070c',
          900: '#0b0f19',
          800: '#111827',
          700: '#1f2937',
        },
        brand: {
          50: '#eef8ff',
          100: '#d8efff',
          200: '#b9e3ff',
          300: '#8ad1ff',
          400: '#54b4ff',
          500: '#2c93ff',
          600: '#1673eb',
          700: '#105bcb',
          800: '#134ca4',
          900: '#164282',
        }
      }
    },
  },
  plugins: [],
}
