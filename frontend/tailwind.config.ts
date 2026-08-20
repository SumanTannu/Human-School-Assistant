import type { Config } from "tailwindcss";
const config: Config = { content: ["./app/**/*.{js,ts,jsx,tsx}"], theme: { extend: { colors: { ink: "#10233f", sky: "#e9f6ff", teal: "#21a59b" }, boxShadow: { soft: "0 16px 50px rgba(32, 77, 119, .12)" } } }, plugins: [] };
export default config;
