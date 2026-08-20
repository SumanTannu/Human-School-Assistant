import type { Metadata } from "next";
import "./globals.css";
export const metadata: Metadata = { title: "XYZ AI | School Assistant", description: "Human-like AI school assistant" };
export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) { return <html lang="en"><body>{children}</body></html>; }
