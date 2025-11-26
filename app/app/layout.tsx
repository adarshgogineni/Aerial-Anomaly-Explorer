import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "UAP Explorer",
  description: "Interactive visualization of UAP/UFO sighting reports with ML-powered anomaly detection",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        <header className="bg-gray-900 text-white p-4 border-b border-gray-700">
          <div className="container mx-auto">
            <h1 className="text-2xl font-bold">UAP Explorer</h1>
          </div>
        </header>
        {children}
      </body>
    </html>
  );
}
