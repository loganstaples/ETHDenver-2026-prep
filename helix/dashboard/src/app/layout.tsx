import { Providers } from "@/components/Providers";
import AppShell from "@/components/layout/AppShell";
import './globals.css';
import { GeistSans } from 'geist/font/sans';
import { GeistMono } from 'geist/font/mono';

export const metadata = {
    title: 'HELIX',
    description: 'Verifiable Machine Learning Network',
};

export default function RootLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    return (
        <html lang="en" className={`${GeistSans.variable} ${GeistMono.variable}`}>
            <body className="font-sans">
                <Providers>
                    <AppShell>{children}</AppShell>
                </Providers>
            </body>
        </html>
    );
}
