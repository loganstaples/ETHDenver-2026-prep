/** @type {import('tailwindcss').Config} */
module.exports = {
    content: [
        "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
        "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
        "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
    ],
    theme: {
        extend: {
            colors: {
                'helix-bg': '#09090b',
                'helix-surface': '#131316',
                'helix-surface2': '#1a1a1e',
                'helix-border': '#262630',
                'helix-border2': '#363640',
                'helix-muted': '#8b8b9a',
                'helix-dim': '#5e5e6e',
                'helix-text': '#fafafa',
                'helix-text2': '#b8b8c4',
            },
            fontFamily: {
                sans: ['var(--font-geist-sans)'],
                mono: ['var(--font-geist-mono)'],
            },
            fontSize: {
                '2xs': ['0.8125rem', { lineHeight: '1.25rem' }],
            },
            animation: {
                'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
            },
        },
    },
    plugins: [],
};
