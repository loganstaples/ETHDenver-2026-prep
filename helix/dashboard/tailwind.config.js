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
                'helix-surface': '#111113',
                'helix-surface2': '#161618',
                'helix-border': '#1e1e22',
                'helix-border2': '#2a2a2e',
                'helix-muted': '#63636e',
                'helix-dim': '#3e3e44',
                'helix-text': '#fafafa',
                'helix-text2': '#a1a1a6',
            },
            fontFamily: {
                sans: ['var(--font-geist-sans)'],
                mono: ['var(--font-geist-mono)'],
            },
            fontSize: {
                '2xs': ['0.6875rem', { lineHeight: '1rem' }],
            },
            animation: {
                'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
            },
        },
    },
    plugins: [],
};
