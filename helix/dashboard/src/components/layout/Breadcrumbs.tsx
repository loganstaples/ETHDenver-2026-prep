'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { cn } from '@/lib/utils';

interface Crumb {
  label: string;
  href?: string;
}

function buildCrumbs(pathname: string): Crumb[] {
  const crumbs: Crumb[] = [{ label: 'HELIX', href: '/' }];

  if (pathname === '/') {
    crumbs.push({ label: 'Models' });
    return crumbs;
  }

  const segments = pathname.split('/').filter(Boolean);

  if (segments[0] === 'models' && segments.length >= 2) {
    crumbs.push({ label: 'Models', href: '/' });
    crumbs.push({ label: `Model #${segments[1]}` });
    return crumbs;
  }

  if (segments[0] === 'network') {
    crumbs.push({ label: 'Network' });
    return crumbs;
  }

  if (segments[0] === 'settings') {
    crumbs.push({ label: 'Settings' });
    return crumbs;
  }

  // Fallback: capitalize each segment
  for (let i = 0; i < segments.length; i++) {
    const isLast = i === segments.length - 1;
    const label = segments[i].charAt(0).toUpperCase() + segments[i].slice(1);
    crumbs.push(
      isLast ? { label } : { label, href: '/' + segments.slice(0, i + 1).join('/') },
    );
  }

  return crumbs;
}

export default function Breadcrumbs() {
  const pathname = usePathname();
  const crumbs = buildCrumbs(pathname);

  return (
    <nav aria-label="Breadcrumb" className="flex items-center gap-1.5 text-sm">
      {crumbs.map((crumb, i) => {
        const isLast = i === crumbs.length - 1;

        return (
          <span key={i} className="flex items-center gap-1.5">
            {i > 0 && (
              <span className="text-helix-dim select-none">/</span>
            )}
            {isLast || !crumb.href ? (
              <span className={cn(isLast ? 'text-white' : 'text-helix-muted')}>
                {crumb.label}
              </span>
            ) : (
              <Link
                href={crumb.href}
                className="text-helix-muted hover:text-helix-text2 transition-colors"
              >
                {crumb.label}
              </Link>
            )}
          </span>
        );
      })}
    </nav>
  );
}
