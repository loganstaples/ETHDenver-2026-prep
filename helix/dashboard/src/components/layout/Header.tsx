'use client';

import { ConnectButton } from '@rainbow-me/rainbowkit';
import Breadcrumbs from './Breadcrumbs';

export default function Header() {
  return (
    <header className="h-14 bg-helix-bg/80 backdrop-blur-sm sticky top-0 z-40 border-b border-helix-border">
      <div className="flex items-center justify-between h-full px-6">
        <Breadcrumbs />
        <ConnectButton
          showBalance={false}
          chainStatus="icon"
          accountStatus="address"
        />
      </div>
    </header>
  );
}
