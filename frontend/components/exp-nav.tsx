"use client";

import clsx from "clsx";
import { Home } from "lucide-react";
import { EXPERIMENTS } from "@/lib/experiments";

// Sticky switcher so you can jump straight between experiments without going
// back to the home page first.
export function ExpNav({ active }: { active: string }) {
  const items = [...EXPERIMENTS].sort((a, b) => a.order - b.order);
  return (
    <nav className="exp-nav" aria-label="Experiments">
      <a href="#" className="exp-nav-pill exp-nav-home" aria-label="Home">
        <Home size={15} />
      </a>
      {items.map((e) => (
        <a
          key={e.slug}
          href={`#${e.slug}`}
          className={clsx("exp-nav-pill", e.slug === active && "is-active")}
          style={e.slug === active ? { borderColor: e.accent, color: e.accent } : undefined}
        >
          {e.navLabel}
        </a>
      ))}
    </nav>
  );
}
