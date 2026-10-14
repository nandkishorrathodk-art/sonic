import React from "react";
import Link from "next/link";

interface BrandMarkProps {
  size?: number;
  withWordmark?: boolean;
  href?: string;
  className?: string;
}

export function BrandMark({ size = 32, withWordmark = false, href, className = "" }: BrandMarkProps) {
  const mark = (
    <div
      className="relative flex items-center gap-2.5"
      style={{ height: size }}
    >
      <div
        className="relative grid place-items-center rounded-[11px] border border-white/10 bg-white/[0.035]"
        style={{ width: size, height: size }}
      >
        <div className="relative grid h-full w-full place-items-center">
          <span
            className="font-sans font-semibold text-primary-400 leading-none"
            style={{ fontSize: size * 0.42 }}
          >
            S
          </span>
        </div>
      </div>
      {withWordmark && (
        <div className="flex flex-col leading-none">
          <span className="font-semibold tracking-[0.08em] text-white text-sm">
            SONIC
          </span>
          <span className="mt-1 text-[8px] font-medium uppercase tracking-[0.16em] text-muted-dim">
            Computer-native AI
          </span>
        </div>
      )}
    </div>
  );

  if (href) {
    return (
      <Link href={href} className={`group ${className}`}>
        <span className="block transition group-hover:scale-[1.03]">{mark}</span>
      </Link>
    );
  }
  return <div className={className}>{mark}</div>;
}
