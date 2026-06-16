import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-medium",
  {
    variants: {
      variant: {
        default: "border-line bg-white/5 text-ink-muted",
        accent: "border-accent/30 bg-accent/10 text-accent",
        violet: "border-accent-violet/30 bg-accent-violet/10 text-accent-violet",
        amber: "border-accent-amber/30 bg-accent-amber/10 text-accent-amber",
        rose: "border-accent-rose/30 bg-accent-rose/10 text-accent-rose",
        success: "border-emerald-400/30 bg-emerald-400/10 text-emerald-300",
      },
    },
    defaultVariants: { variant: "default" },
  },
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof badgeVariants> {}

export function Badge({ className, variant, ...props }: BadgeProps) {
  return <span className={cn(badgeVariants({ variant }), className)} {...props} />;
}
