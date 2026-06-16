import * as React from "react";
import { cn } from "@/lib/utils";

export const Input = React.forwardRef<
  HTMLInputElement,
  React.InputHTMLAttributes<HTMLInputElement>
>(({ className, ...props }, ref) => (
  <input
    ref={ref}
    className={cn(
      "h-9 w-full rounded-lg border border-line bg-base-800/60 px-3 text-sm text-ink placeholder:text-ink-faint",
      "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/60",
      className,
    )}
    {...props}
  />
));
Input.displayName = "Input";

export const Select = React.forwardRef<
  HTMLSelectElement,
  React.SelectHTMLAttributes<HTMLSelectElement>
>(({ className, ...props }, ref) => (
  <select
    ref={ref}
    className={cn(
      "h-9 w-full rounded-lg border border-line bg-base-800/60 px-3 text-sm text-ink",
      "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/60",
      className,
    )}
    {...props}
  />
));
Select.displayName = "Select";
