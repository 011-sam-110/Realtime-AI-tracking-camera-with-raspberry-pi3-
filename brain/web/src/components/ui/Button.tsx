import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-lg text-sm font-medium transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/60 disabled:pointer-events-none disabled:opacity-50 active:scale-[0.98]",
  {
    variants: {
      variant: {
        primary:
          "bg-accent/90 text-base-900 hover:bg-accent shadow-glow font-semibold",
        secondary:
          "border border-line bg-white/5 text-ink hover:bg-white/10 backdrop-blur",
        ghost: "text-ink-muted hover:bg-white/5 hover:text-ink",
        danger:
          "border border-accent-rose/30 bg-accent-rose/10 text-accent-rose hover:bg-accent-rose/20",
        outline:
          "border border-line bg-transparent text-ink hover:bg-white/5",
      },
      size: {
        sm: "h-8 px-3 text-xs",
        md: "h-9 px-4",
        lg: "h-11 px-6 text-base",
        icon: "h-9 w-9",
      },
    },
    defaultVariants: { variant: "secondary", size: "md" },
  },
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, ...props }, ref) => (
    <button
      ref={ref}
      className={cn(buttonVariants({ variant, size }), className)}
      {...props}
    />
  ),
);
Button.displayName = "Button";

export { buttonVariants };
