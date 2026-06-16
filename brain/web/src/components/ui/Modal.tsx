import * as React from "react";
import { X } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "./Button";

export interface ModalProps {
  open: boolean;
  onClose: () => void;
  title?: string;
  children: React.ReactNode;
  className?: string;
}

/** Lightweight accessible modal (no portal dep; renders inline w/ a fixed overlay). */
export function Modal({ open, onClose, title, children, className }: ModalProps) {
  React.useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      role="dialog"
      aria-modal="true"
    >
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm animate-fade-in"
        onClick={onClose}
      />
      <div
        className={cn(
          "glass relative z-10 w-full max-w-md animate-fade-in p-6",
          className,
        )}
      >
        <div className="mb-4 flex items-center justify-between">
          {title && (
            <h2 className="text-base font-semibold text-ink">{title}</h2>
          )}
          <Button
            variant="ghost"
            size="icon"
            onClick={onClose}
            aria-label="Close"
            className="ml-auto"
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
        {children}
      </div>
    </div>
  );
}
