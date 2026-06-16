import * as React from "react";
import { faceUrl } from "@/api/client";
import { colorForId, initials } from "@/lib/utils";
import { cn } from "@/lib/utils";

/**
 * Round (or filled) avatar showing a person's newest face crop
 * (/api/faces/{id}/0.jpg), falling back to coloured initials if the crop is
 * missing — a graceful empty state for the gallery and live panel.
 *
 * size: pixel diameter for the round avatar (ignored when fill=true).
 * fill: stretch to the parent box (used for the square gallery thumbnail).
 */
export function PersonAvatar({
  id,
  label,
  size = 40,
  fill = false,
  className,
}: {
  id: number;
  label: string;
  size?: number;
  fill?: boolean;
  className?: string;
}) {
  const [failed, setFailed] = React.useState(false);
  const color = colorForId(id);
  const dim = fill ? undefined : { width: `${size}px`, height: `${size}px` };
  const initialFontPx = fill ? 40 : size * 0.4;

  return (
    <div
      className={cn(
        "relative flex shrink-0 items-center justify-center overflow-hidden",
        fill ? "h-full w-full" : "rounded-full border border-line",
        className,
      )}
      style={dim}
    >
      {!failed ? (
        <img
          src={faceUrl(id, 0)}
          alt={label}
          loading="lazy"
          className="h-full w-full object-cover"
          onError={() => setFailed(true)}
        />
      ) : (
        <div
          className="flex h-full w-full items-center justify-center font-semibold"
          style={{
            background: `${color}22`,
            color,
            fontSize: initialFontPx,
          }}
        >
          {initials(label)}
        </div>
      )}
    </div>
  );
}
