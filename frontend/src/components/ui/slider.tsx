import * as React from "react"

import { cn } from "@/lib/utils"

// Simple slider component without Radix UI for now
export const Slider = React.forwardRef<
  HTMLInputElement,
  {
    value: number[];
    onValueChange: (value: number[]) => void;
    min?: number;
    max?: number;
    step?: number;
    className?: string;
  }
>(({ className, value, onValueChange, min, max, step, ...props }, ref) => (
  <input
    type="range"
    className={cn(
      "w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider",
      className
    )}
    ref={ref}
    value={value[0] || 0}
    onChange={(e) => onValueChange([parseFloat(e.target.value)])}
    min={min}
    max={max}
    step={step}
    {...props}
  />
));
Slider.displayName = "Slider"
