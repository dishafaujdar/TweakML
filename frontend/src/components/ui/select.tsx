import * as React from "react"

import { cn } from "@/lib/utils"

// Simple select components without Radix UI for now
export const Select = React.forwardRef<
  HTMLSelectElement,
  React.SelectHTMLAttributes<HTMLSelectElement> & {
    onValueChange?: (value: string) => void;
    children: React.ReactNode;
  }
>(({ className, onValueChange, onChange, children, ...props }, ref) => (
  <select
    className={cn(
      "flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50",
      className
    )}
    ref={ref}
    onChange={(e) => {
      onChange?.(e);
      onValueChange?.(e.target.value);
    }}
    {...props}
  >
    {children}
  </select>
));
Select.displayName = "Select"

export const SelectTrigger = Select
export const SelectContent = ({ children }: { children: React.ReactNode }) => <>{children}</>
export const SelectItem = ({ children, value }: { children: React.ReactNode, value: string }) => (
  <option value={value}>{children}</option>
)
export const SelectValue = ({ placeholder }: { placeholder?: string }) => (
  <option value="" disabled>{placeholder}</option>
)
