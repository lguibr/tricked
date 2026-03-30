import { useState, useEffect } from 'react';
import { Slider } from '@/components/ui/slider';
import { Label } from '@/components/ui/label';

interface LogarithmicSliderProps {
  label: string;
  description?: string;
  min: number;
  max: number;
  value: number;
  onChange: (value: number) => void;
  disabled?: boolean;
}

export function LogarithmicSlider({ label, description, min, max, value, onChange, disabled }: LogarithmicSliderProps) {
  const logMin = Math.log10(min);
  const logMax = Math.log10(max);

  const getLinear = (val: number) => {
    const logVal = Math.log10(val);
    return ((logVal - logMin) / (logMax - logMin)) * 100;
  };

  const getReal = (lin: number) => {
    const logVal = logMin + (lin / 100) * (logMax - logMin);
    return Math.pow(10, logVal);
  };

  const [linearValue, setLinearValue] = useState([getLinear(value)]);

  useEffect(() => {
    setLinearValue([getLinear(value)]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value, min, max]);

  const handleLinearChange = (val: number | readonly number[]) => {
    const newVal = Array.isArray(val) ? val : [val];
    setLinearValue(newVal as number[]);
    onChange(getReal(newVal[0]));
  };

  return (
    <div className="grid gap-2">
      <div className="flex items-center justify-between">
        <div className="space-y-0.5">
          <Label className="text-sm font-semibold text-muted-foreground">{label}</Label>
          {description && <p className="text-[10px] text-muted-foreground/60">{description}</p>}
        </div>
        <span className="text-sm font-mono text-primary font-bold">
          {value < 0.01 ? value.toExponential(2) : value.toLocaleString(undefined, { maximumFractionDigits: 4 })}
        </span>
      </div>
      <Slider
        min={0}
        max={100}
        step={0.1}
        value={linearValue}
        onValueChange={handleLinearChange}
        disabled={disabled}
        className="cursor-pointer w-full"
      />
    </div>
  );
}
