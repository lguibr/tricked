import { Field, FieldLabel } from "@/components/ui/field";
import { Slider } from "@/components/ui/slider";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Info } from "lucide-react";

export interface FieldDef {
  key: string;
  label: string;
  min: number;
  max: number;
  step?: number;
  tooltip?: string;
}

export interface GroupDef {
  title: string;
  color?: string;
  icon?: any;
  fields: FieldDef[];
}

export interface ParameterFormProps {
  mode: "single" | "bounds";
  groups: GroupDef[];
  value: Record<string, any>;
  onChange: (val: Record<string, any>) => void;
}

export function ParameterForm({
  mode,
  groups,
  value,
  onChange,
}: ParameterFormProps) {
  const handleSingleChange = (key: string, val: number[]) => {
    onChange({ ...value, [key]: val[0] });
  };

  const handleBoundsChange = (
    key: string,
    boundType: "min" | "max",
    val: number[],
    fieldDef: FieldDef,
  ) => {
    const currentBounds = value[key] || {
      min: fieldDef.min,
      max: fieldDef.max,
    };
    let newMin = currentBounds.min;
    let newMax = currentBounds.max;

    if (boundType === "min") {
      newMin = val[0];
      if (newMin > newMax) newMax = newMin;
    } else {
      newMax = val[0];
      if (newMax < newMin) newMin = newMax;
    }

    onChange({ ...value, [key]: { min: newMin, max: newMax } });
  };

  return (
    <TooltipProvider>
      <div className="flex flex-col gap-3">
        {groups.map((group, idx) => (
          <div
            key={idx}
            className="flex flex-col gap-2 p-3 bg-zinc-950 rounded-lg border border-border/40"
          >
            <div className="flex flex-row items-center gap-2">
              {group.icon && (
                <group.icon
                  className={`w-4 h-4 ${group.color || "text-zinc-300"}`}
                />
              )}
              <span
                className={`text-xs font-bold uppercase tracking-wider ${group.color || "text-zinc-300"}`}
              >
                {group.title}
              </span>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-1 p-3 bg-[#0a0a0c] rounded border border-border/20">
              {group.fields.map((f) => {
                const labelContent = (
                  <div className="flex items-center gap-1">
                    {f.label}
                    {f.tooltip && (
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Info className="w-3 h-3 text-zinc-500 cursor-help" />
                        </TooltipTrigger>
                        <TooltipContent side="top">
                          <p className="max-w-[200px] text-xs leading-relaxed">
                            {f.tooltip}
                          </p>
                        </TooltipContent>
                      </Tooltip>
                    )}
                  </div>
                );

                if (mode === "single") {
                  const currentVal =
                    value[f.key] !== undefined ? value[f.key] : f.min;
                  return (
                    <Field key={f.key}>
                      <div className="flex justify-between w-full mb-1">
                        <FieldLabel className="text-[10px] text-zinc-400 max-w-[80%]">
                          {labelContent}
                        </FieldLabel>
                        <span className="text-[10px] font-mono text-zinc-500">
                          {currentVal}
                        </span>
                      </div>
                      <Slider
                        value={[currentVal]}
                        onValueChange={(v) => handleSingleChange(f.key, v)}
                        min={f.min}
                        max={f.max}
                        step={f.step || 1}
                        className="py-1"
                      />
                    </Field>
                  );
                } else {
                  // bounds mode
                  const currentBounds = value[f.key] || {
                    min: f.min,
                    max: f.max,
                  };
                  return (
                    <Field key={f.key} className="col-span-1 sm:col-span-2">
                      <div className="flex justify-between mb-1">
                        <FieldLabel className="text-[10px] text-zinc-400">
                          <div className="flex items-center gap-1">
                            {f.label} (Min/Max)
                            {f.tooltip && (
                              <Tooltip>
                                <TooltipTrigger asChild>
                                  <Info className="w-3 h-3 text-zinc-500 cursor-help" />
                                </TooltipTrigger>
                                <TooltipContent side="top">
                                  <p className="max-w-[200px] text-xs leading-relaxed">
                                    {f.tooltip}
                                  </p>
                                </TooltipContent>
                              </Tooltip>
                            )}
                          </div>
                        </FieldLabel>
                        <span className="text-[10px] font-mono text-zinc-500">
                          {currentBounds.min} - {currentBounds.max}
                        </span>
                      </div>
                      <div className="flex gap-4">
                        <Slider
                          value={[currentBounds.min]}
                          onValueChange={(v) =>
                            handleBoundsChange(f.key, "min", v, f)
                          }
                          min={f.min}
                          max={f.max}
                          step={f.step || 1}
                          className="py-1 flex-1"
                        />
                        <Slider
                          value={[currentBounds.max]}
                          onValueChange={(v) =>
                            handleBoundsChange(f.key, "max", v, f)
                          }
                          min={f.min}
                          max={f.max}
                          step={f.step || 1}
                          className="py-1 flex-1"
                        />
                      </div>
                    </Field>
                  );
                }
              })}
            </div>
          </div>
        ))}
      </div>
    </TooltipProvider>
  );
}
