import { useState } from "react";
import { Check, Copy, Zap, Network } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useAppStore } from "@/store/useAppStore";
import { useTuningStore } from "@/store/useTuningStore";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { categorizeParams, GROUP_THEMES } from "@/lib/config-schema";
import { ConfigBadge } from "./ConfigBadge";

export function HydraConfigViewer({ configStr }: { configStr: string }) {
  const [copied, setCopied] = useState(false);

  let configObj: Record<string, any> = {};
  try {
    configObj = JSON.parse(configStr);
  } catch (e) {
    return <span className="text-red-400">Invalid config JSON.</span>;
  }

  const { hardware, network, mcts, training, other } =
    categorizeParams(configObj);

  const handleCopy = (e: React.MouseEvent) => {
    e.stopPropagation();
    navigator.clipboard.writeText(JSON.stringify(configObj, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const renderGroup = (title: string, items: [string, any][]) => {
    if (items.length === 0) return null;
    const theme = GROUP_THEMES[title] || GROUP_THEMES["Other Setup"];

    return (
      <div className="mb-4 last:mb-0">
        <h4
          className={`text-[11px] font-black uppercase tracking-widest mb-2 px-1 border-b pb-1 flex items-center gap-1.5 ${theme.badgeBorder} ${theme.iconColor}`}
        >
          {title}
        </h4>
        <div className="flex flex-wrap gap-2">
          {items.map(([key, value]) => (
            <ConfigBadge
              key={key}
              paramKey={key}
              value={value}
              groupTitle={title}
              compact={false}
            />
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className="relative group/config mt-1 bg-[#09090b]/80 border border-white/5 rounded-lg p-4 flex flex-col gap-2 max-h-[350px] overflow-y-auto shadow-inner bg-gradient-to-br from-black/40 to-transparent">
      <div className="flex items-center justify-between mb-2">
        <span className="text-[10px] font-black text-zinc-500 uppercase tracking-widest px-1">
          Active Engine Parameters
        </span>
        <Button
          variant="secondary"
          size="sm"
          className="h-6 px-2 text-[10px] uppercase font-bold tracking-widest bg-zinc-800/80 hover:bg-zinc-700/80 text-zinc-300 transition-colors shadow-sm"
          onClick={handleCopy}
        >
          {copied ? (
            <Check className="w-3 h-3 mr-1.5 text-emerald-400" />
          ) : (
            <Copy className="w-3 h-3 mr-1.5 text-sky-400" />
          )}
          {copied ? "Copied!" : "Copy Payload"}
        </Button>
      </div>

      <TooltipProvider>
        {renderGroup("Hardware & Distribution", hardware)}
        {renderGroup("Network Architecture", network)}
        {renderGroup("MCTS Search Logic", mcts)}
        {renderGroup("Training Hyperparameters", training)}
        {renderGroup("Other Setup", other)}
      </TooltipProvider>
    </div>
  );
}

export function CompactTrialParams({
  params,
}: {
  params: Record<string, any>;
}) {
  const [copied, setCopied] = useState(false);
  const setInitialRunConfig = useAppStore((s) => s.setInitialRunConfig);
  const setIsCreatingRun = useAppStore((s) => s.setIsCreatingRun);
  const setInitialRefineConfig = useTuningStore(
    (s) => s.setInitialRefineConfig,
  );
  const setIsCreatingStudy = useAppStore((s) => s.setIsCreatingStudy);

  if (!params || Object.keys(params).length === 0) {
    return <span className="text-zinc-600 italic">No parameters</span>;
  }

  const { hardware, network, mcts, training, other } = categorizeParams(params);

  const handleCopy = (e: React.MouseEvent) => {
    e.stopPropagation();
    navigator.clipboard.writeText(JSON.stringify(params, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const renderCompactGroup = (title: string, items: [string, any][]) => {
    if (items.length === 0) return null;
    return (
      <>
        {items.map(([key, value]) => (
          <ConfigBadge
            key={key}
            paramKey={key}
            value={value}
            groupTitle={title}
            compact={true}
          />
        ))}
      </>
    );
  };

  return (
    <TooltipProvider>
      <div className="flex items-start gap-2">
        <div className="flex flex-wrap gap-1 max-w-[400px]">
          {renderCompactGroup("Hardware & Distribution", hardware)}
          {renderCompactGroup("Network Architecture", network)}
          {renderCompactGroup("MCTS Search Logic", mcts)}
          {renderCompactGroup("Training Hyperparameters", training)}
          {renderCompactGroup("Other Setup", other)}
        </div>
        <div className="flex flex-col gap-1 mt-0.5 shrink-0">
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="ghost"
                size="icon"
                onClick={(e) => {
                  e.stopPropagation();
                  setInitialRunConfig(params);
                  setIsCreatingRun(true);
                }}
                className="h-5 w-5 hover:bg-emerald-500/20 text-emerald-400"
              >
                <Zap className="w-3 h-3" />
              </Button>
            </TooltipTrigger>
            <TooltipContent side="right" className="text-[10px] bg-black">
              Run Config
            </TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="ghost"
                size="icon"
                onClick={(e) => {
                  e.stopPropagation();
                  setInitialRefineConfig(params);
                  setIsCreatingStudy(true);
                }}
                className="h-5 w-5 hover:bg-purple-500/20 text-purple-400"
              >
                <Network className="w-3 h-3" />
              </Button>
            </TooltipTrigger>
            <TooltipContent side="right" className="text-[10px] bg-black">
              Refine Tuning Bounds
            </TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="ghost"
                size="icon"
                onClick={handleCopy}
                className="h-5 w-5 hover:bg-white/10"
              >
                {copied ? (
                  <Check className="w-3 h-3 text-emerald-400" />
                ) : (
                  <Copy className="w-3 h-3 text-zinc-400" />
                )}
              </Button>
            </TooltipTrigger>
            <TooltipContent side="right" className="text-[10px] bg-black">
              Copy JSON
            </TooltipContent>
          </Tooltip>
        </div>
      </div>
    </TooltipProvider>
  );
}
