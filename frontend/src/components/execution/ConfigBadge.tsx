import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  getIcon,
  GROUP_THEMES,
  EXPLANATIONS,
  renderValue,
} from "@/lib/config-schema";

export function ConfigBadge({
  paramKey,
  value,
  groupTitle,
  compact = false,
}: {
  paramKey: string;
  value: any;
  groupTitle: string;
  compact?: boolean;
}) {
  const theme = GROUP_THEMES[groupTitle] || GROUP_THEMES["Other Setup"];
  const Icon = getIcon(paramKey);
  const explanation =
    EXPLANATIONS[paramKey] ||
    "Custom parameter passed to the engine process overrides.";

  if (compact) {
    return (
      <Tooltip delayDuration={100}>
        <TooltipTrigger asChild>
          <div
            className={`flex items-center gap-1.5 px-1.5 py-[2px] ${theme.badgeBg} border ${theme.badgeBorder} rounded text-[9px] font-mono cursor-help hover:brightness-125 transition-all truncate`}
          >
            <Icon className={`w-3 h-3 ${theme.iconColor}`} />
            <span className={`${theme.labelText} font-medium`}>
              {paramKey}:
            </span>
            <span className="font-bold text-white drop-shadow-md">
              {renderValue(value)}
            </span>
          </div>
        </TooltipTrigger>
        <TooltipContent
          side="top"
          className="bg-zinc-950 border-white/10 text-[10px] p-2 shadow-xl"
        >
          <span className={theme.iconColor}>{groupTitle}</span>
        </TooltipContent>
      </Tooltip>
    );
  }

  return (
    <Tooltip delayDuration={300}>
      <TooltipTrigger asChild>
        <div
          className={`flex items-center gap-2 px-2.5 py-1 ${theme.badgeBg} border ${theme.badgeBorder} rounded-md text-[10px] sm:text-xs font-mono cursor-help hover:brightness-125 transition-all shadow-sm`}
        >
          <Icon className={`w-3.5 h-3.5 ${theme.iconColor}`} />
          <span
            className={`${theme.labelText} font-semibold truncate max-w-[140px]`}
          >
            {paramKey}
          </span>
          <div className="w-[1px] h-3 bg-white/20 mx-0.5" />
          <span className="font-bold truncate max-w-[140px] drop-shadow-md">
            {renderValue(value)}
          </span>
        </div>
      </TooltipTrigger>
      <TooltipContent
        side="bottom"
        className="max-w-[280px] bg-zinc-950 border-white/10 text-xs text-zinc-300 shadow-2xl p-3"
      >
        <p
          className={`font-black uppercase tracking-wider mb-1 ${theme.iconColor}`}
        >
          {paramKey}
        </p>
        <p className="text-zinc-400 leading-relaxed font-sans">{explanation}</p>
      </TooltipContent>
    </Tooltip>
  );
}
