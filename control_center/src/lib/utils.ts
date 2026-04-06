import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function getProcessColorVariation(baseColor: string, processName: string): string {
  const nameL = processName.toLowerCase();

  // Basic parsing assuming "#RRGGBB"
  let r = 59; let g = 130; let b = 246; // fallback blue
  if (baseColor.startsWith('#') && baseColor.length >= 7) {
    r = parseInt(baseColor.slice(1, 3), 16) || r;
    g = parseInt(baseColor.slice(3, 5), 16) || g;
    b = parseInt(baseColor.slice(5, 7), 16) || b;
  } else if (baseColor.includes('rgb')) {
    const match = baseColor.match(/\d+/g);
    if (match && match.length >= 3) {
      r = parseInt(match[0], 10);
      g = parseInt(match[1], 10);
      b = parseInt(match[2], 10);
    }
  }

  let luminanceMultiplier = 1.0;
  let alpha = 1.0;

  if (nameL.includes('inference')) {
    luminanceMultiplier = 1.3;
  } else if (nameL.includes('reanalyze')) {
    luminanceMultiplier = 0.6;
  } else if (nameL.includes('prefetch')) {
    luminanceMultiplier = 1.5;
  } else if (nameL.includes('mcts')) {
    luminanceMultiplier = 1.0;
  } else if (nameL.includes('self') || nameL.includes('engine')) {
    luminanceMultiplier = 0.8;
  } else {
    // Deterministic jitter
    let hash = 0;
    for (let i = 0; i < processName.length; i++) {
      hash = processName.charCodeAt(i) + ((hash << 5) - hash);
    }
    const jitter = (Math.abs(hash) % 40) / 100;
    luminanceMultiplier = 0.8 + jitter;
  }

  r = Math.min(255, Math.max(0, Math.floor(r * luminanceMultiplier)));
  g = Math.min(255, Math.max(0, Math.floor(g * luminanceMultiplier)));
  b = Math.min(255, Math.max(0, Math.floor(b * luminanceMultiplier)));

  return alpha === 1.0 ? `rgb(${r}, ${g}, ${b})` : `rgba(${r}, ${g}, ${b}, ${alpha})`;
}
