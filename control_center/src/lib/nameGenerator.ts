const ADJECTIVES = [
  "bold", "swift", "lunar", "quantum", "crimson", "silent", "astral", "neon", 
  "cosmic", "cyber", "apex", "iron", "phantom", "echo", "nova", "solar",
  "glacial", "void", "stellar", "amber", "jade", "cobalt"
];

const NOUNS = [
  "cricket", "forge", "engine", "pulse", "core", "matrix", "nexus", "vertex", 
  "spark", "drift", "wave", "horizon", "beacon", "titan", "cipher", "orbit",
  "prism", "flare", "cascade", "bastion"
];

export function generateRunName(): string {
  const adj = ADJECTIVES[Math.floor(Math.random() * ADJECTIVES.length)];
  const noun = NOUNS[Math.floor(Math.random() * NOUNS.length)];
  const num = Math.floor(Math.random() * 900) + 100; // 100-999
  
  return `${adj}-${noun}-${num}`;
}
