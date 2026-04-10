import { Server, Cpu, Database, Network } from "lucide-react";

export function StudyDiagnosticsUninitialized() {
  return (
    <div className="p-8 w-full h-full flex flex-col items-center justify-center bg-[#0a0a0a] text-zinc-400 relative overflow-hidden">
      <div className="absolute top-1/4 left-1/2 -translate-x-1/2 w-[600px] h-[400px] bg-emerald-500/5 blur-[120px] rounded-full pointer-events-none" />

      <Server className="w-16 h-16 text-zinc-700 mb-6 block relative z-10" />
      <h2 className="text-xl font-black text-zinc-200 mb-2 uppercase tracking-widest text-center relative z-10">
        Diagnostics Uninitialized
      </h2>
      <p className="text-sm max-w-md text-center text-zinc-500 mb-8 leading-relaxed relative z-10">
        Baseline tuning relies on recursive optimization to precisely estimate
        maximal hardware concurrency limits, memory alignment paths, and deep
        Monte Carlo search capacities.
      </p>

      <div className="grid grid-cols-3 gap-6 max-w-3xl w-full relative z-10">
        <div className="bg-black/40 backdrop-blur-md border border-white/5 shadow-2xl rounded-xl p-5 flex flex-col items-center hover:border-emerald-500/30 transition-colors">
          <Cpu className="w-8 h-8 text-emerald-500/80 mb-4" />
          <h3 className="text-[11px] font-black text-zinc-200 uppercase tracking-widest mb-1.5">
            Compute Estimates
          </h3>
          <p className="text-[10px] text-zinc-500 text-center uppercase tracking-wider">
            Batch: 256-4096
            <br />
            Workers: 8-32
          </p>
        </div>
        <div className="bg-black/40 backdrop-blur-md border border-white/5 shadow-2xl rounded-xl p-5 flex flex-col items-center hover:border-amber-500/30 transition-colors">
          <Network className="w-8 h-8 text-amber-500/80 mb-4" />
          <h3 className="text-[11px] font-black text-zinc-200 uppercase tracking-widest mb-1.5">
            Search Space
          </h3>
          <p className="text-[10px] text-zinc-500 text-center uppercase tracking-wider">
            Simulations: 10-200
            <br />
            C_puct: 1.0-5.0
          </p>
        </div>
        <div className="bg-black/40 backdrop-blur-md border border-white/5 shadow-2xl rounded-xl p-5 flex flex-col items-center hover:border-purple-500/30 transition-colors">
          <Database className="w-8 h-8 text-purple-500/80 mb-4" />
          <h3 className="text-[11px] font-black text-zinc-200 uppercase tracking-widest mb-1.5">
            Memory Constraints
          </h3>
          <p className="text-[10px] text-zinc-500 text-center uppercase tracking-wider">
            Buffer: 100k
            <br />
            Max Gumbel: 16
          </p>
        </div>
      </div>

      <p className="text-[10px] font-mono tracking-widest uppercase bg-black border border-white/10 p-3 rounded-lg text-center max-w-sm mt-12 text-zinc-500 relative z-10">
        Deploy a scan to begin empirical measurement.
      </p>
    </div>
  );
}
