import type { ReactNode } from 'react';
import { Link } from 'react-router-dom';
import { NetworkBackground } from '@/components/NetworkBackground';
import { useEngineStore } from '@/store/useEngineStore';
import { motion } from 'framer-motion';

export function Layout({ children }: { children: ReactNode }) {
    const trainingInfo = useEngineStore((state) => state.trainingInfo);

    return (
        <div className="min-h-screen bg-[#050505] text-white font-sans antialiased selection:bg-primary/30 relative flex flex-col items-center overflow-x-hidden">
            <NetworkBackground />
            <nav className="w-full flex justify-center border-b border-white/5 bg-black/40 backdrop-blur-md sticky top-0 z-50">
                <div className="w-full max-w-7xl flex items-center justify-between p-6">
                    <div className="font-black text-2xl tracking-tighter flex items-center drop-shadow-md">
                        <Link to="/" className="flex items-center hover:opacity-90 transition-opacity">
                            <img src="/logo.png" alt="Tricked AI Logo" className="h-14 w-auto mr-3" />
                        </Link>
                        {trainingInfo?.exp_name && (
                            <div className="flex items-center ml-2 px-3 py-1 rounded-full bg-primary/10 border border-primary/30 text-primary text-xs font-mono tracking-normal">
                                <span className="w-2 h-2 rounded-full bg-primary animate-pulse mr-2" />
                                {trainingInfo.exp_name}
                            </div>
                        )}
                    </div>
                    <div className="flex gap-8 font-bold text-xs uppercase tracking-widest">
                    </div>
                </div>
            </nav>
            <main className="w-full max-w-7xl flex-grow flex flex-col py-10 px-6 z-10 relative">
                <motion.div
                    initial={{ opacity: 0, y: 10, filter: 'blur(4px)' }}
                    animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
                    exit={{ opacity: 0, y: -10, filter: 'blur(4px)' }}
                    transition={{ duration: 0.4, ease: 'easeOut' }}
                    className="w-full flex-grow flex flex-col"
                >
                    {children}
                </motion.div>
            </main>
        </div>
    );
}
