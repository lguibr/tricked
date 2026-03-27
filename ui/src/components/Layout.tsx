import type { ReactNode } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Activity, Hammer, Lock } from 'lucide-react';
import { NetworkBackground } from '@/components/NetworkBackground';
import { motion } from 'framer-motion';

function NavLink({ to, icon: Icon, label }: { to: string; icon: any; label: string }) {
    const location = useLocation();
    const isActive = location.pathname === to;

    return (
        <Link
            to={to}
            className={`flex items-center gap-2 transition-all duration-300 ${isActive
                ? 'text-primary drop-shadow-[0_0_8px_rgba(16,185,129,0.8)]'
                : 'text-muted-foreground hover:text-primary'
                }`}
        >
            <Icon className="w-4 h-4" />
            {label}
        </Link>
    );
}

export function Layout({ children }: { children: ReactNode }) {
    return (
        <div className="min-h-screen bg-[#050505] text-white font-sans antialiased selection:bg-primary/30 relative flex flex-col items-center overflow-x-hidden">
            <NetworkBackground />
            <nav className="w-full flex justify-center border-b border-white/5 bg-black/40 backdrop-blur-md sticky top-0 z-50">
                <div className="w-full max-w-7xl flex items-center justify-between p-6">
                    <div className="font-black text-2xl tracking-tighter flex items-center drop-shadow-md">
                        <Link to="/" className="flex items-center hover:opacity-90 transition-opacity">
                            <img src="/logo.png" alt="Tricked AI Logo" className="h-8 w-auto mr-3" />
                        </Link>
                    </div>
                    <div className="flex gap-8 font-bold text-xs uppercase tracking-widest">
                        <NavLink to="/" icon={Activity} label="Control" />
                        <NavLink to="/forge" icon={Hammer} label="Forge" />
                        <NavLink to="/vault" icon={Lock} label="Vault" />
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
