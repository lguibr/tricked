import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { Activity, Hammer, Lock } from 'lucide-react';
import { NetworkBackground } from '@/components/NetworkBackground';
import { MissionControl } from '@/pages/MissionControl';
import { Forge } from '@/pages/Forge';
import { Vault } from '@/pages/Vault';

function Navigation() {
  return (
    <nav className="w-full flex items-center justify-between p-4 border-b border-white/10 bg-background/30 backdrop-blur top-0 sticky z-50 shadow-sm">
      <div className="font-black text-2xl tracking-tighter flex items-center text-foreground drop-shadow-md">
        <Link to="/" className="flex items-center hover:opacity-90 transition-opacity">
          <img src="/logo.png" alt="Tricked AI Logo" className="h-10 w-auto mr-3 drop-shadow-[0_0_8px_#00fbfb]" />
        </Link>
      </div>
      <div className="flex gap-8 font-bold text-sm tracking-wide">
        <Link
          to="/"
          className="flex items-center gap-2 text-muted-foreground hover:text-primary hover:drop-shadow-[0_0_8px_#00fbfb] transition-all"
        >
          <Activity className="w-5 h-5" /> CONTROL
        </Link>
        <Link
          to="/forge"
          className="flex items-center gap-2 text-muted-foreground hover:text-primary hover:drop-shadow-[0_0_8px_#00fbfb] transition-all"
        >
          <Hammer className="w-5 h-5" /> FORGE
        </Link>
        <Link
          to="/vault"
          className="flex items-center gap-2 text-muted-foreground hover:text-primary hover:drop-shadow-[0_0_8px_#00fbfb] transition-all"
        >
          <Lock className="w-5 h-5" /> VAULT
        </Link>
      </div>
    </nav>
  );
}

function AnimatedRoutes() {
  const location = useLocation();

  return (
    <AnimatePresence mode="wait">
      <Routes location={location} key={location.pathname}>
        <Route
          path="/"
          element={
            <MotionWrapper>
              <MissionControl />
            </MotionWrapper>
          }
        />
        <Route
          path="/forge"
          element={
            <MotionWrapper>
              <Forge />
            </MotionWrapper>
          }
        />
        <Route
          path="/vault"
          element={
            <MotionWrapper>
              <Vault />
            </MotionWrapper>
          }
        />
      </Routes>
    </AnimatePresence>
  );
}

export default function App() {
  return (
    <Router>
      <div className="min-h-screen bg-[#0e131f] text-white font-sans antialiased selection:bg-primary/30 relative">
        <NetworkBackground />
        <Navigation />
        <AnimatedRoutes />
      </div>
    </Router>
  );
}

function MotionWrapper({ children }: { children: React.ReactNode }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10, filter: 'blur(4px)' }}
      animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
      exit={{ opacity: 0, y: -10, filter: 'blur(4px)' }}
      transition={{ duration: 0.4, ease: 'easeOut' }}
      className="w-full flex-grow"
    >
      {children}
    </motion.div>
  );
}
