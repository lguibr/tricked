import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
import { AnimatePresence } from 'framer-motion';
import { MissionControl } from '@/pages/MissionControl';
import { Forge } from '@/pages/Forge';
import { Vault } from '@/pages/Vault';
import { Layout } from '@/components/Layout';

function AnimatedRoutes() {
  const location = useLocation();

  return (
    <AnimatePresence mode="wait">
      <Routes location={location} key={location.pathname}>
        <Route path="/" element={<MissionControl />} />
        <Route path="/forge" element={<Forge />} />
        <Route path="/vault" element={<Vault />} />
      </Routes>
    </AnimatePresence>
  );
}

export default function App() {
  return (
    <Router>
      <Layout>
        <AnimatedRoutes />
      </Layout>
    </Router>
  );
}
