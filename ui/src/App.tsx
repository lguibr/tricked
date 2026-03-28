import { useEffect } from 'react';
import { BrowserRouter as Router } from 'react-router-dom';
import { Forge } from '@/pages/Forge';
import { Vault } from '@/pages/Vault';
import { Layout } from '@/components/Layout';
import { useEngineStore } from '@/store/useEngineStore';

export default function App() {
  const startPolling = useEngineStore((s) => s.startPolling);

  useEffect(() => {
    startPolling();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <Router>
      <Layout>
        <div className="flex flex-col gap-24 w-full divider-y max-w-7xl mx-auto px-4 md:px-8">
          <Forge />
          <Vault />
        </div>
      </Layout>
    </Router>
  );
}
