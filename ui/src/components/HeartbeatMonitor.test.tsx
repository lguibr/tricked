import { render, screen } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { HeartbeatMonitor } from './HeartbeatMonitor';

// Mock the Zustand store so it runs without actual WebSocket states
vi.mock('@/store/useEngineStore', () => ({
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  useEngineStore: (selector: any) =>
    selector({
      trainingInfo: { games_per_second: 42.5, gps_history: [] },
      isTraining: true,
    }),
}));

// Mock Recharts ResponsiveContainer to prevent JSDOM resize observer errors
vi.mock('recharts', async () => {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const OriginalModule = await vi.importActual<any>('recharts');
  return {
    ...OriginalModule,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    ResponsiveContainer: ({ children }: any) => <div style={{ width: 800, height: 600 }}>{children}</div>,
  };
});

describe('HeartbeatMonitor', () => {
  it('renders the current GPS correctly when training', () => {
    render(<HeartbeatMonitor />);

    // Validate engine vitals display
    expect(screen.getByText('42.5')).toBeInTheDocument();
    expect(screen.getByText(/Engine Vitals/i)).toBeInTheDocument();
  });
});
