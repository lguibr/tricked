import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { ExecutionTab } from './components/ExecutionTab';

// Mock matchMedia for interactive Shadcn components
Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: vi.fn().mockImplementation(query => ({
        matches: false,
        media: query,
        onchange: null,
        addListener: vi.fn(),
        removeListener: vi.fn(),
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        dispatchEvent: vi.fn(),
    })),
});

const { mockInvoke } = vi.hoisted(() => {
    return {
        mockInvoke: vi.fn(async (cmd: string, args?: any) => {
            if (cmd === 'list_runs') return [];
            if (cmd === 'create_run') return { id: 'test-run', name: args.name, status: 'WAITING', type: args.type, config: '{}' };
            if (cmd === 'get_tuning_study') return "[]";
            return [];
        })
    };
});

vi.mock('@tauri-apps/api/core', () => ({
    invoke: (cmd: string, args: any) => mockInvoke(cmd, args)
}));

vi.mock('@tauri-apps/api/event', () => ({
    listen: vi.fn(() => Promise.resolve(() => { }))
}));

vi.mock('echarts-for-react', () => ({
    default: () => <div data-testid="echarts-mock" />
}));

// ResizeObserver mock
global.ResizeObserver = class ResizeObserver {
    observe() { }
    unobserve() { }
    disconnect() { }
};

describe('ExecutionTab UI Interactions Phase', () => {
    beforeEach(() => {
        mockInvoke.mockClear();
    });

    it('requests the list of runs on mount', async () => {
        render(<ExecutionTab />);
        await waitFor(() => {
            expect(mockInvoke).toHaveBeenCalledWith('list_runs', undefined);
        });
    });

    it('creates a new run and passes payload config to tauri invoke', async () => {
        render(<ExecutionTab />);

        // The trigger is an icon button, fallback to querying by svg class if needed, or index
        const buttons = screen.getAllByRole('button');
        // Fallback simulate clicking the "plus" icon trigger
        fireEvent.click(buttons[0]);

        await waitFor(() => {
            expect(screen.getByText('Create Run/Experiment')).toBeInTheDocument();
        });

        // Input Run Name
        const nameInput = screen.getByLabelText('Name');
        fireEvent.change(nameInput, { target: { value: 'integration_test_run' } });

        // Select Preset
        const selectPreset = screen.getByLabelText(/Base Config \/ Hydra Payload/i);
        fireEvent.change(selectPreset, { target: { value: 'big' } });

        // Submit Form
        const submitBtn = screen.getByRole('button', { name: 'Create' });
        fireEvent.click(submitBtn);

        await waitFor(() => {
            expect(mockInvoke).toHaveBeenCalledWith('create_run', {
                name: 'integration_test_run',
                type: 'SINGLE',
                preset: 'big'
            });
        });
    });
});
