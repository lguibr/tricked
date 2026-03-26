import { describe, it, expect, vi } from 'vitest';
import { render } from '@testing-library/react';
import { BoardVisualizer } from './BoardVisualizer';
import { useEngineStore } from '../store/useEngineStore';
import React from 'react';

vi.mock('../store/useEngineStore', () => ({
    useEngineStore: vi.fn(),
}));

describe('BoardVisualizer', () => {
    it('renders hole logits with pulse animation when enabled', () => {
        vi.mocked(useEngineStore).mockReturnValue({
            board_state: '0',
            hole_logits: Array(96).fill(0.8),
        });

        const { container } = render(<BoardVisualizer showHoles={true} />);
        const polygons = container.querySelectorAll('polygon');

        expect(polygons.length).toBe(96);
        expect(polygons[0].className.baseVal).toContain('animate-pulse');
        expect(polygons[0].className.baseVal).toContain('fill-red-500');
    });

    it('maps policy probability to opacity correctly', () => {
        const probs = Array(96).fill(0);
        probs[0] = 0.75;
        vi.mocked(useEngineStore).mockReturnValue({
            board_state: '0',
            policy_probs: probs,
        });

        const { container } = render(<BoardVisualizer showPolicy={true} />);
        const polygons = container.querySelectorAll('polygon');

        expect(polygons[0].className.baseVal).toContain('fill-cyan-500');
        expect(polygons[0].style.opacity).toBe('0.75');

        expect(polygons[1].className.baseVal).not.toContain('fill-cyan-500');
    });
});
