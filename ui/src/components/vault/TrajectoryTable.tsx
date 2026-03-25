import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Button } from '@/components/ui/button';
import { Search } from 'lucide-react';
import { useEngineStore } from '@/store/useEngineStore';

const MOCK_GAMES = [
  { id: 1042, score: 85200, steps: 142, difficulty: 'High', date: '2026-03-24' },
  { id: 1089, score: 71400, steps: 118, difficulty: 'Medium', date: '2026-03-24' },
  { id: 994, score: 12050, steps: 24, difficulty: 'High', date: '2026-03-23' }, // A death trap example
];

export function TrajectoryTable() {
  const loadReplay = useEngineStore((s) => s.loadReplay);

  return (
    <div className="w-full bg-background border border-border/50 rounded-xl overflow-hidden shadow-lg">
      <div className="p-4 border-b border-border/50 bg-muted/20 flex justify-between items-center">
        <h3 className="font-bold text-white flex items-center gap-2">
          <Search className="w-4 h-4 text-primary" />
          Redis Trajectory Cache
        </h3>
      </div>
      <Table>
        <TableHeader>
          <TableRow className="hover:bg-transparent">
            <TableHead className="w-[100px]">ID</TableHead>
            <TableHead>Score</TableHead>
            <TableHead>Steps</TableHead>
            <TableHead>Difficulty</TableHead>
            <TableHead className="text-right">Action</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {MOCK_GAMES.map((game) => (
            <TableRow key={game.id} className="border-border/40 hover:bg-primary/5 transition-colors">
              <TableCell className="font-mono text-muted-foreground">#{game.id}</TableCell>
              <TableCell className="font-bold text-white">{game.score.toLocaleString()}</TableCell>
              <TableCell className="text-muted-foreground">{game.steps}</TableCell>
              <TableCell>
                <span
                  className={`px-2 py-1 rounded text-xs font-semibold ${game.difficulty === 'High' ? 'bg-red-500/20 text-red-400' : 'bg-yellow-500/20 text-yellow-400'}`}
                >
                  {game.difficulty}
                </span>
              </TableCell>
              <TableCell className="text-right">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => loadReplay(game.id)}
                  className="hover:text-primary hover:bg-primary/10"
                >
                  Load
                </Button>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
