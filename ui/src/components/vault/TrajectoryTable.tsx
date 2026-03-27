import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Button } from '@/components/ui/button';
import { Search } from 'lucide-react';
import { useEngineStore } from '@/store/useEngineStore';


export function TrajectoryTable() {
  const loadReplay = useEngineStore((s) => s.loadReplay);
  const trainingInfo = useEngineStore((s) => s.trainingInfo);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const games = trainingInfo?.top_games || [];

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
          {/* eslint-disable-next-line @typescript-eslint/no-explicit-any */}
          {games.map((game: any) => (
            <TableRow key={game.global_start_idx} className="border-border/40 hover:bg-primary/5 transition-colors">
              <TableCell className="font-mono text-muted-foreground">#{game.global_start_idx}</TableCell>
              <TableCell className="font-bold text-white">{game.score?.toLocaleString()}</TableCell>
              <TableCell className="text-muted-foreground">{game.length}</TableCell>
              <TableCell>
                <span
                  className={`px-2 py-1 rounded text-xs font-semibold ${game.difficulty >= 2 ? 'bg-red-500/20 text-red-400' : 'bg-yellow-500/20 text-yellow-400'}`}
                >
                  {game.difficulty >= 2 ? 'High' : 'Medium'}
                </span>
              </TableCell>
              <TableCell className="text-right">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => loadReplay(game.global_start_idx)}
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
