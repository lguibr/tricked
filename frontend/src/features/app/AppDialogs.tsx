import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
  DialogDescription,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Field, FieldLabel, FieldSet } from "@/components/ui/field";
import { Button } from "@/components/ui/button";
import { useAppStore } from "@/store/useAppStore";
import { useAppActions } from "@/features/app/useAppActions";

export function AppDialogs() {
  const runToRename = useAppStore((state) => state.runToRename);
  const setRunToRename = useAppStore((state) => state.setRunToRename);
  const newName = useAppStore((state) => state.newName);
  const setNewName = useAppStore((state) => state.setNewName);
  const runToDelete = useAppStore((state) => state.runToDelete);
  const setRunToDelete = useAppStore((state) => state.setRunToDelete);
  const runToFlush = useAppStore((state) => state.runToFlush);
  const setRunToFlush = useAppStore((state) => state.setRunToFlush);

  const { handleRename, handleDelete, handleFlush } = useAppActions();

  return (
    <>
      <Dialog
        open={!!runToRename}
        onOpenChange={(open) => !open && setRunToRename(null)}
      >
        <DialogContent className="sm:max-w-[350px] border-border/20 bg-[#0a0a0a]">
          <DialogHeader>
            <DialogTitle>Rename Run</DialogTitle>
          </DialogHeader>
          <FieldSet>
            <Field>
              <FieldLabel>New Name</FieldLabel>
              <Input
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                className="bg-zinc-900 border-border/30"
              />
            </Field>
          </FieldSet>
          <DialogFooter>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setRunToRename(null)}
            >
              Cancel
            </Button>
            <Button size="sm" onClick={handleRename}>
              Save
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog
        open={!!runToDelete}
        onOpenChange={(open) => !open && setRunToDelete(null)}
      >
        <DialogContent className="sm:max-w-[400px] border-border/20 bg-[#0a0a0a]">
          <DialogHeader>
            <DialogTitle>Delete Config</DialogTitle>
            <DialogDescription>
              This deletes the configuration entirely. Cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setRunToDelete(null)}
            >
              Cancel
            </Button>
            <Button variant="destructive" size="sm" onClick={handleDelete}>
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog
        open={!!runToFlush}
        onOpenChange={(open) => !open && setRunToFlush(null)}
      >
        <DialogContent className="sm:max-w-[400px] border-border/20 bg-[#0a0a0a]">
          <DialogHeader>
            <DialogTitle>Flush Data</DialogTitle>
            <DialogDescription>
              Clears all metrics, checkpoints, and logs for this run but keeps
              the config.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setRunToFlush(null)}
            >
              Cancel
            </Button>
            <Button variant="destructive" size="sm" onClick={handleFlush}>
              Flush
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
