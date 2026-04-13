import {
  listRunsApiRunsGet,
  startRunApiRunsStartPost,
  stopRunApiRunsStopPost,
  renameRunApiRunsRenamePost,
  deleteRunApiRunsDeletePost,
  flushRunApiRunsFlushPost,
  createRunApiRunsCreatePost,
  saveConfigApiRunsSaveConfigPost,
  startStudyApiStudiesStartPost,
} from "@/client";
import { client } from "@/client/client.gen";

// Configure base URL so requests escape the Vite dev server
client.setConfig({ baseUrl: "http://127.0.0.1:8000" });

// A bridge to route legacy invocations natively to our OpenAPI Python backend!
export const invoke = async <T>(
  cmd: string,
  args: Record<string, any> = {},
): Promise<T> => {
  try {
    switch (cmd) {
      case "list_runs":
        return (await listRunsApiRunsGet()).data as unknown as T;
      case "start_run":
        return (await startRunApiRunsStartPost({ body: { id: args.id } }))
          .data as unknown as T;
      case "stop_run":
        return (
          await stopRunApiRunsStopPost({
            body: { id: args.id, force: args.force || false },
          })
        ).data as unknown as T;
      case "rename_run":
        return (
          await renameRunApiRunsRenamePost({
            body: { id: args.id, newName: args.newName },
          })
        ).data as unknown as T;
      case "delete_run":
        return (await deleteRunApiRunsDeletePost({ body: { id: args.id } }))
          .data as unknown as T;
      case "flush_run":
        return (await flushRunApiRunsFlushPost({ body: { id: args.id } }))
          .data as unknown as T;
      case "create_run":
        return (
          await createRunApiRunsCreatePost({
            body: {
              name: args.name,
              type: args.type,
              preset: args.preset || "default",
            },
          })
        ).data as unknown as T;
      case "save_config":
        return (
          await saveConfigApiRunsSaveConfigPost({
            body: { id: args.id, config: args.config },
          })
        ).data as unknown as T;
      case "start_study":
        return (
          await startStudyApiStudiesStartPost({
            body: { id: args.id, baseConfig: args.baseConfig } as any,
          })
        ).data as unknown as T;

      // Mocks for missing features in backend decapitation
      case "list_checkpoints": {
        const res = await fetch(
          `http://127.0.0.1:8000/api/runs/${args.id}/checkpoints`,
        );
        if (!res.ok) throw new Error("Failed to fetch checkpoints");
        return (await res.json()) as unknown as T;
      }
      case "start_evaluation":
        return true as unknown as T;
      case "stop_evaluation":
        return true as unknown as T;
      case "playground_start_game":
        const r1 = await fetch("http://127.0.0.1:8000/api/playground/start", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            difficulty: args.difficulty,
            clutter: args.clutter,
          }),
        });
        return (await r1.json()) as unknown as T;
      case "playground_apply_move":
        const r2 = await fetch(
          "http://127.0.0.1:8000/api/playground/apply_move",
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(args),
          },
        );
        return (await r2.json()) as unknown as T;
      case "evaluation_step":
        const r3 = await fetch("http://127.0.0.1:8000/api/evaluation/step", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(args),
        });
        return (await r3.json()) as unknown as T;
      case "playground_commit_to_vault": {
        const r4 = await fetch(
          "http://127.0.0.1:8000/api/playground/commit_to_vault",
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(args),
          },
        );
        return (await r4.json()) as unknown as T;
      }
      case "get_vault_games": {
        const r5 = await fetch("http://127.0.0.1:8000/api/vault/global");
        if (!r5.ok) throw new Error("Failed to fetch vault games");
        return (await r5.json()) as unknown as T;
      }
      case "flush_global_vault": {
        const res = await fetch("http://127.0.0.1:8000/api/vault/flush", { method: "POST" });
        if (!res.ok) throw new Error("Failed to flush vault");
        return (await res.json()) as unknown as T;
      }
      case "empty_all_vaults": {
        const res = await fetch("http://127.0.0.1:8000/api/vault/empty", { method: "POST" });
        if (!res.ok) throw new Error("Failed to empty vault");
        return (await res.json()) as unknown as T;
      }
      case "remove_vault_game": {
        const res = await fetch("http://127.0.0.1:8000/api/vault/remove", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(args)
        });
        if (!res.ok) throw new Error("Failed to remove game");
        return (await res.json()) as unknown as T;
      }
      case "get_study_status":
        return false as unknown as T;
      case "get_active_study":
        return false as unknown as T;
      case "stop_study":
        return null as unknown as T;
      case "flush_study":
        return null as unknown as T;

      default:
        console.warn(
          `[API Bridge] Attempted to invoke unimplemented command: ${cmd}`,
          args,
        );
        return null as unknown as T;
    }
  } catch (error) {
    console.error(`[API Bridge] Error invoking ${cmd}:`, error);
    throw error;
  }
};
