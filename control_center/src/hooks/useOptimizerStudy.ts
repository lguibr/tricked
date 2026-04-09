import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";

export type { TrialData as Trial } from "../bindings/TrialData";
export type { StudyData } from "../bindings/StudyData";
import { type StudyData } from "../bindings/StudyData";

export function useOptimizerStudy(selectedRunId: string | null) {
  const [study, setStudy] = useState<StudyData | null>(null);

  useEffect(() => {
    let active = true;
    const fetchStudy = async () => {
      try {
        if (!selectedRunId) return;
        const data = await invoke<any>("get_tuning_study", {
          studyId: selectedRunId,
        });
        if (active) {
          if (Array.isArray(data)) {
            setStudy({ trials: data, importance: {} });
          } else if (data.trials) {
            setStudy(data);
          }
        }
      } catch (e) {
        console.error("Failed to fetch optimizer study:", e);
      }
    };
    if (selectedRunId) {
      fetchStudy();
    } else {
      setStudy(null);
    }
    const interval = setInterval(() => {
      if (selectedRunId) fetchStudy();
    }, 2000);
    return () => {
      active = false;
      clearInterval(interval);
    };
  }, [selectedRunId]);

  return study;
}
