import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";

export interface Trial {
  number: number;
  state: string;
  value: number | number[] | null;
  params: Record<string, string | number>;
  intermediate_values: Record<string, number>;
}

export interface StudyData {
  trials: Trial[];
  importance: Record<string, number>;
}

export function useOptunaStudy(selectedRunId: string | null) {
  const [study, setStudy] = useState<StudyData | null>(null);

  useEffect(() => {
    let active = true;
    const fetchStudy = async () => {
      try {
        if (!selectedRunId) return;
        const jsonStr = await invoke<string>("get_tuning_study", {
          studyId: selectedRunId,
        });
        const data = JSON.parse(jsonStr);
        if (active) {
          if (Array.isArray(data)) {
            setStudy({ trials: data, importance: {} });
          } else if (data.trials) {
            setStudy(data);
          }
        }
      } catch (e) {
        console.error("Failed to fetch optuna study:", e);
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
