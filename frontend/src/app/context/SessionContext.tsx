import React, { createContext, useContext, useState, useCallback } from "react";
import {
  Shot,
  Drill,
  VideoAnalysisResult,
  uploadVideo,
  generateMockShotDetail,
  USE_MOCK,
} from "../../api";

// ── State ─────────────────────────────────────────────────────────────────────

export type UploadStatus = "idle" | "uploading" | "done" | "error";

interface AnalysisState {
  uploadStatus: UploadStatus;
  analysisResult: VideoAnalysisResult | null;
  shots: Shot[];
  drills: Drill[];
  errorMessage: string | null;
}

interface AnalysisContextValue extends AnalysisState {
  startUpload: (file: File) => Promise<void>;
  clearAnalysis: () => void;
}

const SessionContext = createContext<AnalysisContextValue | null>(null);

// ── Provider ──────────────────────────────────────────────────────────────────

export function SessionProvider({ children }: { children: React.ReactNode }) {
  const [state, setState] = useState<AnalysisState>({
    uploadStatus:   "idle",
    analysisResult: null,
    shots:          [],
    drills:         [],
    errorMessage:   null,
  });

  const startUpload = useCallback(async (file: File) => {
    setState((prev) => ({
      ...prev,
      uploadStatus: "uploading",
      errorMessage: null,
    }));

    try {
      const result = await uploadVideo(file);
      setState({
        uploadStatus:   "done",
        analysisResult: result,
        shots:          result.shots,
        drills:         result.drills,
        errorMessage:   null,
      });
    } catch (err: any) {
      setState((prev) => ({
        ...prev,
        uploadStatus: "error",
        errorMessage: err?.message ?? "Analysis failed. Please try again.",
      }));
    }
  }, []);

  const clearAnalysis = useCallback(() => {
    setState({
      uploadStatus:   "idle",
      analysisResult: null,
      shots:          [],
      drills:         [],
      errorMessage:   null,
    });
  }, []);

  return (
    <SessionContext.Provider value={{ ...state, startUpload, clearAnalysis }}>
      {children}
    </SessionContext.Provider>
  );
}

export function useSession() {
  const ctx = useContext(SessionContext);
  if (!ctx) throw new Error("useSession must be used inside SessionProvider");
  return ctx;
}
