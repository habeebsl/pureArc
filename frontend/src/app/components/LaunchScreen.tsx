import { checkHealth, USE_MOCK } from "../../api";
import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router";
import { motion, AnimatePresence } from "motion/react";
import {
  AlertTriangle,
  CheckCircle2,
  ChevronRight,
  RefreshCw,
  Upload,
  Video,
  Wifi,
  WifiOff,
} from "lucide-react";
import { useSession } from "../context/SessionContext";

type HealthState = "checking" | "online" | "offline";

export function LaunchScreen() {
  const navigate                        = useNavigate();
  const { uploadStatus, errorMessage, startUpload } = useSession();
  const [health, setHealth]             = useState<HealthState>("checking");
  const [dragOver, setDragOver]         = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const fileInputRef                    = useRef<HTMLInputElement>(null);

  const runHealthCheck = async () => {
    setHealth("checking");
    try {
      const res = await checkHealth();
      setHealth(res.status === "offline" ? "offline" : "online");
    } catch {
      setHealth("offline");
    }
  };

  useEffect(() => { runHealthCheck(); }, []);

  // Navigate to results when analysis is complete
  useEffect(() => {
    if (uploadStatus === "done") navigate("/summary");
  }, [uploadStatus, navigate]);

  const handleFile = (file: File) => {
    if (!file.type.startsWith("video/") && !file.name.match(/\.(mp4|mov|avi|mkv|webm)$/i)) {
      return;
    }
    setSelectedFile(file);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;
    await startUpload(selectedFile);
  };

  const isLoading = uploadStatus === "uploading";
  const canStart  = health !== "checking" && !isLoading && selectedFile !== null;

  return (
    <div
      className="min-h-screen flex flex-col items-center justify-center relative overflow-hidden"
      style={{ background: "#0A0A0A", fontFamily: "'Inter', sans-serif" }}
    >
      {/* Subtle court-line background */}
      <div className="absolute inset-0 pointer-events-none select-none overflow-hidden opacity-[0.03]">
        <svg width="100%" height="100%" xmlns="http://www.w3.org/2000/svg">
          <circle cx="50%" cy="50%" r="180" fill="none" stroke="#E8602C" strokeWidth="2" />
          <circle cx="50%" cy="50%" r="60"  fill="none" stroke="#E8602C" strokeWidth="2" />
          <line x1="0" y1="50%" x2="100%" y2="50%" stroke="#E8602C" strokeWidth="1.5" />
          <rect x="35%" y="20%" width="30%" height="30%" fill="none" stroke="#E8602C" strokeWidth="1.5" />
          <rect x="35%" y="50%" width="30%" height="30%" fill="none" stroke="#E8602C" strokeWidth="1.5" />
        </svg>
      </div>

      <motion.div
        initial={{ opacity: 0, y: 24 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, ease: "easeOut" }}
        className="w-full max-w-md px-6 flex flex-col items-center gap-8"
      >
        {/* Logo */}
        <div className="flex flex-col items-center gap-3">
          <div className="w-16 h-16 rounded-2xl flex items-center justify-center" style={{ background: "#E8602C" }}>
            <svg width="34" height="34" viewBox="0 0 34 34" fill="none">
              <circle cx="17" cy="17" r="13" stroke="white" strokeWidth="2" />
              <path d="M4 17 Q17 4 30 17"  stroke="white" strokeWidth="2" fill="none" />
              <path d="M4 17 Q17 30 30 17" stroke="white" strokeWidth="2" fill="none" />
              <line x1="17" y1="4" x2="17" y2="30" stroke="white" strokeWidth="2" />
            </svg>
          </div>
          <div className="text-center">
            <p className="tracking-[0.2em] uppercase text-xs mb-1"
               style={{ color: "#E8602C", fontFamily: "'Barlow Condensed', sans-serif", fontWeight: 600 }}>
              Shot Coaching
            </p>
            <h1 className="tracking-tight"
                style={{ fontFamily: "'Barlow Condensed', sans-serif", fontWeight: 700, fontSize: "2.8rem", color: "#F2EDE8", lineHeight: 1 }}>
              PUREARC
            </h1>
          </div>
        </div>

        {/* Backend status */}
        <div className="w-full rounded-2xl p-5 flex flex-col gap-4" style={{ background: "#141414", border: "1px solid #252525" }}>
          <AnimatePresence mode="wait">
            {health === "checking" && (
              <motion.div key="checking" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex items-center gap-3">
                <motion.div animate={{ rotate: 360 }} transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                  className="w-8 h-8 rounded-full border-2 border-t-transparent"
                  style={{ borderColor: "#2C2C2C", borderTopColor: "#E8602C" }} />
                <div>
                  <p className="text-sm" style={{ color: "#F2EDE8" }}>Checking service…</p>
                  <p className="text-xs mt-0.5" style={{ color: "#6E6862" }}>Connecting to PureArc backend</p>
                </div>
              </motion.div>
            )}
            {health === "online" && (
              <motion.div key="online" initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0 }} className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0" style={{ background: "#1A2E23" }}>
                  <Wifi className="w-4 h-4" style={{ color: "#3DB87A" }} />
                </div>
                <div>
                  <p className="text-sm" style={{ color: "#F2EDE8" }}>
                    Ready to Analyze
                    {USE_MOCK && <span className="ml-2 text-xs px-1.5 py-0.5 rounded" style={{ background: "#2A1F0F", color: "#E8602C" }}>DEMO</span>}
                  </p>
                  <p className="text-xs mt-0.5" style={{ color: "#3DB87A" }}>
                    {USE_MOCK ? "Demo mode — video analysis simulated" : "Backend online · All systems operational"}
                  </p>
                </div>
              </motion.div>
            )}
            {health === "offline" && (
              <motion.div key="offline" initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0 }} className="flex flex-col gap-3">
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0" style={{ background: "#2A1818" }}>
                    <WifiOff className="w-4 h-4" style={{ color: "#D95C5C" }} />
                  </div>
                  <div>
                    <p className="text-sm" style={{ color: "#F2EDE8" }}>Service Offline</p>
                    <p className="text-xs mt-0.5" style={{ color: "#D95C5C" }}>Cannot reach PureArc backend</p>
                  </div>
                </div>
                <button onClick={runHealthCheck}
                  className="flex items-center justify-center gap-2 rounded-xl py-2.5 text-sm transition-colors"
                  style={{ background: "#1C1C1C", color: "#F2EDE8", border: "1px solid #2C2C2C" }}>
                  <RefreshCw className="w-4 h-4" /> Retry Connection
                </button>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Drop zone */}
        <div className="w-full">
          <input ref={fileInputRef} type="file" accept="video/*,.mp4,.mov,.avi,.mkv,.webm"
            className="hidden" onChange={handleInputChange} />

          <motion.div
            onClick={() => !isLoading && fileInputRef.current?.click()}
            onDragOver={(e) => { e.preventDefault(); if (!isLoading) setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
            animate={{ borderColor: dragOver ? "#E8602C" : selectedFile ? "#3DB87A" : "#2C2C2C" }}
            transition={{ duration: 0.15 }}
            className="w-full rounded-2xl p-7 flex flex-col items-center gap-3 cursor-pointer transition-colors"
            style={{ background: dragOver ? "#161006" : selectedFile ? "#0C1A10" : "#141414",
                     border: `2px dashed ${dragOver ? "#E8602C" : selectedFile ? "#1E4030" : "#2C2C2C"}` }}
          >
            {selectedFile ? (
              <>
                <div className="w-11 h-11 rounded-xl flex items-center justify-center" style={{ background: "#122416" }}>
                  <Video className="w-5 h-5" style={{ color: "#3DB87A" }} />
                </div>
                <div className="text-center">
                  <p className="text-sm font-medium" style={{ color: "#F2EDE8" }}>{selectedFile.name}</p>
                  <p className="text-xs mt-0.5" style={{ color: "#3DB87A" }}>
                    {(selectedFile.size / 1024 / 1024).toFixed(1)} MB · Click to change
                  </p>
                </div>
              </>
            ) : (
              <>
                <div className="w-11 h-11 rounded-xl flex items-center justify-center" style={{ background: "#1E1E1E" }}>
                  <Upload className="w-5 h-5" style={{ color: "#6E6862" }} />
                </div>
                <div className="text-center">
                  <p className="text-sm font-medium" style={{ color: "#C8C2BC" }}>
                    Drop your shooting video here
                  </p>
                  <p className="text-xs mt-1" style={{ color: "#4A4540" }}>
                    MP4, MOV, AVI · 10–60 seconds works best
                  </p>
                </div>
              </>
            )}
          </motion.div>
        </div>

        {/* Processing status */}
        <AnimatePresence>
          {isLoading && (
            <motion.div key="processing" initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
              className="w-full rounded-xl px-4 py-3 flex items-center gap-3"
              style={{ background: "#141008", border: "1px solid #302010" }}>
              <motion.div animate={{ rotate: 360 }} transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                className="w-5 h-5 rounded-full border-2 border-t-transparent flex-shrink-0"
                style={{ borderColor: "#3A2A10", borderTopColor: "#E8602C" }} />
              <div>
                <p className="text-sm" style={{ color: "#F2EDE8" }}>Analyzing shots…</p>
                <p className="text-xs mt-0.5" style={{ color: "#8A7050" }}>
                  This may take a minute on CPU · Detecting every shot attempt
                </p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Error */}
        {uploadStatus === "error" && errorMessage && (
          <motion.div key="error" initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }}
            className="w-full rounded-xl px-4 py-3 flex items-start gap-2.5"
            style={{ background: "#2A1818", border: "1px solid #3D2020" }}>
            <AlertTriangle className="w-4 h-4 flex-shrink-0 mt-0.5" style={{ color: "#D95C5C" }} />
            <p className="text-sm" style={{ color: "#D95C5C" }}>{errorMessage}</p>
          </motion.div>
        )}

        {/* Analyze button */}
        <motion.button
          onClick={handleAnalyze}
          disabled={!canStart}
          whileHover={canStart ? { scale: 1.02 } : {}}
          whileTap={canStart ? { scale: 0.98 } : {}}
          className="w-full flex items-center justify-center gap-3 rounded-2xl py-4 transition-all"
          style={{
            background: canStart ? "#E8602C" : "#2A2A2A",
            color: canStart ? "#fff" : "#4A4A4A",
            fontFamily: "'Barlow Condensed', sans-serif",
            fontWeight: 700,
            fontSize: "1.15rem",
            letterSpacing: "0.08em",
            cursor: canStart ? "pointer" : "not-allowed",
          }}
        >
          {isLoading ? (
            <>
              <motion.div animate={{ rotate: 360 }} transition={{ duration: 0.8, repeat: Infinity, ease: "linear" }}
                className="w-5 h-5 rounded-full border-2 border-t-transparent"
                style={{ borderColor: "#6A6A6A", borderTopColor: "#AAA" }} />
              ANALYZING…
            </>
          ) : (
            <>
              <Video className="w-5 h-5" />
              ANALYZE VIDEO
              <ChevronRight className="w-5 h-5" />
            </>
          )}
        </motion.button>

        <p className="text-xs text-center" style={{ color: "#4A4540" }}>
          No account required · Results shown instantly after processing
        </p>
      </motion.div>
    </div>
  );
}
