import { useState, useRef } from "react";
import { Mic, Upload, FileAudio, Loader2, AlertCircle, CheckCircle2, Bot, User, Info } from "lucide-react";
import heroBg from "@/assets/hero-bg.jpg";

const API_URL = "https://prem678-ai-buildathon.hf.space/api/voice-detection";
const API_KEY = "sk_test_123456789";
const LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"];

const convertToBase64 = (file: File): Promise<string> =>
  new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => resolve((reader.result as string).split(",")[1]);
    reader.onerror = reject;
  });

type ResultData = {
  status?: string;
  language?: string;
  classification?: string;
  confidenceScore?: number;
  explanation?: string;
};

export default function VoiceDetector() {
  const [language, setLanguage] = useState("English");
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [base64Input, setBase64Input] = useState("");
  const [inputMode, setInputMode] = useState<"file" | "base64">("file");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ResultData | null>(null);
  const [error, setError] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setResult(null);

    if (inputMode === "file" && !audioFile) return setError("Please upload an audio file.");
    if (inputMode === "base64" && !base64Input.trim()) return setError("Please paste a Base64 string.");

    setLoading(true);
    try {
      const audioBase64 = inputMode === "file" ? await convertToBase64(audioFile!) : base64Input.trim();
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json", "x-api-key": API_KEY },
        body: JSON.stringify({ language, audioFormat: "mp3", audioBase64 }),
      });
      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      setResult(await res.json());
    } catch (err: any) {
      setError(err.message || "Request failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const confidence = result?.confidenceScore;
  const isAI = result?.classification === "AI_GENERATED";
  const confPercent = confidence != null ? Math.round(confidence * 100) : null;

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("audio/")) setAudioFile(file);
  };

  return (
    <div className="relative min-h-screen overflow-hidden">
      {/* Background */}
      <div className="fixed inset-0 -z-10">
        <img src={heroBg} alt="" className="h-full w-full object-cover opacity-30" />
        <div className="absolute inset-0 bg-background/60" />
      </div>

      <div className="mx-auto flex max-w-2xl flex-col items-center gap-8 px-5 py-16">
        {/* Header */}
        <div className="animate-fade-up text-center">
          <div className="mx-auto mb-5 flex h-14 w-14 items-center justify-center rounded-2xl bg-primary/10">
            <Mic className="h-7 w-7 text-primary" />
          </div>
          <h1 className="text-3xl font-bold tracking-tight text-foreground">
            AI Voice Detector
          </h1>
          <p className="mt-2 text-muted-foreground">
            Detect whether an audio clip is AI-generated or human.
          </p>
        </div>

        {/* Main Card */}
        <div className="glass-card animate-fade-up-delay w-full rounded-2xl p-8">
          <form onSubmit={handleSubmit} className="flex flex-col gap-6">
            {/* Language */}
            <div className="flex flex-col gap-2">
              <label className="text-sm font-medium text-foreground">Language Model</label>
              <select
                className="rounded-xl border border-border bg-background/50 px-4 py-3 text-sm text-foreground outline-none transition-all focus:border-primary focus:ring-2 focus:ring-primary/20"
                value={language}
                onChange={(e) => setLanguage(e.target.value)}
              >
                {LANGUAGES.map((l) => (
                  <option key={l} value={l}>{l}</option>
                ))}
              </select>
            </div>

            {/* Info */}
            <div className="flex items-center gap-3 rounded-xl bg-primary/5 px-4 py-3 text-sm text-muted-foreground">
              <Info className="h-4 w-4 shrink-0 text-primary" />
              Any audio format is supported and will be seamlessly converted.
            </div>

            {/* Toggle */}
            <div className="flex flex-col gap-2">
              <label className="text-sm font-medium text-foreground">Input Method</label>
              <div className="flex gap-2">
                {([["file", "Upload File"], ["base64", "Paste Base64"]] as const).map(([val, lbl]) => (
                  <button
                    key={val}
                    type="button"
                    className={`flex-1 rounded-xl border px-4 py-2.5 text-sm font-medium transition-all active:scale-[0.97] ${
                      inputMode === val
                        ? "border-primary bg-primary text-primary-foreground shadow-sm"
                        : "border-border bg-background/50 text-muted-foreground hover:bg-secondary"
                    }`}
                    onClick={() => { setInputMode(val); setAudioFile(null); setBase64Input(""); }}
                  >
                    {lbl}
                  </button>
                ))}
              </div>
            </div>

            {/* File / Base64 Input */}
            {inputMode === "file" ? (
              <div
                className={`group cursor-pointer rounded-2xl border-2 border-dashed p-8 text-center transition-all active:scale-[0.98] ${
                  dragOver ? "border-primary bg-primary/5" : "border-border hover:border-primary/40 hover:bg-primary/[0.02]"
                }`}
                onClick={() => fileRef.current?.click()}
                onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={handleDrop}
              >
                <input
                  ref={fileRef}
                  type="file"
                  accept="audio/*"
                  className="hidden"
                  onChange={(e) => setAudioFile(e.target.files?.[0] || null)}
                />
                {audioFile ? (
                  <div className="flex flex-col items-center gap-3">
                    <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-primary/10">
                      <FileAudio className="h-6 w-6 text-primary" />
                    </div>
                    <div>
                      <p className="font-medium text-foreground">{audioFile.name}</p>
                      <p className="mt-0.5 text-xs text-muted-foreground">{(audioFile.size / 1024).toFixed(1)} KB</p>
                    </div>
                  </div>
                ) : (
                  <div className="flex flex-col items-center gap-3">
                    <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-secondary transition-colors group-hover:bg-primary/10">
                      <Upload className="h-5 w-5 text-muted-foreground transition-colors group-hover:text-primary" />
                    </div>
                    <div>
                      <p className="font-medium text-foreground">Click or drag to upload audio</p>
                      <p className="mt-0.5 text-xs text-muted-foreground">Supports MP3, WAV, FLAC, M4A & more</p>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <textarea
                className="min-h-[120px] resize-none rounded-xl border border-border bg-background/50 px-4 py-3 text-sm text-foreground outline-none transition-all placeholder:text-muted-foreground/60 focus:border-primary focus:ring-2 focus:ring-primary/20"
                placeholder="Paste your Base64 encoded audio string here..."
                value={base64Input}
                onChange={(e) => setBase64Input(e.target.value)}
              />
            )}

            {/* Error */}
            {error && (
              <div className="flex items-center gap-2 rounded-xl bg-destructive/10 px-4 py-3 text-sm text-destructive">
                <AlertCircle className="h-4 w-4 shrink-0" />
                {error}
              </div>
            )}

            {/* Submit */}
            <button
              type="submit"
              disabled={loading}
              className="flex items-center justify-center gap-2 rounded-xl bg-primary px-6 py-3.5 text-sm font-semibold text-primary-foreground shadow-sm transition-all hover:shadow-md active:scale-[0.97] disabled:opacity-60 disabled:pointer-events-none"
            >
              {loading ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin-slow" />
                  Analyzing Audio…
                </>
              ) : (
                "Analyze Audio"
              )}
            </button>
          </form>
        </div>

        {/* Result Card */}
        {result && (
          <div className="glass-card animate-fade-up w-full rounded-2xl p-8">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold text-foreground">Analysis Overview</h2>
              <span className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium ${
                isAI
                  ? "bg-destructive/10 text-destructive"
                  : "bg-success/10 text-success"
              }`}>
                {isAI ? <Bot className="h-3.5 w-3.5" /> : <User className="h-3.5 w-3.5" />}
                {isAI ? "AI Generated" : "Human Voice"}
              </span>
            </div>

            {confPercent != null && (
              <div className="mt-6">
                <div className="flex items-baseline justify-between text-sm">
                  <span className="text-muted-foreground">Confidence</span>
                  <span className="text-xl font-bold tabular-nums text-foreground">{confPercent}%</span>
                </div>
                <div className="mt-2 h-2 overflow-hidden rounded-full bg-secondary">
                  <div
                    className={`animate-progress h-full rounded-full ${isAI ? "bg-destructive" : "bg-success"}`}
                    style={{ width: `${confPercent}%` }}
                  />
                </div>
              </div>
            )}

            <div className="mt-6 grid grid-cols-2 gap-3">
              {([
                ["Status", result.status],
                ["Language", result.language],
                ["Classification", result.classification?.replace("_", " ")],
                ["Raw Score", confidence?.toFixed(4)],
              ] as const).map(([k, v]) => (
                <div key={k} className="rounded-xl bg-secondary/60 px-4 py-3">
                  <p className="text-xs text-muted-foreground">{k}</p>
                  <p className="mt-1 text-sm font-semibold text-foreground">{v ?? "—"}</p>
                </div>
              ))}
            </div>

            {result.explanation && (
              <div className="mt-6 rounded-xl bg-secondary/60 px-4 py-4">
                <div className="mb-1 flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
                  <CheckCircle2 className="h-3.5 w-3.5" />
                  Reasoning
                </div>
                <p className="text-sm leading-relaxed text-foreground">{result.explanation}</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
