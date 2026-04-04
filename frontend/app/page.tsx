"use client"

import { useState, useRef } from "react"
import {
  Upload, CheckCircle, AlertTriangle, Loader2,
  FileText, Cpu, LayoutTemplate, ChevronDown, ChevronRight,
  Info
} from "lucide-react"

// ── Types ─────────────────────────────────────────────────────────────────────
interface ModelResult {
  label: string
  confidence: number
  probabilities: Record<string, number>
  available?: boolean
  method?: string
}

interface AnalysisResult {
  filename: string
  is_ai_generated: boolean
  label: "human_written" | "ai_generated" | "template_based"
  confidence: number
  explanation: string[]
  matched_ai_phrases: string[]
  matched_template_patterns: string[]
  raw_heuristics: Record<string, unknown>
  features: Record<string, number>
  feature_importances: Record<string, number>
  model_results: {
    random_forest: ModelResult
    xgboost: ModelResult
    ensemble: ModelResult
  }
  debug_info: {
    extracted_text_preview: string
    preprocessed_text_preview: string
  }
}

// ── Label config ──────────────────────────────────────────────────────────────
const LABEL_CFG = {
  human_written: {
    Icon: CheckCircle,
    title: "Likely Human-Written",
    subtitle: "human written",
    headerBg: "bg-emerald-600",
    cardBg: "bg-emerald-50",
    border: "border-emerald-200",
    bar: "bg-emerald-500",
    badge: "bg-emerald-100 text-emerald-800 border-emerald-300",
    iconColor: "text-emerald-600",
  },
  ai_generated: {
    Icon: Cpu,
    title: "AI-Generated Content Detected",
    subtitle: "ai generated",
    headerBg: "bg-rose-600",
    cardBg: "bg-rose-50",
    border: "border-rose-200",
    bar: "bg-rose-500",
    badge: "bg-rose-100 text-rose-800 border-rose-300",
    iconColor: "text-rose-600",
  },
  template_based: {
    Icon: LayoutTemplate,
    title: "Template / AI-Assisted Resume",
    subtitle: "template based",
    headerBg: "bg-amber-500",
    cardBg: "bg-amber-50",
    border: "border-amber-200",
    bar: "bg-amber-500",
    badge: "bg-amber-100 text-amber-800 border-amber-300",
    iconColor: "text-amber-600",
  },
}

// ── Highlight suspicious sentences ────────────────────────────────────────────
function highlightSuspicious(text: string, aiPhrases: string[]): React.ReactNode[] {
  if (!aiPhrases.length) return [<span key="t">{text}</span>]

  // Split into sentences
  const sentences = text.match(/[^.!?]+[.!?]*/g) ?? [text]

  return sentences.map((sentence, i) => {
    const lower = sentence.toLowerCase()
    const isSuspicious = aiPhrases.some(p => lower.includes(p.toLowerCase()))
    return (
      <span
        key={i}
        title={isSuspicious ? "Suspicious AI-like sentence" : undefined}
        className={isSuspicious
          ? "bg-rose-100 text-rose-900 rounded px-0.5 border-b-2 border-rose-400 cursor-help"
          : "text-slate-700"}
      >
        {sentence}
      </span>
    )
  })
}

// ── Confidence Bar ─────────────────────────────────────────────────────────────
function ConfidenceBar({ value, barClass }: { value: number; barClass: string }) {
  const pct = Math.round(value * 100)
  return (
    <div>
      <div className="flex justify-between text-sm mb-1.5">
        <span className="text-slate-600 font-medium">Confidence Score</span>
        <span className="font-extrabold text-2xl text-slate-900">{pct}%</span>
      </div>
      <div className="w-full bg-slate-200 rounded-full h-4 overflow-hidden">
        <div
          className={`h-4 rounded-full transition-all duration-700 ${barClass}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}

// ── Model Card ────────────────────────────────────────────────────────────────
function ModelCard({
  name, result, active
}: {
  name: string
  result: ModelResult
  active?: boolean
}) {
  const label = result.label.replace(/_/g, " ")
  const conf  = Math.round((result.confidence ?? 0) * 100)
  const unavail = result.label === "unavailable" || result.available === false

  const labelColor =
    result.label === "ai_generated"   ? "text-rose-600"   :
    result.label === "template_based" ? "text-amber-600"  :
    result.label === "human_written"  ? "text-emerald-600": "text-slate-500"

  return (
    <div className={`rounded-xl border p-4 flex flex-col gap-2 ${active ? "border-indigo-400 shadow-md bg-indigo-50" : "border-slate-200 bg-white"}`}>
      <div className="flex items-center justify-between">
        <span className="font-bold text-slate-700 text-sm uppercase tracking-wide">{name}</span>
        {active && <span className="text-xs bg-indigo-100 text-indigo-700 rounded-full px-2 py-0.5 font-semibold">Ensemble</span>}
      </div>
      {unavail ? (
        <span className="text-slate-400 text-xs italic">Not available</span>
      ) : (
        <>
          <span className={`font-bold text-lg capitalize ${labelColor}`}>{label}</span>
          <div className="w-full bg-slate-100 rounded-full h-2">
            <div
              className={`h-2 rounded-full ${
                result.label === "ai_generated"   ? "bg-rose-500"   :
                result.label === "template_based" ? "bg-amber-500"  : "bg-emerald-500"
              }`}
              style={{ width: `${conf}%` }}
            />
          </div>
          <span className="text-xs text-slate-500 font-mono">{conf}% confidence</span>
          {result.probabilities && Object.keys(result.probabilities).length > 0 && (
            <div className="mt-1 space-y-0.5">
              {Object.entries(result.probabilities).map(([lbl, prob]) => (
                <div key={lbl} className="flex justify-between text-xs text-slate-500">
                  <span className="capitalize">{lbl.replace(/_/g, " ")}</span>
                  <span className="font-mono">{Math.round((prob as number) * 100)}%</span>
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  )
}

// ── Feature Importance Bar ─────────────────────────────────────────────────────
function ImportanceBar({ name, value, max }: { name: string; value: number; max: number }) {
  const pct   = max > 0 ? (value / max) * 100 : 0
  const label = name.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase())
  return (
    <div className="mb-2">
      <div className="flex justify-between text-xs mb-0.5">
        <span className="text-slate-600 truncate max-w-[65%]">{label}</span>
        <span className="text-slate-500 font-mono">{(value * 100).toFixed(1)}%</span>
      </div>
      <div className="w-full bg-slate-100 rounded-full h-2.5">
        <div className="h-2.5 rounded-full bg-indigo-500" style={{ width: `${pct}%` }} />
      </div>
    </div>
  )
}

// ── Feature Grid ───────────────────────────────────────────────────────────────
function FeatureGrid({ features }: { features: Record<string, number> }) {
  const items: [string, string][] = [
    ["type_token_ratio",            "Lexical Diversity (TTR)"],
    ["avg_word_length",             "Avg Word Length"],
    ["sentence_length_std",         "Sentence Length Std Dev"],
    ["avg_sentence_length",         "Avg Sentence Length"],
    ["perplexity",                  "Neural Perplexity (DistilGPT2)"],
    ["passive_voice_ratio",         "Passive Voice Ratio"],
    ["readability_flesch",          "Flesch Reading Ease"],
    ["readability_fog",             "Gunning Fog Index"],
    ["first_person_count",          "First-Person Pronouns"],
    ["informal_word_count",         "Informal Words"],
    ["ai_phrase_match_count",       "AI Phrases Matched"],
    ["template_pattern_match_count","Template Patterns Matched"],
  ]
  return (
    <div className="grid grid-cols-2 gap-2">
      {items.filter(([k]) => features[k] !== undefined).map(([k, label]) => (
        <div key={k} className="flex justify-between bg-slate-50 rounded px-3 py-2 border border-slate-100">
          <span className="text-slate-500 text-xs">{label}</span>
          <span className="font-mono font-bold text-slate-800 text-xs">{features[k].toFixed(2)}</span>
        </div>
      ))}
    </div>
  )
}

// ── Collapsible Section ────────────────────────────────────────────────────────
function Section({ title, children, defaultOpen = false }: {
  title: string; children: React.ReactNode; defaultOpen?: boolean
}) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="border border-slate-200 rounded-xl overflow-hidden">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between px-4 py-3 bg-slate-50 hover:bg-slate-100 transition-colors text-left"
      >
        <span className="font-semibold text-slate-700 text-sm">{title}</span>
        {open ? <ChevronDown className="w-4 h-4 text-slate-400" /> : <ChevronRight className="w-4 h-4 text-slate-400" />}
      </button>
      {open && <div className="p-4">{children}</div>}
    </div>
  )
}

// ── Main Page ──────────────────────────────────────────────────────────────────
export default function Home() {
  const [file, setFile]       = useState<File | null>(null)
  const [dragging, setDrag]   = useState(false)
  const [analyzing, setAnal]  = useState(false)
  const [result, setResult]   = useState<AnalysisResult | null>(null)
  const [error, setError]     = useState<string | null>(null)
  const inputRef              = useRef<HTMLInputElement>(null)

  const handleFile = (f: File) => { setFile(f); setResult(null); setError(null) }

  const handleAnalyze = async () => {
    if (!file) return
    setAnal(true); setError(null)
    try {
      const form = new FormData()
      form.append("file", file)
      const up = await fetch("http://localhost:8000/upload/", { method: "POST", body: form })
      if (!up.ok) throw new Error(`Upload failed (${up.status})`)
      const an = await fetch(`http://localhost:8000/analyze/${file.name}`, { method: "POST" })
      if (!an.ok) throw new Error(`Analysis failed (${an.status})`)
      setResult(await an.json())
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Unknown error. Is the backend running at localhost:8000?")
    } finally {
      setAnal(false)
    }
  }

  const cfg    = result ? (LABEL_CFG[result.label] ?? LABEL_CFG.human_written) : null
  const maxImp = result ? Math.max(...Object.values(result.feature_importances), 0.001) : 1

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-indigo-950 py-10 px-4">

      {/* Header */}
      <div className="max-w-3xl mx-auto mb-8 text-center">
        <h1 className="text-4xl font-extrabold text-white tracking-tight mb-2">
          AI Resume Detector
        </h1>
        <p className="text-slate-400 text-sm">
          Dataset A (text corpus) · Dataset B (50 AI phrases + 24 template patterns)
          · Random Forest · XGBoost · DistilGPT2 Perplexity
        </p>
      </div>

      {/* Upload Card */}
      <div className="max-w-3xl mx-auto bg-white rounded-2xl shadow-2xl overflow-hidden">
        <div className="bg-indigo-600 px-6 py-4">
          <h2 className="text-white font-semibold text-lg">Upload Resume</h2>
          <p className="text-indigo-200 text-xs mt-0.5">PDF · DOCX · TXT</p>
        </div>
        <div className="p-6">
          <div
            onDragOver={e => { e.preventDefault(); setDrag(true) }}
            onDragLeave={() => setDrag(false)}
            onDrop={e => { e.preventDefault(); setDrag(false); e.dataTransfer.files[0] && handleFile(e.dataTransfer.files[0]) }}
            onClick={() => inputRef.current?.click()}
            className={`border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-all
              ${dragging ? "border-indigo-400 bg-indigo-50" : "border-slate-300 hover:border-indigo-400 hover:bg-indigo-50"}`}
          >
            <input ref={inputRef} type="file" accept=".pdf,.docx,.txt" className="hidden"
              onChange={e => e.target.files?.[0] && handleFile(e.target.files[0])} />
            <Upload className="w-10 h-10 text-slate-400 mx-auto mb-3" />
            {file
              ? <div className="flex items-center justify-center gap-2"><FileText className="w-5 h-5 text-indigo-600" /><span className="font-medium text-indigo-700">{file.name}</span></div>
              : <><p className="text-slate-600 font-medium">Drag & drop or click to select</p><p className="text-slate-400 text-sm mt-1">PDF, DOCX, or TXT</p></>
            }
          </div>

          <button onClick={handleAnalyze} disabled={!file || analyzing}
            className="w-full mt-4 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed text-white font-bold py-3 rounded-xl transition-colors flex items-center justify-center gap-2 text-base">
            {analyzing ? <><Loader2 className="w-5 h-5 animate-spin" /> Analyzing…</> : "Analyze Resume"}
          </button>

          {error && <div className="mt-4 p-3 bg-rose-50 border border-rose-200 rounded-lg text-rose-700 text-sm">{error}</div>}
        </div>
      </div>

      {/* Results */}
      {result && cfg && (
        <div className="max-w-3xl mx-auto mt-6 space-y-4">

          {/* ── Verdict ── */}
          <div className={`rounded-2xl shadow-xl overflow-hidden border ${cfg.border}`}>
            <div className={`${cfg.headerBg} px-6 py-5 flex items-center gap-4`}>
              <cfg.Icon className="w-9 h-9 text-white shrink-0" />
              <div>
                <h2 className="text-white font-extrabold text-2xl leading-tight">{cfg.title}</h2>
                <span className="text-white/70 text-xs font-mono uppercase tracking-widest">{cfg.subtitle}</span>
              </div>
            </div>

            <div className={`${cfg.cardBg} px-6 py-5 space-y-5`}>
              {/* Confidence */}
              <ConfidenceBar value={result.confidence} barClass={cfg.bar} />

              {/* NL Explanation */}
              <div className={`rounded-xl border ${cfg.border} bg-white/70 p-4`}>
                <div className="flex items-center gap-2 mb-2">
                  <Info className={`w-4 h-4 ${cfg.iconColor}`} />
                  <span className="font-semibold text-slate-800 text-sm">Why this classification?</span>
                </div>
                <ul className="space-y-1.5">
                  {result.explanation.map((r, i) => (
                    <li key={i} className="flex items-start gap-2 text-slate-700 text-sm">
                      <span className={`mt-1.5 w-2 h-2 rounded-full shrink-0 ${cfg.bar}`} />
                      {r}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>

          {/* ── Model Comparison ── */}
          <Section title="Model Comparison — Random Forest vs XGBoost vs Ensemble" defaultOpen>
            <div className="grid grid-cols-3 gap-3">
              <ModelCard name="Random Forest" result={result.model_results.random_forest} />
              <ModelCard name="XGBoost" result={result.model_results.xgboost} />
              <ModelCard name="Ensemble" result={result.model_results.ensemble} active />
            </div>
            <p className="text-xs text-slate-400 mt-3 text-center">
              Ensemble method: {result.model_results.ensemble.method ?? "soft_vote"} · Trained on Dataset A (text corpus + synthetic) · Signals from Dataset B
            </p>
          </Section>

          {/* ── Suspicious Sentence Highlighting ── */}
          {result.matched_ai_phrases.length > 0 && (
            <Section title="Suspicious Sentence Highlighting" defaultOpen>
              <p className="text-xs text-slate-500 mb-3 flex items-center gap-1">
                <span className="inline-block w-3 h-3 bg-rose-200 border-b-2 border-rose-400 rounded" />
                Highlighted sentences contain detected AI-signature phrases
              </p>
              <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 text-sm leading-relaxed">
                {highlightSuspicious(result.debug_info.extracted_text_preview, result.matched_ai_phrases)}
              </div>
            </Section>
          )}

          {/* ── Matched AI Phrases ── */}
          {result.matched_ai_phrases.length > 0 && (
            <Section title={`AI Signature Phrases (Dataset B) — ${result.matched_ai_phrases.length} matched`} defaultOpen>
              <div className="flex flex-wrap gap-2">
                {result.matched_ai_phrases.map((p, i) => (
                  <span key={i} className="px-3 py-1 bg-rose-100 text-rose-700 rounded-full text-xs font-medium border border-rose-200">
                    {p}
                  </span>
                ))}
              </div>
            </Section>
          )}

          {/* ── Template Patterns ── */}
          {result.matched_template_patterns.length > 0 && (
            <Section title={`Template Patterns (Dataset B) — ${result.matched_template_patterns.length} matched`} defaultOpen>
              <div className="flex flex-wrap gap-2">
                {result.matched_template_patterns.map((p, i) => (
                  <span key={i} className="px-3 py-1 bg-amber-100 text-amber-700 rounded-full text-xs font-medium border border-amber-200">
                    {p}
                  </span>
                ))}
              </div>
            </Section>
          )}

          {/* ── Feature Importances ── */}
          {Object.keys(result.feature_importances).length > 0 && (
            <Section title="Top Feature Importances (Random Forest — Dataset A trained)" defaultOpen>
              {Object.entries(result.feature_importances)
                .sort((a, b) => b[1] - a[1])
                .map(([k, v]) => <ImportanceBar key={k} name={k} value={v} max={maxImp} />)}
            </Section>
          )}

          {/* ── Stylometric Features ── */}
          <Section title="Stylometric & Neural Features (25-dim vector)">
            <FeatureGrid features={result.features} />
          </Section>

          {/* ── Weak Supervision Heuristics ── */}
          <Section title="Weak Supervision Heuristics (Dataset B signals)">
            <div className="grid grid-cols-2 gap-2">
              {[
                ["AI Phrase Score",   result.raw_heuristics.ai_phrase_score as number],
                ["Template Score",    result.raw_heuristics.template_score as number],
                ["Weak Label",        result.raw_heuristics.weak_label as string],
                ["Weak Confidence",   result.raw_heuristics.weak_confidence as number],
              ].map(([label, val]) => (
                <div key={label as string} className="bg-slate-50 border border-slate-100 rounded-lg px-3 py-2 flex justify-between">
                  <span className="text-slate-500 text-xs">{label as string}</span>
                  <span className="font-mono font-bold text-slate-800 text-xs">
                    {typeof val === "number" ? val.toFixed(3) : String(val)}
                  </span>
                </div>
              ))}
            </div>
          </Section>

          {/* ── Text Debug ── */}
          <Section title="Text Preview (Debug)">
            <div className="space-y-3">
              <div>
                <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-1">Extracted Raw</p>
                <pre className="bg-slate-900 text-green-300 text-xs rounded-xl p-3 whitespace-pre-wrap max-h-40 overflow-y-auto font-mono">
                  {result.debug_info.extracted_text_preview}
                </pre>
              </div>
              <div>
                <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-1">Preprocessed (Cleaned)</p>
                <pre className="bg-slate-900 text-blue-300 text-xs rounded-xl p-3 whitespace-pre-wrap max-h-40 overflow-y-auto font-mono">
                  {result.debug_info.preprocessed_text_preview}
                </pre>
              </div>
            </div>
          </Section>

        </div>
      )}

      <p className="text-center text-slate-600 text-xs mt-10">
        AI Resume Detector · M.Tech Research Project · RandomForest + XGBoost + DistilGPT2 · Dataset A (text corpus) · Dataset B (50 AI phrases · 24 template patterns)
      </p>
    </main>
  )
}
