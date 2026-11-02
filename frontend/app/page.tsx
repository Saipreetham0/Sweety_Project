"use client"

import { useState, useRef, useEffect } from "react"
import {
  Upload, CheckCircle, AlertTriangle, Loader2,
  FileText, Cpu, LayoutTemplate, ChevronDown, ChevronRight,
  Info, Sparkles, BarChart3, Layers, ShieldCheck, RefreshCw,
  FileCheck, Zap, FileCode
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
  raw_heuristics: {
    ai_phrase_score?: number
    template_score?: number
    weak_label?: string
    weak_confidence?: number
    uses_ai_phrases?: boolean
    has_template_structure?: boolean
  }
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

// ── Label configuration ───────────────────────────────────────────────────────
const LABEL_CFG = {
  human_written: {
    Icon: CheckCircle,
    title: "Human-Authored Content",
    subtitle: "Organic Human Writing Profile",
    headerBg: "bg-emerald-600",
    cardBg: "bg-emerald-50/70",
    border: "border-emerald-300",
    bar: "bg-emerald-500",
    badge: "bg-emerald-100 text-emerald-800 border-emerald-300",
    iconColor: "text-emerald-600",
    description: "Natural syntactic variation, rich varied vocabulary, and absence of synthetic AI patterns."
  },
  ai_generated: {
    Icon: Cpu,
    title: "AI-Generated Content Detected",
    subtitle: "High Neural Predictability & AI Phrase Match",
    headerBg: "bg-rose-600",
    cardBg: "bg-rose-50/70",
    border: "border-rose-300",
    bar: "bg-rose-500",
    badge: "bg-rose-100 text-rose-800 border-rose-300",
    iconColor: "text-rose-600",
    description: "Overused LLM signature phrases, robotic sentence uniformity, or predictable phrasing detected."
  },
  template_based: {
    Icon: LayoutTemplate,
    title: "Template / Assisted Resume",
    subtitle: "Boilerplate Template Structure Detected",
    headerBg: "bg-amber-600",
    cardBg: "bg-amber-50/70",
    border: "border-amber-300",
    bar: "bg-amber-500",
    badge: "bg-amber-100 text-amber-800 border-amber-300",
    iconColor: "text-amber-600",
    description: "Contains placeholder bracket tokens or rigid boilerplates common in pre-made CV templates."
  },
}

// ── Highlight suspicious sentences ────────────────────────────────────────────
function highlightSuspicious(text: string, aiPhrases: string[]): React.ReactNode[] {
  if (!aiPhrases || !aiPhrases.length) return [<span key="t">{text}</span>]

  const sentences = text.match(/[^.!?\n]+[.!?\n]*/g) ?? [text]

  return sentences.map((sentence, i) => {
    const lower = sentence.toLowerCase()
    const matched = aiPhrases.filter(p => lower.includes(p.toLowerCase()))
    const isSuspicious = matched.length > 0

    if (isSuspicious) {
      return (
        <span
          key={i}
          title={`Contains AI phrases: ${matched.join(", ")}`}
          className="bg-rose-100/90 text-rose-950 font-medium px-1 py-0.5 rounded border-b-2 border-rose-500 cursor-help inline-block my-0.5"
        >
          {sentence}
        </span>
      )
    }
    return <span key={i} className="text-slate-700">{sentence} </span>
  })
}

// ── Model Card Component ──────────────────────────────────────────────────────
function ModelCard({
  name, result, active
}: {
  name: string
  result: ModelResult
  active?: boolean
}) {
  const label = (result?.label || "unknown").replace(/_/g, " ")
  const conf  = Math.round((result?.confidence ?? 0) * 100)
  const isAvailable = result?.available !== false && result?.label !== "unavailable"

  const labelColor =
    result?.label === "ai_generated"   ? "text-rose-600"   :
    result?.label === "template_based" ? "text-amber-600"  :
    result?.label === "human_written"  ? "text-emerald-600": "text-slate-500"

  const barColor =
    result?.label === "ai_generated"   ? "bg-rose-500"   :
    result?.label === "template_based" ? "bg-amber-500"  : "bg-emerald-500"

  return (
    <div className={`rounded-2xl border p-5 flex flex-col justify-between transition-all duration-300 ${
      active
        ? "border-indigo-400 bg-indigo-50/60 shadow-lg ring-2 ring-indigo-500/20"
        : "border-slate-200 bg-white hover:border-slate-300 shadow-sm"
    }`}>
      <div>
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-1.5">
            <span className="font-bold text-slate-800 text-sm tracking-wide">{name}</span>
          </div>
          {active && (
            <span className="text-[11px] bg-indigo-600 text-white rounded-full px-2.5 py-0.5 font-semibold shadow-sm">
              Ensemble Vote
            </span>
          )}
        </div>

        {!isAvailable ? (
          <div className="py-4 text-center">
            <span className="text-slate-400 text-xs italic">Model Offline</span>
          </div>
        ) : (
          <>
            <div className="flex items-baseline justify-between mb-2">
              <span className={`font-extrabold text-xl capitalize ${labelColor}`}>
                {label}
              </span>
              <span className="font-mono text-sm font-bold text-slate-700">{conf}%</span>
            </div>

            <div className="w-full bg-slate-100 rounded-full h-2.5 mb-3 overflow-hidden">
              <div
                className={`h-2.5 rounded-full transition-all duration-700 ${barColor}`}
                style={{ width: `${conf}%` }}
              />
            </div>

            {result.probabilities && Object.keys(result.probabilities).length > 0 && (
              <div className="space-y-1.5 pt-2 border-t border-slate-100">
                {Object.entries(result.probabilities).map(([lbl, prob]) => (
                  <div key={lbl} className="flex justify-between items-center text-xs">
                    <span className="text-slate-500 capitalize">{lbl.replace(/_/g, " ")}</span>
                    <span className="font-mono font-medium text-slate-700">{Math.round((prob as number) * 100)}%</span>
                  </div>
                ))}
              </div>
            )}
          </>
        )}
      </div>

      <div className="mt-4 pt-2 text-[11px] text-slate-400 border-t border-slate-100 flex justify-between items-center">
        <span>Method</span>
        <span className="font-mono text-slate-600">{result?.method || (active ? "soft_vote" : "direct")}</span>
      </div>
    </div>
  )
}

// ── Feature Importance Bar ─────────────────────────────────────────────────────
function ImportanceBar({ name, value, max }: { name: string; value: number; max: number }) {
  const pct = max > 0 ? (value / max) * 100 : 0
  const label = name.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase())
  return (
    <div className="mb-3">
      <div className="flex justify-between text-xs mb-1">
        <span className="text-slate-700 font-medium truncate max-w-[70%]">{label}</span>
        <span className="text-indigo-600 font-mono font-bold">{(value * 100).toFixed(1)}%</span>
      </div>
      <div className="w-full bg-slate-100 rounded-full h-2 overflow-hidden">
        <div
          className="h-2 rounded-full bg-indigo-600 transition-all duration-500"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}

// ── Collapsible Section ────────────────────────────────────────────────────────
function Section({
  title, icon: Icon, children, defaultOpen = false, badge
}: {
  title: string
  icon?: React.ElementType
  children: React.ReactNode
  defaultOpen?: boolean
  badge?: string | number
}) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="border border-slate-200 rounded-2xl bg-white shadow-sm overflow-hidden transition-all duration-200">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between px-5 py-4 bg-slate-50/80 hover:bg-slate-100/80 transition-colors text-left"
      >
        <div className="flex items-center gap-2.5">
          {Icon && <Icon className="w-4 h-4 text-indigo-600 shrink-0" />}
          <span className="font-bold text-slate-800 text-sm tracking-tight">{title}</span>
          {badge !== undefined && (
            <span className="text-xs bg-indigo-100 text-indigo-700 font-semibold px-2 py-0.5 rounded-full">
              {badge}
            </span>
          )}
        </div>
        {open ? <ChevronDown className="w-4 h-4 text-slate-400" /> : <ChevronRight className="w-4 h-4 text-slate-400" />}
      </button>
      {open && <div className="p-5 border-t border-slate-100">{children}</div>}
    </div>
  )
}

// ── Main Page Component ────────────────────────────────────────────────────────
export default function Home() {
  const [file, setFile] = useState<File | null>(null)
  const [dragging, setDrag] = useState(false)
  const [analyzing, setAnal] = useState(false)
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [apiOnline, setApiOnline] = useState<boolean | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

  // Health check on mount
  useEffect(() => {
    fetch(`${apiBase}/`)
      .then(r => r.ok ? setApiOnline(true) : setApiOnline(false))
      .catch(() => setApiOnline(false))
  }, [apiBase])

  const handleFile = (f: File) => {
    setFile(f)
    setResult(null)
    setError(null)
  }

  const handleAnalyze = async () => {
    if (!file) return
    setAnal(true)
    setError(null)
    try {
      const form = new FormData()
      form.append("file", file)
      const up = await fetch(`${apiBase}/upload/`, { method: "POST", body: form })
      if (!up.ok) throw new Error(`Upload failed with status ${up.status}`)
      const uploadData = await up.json()
      const targetFilename = uploadData.filename || file.name
      const an = await fetch(`${apiBase}/analyze/${encodeURIComponent(targetFilename)}`, { method: "POST" })
      if (!an.ok) throw new Error(`Analysis failed with status ${an.status}`)
      setResult(await an.json())
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : `Unknown error communicating with backend at ${apiBase}`)
    } finally {
      setAnal(false)
    }
  }

  // Quick preset loaders
  const loadPreset = async (filename: string) => {
    setAnal(true)
    setError(null)
    try {
      const an = await fetch(`${apiBase}/analyze/${encodeURIComponent(filename)}`, { method: "POST" })
      if (!an.ok) throw new Error(`Preset failed (${an.status})`)
      setResult(await an.json())
      setFile(new File(["[loaded from benchmark]"], filename, { type: "text/plain" }))
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to load sample")
    } finally {
      setAnal(false)
    }
  }

  const cfg = result ? (LABEL_CFG[result.label] ?? LABEL_CFG.human_written) : null
  const maxImp = result ? Math.max(...Object.values(result.feature_importances || {}), 0.001) : 1

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100 py-10 px-4 sm:px-6">

      {/* ── Top Bar / Header ── */}
      <header className="max-w-4xl mx-auto mb-10 text-center">
        <div className="inline-flex items-center gap-2 bg-slate-800/80 border border-slate-700/80 px-4 py-1.5 rounded-full text-xs text-slate-300 mb-4 shadow-inner">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
          <span>Research-Grade System</span>
          <span className="text-slate-500">•</span>
          <span>Weakly Supervised Hybrid ML</span>
          {apiOnline !== null && (
            <>
              <span className="text-slate-500">•</span>
              <span className={apiOnline ? "text-emerald-400" : "text-rose-400"}>
                API {apiOnline ? "Online" : "Offline"}
              </span>
            </>
          )}
        </div>

        <h1 className="text-4xl sm:text-5xl font-extrabold tracking-tight text-white mb-3">
          AI Resume Detector
        </h1>
        <p className="text-slate-400 text-sm sm:text-base max-w-2xl mx-auto leading-relaxed">
          Combines <strong className="text-slate-200">Random Forest</strong>, <strong className="text-slate-200">XGBoost</strong>, and <strong className="text-slate-200">Heuristic Weak Supervision</strong> to detect AI-generated resumes with explainable feature attribution.
        </p>

        {/* Preset sample buttons */}
        <div className="mt-6 flex flex-wrap items-center justify-center gap-2">
          <span className="text-xs text-slate-400 mr-1 font-medium">Quick Test:</span>
          <button
            onClick={() => loadPreset("AI_Candidate_1.txt")}
            className="text-xs bg-slate-800 hover:bg-slate-700 text-rose-300 border border-rose-900/60 px-3 py-1.5 rounded-lg transition-colors flex items-center gap-1.5 font-medium"
          >
            <Cpu className="w-3.5 h-3.5" /> Sample AI Resume
          </button>
          <button
            onClick={() => loadPreset("Human_Candidate_1.txt")}
            className="text-xs bg-slate-800 hover:bg-slate-700 text-emerald-300 border border-emerald-900/60 px-3 py-1.5 rounded-lg transition-colors flex items-center gap-1.5 font-medium"
          >
            <CheckCircle className="w-3.5 h-3.5" /> Sample Human Resume
          </button>
        </div>
      </header>

      {/* ── Upload Area Card ── */}
      <section className="max-w-4xl mx-auto bg-white text-slate-900 rounded-3xl shadow-2xl border border-slate-200 overflow-hidden mb-8">
        <div className="bg-indigo-600 px-6 sm:px-8 py-5 flex items-center justify-between">
          <div>
            <h2 className="text-white font-extrabold text-lg sm:text-xl">Upload Resume Document</h2>
            <p className="text-indigo-100 text-xs mt-0.5">Supports PDF (with layout mode), DOCX, and TXT files</p>
          </div>
          <span className="text-xs bg-white/20 text-white px-3 py-1 rounded-full font-medium backdrop-blur-sm">
            Max 10MB
          </span>
        </div>

        <div className="p-6 sm:p-8">
          <div
            onDragOver={e => { e.preventDefault(); setDrag(true) }}
            onDragLeave={() => setDrag(false)}
            onDrop={e => {
              e.preventDefault()
              setDrag(false)
              if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0])
            }}
            onClick={() => inputRef.current?.click()}
            className={`border-2 border-dashed rounded-2xl p-10 text-center cursor-pointer transition-all duration-200 ${
              dragging
                ? "border-indigo-500 bg-indigo-50/80 scale-[0.99]"
                : "border-slate-300 hover:border-indigo-400 hover:bg-slate-50/80"
            }`}
          >
            <input
              ref={inputRef}
              type="file"
              accept=".pdf,.docx,.txt"
              className="hidden"
              onChange={e => e.target.files?.[0] && handleFile(e.target.files[0])}
            />

            <div className="w-14 h-14 mx-auto mb-4 bg-indigo-50 rounded-2xl flex items-center justify-center text-indigo-600 border border-indigo-100 shadow-sm">
              <Upload className="w-7 h-7" />
            </div>

            {file ? (
              <div className="inline-flex items-center gap-2 bg-indigo-50 border border-indigo-200 rounded-xl px-4 py-2">
                <FileCheck className="w-5 h-5 text-indigo-600" />
                <span className="font-bold text-indigo-900 text-sm">{file.name}</span>
                <span className="text-xs text-indigo-600">({(file.size / 1024).toFixed(1)} KB)</span>
              </div>
            ) : (
              <div>
                <p className="text-slate-800 font-bold text-base">Drag & drop your resume here, or click to browse</p>
                <p className="text-slate-400 text-xs mt-1.5 font-medium">Accepts standard PDF, Microsoft Word (.docx), or plain text (.txt)</p>
              </div>
            )}
          </div>

          <div className="mt-5 flex gap-3">
            <button
              onClick={handleAnalyze}
              disabled={!file || analyzing}
              className="flex-1 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed text-white font-extrabold py-3.5 px-6 rounded-2xl transition-all shadow-md flex items-center justify-center gap-2.5 text-base"
            >
              {analyzing ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  <span>Extracting Features & Analyzing Models…</span>
                </>
              ) : (
                <>
                  <Zap className="w-5 h-5" />
                  <span>Analyze Resume</span>
                </>
              )}
            </button>

            {file && !analyzing && (
              <button
                onClick={() => { setFile(null); setResult(null); setError(null) }}
                className="px-4 py-3 border border-slate-200 hover:bg-slate-100 text-slate-600 rounded-2xl font-bold text-sm transition-colors"
                title="Reset"
              >
                <RefreshCw className="w-4 h-4" />
              </button>
            )}
          </div>

          {error && (
            <div className="mt-4 p-4 bg-rose-50 border border-rose-200 rounded-xl text-rose-700 text-sm flex items-start gap-3">
              <AlertTriangle className="w-5 h-5 text-rose-500 shrink-0 mt-0.5" />
              <div>
                <p className="font-bold">Analysis Error</p>
                <p className="text-rose-600 text-xs mt-0.5">{error}</p>
              </div>
            </div>
          )}
        </div>
      </section>

      {/* ── Results Dashboard ── */}
      {result && cfg && (
        <section className="max-w-4xl mx-auto space-y-6 text-slate-900 animate-fadeIn">

          {/* ── Primary Verdict Banner ── */}
          <div className={`rounded-3xl shadow-2xl overflow-hidden border ${cfg.border} bg-white`}>
            <div className={`${cfg.headerBg} px-6 sm:px-8 py-6 text-white flex items-center gap-5`}>
              <div className="p-3 bg-white/20 backdrop-blur-md rounded-2xl shrink-0">
                <cfg.Icon className="w-10 h-10 text-white" />
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2.5 mb-1">
                  <span className="text-xs font-mono uppercase tracking-widest px-2.5 py-0.5 rounded-full bg-white/20 font-bold">
                    {result.filename}
                  </span>
                </div>
                <h2 className="text-2xl sm:text-3xl font-black tracking-tight leading-tight">{cfg.title}</h2>
                <p className="text-white/80 text-xs sm:text-sm mt-1">{cfg.subtitle} · {cfg.description}</p>
              </div>
            </div>

            <div className={`${cfg.cardBg} p-6 sm:p-8 space-y-6`}>
              {/* Confidence Meter */}
              <div>
                <div className="flex justify-between items-baseline mb-2">
                  <span className="text-slate-700 font-bold text-sm">Classification Confidence</span>
                  <span className="font-black text-3xl font-mono text-slate-900">
                    {Math.round(result.confidence * 100)}%
                  </span>
                </div>
                <div className="w-full bg-slate-200/80 rounded-full h-4 overflow-hidden p-0.5 border border-slate-300/50">
                  <div
                    className={`h-3 rounded-full transition-all duration-1000 ${cfg.bar}`}
                    style={{ width: `${Math.round(result.confidence * 100)}%` }}
                  />
                </div>
              </div>

              {/* Explanations */}
              <div className="bg-white rounded-2xl p-5 border border-slate-200/80 shadow-sm">
                <div className="flex items-center gap-2 mb-3">
                  <Info className={`w-4 h-4 ${cfg.iconColor}`} />
                  <span className="font-bold text-slate-800 text-sm">Decision Rationale & Explainable Attributes</span>
                </div>
                <ul className="space-y-2">
                  {result.explanation.map((reason, i) => (
                    <li key={i} className="flex items-start gap-2.5 text-slate-700 text-sm leading-relaxed">
                      <span className={`w-2 h-2 rounded-full mt-2 shrink-0 ${cfg.bar}`} />
                      <span>{reason}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>

          {/* ── 3-Model Comparison Grid ── */}
          <Section title="Model Comparison — Random Forest • XGBoost • Soft-Voting Ensemble" icon={Layers} defaultOpen>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <ModelCard
                name="Random Forest"
                result={result.model_results.random_forest}
              />
              <ModelCard
                name="XGBoost Classifier"
                result={result.model_results.xgboost}
              />
              <ModelCard
                name="Soft-Voting Ensemble"
                result={result.model_results.ensemble}
                active
              />
            </div>
            <p className="text-xs text-slate-500 mt-4 text-center">
              The ensemble combines probabilities using soft voting and weak supervision signals from Dataset B.
            </p>
          </Section>

          {/* ── Suspicious Sentence Highlighting ── */}
          {result.matched_ai_phrases.length > 0 && (
            <Section
              title="Sentence-Level AI Phrase Heatmap"
              icon={Sparkles}
              defaultOpen
              badge={`${result.matched_ai_phrases.length} phrases`}
            >
              <p className="text-xs text-slate-500 mb-3 flex items-center gap-1.5">
                <span className="inline-block w-3.5 h-3.5 bg-rose-200 border-b-2 border-rose-500 rounded" />
                <span>Highlighted regions contain detected AI signature keywords from Dataset B:</span>
              </p>
              <div className="bg-slate-50 border border-slate-200 rounded-2xl p-5 text-sm leading-relaxed max-h-72 overflow-y-auto">
                {highlightSuspicious(result.debug_info.extracted_text_preview, result.matched_ai_phrases)}
              </div>
            </Section>
          )}

          {/* ── Matched AI Signature Phrases ── */}
          {result.matched_ai_phrases.length > 0 && (
            <Section title="Detected AI Signature Phrases (Dataset B)" icon={Cpu} defaultOpen badge={result.matched_ai_phrases.length}>
              <div className="flex flex-wrap gap-2">
                {result.matched_ai_phrases.map((phrase, i) => (
                  <span
                    key={i}
                    className="px-3.5 py-1.5 bg-rose-50 text-rose-800 border border-rose-200 rounded-xl text-xs font-bold tracking-tight shadow-sm"
                  >
                    "{phrase}"
                  </span>
                ))}
              </div>
            </Section>
          )}

          {/* ── Top Decision Factors ── */}
          {Object.keys(result.feature_importances || {}).length > 0 && (
            <Section title="Top Feature Importances (Random Forest Attribution)" icon={BarChart3} defaultOpen>
              <div className="space-y-1">
                {Object.entries(result.feature_importances)
                  .sort((a, b) => b[1] - a[1])
                  .map(([name, val]) => (
                    <ImportanceBar key={name} name={name} value={val} max={maxImp} />
                  ))}
              </div>
            </Section>
          )}

          {/* ── 24-Feature Stylometric Breakdown ── */}
          <Section title="Stylometric & Structural Profile" icon={FileText}>
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
              {[
                ["Lexical Diversity (TTR)", result.features.type_token_ratio?.toFixed(3)],
                ["Average Word Length", result.features.avg_word_length?.toFixed(2)],
                ["Function Word Ratio", result.features.function_word_ratio?.toFixed(3)],
                ["Comma Density", result.features.comma_density?.toFixed(3)],
                ["Bullet Count", result.features.bullet_count?.toFixed(0)],
                ["Bullet Word Count (avg)", result.features.avg_bullet_word_count?.toFixed(1)],
                ["Bullet Length Variance", result.features.bullet_length_variance?.toFixed(2)],
                ["Detected Sections", result.features.section_count?.toFixed(0)],
                ["All-Caps Line Ratio", result.features.all_caps_ratio?.toFixed(3)],
                ["Sentence Length (avg)", result.features.avg_sentence_length?.toFixed(1)],
                ["Sentence Length StdDev", result.features.sentence_length_std?.toFixed(2)],
                ["Flesch Reading Ease", result.features.readability_flesch?.toFixed(1)],
                ["Gunning Fog Index", result.features.readability_fog?.toFixed(1)],
                ["First-Person Pronouns", result.features.first_person_count?.toFixed(0)],
                ["Informal Contractions", result.features.informal_word_count?.toFixed(0)],
                ["Matched AI Phrases", result.features.ai_phrase_match_count?.toFixed(0)],
                ["Template Patterns", result.features.template_pattern_match_count?.toFixed(0)],
              ].map(([lbl, val]) => (
                <div key={lbl as string} className="bg-slate-50 border border-slate-100 rounded-xl p-3 flex flex-col justify-between">
                  <span className="text-[11px] font-medium text-slate-500">{lbl}</span>
                  <span className="font-mono font-bold text-slate-800 text-sm mt-1">{val ?? "—"}</span>
                </div>
              ))}
            </div>
          </Section>

          {/* ── Raw Text Debug Preview ── */}
          <Section title="Extracted & Preprocessed Text Inspector" icon={FileCode}>
            <div className="space-y-4">
              <div>
                <p className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-1.5">Extracted Raw Text</p>
                <pre className="bg-slate-900 text-emerald-400 text-xs rounded-xl p-4 whitespace-pre-wrap max-h-48 overflow-y-auto font-mono">
                  {result.debug_info.extracted_text_preview}
                </pre>
              </div>
              <div>
                <p className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-1.5">Preprocessed (Cleaned Structural Text)</p>
                <pre className="bg-slate-900 text-cyan-300 text-xs rounded-xl p-4 whitespace-pre-wrap max-h-48 overflow-y-auto font-mono">
                  {result.debug_info.preprocessed_text_preview}
                </pre>
              </div>
            </div>
          </Section>

        </section>
      )}

      {/* ── Footer ── */}
      <footer className="max-w-4xl mx-auto mt-16 text-center text-xs text-slate-500 space-y-2">
        <p>AI Resume Detector · Weakly Supervised Hybrid ML Pipeline</p>
        <p className="text-slate-600">
          Features: Random Forest • XGBoost Classifier • Soft-Voting Ensemble • Dataset A Corpus • Dataset B Heuristics
        </p>
      </footer>
    </main>
  )
}
