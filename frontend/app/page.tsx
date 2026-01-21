"use client"

import { useState } from "react"
import { Upload, FileText, CheckCircle, AlertTriangle, Loader2 } from "lucide-react"

export default function Home() {
  const [file, setFile] = useState<File | null>(null)
  const [analyzing, setAnalyzing] = useState(false)
  const [result, setResult] = useState<any>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0])
      setResult(null)
    }
  }

  const handleAnalyze = async () => {
    if (!file) return

    setAnalyzing(true)

    // Simulate upload and analysis for now
    // In real flow: POST /api/upload then POST /api/analyze

    try {
      const formData = new FormData()
      formData.append("file", file)

      const uploadRes = await fetch("http://localhost:8000/upload/", {
        method: "POST",
        body: formData
      })

      if (!uploadRes.ok) throw new Error("Upload failed")

      const analyzeRes = await fetch(`http://localhost:8000/analyze/${file.name}`, {
        method: "POST"
      })

      const data = await analyzeRes.json()
      setResult(data)
    } catch (error) {
      console.error(error)
      alert("Analysis failed. Make sure backend is running.")
    } finally {
      setAnalyzing(false)
    }
  }

  return (
    <main className="flex min-h-screen flex-col items-center p-24 bg-slate-50">
      <div className="z-10 max-w-5xl w-full items-center justify-between font-mono text-sm lg:flex">
        <h1 className="text-4xl font-bold text-slate-900 mb-8">Resume AI Detector</h1>
      </div>

      <div className="bg-white p-8 rounded-xl shadow-lg w-full max-w-2xl">
        <div className="border-2 border-dashed border-slate-300 rounded-lg p-12 text-center hover:bg-slate-50 transition-colors relative">
          <input
            type="file"
            accept=".pdf,.docx,.txt"
            onChange={handleFileChange}
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          />
          <div className="flex flex-col items-center gap-4">
            <Upload className="w-12 h-12 text-slate-400" />
            <p className="text-lg text-slate-600">
              {file ? file.name : "Drag & drop or click to upload resume"}
            </p>
            <p className="text-sm text-slate-400">Supported: PDF, DOCX, TXT</p>
          </div>
        </div>

        <button
          onClick={handleAnalyze}
          disabled={!file || analyzing}
          className="w-full mt-6 bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-lg transition-colors flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {analyzing ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" /> Analyzing...
            </>
          ) : (
            "Analyze Resume"
          )}
        </button>

        {result && (
          <div className={`mt-8 p-6 rounded-lg border ${result.is_ai_generated ? "bg-red-50 border-red-200" : "bg-green-50 border-green-200"}`}>
            <div className="flex items-center gap-3 mb-4">
              {result.is_ai_generated ? (
                <AlertTriangle className="w-8 h-8 text-red-600" />
              ) : (
                <CheckCircle className="w-8 h-8 text-green-600" />
              )}
              <h2 className="text-2xl font-bold text-slate-900">
                {result.is_ai_generated ? "AI-Generated Content Detected" : "Likely Human-Written"}
              </h2>
            </div>

            <div className="space-y-2">
              <p className="text-slate-700">
                <span className="font-semibold">Confidence Score:</span> {(result.confidence * 100).toFixed(1)}%
              </p>

              {result.explanation && result.explanation.length > 0 && (
                <div className="mt-4">
                  <h3 className="font-semibold text-slate-900 mb-2">Analysis:</h3>
                  <ul className="list-disc list-inside text-slate-700 space-y-1">
                    {result.explanation.map((reason: string, idx: number) => (
                      <li key={idx}>{reason}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </main>
  )
}
