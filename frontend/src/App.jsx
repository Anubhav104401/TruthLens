import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import axios from 'axios';
import { motion } from 'framer-motion';
import {
  FiBarChart2,
  FiBookOpen,
  FiClock,
  FiDatabase,
  FiFileText,
  FiInfo,
  FiShield,
  FiType,
} from 'react-icons/fi';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

const cardPanel = 'rounded-[18px] border border-[rgba(148,163,184,0.08)] bg-[rgba(255,255,255,0.54)] backdrop-blur-[4px] shadow-[0_2px_10px_rgba(15,23,42,0.02)]';
const subCard = 'rounded-[16px] border border-[rgba(148,163,184,0.08)] bg-[rgba(255,255,255,0.68)] backdrop-blur-[4px] shadow-[0_2px_8px_rgba(15,23,42,0.02)]';

const SAMPLE_ARTICLE = `City officials approved a new housing initiative on Monday after months of hearings, according to documents posted on the municipal website. The program will convert an unused warehouse district into mixed-income apartments, with a portion reserved for teachers, transit workers, and first responders.

Supporters say the plan could reduce vacancy pressure and bring new foot traffic to nearby businesses, while critics argue the city should focus on existing infrastructure before expanding development incentives.`;

function App() {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [modelInfo, setModelInfo] = useState(null);
  const [loadingPrediction, setLoadingPrediction] = useState(false);
  const [error, setError] = useState('');

  const textareaRef = useRef(null);

  const charCount = useMemo(() => text.trim().length, [text]);
  const wordCount = useMemo(() => text.trim().split(/\s+/).filter(Boolean).length, [text]);
  const canAnalyze = charCount >= 30 && wordCount >= 5 && !loadingPrediction;

  const loadModelInfo = useCallback(async () => {
    try {
      const response = await axios.get(`${API_URL}/model-info`);
      setModelInfo(response.data);
    } catch {
      setModelInfo(null);
    }
  }, []);

  const handlePredict = useCallback(async () => {
    if (!canAnalyze) {
      return;
    }

    setLoadingPrediction(true);
    setError('');

    try {
      const response = await axios.post(`${API_URL}/predict`, { text });
      const nextResult = response.data;
      setResult(nextResult);
      setHistory((current) => [
        {
          id: crypto.randomUUID?.() || `${Date.now()}-${Math.random()}`,
          text,
          prediction: nextResult.prediction,
          confidence: nextResult.confidence,
          timestamp: new Date().toISOString(),
          result: nextResult,
        },
        ...current,
      ].slice(0, 8));
    } catch (predictError) {
      setError(predictError.response?.data?.detail || 'Could not analyze this article.');
    } finally {
      setLoadingPrediction(false);
    }
  }, [canAnalyze, text]);

  useEffect(() => {
    loadModelInfo();
  }, [loadModelInfo]);

  useEffect(() => {
    const handleKeyDown = (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
        event.preventDefault();
        handlePredict();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handlePredict]);

  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) {
      return;
    }

    textarea.style.height = 'auto';
    textarea.style.height = `${Math.max(textarea.scrollHeight, 260)}px`;
  }, [text]);

  const handlePaste = async () => {
    try {
      const clipboardText = await navigator.clipboard.readText();
      if (clipboardText.trim()) {
        setText(clipboardText);
      }
    } catch {
      setError('Clipboard access is unavailable in this browser context.');
    }
  };

  const handleSample = () => {
    setText(SAMPLE_ARTICLE);
    setError('');
  };

  const handleClear = () => {
    setText('');
    setResult(null);
    setError('');
  };

  const reopenResult = (item) => {
    setText(item.text);
    setResult(item.result);
  };

  return (
    <div className="relative min-h-screen text-slate-900">
      <main className="relative z-10 mx-auto max-w-[1260px] px-4 py-3 sm:px-6 sm:py-4 lg:px-8 lg:py-5">
        <Hero modelInfo={modelInfo} />

        <section className="grid gap-2 py-2.5 sm:grid-cols-[minmax(0,1.05fr)_minmax(0,0.95fr)]">
          <ArticleInput
            text={text}
            textareaRef={textareaRef}
            charCount={charCount}
            wordCount={wordCount}
            canAnalyze={canAnalyze}
            loadingPrediction={loadingPrediction}
            error={error}
            onChange={setText}
            onAnalyze={handlePredict}
            onPaste={handlePaste}
            onSample={handleSample}
            onClear={handleClear}
          />

          <PredictionResult result={result} />
        </section>

        <section className="grid gap-3 pt-1 sm:grid-cols-[minmax(0,1.05fr)_minmax(0,0.95fr)]">
          <RecentAnalyses history={history} onOpen={reopenResult} />
          <ModelInformation modelInfo={modelInfo} />
        </section>

        <section className="pt-0.5">
          <WorkflowNotes />
        </section>
      </main>
    </div>
  );
}


function Hero({ modelInfo }) {
  return (
    <motion.section
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
      className="rounded-[18px] p-2 bg-transparent border-none shadow-none"
    >
      <div className="max-w-3xl space-y-0.5">
        <div className="flex items-center gap-2">
          <div className="flex h-10 w-10 items-center justify-center rounded-[12px] border border-[rgba(148,163,184,0.14)] bg-[rgba(255,255,255,0.85)] text-slate-900">
            <FiFileText className="h-5 w-5" />
          </div>
          <h1 className="text-[1.45rem] font-semibold tracking-tight text-slate-900 sm:text-[1.75rem]">TruthLens</h1>
        </div>

        <p className="text-[0.95rem] font-semibold leading-[1.02] text-slate-900 sm:text-[1.12rem]">
          Analyze articles.
          <br />
          Understand predictions.
          <span className="text-blue-700"> Make informed decisions.</span>
        </p>
        <p className="max-w-2xl text-[0.82rem] leading-5 text-slate-500">TruthLens uses machine learning and explainable AI to estimate the credibility of news articles.</p>
      </div>

      <div className="mt-1 grid gap-2 sm:grid-cols-3">
        <InfoCard icon={<FiDatabase />} label="53,847" value="Dataset Rows" sublabel="Dataset Rows" />
        <InfoCard icon={<FiShield />} label="Model" value={modelInfo?.model_name || 'linear_svm'} sublabel="linear_svm" />
        <InfoCard icon={<FiInfo />} label="This is an AI model." value="AI-assisted Analysis" sublabel="Not a fact-checker." />
      </div>
    </motion.section>
  );
}

function ArticleInput({
  text,
  textareaRef,
  charCount,
  wordCount,
  canAnalyze,
  loadingPrediction,
  error,
  onChange,
  onAnalyze,
  onPaste,
  onSample,
  onClear,
}) {
  return (
    <motion.section
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
      className={`${cardPanel} p-2.5 min-h-[22rem]`}
    >
      <SectionTitle icon={<FiType />} title="Article Input" />

      <div className="mt-2.5 space-y-2">
        <label className="block">
          <span className="sr-only">Article text</span>
          <textarea
            ref={textareaRef}
            value={text}
            onChange={(event) => onChange(event.target.value)}
            placeholder="Paste a news article, report, or statement here..."
            className="min-h-[15rem] w-full resize-none rounded-[14px] border border-[rgba(148,163,184,0.12)] bg-[rgba(255,255,255,0.72)] px-4 py-2.5 text-[15px] leading-6 text-slate-900 outline-none transition placeholder:text-slate-400 focus:border-blue-300 focus:ring-2 focus:ring-blue-200/50"
          />
        </label>

        <div className="flex items-center gap-3 text-[0.78rem] text-slate-500">
          <span>{charCount} characters</span>
          <span>{wordCount} words</span>
        </div>

        <div className="grid gap-2 sm:grid-cols-[repeat(3,minmax(0,1fr))]">
          <ActionButton label="Paste" onClick={onPaste} />
          <ActionButton label="Sample Article" onClick={onSample} />
          <ActionButton label="Clear" onClick={onClear} muted />
        </div>

        <div>
          <ActionButton
            label={loadingPrediction ? 'Analyzing...' : 'Analyze Article'}
            onClick={onAnalyze}
            primary
            disabled={!canAnalyze || loadingPrediction}
            fullWidth
          />
        </div>

        {error ? <div className="rounded-[12px] border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">{error}</div> : null}

        <p className="text-xs leading-5 text-slate-500">Enter at least a few words before analyzing. Press Ctrl/Command + Enter to submit.</p>
      </div>
    </motion.section>
  );
}

function PredictionResult({ result }) {
  return (
    <motion.section
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25, delay: 0.05 }}
      className={`${cardPanel} p-3 min-h-[24.5rem]`}
    >
      <SectionTitle icon={<FiBarChart2 />} title="Prediction Result" />

      {!result ? (
        <div className="flex min-h-[26rem] flex-col items-center justify-center px-6 text-center text-slate-400">
          <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-full border border-[rgba(148,163,184,0.15)] bg-[rgba(255,255,255,0.82)] text-blue-700">
            <FiFileText className="h-6 w-6" />
          </div>
          <p className="text-sm leading-6 text-slate-500">Your analysis results will appear here after you submit an article.</p>
        </div>
      ) : (
        <div className="mt-3 space-y-3">
          <div className={`${subCard} p-2`}>            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <p className="text-[0.7rem] uppercase tracking-[0.18em] text-slate-500">Prediction</p>
                <StatusPill prediction={result.prediction} />
              </div>
              <div className="text-right">
                <p className="text-[0.7rem] uppercase tracking-[0.18em] text-slate-500">Confidence</p>
                <p className="mt-1 text-lg font-semibold text-slate-900">{Number(result.confidence ?? 0).toFixed(0)}%</p>
              </div>
            </div>
          </div>

          <MetricRow label="Confidence" value={Number(result.confidence ?? 0)} tone="blue" />
          <MetricRow label="Risk" value={result.risk_level || 'Unknown'} tone="amber" asText />

          <div className="grid gap-3 sm:grid-cols-2">
            <SmallStat label="Fake probability" value={`${Number(result.fake_probability ?? 0).toFixed(0)}%`} tone="red" />
            <SmallStat label="Real probability" value={`${Number(result.real_probability ?? 0).toFixed(0)}%`} tone="green" />
            <SmallStat label="ML confidence" value={`${Number(result.ml_confidence ?? 0).toFixed(0)}%`} tone="blue" />
            <SmallStat label="Rule adjustment" value={signedValue(Number(result.rule_penalty ?? 0))} tone={Number(result.rule_penalty ?? 0) > 0 ? 'red' : 'green'} />
          </div>

          {result.weights_used && (
            <div className={`${subCard} p-4`}>
              <h3 className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Score Breakdown & Weights</h3>
              <div className="mt-3 grid gap-3 sm:grid-cols-3">
                <IndicatorBar label="Machine Learning" value={Number(result.weights_used.ml ?? 0) * 100} tone="blue" />
                <IndicatorBar label="Web Verification" value={Number(result.weights_used.gemini ?? 0) * 100} tone="amber" />
                <IndicatorBar label="Rule Heuristics" value={Number(result.weights_used.rule ?? 0) * 100} tone="green" />
              </div>
            </div>
          )}

          {result.gemini_enabled && (result.gemini_confidence > 0 || result.gemini_sources?.length > 0) && (
            <div className={`${subCard} p-4`}>
              <h3 className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Web Verification Sources (Gemini)</h3>
              {result.gemini_sources?.length ? (
                <div className="mt-3 space-y-3">
                  {result.gemini_sources.map((source, index) => (
                    <div key={index} className="rounded-[12px] border border-[rgba(148,163,184,0.12)] bg-[rgba(255,255,255,0.7)] p-3 shadow-sm">
                      <a
                        href={source.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-sm font-semibold text-blue-700 hover:underline line-clamp-1"
                      >
                        {source.title}
                      </a>
                      <p className="mt-1 text-xs leading-5 text-slate-600 line-clamp-2">{source.snippet}</p>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="mt-3 text-sm text-slate-500">No specific sources were cited.</p>
              )}
            </div>
          )}

          <div className={`${subCard} p-4`}>
            <h3 className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Explanation</h3>
            {result.reasons?.length ? (
              <ul className="mt-3 space-y-2 text-sm leading-6 text-slate-700">
                {result.reasons.map((reason, index) => (
                  <li key={`${reason}-${index}`} className="flex gap-2">
                    <span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-blue-600" />
                    <span>{reason}</span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="mt-3 text-sm text-slate-500">No explanation was returned for this analysis.</p>
            )}
          </div>

          <div className={`${subCard} p-4`}>
            <h3 className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Supporting Indicators</h3>
            <div className="mt-3 space-y-3">
              <IndicatorBar label="Fake risk" value={Number(result.fake_probability ?? 0)} tone="red" />
              <IndicatorBar label="Real probability" value={Number(result.real_probability ?? 0)} tone="green" />
              <IndicatorBar label="Decision confidence" value={Number(result.confidence ?? 0)} tone="blue" />
            </div>
          </div>
        </div>
      )}
    </motion.section>
  );
}

function RecentAnalyses({ history, onOpen }) {
  return (
    <motion.section
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25, delay: 0.1 }}
      className={`${cardPanel} p-3`}
    >
      <SectionTitle icon={<FiClock />} title="Recent Analyses" />

      <div className="mt-3 space-y-2.5">
        {history.length ? (
          history.map((item) => (
            <button
              key={item.id}
              type="button"
              onClick={() => onOpen(item)}
              className="w-full rounded-[16px] border border-[rgba(148,163,184,0.14)] bg-[rgba(255,255,255,0.86)] px-4 py-3 text-left transition hover:border-[rgba(148,163,184,0.22)] hover:bg-[rgba(255,255,255,0.92)] focus:outline-none focus:ring-2 focus:ring-blue-200/60"
            >
              <div className="flex items-start justify-between gap-4">
                <div className="min-w-0 space-y-1">
                  <p className="truncate text-sm leading-6 text-slate-800">{item.text}</p>
                  <p className="text-xs text-slate-500">{formatTimestamp(item.timestamp)}</p>
                </div>
                <div className="shrink-0 text-right">
                  <p className="text-sm font-medium text-slate-900">{item.prediction}</p>
                  <p className="text-xs text-slate-500">{Number(item.confidence ?? 0).toFixed(0)}%</p>
                </div>
              </div>
            </button>
          ))
        ) : (
          <div className="rounded-[16px] border border-dashed border-[rgba(148,163,184,0.14)] bg-[rgba(255,255,255,0.86)] px-4 py-6 text-sm text-slate-500">
            <p className="font-semibold text-slate-800">No recent analyses.</p>
            <p className="mt-2 text-xs text-slate-500">Results you generate during this browser session will appear here.</p>
          </div>
        )}
      </div>
    </motion.section>
  );
}

function ModelInformation({ modelInfo }) {
  const rows = [
    { icon: <FiDatabase />, label: 'Dataset Size', value: modelInfo?.dataset_rows ? modelInfo.dataset_rows.toLocaleString() : 'Unavailable' },
    { icon: <FiShield />, label: 'Model', value: modelInfo?.model_name || 'Unavailable' },
    { icon: <FiClock />, label: 'Version', value: modelInfo?.trained_at ? new Date(modelInfo.trained_at).toLocaleDateString() : 'Unavailable' },
    { icon: <FiInfo />, label: 'Description', value: modelInfo?.warning || 'This model estimates credibility using machine learning and explainability signals.' },
  ];

  return (
    <motion.section
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25, delay: 0.15 }}
      className={`${cardPanel} p-3`}
    >
      <SectionTitle icon={<FiDatabase />} title="Model Information" />

      <div className="mt-3 space-y-2">
        {rows.map((row) => (
          <div key={row.label} className="flex items-start gap-3 rounded-[14px] border border-[rgba(148,163,184,0.12)] bg-[rgba(255,255,255,0.84)] px-3 py-2.5">
            <span className="mt-0.5 text-blue-700">{row.icon}</span>
            <div className="min-w-0 flex-1">
              <p className="text-[0.68rem] font-medium uppercase tracking-[0.18em] text-slate-500">{row.label}</p>
              <p className="mt-1 text-sm leading-5 text-slate-700">{row.value}</p>
            </div>
          </div>
        ))}
      </div>
    </motion.section>
  );
}

function WorkflowNotes() {
  return (
    <motion.section
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25, delay: 0.2 }}
      className={`${cardPanel} p-3`}
    >
      <SectionTitle icon={<FiBookOpen />} title="Workflow Notes" />

      <ul className="mt-3 space-y-1.5 text-sm leading-6 text-slate-700">
        <li className="flex gap-2"><span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-blue-600" />Estimates article credibility</li>
        <li className="flex gap-2"><span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-blue-600" />Verify with trusted sources</li>
        <li className="flex gap-2"><span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-blue-600" />Supports decision-making</li>
      </ul>
    </motion.section>
  );
}

function SectionTitle({ icon, title }) {
  return (
    <div className="flex items-center gap-2 text-[0.95rem] font-semibold text-slate-900">
      <span className="text-blue-700">{icon}</span>
      <span>{title}</span>
    </div>
  );
}

function InfoCard({ icon, label, value, sublabel }) {
  return (
    <div className="rounded-[18px] border border-[rgba(148,163,184,0.1)] bg-[rgba(255,255,255,0.72)] backdrop-blur-[4px] px-4 py-2.5 shadow-[0_3px_10px_rgba(15,23,42,0.02)]">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-[0.68rem] font-semibold uppercase tracking-[0.18em] text-slate-500">{label}</p>
          <p className="mt-1.5 text-sm font-semibold text-slate-900">{value}</p>
          <p className="mt-1 text-[0.75rem] text-slate-500">{sublabel}</p>
        </div>
        <span className="rounded-[10px] border border-blue-100 bg-[rgba(235,245,255,0.95)] p-2 text-blue-700">{icon}</span>
      </div>
    </div>
  );
}

function ActionButton({ label, onClick, primary = false, muted = false, disabled = false, fullWidth = false }) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={`inline-flex h-10 items-center justify-center rounded-[12px] border px-4 text-sm font-semibold transition focus:outline-none focus:ring-2 focus:ring-blue-200/70 disabled:cursor-not-allowed disabled:opacity-60 ${fullWidth ? 'w-full' : ''} ${
        primary
          ? 'border-blue-600 bg-blue-600 text-white hover:bg-blue-700'
          : muted
            ? 'border-[rgba(148,163,184,0.16)] bg-[rgba(255,255,255,0.88)] text-slate-700 hover:bg-[rgba(255,255,255,0.94)]'
            : 'border-[rgba(148,163,184,0.16)] bg-[rgba(255,255,255,0.88)] text-slate-700 hover:bg-[rgba(255,255,255,0.94)]'
      }`}
    >
      {label}
    </button>
  );
}

function StatusPill({ prediction }) {
  const tone = prediction?.includes('Fake') ? 'red' : prediction?.includes('Real') ? 'green' : 'amber';
  const classes = {
    red: 'border-red-200 bg-red-50 text-red-700',
    green: 'border-emerald-200 bg-emerald-50 text-emerald-700',
    amber: 'border-amber-200 bg-amber-50 text-amber-700',
  }[tone];

  return <span className={`mt-2 inline-flex rounded-full border px-3 py-1.5 text-sm font-medium ${classes}`}>{prediction || 'Unknown'}</span>;
}

function MetricRow({ label, value, tone = 'blue', asText = false }) {
  const colors = {
    blue: 'bg-blue-600',
    green: 'bg-emerald-600',
    amber: 'bg-amber-500',
    red: 'bg-red-600',
  };
  const textColors = {
    blue: 'text-blue-700',
    green: 'text-emerald-700',
    amber: 'text-amber-700',
    red: 'text-red-700',
  };

  return (
    <div className="rounded-[16px] border border-[rgba(148,163,184,0.14)] bg-[rgba(255,255,255,0.84)] p-3">
      <div className="flex items-center justify-between gap-4 text-sm">
        <span className="font-medium text-slate-600">{label}</span>
        <span className={`font-semibold ${textColors[tone]}`}>{asText ? value : `${Math.max(0, Math.min(100, Number(value)))}%`}</span>
      </div>
      {!asText ? (
        <div className="mt-3 h-2 overflow-hidden rounded-full bg-slate-200">
          <div className={`h-full rounded-full ${colors[tone]}`} style={{ width: `${Math.max(0, Math.min(100, Number(value)))}%` }} />
        </div>
      ) : null}
    </div>
  );
}

function SmallStat({ label, value, tone = 'blue' }) {
  const textColors = {
    blue: 'text-blue-700',
    green: 'text-emerald-700',
    amber: 'text-amber-700',
    red: 'text-red-700',
  };

  return (
    <div className="rounded-[16px] border border-[rgba(148,163,184,0.14)] bg-[rgba(255,255,255,0.86)] p-4">
      <p className="text-[0.7rem] font-medium uppercase tracking-[0.18em] text-slate-500">{label}</p>
      <p className={`mt-2 text-base font-semibold ${textColors[tone]}`}>{value}</p>
    </div>
  );
}

function IndicatorBar({ label, value, tone = 'blue' }) {
  const colors = {
    blue: 'bg-blue-600',
    green: 'bg-emerald-600',
    amber: 'bg-amber-500',
    red: 'bg-red-600',
  };

  return (
    <div>
      <div className="flex items-center justify-between gap-3 text-xs text-slate-500">
        <span>{label}</span>
        <span>{Math.max(0, Math.min(100, value)).toFixed(0)}%</span>
      </div>
      <div className="mt-2 h-2 overflow-hidden rounded-full bg-slate-200">
        <div className={`h-full rounded-full ${colors[tone]}`} style={{ width: `${Math.max(0, Math.min(100, value))}%` }} />
      </div>
    </div>
  );
}

function signedValue(value) {
  if (value > 0) return `+${value}`;
  return `${value}`;
}

function formatTimestamp(timestamp) {
  try {
    return new Date(timestamp).toLocaleString([], {
      month: 'short',
      day: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
    });
  } catch {
    return 'Recently';
  }
}

export default App;
