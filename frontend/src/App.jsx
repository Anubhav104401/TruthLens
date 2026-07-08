import React, { useEffect, useMemo, useState } from 'react';
import axios from 'axios';
import { motion } from 'framer-motion';
import {
  FiActivity,
  FiAlertCircle,
  FiBarChart2,
  FiCheckCircle,
  FiClock,
  FiDatabase,
  FiInfo,
  FiRefreshCw,
  FiSearch,
} from 'react-icons/fi';
import { Doughnut } from 'react-chartjs-2';
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';

ChartJS.register(ArcElement, Tooltip, Legend);

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

function toneFor(prediction = '') {
  if (prediction.includes('Fake')) {
    return {
      icon: FiAlertCircle,
      text: 'text-rose-400',
      bg: 'bg-rose-500/10',
      border: 'border-rose-500/30',
      chart: ['rgba(244, 63, 94, 0.85)', 'rgba(34, 197, 94, 0.65)'],
    };
  }

  if (prediction.includes('Real')) {
    return {
      icon: FiCheckCircle,
      text: 'text-emerald-400',
      bg: 'bg-emerald-500/10',
      border: 'border-emerald-500/30',
      chart: ['rgba(244, 63, 94, 0.65)', 'rgba(34, 197, 94, 0.85)'],
    };
  }

  return {
    icon: FiInfo,
    text: 'text-amber-300',
    bg: 'bg-amber-500/10',
    border: 'border-amber-500/30',
    chart: ['rgba(245, 158, 11, 0.85)', 'rgba(82, 82, 91, 0.8)'],
  };
}

function formatAdjustment(value) {
  if (value > 0) return `+${value}`;
  return `${value}`;
}

function ExplanationHighlighter({ text, explanation }) {
  const wordWeights = useMemo(() => {
    const weights = new Map();
    explanation?.forEach(([word, weight]) => {
      weights.set(String(word).toLowerCase().replace(/[^a-z0-9\s'-]/g, '').trim(), weight);
    });
    return weights;
  }, [explanation]);

  if (!explanation || explanation.length === 0) {
    return <p className="leading-7 text-zinc-300">{text}</p>;
  }

  return (
    <p className="leading-7 text-zinc-300">
      {text.split(/(\s+)/).map((word, idx) => {
        const cleanWord = word.toLowerCase().replace(/[^a-z0-9'-]/g, '');
        const weight = wordWeights.get(cleanWord);

        if (!weight) {
          return <span key={idx}>{word}</span>;
        }

        const isFakeSignal = weight > 0;
        const colorClass = isFakeSignal
          ? 'bg-rose-500/25 text-rose-100'
          : 'bg-emerald-500/20 text-emerald-100';

        return (
          <span key={idx} className={`${colorClass} rounded px-1 font-medium`} title={`Weight: ${weight.toFixed(3)}`}>
            {word}
          </span>
        );
      })}
    </p>
  );
}

function RiskChart({ result }) {
  const tone = toneFor(result.prediction);
  const data = {
    labels: ['Fake risk', 'Real likelihood'],
    datasets: [
      {
        data: [result.fake_probability, result.real_probability],
        backgroundColor: tone.chart,
        borderColor: ['rgba(24, 24, 27, 1)', 'rgba(24, 24, 27, 1)'],
        borderWidth: 2,
      },
    ],
  };

  return (
    <Doughnut
      data={data}
      options={{
        cutout: '68%',
        plugins: {
          legend: {
            position: 'bottom',
            labels: { color: '#d4d4d8', boxWidth: 10, padding: 12 },
          },
        },
      }}
    />
  );
}

function App() {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [stats, setStats] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    axios
      .get(`${API_URL}/model-info`)
      .then((response) => setModelInfo(response.data))
      .catch(() => setModelInfo(null));
  }, []);

  const ensureDemoToken = async (forceRefresh = false) => {
    let token = localStorage.getItem('token');
    if (token && !forceRefresh) return token;

    if (forceRefresh) {
      localStorage.removeItem('token');
    }

    try {
      await axios.post(`${API_URL}/register`, {
        username: 'demo_user',
        email: 'demo@example.com',
        password: 'password123',
      });
    } catch (err) {
      // The demo account probably already exists in the in-memory store.
    }

    const response = await axios.post(`${API_URL}/login`, {
      email: 'demo@example.com',
      password: 'password123',
    });
    token = response.data.access_token;
    localStorage.setItem('token', token);
    return token;
  };

  const loadDashboard = async (token) => {
    try {
      const [historyResponse, statsResponse] = await Promise.all([
        axios.get(`${API_URL}/history`, { headers: { Authorization: `Bearer ${token}` } }),
        axios.get(`${API_URL}/stats`, { headers: { Authorization: `Bearer ${token}` } }),
      ]);
      setHistory(historyResponse.data.history || []);
      setStats(statsResponse.data);
    } catch (err) {
      setHistory([]);
      setStats(null);
    }
  };

  const handlePredict = async (event) => {
    event.preventDefault();
    if (!text.trim()) return;

    setLoading(true);
    setError('');

    try {
      const token = await ensureDemoToken();
      let response;
      try {
        response = await axios.post(
          `${API_URL}/predict`,
          { text },
          { headers: { Authorization: `Bearer ${token}` } },
        );
      } catch (err) {
        if (err.response?.status !== 401) {
          throw err;
        }

        const refreshedToken = await ensureDemoToken(true);
        response = await axios.post(
          `${API_URL}/predict`,
          { text },
          { headers: { Authorization: `Bearer ${refreshedToken}` } },
        );
      }

      setResult(response.data);
      await loadDashboard(localStorage.getItem('token'));
    } catch (err) {
      setError(err.response?.data?.detail || 'Could not analyze this text.');
    } finally {
      setLoading(false);
    }
  };

  const ResultIcon = toneFor(result?.prediction).icon;
  const resultTone = toneFor(result?.prediction);

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 selection:bg-cyan-500/30">
      <header className="border-b border-zinc-800 bg-zinc-950/95">
        <div className="mx-auto flex max-w-7xl flex-col gap-4 px-5 py-5 md:flex-row md:items-center md:justify-between">
          <div>
            <h1 className="text-2xl font-semibold tracking-normal text-white">TruthLens</h1>
            <p className="mt-1 text-sm text-zinc-400">Hybrid fake-risk analysis for news text</p>
          </div>
          <div className="flex flex-wrap items-center gap-3 text-sm text-zinc-300">
            <span className="inline-flex items-center gap-2 rounded border border-zinc-800 px-3 py-2">
              <FiDatabase className="text-cyan-300" />
              {modelInfo?.dataset_rows ? `${modelInfo.dataset_rows.toLocaleString()} training rows` : 'Model metadata unavailable'}
            </span>
            <span className="inline-flex items-center gap-2 rounded border border-zinc-800 px-3 py-2">
              <FiActivity className="text-emerald-300" />
              {modelInfo?.model_name || 'model'}
            </span>
          </div>
        </div>
      </header>

      <main className="mx-auto grid max-w-7xl gap-5 px-5 py-5 xl:grid-cols-[minmax(0,1.05fr)_minmax(380px,0.95fr)]">
        <section className="rounded-lg border border-zinc-800 bg-zinc-900/70">
          <div className="flex items-center justify-between border-b border-zinc-800 px-5 py-4">
            <h2 className="flex items-center gap-2 text-base font-semibold">
              <FiSearch className="text-cyan-300" />
              Article Input
            </h2>
            <button
              type="button"
              onClick={() => {
                setText('');
                setResult(null);
                setError('');
              }}
              className="inline-flex h-9 w-9 items-center justify-center rounded border border-zinc-700 text-zinc-300 transition hover:border-zinc-500 hover:text-white"
              title="Clear"
            >
              <FiRefreshCw />
            </button>
          </div>

          <form onSubmit={handlePredict} className="p-5">
            <textarea
              value={text}
              onChange={(event) => setText(event.target.value)}
              placeholder="Paste a news article here..."
              className="h-[440px] w-full resize-none rounded border border-zinc-700 bg-zinc-950 p-4 text-sm leading-7 text-zinc-100 outline-none transition placeholder:text-zinc-600 focus:border-cyan-400"
            />
            {error ? (
              <div className="mt-3 rounded border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-200">
                {error}
              </div>
            ) : null}
            <button
              type="submit"
              disabled={loading || !text.trim()}
              className="mt-4 inline-flex h-11 w-full items-center justify-center gap-2 rounded bg-cyan-500 px-4 text-sm font-semibold text-zinc-950 transition hover:bg-cyan-400 disabled:cursor-not-allowed disabled:bg-zinc-700 disabled:text-zinc-400"
            >
              {loading ? (
                <span className="h-5 w-5 animate-spin rounded-full border-2 border-zinc-950 border-t-transparent" />
              ) : (
                <FiSearch />
              )}
              Analyze
            </button>
          </form>
        </section>

        <section className="space-y-5">
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            className="rounded-lg border border-zinc-800 bg-zinc-900/70"
          >
            <div className="border-b border-zinc-800 px-5 py-4">
              <h2 className="flex items-center gap-2 text-base font-semibold">
                <FiBarChart2 className="text-cyan-300" />
                Result
              </h2>
            </div>

            {result ? (
              <div className="p-5">
                <div className={`rounded border ${resultTone.border} ${resultTone.bg} p-4`}>
                  <div className="flex items-center justify-between gap-4">
                    <div className="flex min-w-0 items-center gap-3">
                      <ResultIcon className={`h-9 w-9 shrink-0 ${resultTone.text}`} />
                      <div>
                        <p className="text-xs uppercase tracking-wide text-zinc-400">Prediction</p>
                        <p className={`mt-1 text-2xl font-semibold ${resultTone.text}`}>{result.prediction}</p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="text-xs uppercase tracking-wide text-zinc-400">Fake Risk</p>
                      <p className="mt-1 text-2xl font-semibold text-white">{result.final_score}%</p>
                    </div>
                  </div>
                </div>

                <div className="mt-5 grid gap-5 md:grid-cols-[180px_1fr]">
                  <div className="mx-auto h-44 w-44">
                    <RiskChart result={result} />
                  </div>
                  <div className="grid content-start gap-3 text-sm">
                    <div className="flex justify-between border-b border-zinc-800 pb-2">
                      <span className="text-zinc-400">Model fake risk</span>
                      <span className="font-medium text-zinc-100">{result.ml_confidence}%</span>
                    </div>
                    <div className="flex justify-between border-b border-zinc-800 pb-2">
                      <span className="text-zinc-400">Rule adjustment</span>
                      <span className={result.rule_penalty > 0 ? 'font-medium text-rose-300' : 'font-medium text-emerald-300'}>
                        {formatAdjustment(result.rule_penalty)}
                      </span>
                    </div>
                    <div className="flex justify-between border-b border-zinc-800 pb-2">
                      <span className="text-zinc-400">Decision confidence</span>
                      <span className="font-medium text-zinc-100">{result.confidence}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-zinc-400">Risk level</span>
                      <span className="font-medium text-zinc-100">{result.risk_level}</span>
                    </div>
                  </div>
                </div>

                <div className="mt-5 border-t border-zinc-800 pt-5">
                  <h3 className="text-sm font-semibold text-zinc-100">Reasons</h3>
                  <ul className="mt-3 space-y-2 text-sm text-zinc-300">
                    {result.reasons.map((reason, index) => (
                      <li key={`${reason}-${index}`} className="flex gap-2">
                        <span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-cyan-300" />
                        <span>{reason}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                <div className="mt-5 border-t border-zinc-800 pt-5">
                  <h3 className="text-sm font-semibold text-zinc-100">Explanation Heatmap</h3>
                  <div className="mt-3 max-h-52 overflow-y-auto pr-2 text-sm">
                    <ExplanationHighlighter text={text} explanation={result.explanation} />
                  </div>
                </div>

                <p className="mt-5 border-t border-zinc-800 pt-4 text-xs leading-5 text-zinc-500">
                  {result.model_warning}
                </p>
              </div>
            ) : (
              <div className="flex min-h-[520px] flex-col items-center justify-center gap-3 p-8 text-zinc-500">
                <FiClock className="h-12 w-12 opacity-50" />
                <p className="text-sm">Awaiting article submission</p>
              </div>
            )}
          </motion.div>

          <div className="rounded-lg border border-zinc-800 bg-zinc-900/70">
            <div className="flex items-center justify-between border-b border-zinc-800 px-5 py-4">
              <h2 className="text-base font-semibold">Session History</h2>
              <span className="text-sm text-zinc-400">{stats?.total_predictions || 0} checked</span>
            </div>
            <div className="divide-y divide-zinc-800">
              {history.length ? (
                history.slice(0, 5).map((item) => (
                  <div key={item._id} className="grid grid-cols-[1fr_auto] gap-3 px-5 py-3 text-sm">
                    <span className="truncate text-zinc-300">{item.text}</span>
                    <span className={toneFor(item.prediction).text}>
                      {item.prediction} · {item.final_score}%
                    </span>
                  </div>
                ))
              ) : (
                <p className="px-5 py-5 text-sm text-zinc-500">No predictions in this backend session</p>
              )}
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
