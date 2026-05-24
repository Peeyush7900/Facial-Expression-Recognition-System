import React, { useState, useEffect, useRef } from 'react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, Legend } from 'recharts';
import { Cpu, Settings, Play, RefreshCw, CheckCircle, AlertTriangle } from 'lucide-react';

export default function ModelMonitor({ backendUrl, onTrainingComplete }) {
  const [epochs, setEpochs] = useState(10);
  const [lr, setLr] = useState(0.001);
  const [batchSize, setBatchSize] = useState(32);
  const [status, setStatus] = useState({
    status: 'idle',
    current_epoch: 0,
    total_epochs: 0,
    logs: '',
    metrics: {}
  });

  const [triggerLoading, setTriggerLoading] = useState(false);
  const logEndRef = useRef(null);
  let pollingInterval = useRef(null);

  // Poll training status
  const pollStatus = async () => {
    try {
      const res = await fetch(`${backendUrl}/api/train-status`);
      if (res.ok) {
        const data = await res.json();
        setStatus(data);
        
        if (data.status !== 'training') {
          stopPolling();
          if (data.status === 'completed') {
            onTrainingComplete(); // Notify parent to refresh analytics
          }
        }
      }
    } catch (err) {
      console.error("Error polling train status:", err);
    }
  };

  const startPolling = () => {
    stopPolling();
    pollStatus();
    pollingInterval.current = setInterval(pollStatus, 2000);
  };

  const stopPolling = () => {
    if (pollingInterval.current) {
      clearInterval(pollingInterval.current);
      pollingInterval.current = null;
    }
  };

  // Trigger training run
  const handleStartTraining = async () => {
    setTriggerLoading(true);
    try {
      const res = await fetch(`${backendUrl}/api/train-model?epochs=${epochs}&batch_size=${batchSize}&lr=${lr}`, {
        method: 'POST'
      });
      if (res.ok) {
        const data = await res.json();
        setStatus(data.status);
        startPolling();
      }
    } catch (err) {
      console.error(err);
    } finally {
      setTriggerLoading(false);
    }
  };

  // Auto-scroll logs to bottom
  useEffect(() => {
    if (logEndRef.current) {
      logEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [status.logs]);

  // Initial status check & cleanup
  useEffect(() => {
    pollStatus();
    // If it's already training, start polling
    fetch(`${backendUrl}/api/train-status`)
      .then(res => res.json())
      .then(data => {
        setStatus(data);
        if (data.status === 'training') {
          startPolling();
        }
      })
      .catch(console.error);

    return () => stopPolling();
  }, []);

  // Format Recharts history curves
  const getCurvesData = () => {
    const history = status.metrics?.history || status.metrics?.trained_metrics?.history;
    if (!history) return [];
    
    return history.epochs.map((epoch, idx) => ({
      epoch,
      trainLoss: parseFloat(history.train_loss[idx].toFixed(4)),
      valLoss: parseFloat(history.val_loss[idx].toFixed(4)),
      trainAcc: parseFloat(history.train_acc[idx].toFixed(2)),
      valAcc: parseFloat(history.val_acc[idx].toFixed(2)),
    }));
  };

  const curvesData = getCurvesData();

  return (
    <div className="flex flex-col gap-8 w-full">
      {/* Hyperparameters Config Panel */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="glass-panel rounded-2xl p-5 flex flex-col gap-4">
          <div className="flex items-center gap-2 mb-2">
            <Settings className="w-5 h-5 text-slate-400" />
            <h3 className="font-semibold text-sm text-slate-200">Training Configurations</h3>
          </div>

          {/* Epochs slider */}
          <div className="flex flex-col gap-1.5">
            <div className="flex justify-between text-xs">
              <span className="text-slate-400 font-medium">Epochs</span>
              <span className="font-mono text-brand-400 font-bold">{epochs}</span>
            </div>
            <input
              type="range"
              min="5"
              max="50"
              step="5"
              value={epochs}
              onChange={(e) => setEpochs(parseInt(e.target.value))}
              disabled={status.status === 'training'}
              className="w-full h-1 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-brand-500 disabled:opacity-40"
            />
          </div>

          {/* Learning Rate input */}
          <div className="flex flex-col gap-1.5">
            <div className="flex justify-between text-xs">
              <span className="text-slate-400 font-medium">Learning Rate (alpha)</span>
              <span className="font-mono text-brand-400 font-bold">{lr}</span>
            </div>
            <select
              value={lr}
              onChange={(e) => setLr(parseFloat(e.target.value))}
              disabled={status.status === 'training'}
              className="w-full px-3 py-2 bg-slate-950 border border-slate-800 rounded-xl text-xs text-slate-300 font-mono focus:outline-none focus:border-brand-500 disabled:opacity-40"
            >
              <option value={0.01}>0.01 (Fast)</option>
              <option value={0.005}>0.005</option>
              <option value={0.001}>0.001 (Recommended)</option>
              <option value={0.0005}>0.0005</option>
              <option value={0.0001}>0.0001 (Precise)</option>
            </select>
          </div>

          {/* Batch Size select */}
          <div className="flex flex-col gap-1.5">
            <div className="flex justify-between text-xs">
              <span className="text-slate-400 font-medium">Batch Size</span>
              <span className="font-mono text-brand-400 font-bold">{batchSize}</span>
            </div>
            <select
              value={batchSize}
              onChange={(e) => setBatchSize(parseInt(e.target.value))}
              disabled={status.status === 'training'}
              className="w-full px-3 py-2 bg-slate-950 border border-slate-800 rounded-xl text-xs text-slate-300 font-mono focus:outline-none focus:border-brand-500 disabled:opacity-40"
            >
              <option value={16}>16</option>
              <option value={32}>32 (Recommended)</option>
              <option value={64}>64</option>
            </select>
          </div>

          <button
            onClick={handleStartTraining}
            disabled={status.status === 'training' || triggerLoading}
            className={`w-full flex items-center justify-center gap-2 py-3 rounded-xl font-medium transition-all shadow-lg ${
              status.status === 'training'
                ? 'bg-slate-800 text-slate-500 border border-slate-700 cursor-not-allowed'
                : 'bg-brand-600 hover:bg-brand-700 text-white shadow-brand-600/10'
            }`}
          >
            {status.status === 'training' || triggerLoading ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin text-slate-500" />
                Training CNN...
              </>
            ) : (
              <>
                <Play className="w-4 h-4 fill-white" />
                Train Emotion Model
              </>
            )}
          </button>
        </div>

        {/* Live Logs Console */}
        <div className="lg:col-span-2 glass-panel rounded-2xl p-5 flex flex-col h-[320px]">
          <div className="flex justify-between items-center mb-3">
            <div className="flex items-center gap-2">
              <Cpu className="w-5 h-5 text-emerald-400" />
              <h3 className="font-semibold text-sm text-slate-200">Execution Logs Console</h3>
            </div>
            {status.status === 'training' && (
              <span className="text-[10px] font-mono bg-emerald-950/30 text-emerald-400 border border-emerald-900/30 px-2 py-0.5 rounded animate-pulse">
                EPOCH {status.current_epoch}/{status.total_epochs}
              </span>
            )}
          </div>

          {/* Console Textarea */}
          <div className="flex-1 bg-slate-950 border border-slate-900 rounded-xl p-4 overflow-y-auto font-mono text-[10px] text-emerald-500 leading-normal">
            <div className="whitespace-pre-line">
              {status.logs || "Console idle. Awaiting model execution command..."}
              {status.status === 'training' && (
                <span className="inline-block w-1.5 h-3 bg-emerald-500 ml-1 animate-pulse" />
              )}
            </div>
            <div ref={logEndRef} />
          </div>
        </div>
      </div>

      {/* Metrics Curves (Loss & Accuracy Plots) */}
      {curvesData.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Loss Curve */}
          <div className="glass-panel rounded-2xl p-5 flex flex-col h-80">
            <h4 className="font-semibold text-xs text-slate-400 mb-4">Training vs Validation Loss</h4>
            <div className="flex-1 min-h-0">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={curvesData}>
                  <XAxis dataKey="epoch" stroke="#475569" fontSize={9} tickLine={false} />
                  <YAxis stroke="#475569" fontSize={9} tickLine={false} />
                  <Tooltip
                    contentStyle={{ background: '#0f172a', border: '1px solid #334155', borderRadius: '8px' }}
                    labelStyle={{ color: '#64748b', fontSize: '10px' }}
                    itemStyle={{ color: '#f8fafc', fontSize: '12px' }}
                  />
                  <Legend wrapperStyle={{ fontSize: '10px', marginTop: '10px' }} />
                  <Line type="monotone" dataKey="trainLoss" name="Train Loss" stroke="#3b82f6" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="valLoss" name="Val Loss" stroke="#ef4444" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Accuracy Curve */}
          <div className="glass-panel rounded-2xl p-5 flex flex-col h-80">
            <h4 className="font-semibold text-xs text-slate-400 mb-4">Training vs Validation Accuracy (%)</h4>
            <div className="flex-1 min-h-0">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={curvesData}>
                  <XAxis dataKey="epoch" stroke="#475569" fontSize={9} tickLine={false} />
                  <YAxis stroke="#475569" fontSize={9} tickLine={false} domain={[0, 100]} />
                  <Tooltip
                    contentStyle={{ background: '#0f172a', border: '1px solid #334155', borderRadius: '8px' }}
                    labelStyle={{ color: '#64748b', fontSize: '10px' }}
                    itemStyle={{ color: '#f8fafc', fontSize: '12px' }}
                  />
                  <Legend wrapperStyle={{ fontSize: '10px', marginTop: '10px' }} />
                  <Line type="monotone" dataKey="trainAcc" name="Train Acc" stroke="#10b981" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="valAcc" name="Val Acc" stroke="#f59e0b" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
