import React from 'react';
import { ResponsiveContainer, PieChart, Pie, Cell, AreaChart, Area, XAxis, YAxis, Tooltip } from 'recharts';
import { BarChart2, TrendingUp, History, Grid, Trash2 } from 'lucide-react';

const EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"];
const EMOTION_COLORS = {
  angry: '#ef4444',    // Red
  disgust: '#a855f7',  // Purple
  fear: '#f97316',     // Orange
  happy: '#10b981',    // Emerald
  sad: '#3b82f6',      // Blue
  surprise: '#eab308', // Yellow
  neutral: '#64748b'   // Slate
};

export default function AnalyticsPanel({ analyticsData, onClearSession, backendUrl }) {
  const { session_history, session_distribution, session_avg_confidence, trained_metrics } = analyticsData;

  // Format distribution for Pie Chart
  const pieData = Object.entries(session_distribution || {})
    .filter(([_, val]) => val > 0)
    .map(([key, val]) => ({
      name: key.toUpperCase(),
      value: Math.round(val * 100)
    }));

  // Format session history for timeline (Area Chart)
  const areaData = (session_history || []).map((item, index) => ({
    frame: index + 1,
    confidence: Math.round(item.confidence * 100),
    emotion: item.emotion.toUpperCase()
  }));

  // Retrieve Confusion Matrix from trained_metrics
  const confusionMatrix = trained_metrics?.confusion_matrix || null;
  const metricsEmotions = trained_metrics?.emotions || EMOTIONS;

  // Clear session history helper
  const handleClearHistory = async () => {
    try {
      const res = await fetch(`${backendUrl}/api/clear-session`, { method: 'POST' });
      if (res.ok) {
        onClearSession();
      }
    } catch (err) {
      console.error("Error clearing session:", err);
    }
  };

  return (
    <div className="flex flex-col gap-8 w-full">
      {/* Session Metrics Summary cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
        <div className="glass-panel rounded-2xl p-5 flex flex-col gap-1">
          <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">Frames Analyzed</span>
          <span className="text-3xl font-bold font-mono text-slate-100">{(session_history || []).length}</span>
        </div>
        <div className="glass-panel rounded-2xl p-5 flex flex-col gap-1">
          <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">Session Avg Confidence</span>
          <span className="text-3xl font-bold font-mono text-brand-400">
            {session_history?.length > 0 ? `${Math.round(session_avg_confidence * 100)}%` : '0%'}
          </span>
        </div>
        <div className="glass-panel rounded-2xl p-5 flex flex-col gap-1">
          <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">Model Accuracy</span>
          <span className="text-3xl font-bold font-mono text-emerald-400">
            {trained_metrics?.final_val_acc ? `${trained_metrics.final_val_acc.toFixed(1)}%` : 'N/A'}
          </span>
        </div>
      </div>

      {/* Recharts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Emotion prevalence chart */}
        <div className="glass-panel rounded-2xl p-5 flex flex-col h-80">
          <div className="flex items-center gap-2 mb-4">
            <BarChart2 className="w-5 h-5 text-brand-400" />
            <h3 className="font-semibold text-sm text-slate-200">Session Emotion Prevalence (%)</h3>
          </div>
          <div className="flex-1 min-h-0 relative">
            {pieData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    innerRadius={55}
                    outerRadius={80}
                    paddingAngle={3}
                    dataKey="value"
                  >
                    {pieData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={EMOTION_COLORS[entry.name.toLowerCase()] || '#3b82f6'} />
                    ))}
                  </Pie>
                  <Tooltip 
                    contentStyle={{ background: '#0f172a', border: '1px solid #334155', borderRadius: '8px' }}
                    itemStyle={{ color: '#f8fafc' }}
                  />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <div className="absolute inset-0 flex items-center justify-center text-xs text-slate-650">No session data available</div>
            )}
            
            {/* Custom side legend */}
            {pieData.length > 0 && (
              <div className="absolute right-4 top-1/2 -translate-y-1/2 flex flex-col gap-1.5 text-[10px] text-slate-400 max-h-full overflow-y-auto">
                {pieData.map((entry) => (
                  <div key={entry.name} className="flex items-center gap-2">
                    <span 
                      className="w-2.5 h-2.5 rounded-full shrink-0" 
                      style={{ backgroundColor: EMOTION_COLORS[entry.name.toLowerCase()] }} 
                    />
                    <span>{entry.name}: {entry.value}%</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Confidence Timeline */}
        <div className="glass-panel rounded-2xl p-5 flex flex-col h-80">
          <div className="flex items-center gap-2 mb-4">
            <TrendingUp className="w-5 h-5 text-emerald-400" />
            <h3 className="font-semibold text-sm text-slate-200">Confidence Over Timeline</h3>
          </div>
          <div className="flex-1 min-h-0 relative">
            {areaData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={areaData}>
                  <defs>
                    <linearGradient id="colorConfidence" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.4}/>
                      <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.0}/>
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="frame" stroke="#475569" fontSize={9} tickLine={false} />
                  <YAxis domain={[0, 100]} stroke="#475569" fontSize={9} tickLine={false} />
                  <Tooltip
                    contentStyle={{ background: '#0f172a', border: '1px solid #334155', borderRadius: '8px' }}
                    labelStyle={{ color: '#64748b', fontSize: '10px' }}
                    itemStyle={{ color: '#f8fafc', fontSize: '12px' }}
                    formatter={(value, name, props) => [`${value}% (${props.payload.emotion})`, 'Confidence']}
                  />
                  <Area type="monotone" dataKey="confidence" stroke="#3b82f6" strokeWidth={2} fillOpacity={1} fill="url(#colorConfidence)" />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <div className="absolute inset-0 flex items-center justify-center text-xs text-slate-650">No timeline data available</div>
            )}
          </div>
        </div>
      </div>

      {/* Confusion Matrix Heatmap Panel */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2 glass-panel rounded-2xl p-5 flex flex-col">
          <div className="flex items-center gap-2 mb-5">
            <Grid className="w-5 h-5 text-amber-400" />
            <h3 className="font-semibold text-sm text-slate-200">Trained CNN Confusion Matrix Heatmap</h3>
          </div>
          
          {confusionMatrix ? (
            <div className="flex flex-col items-center overflow-x-auto w-full">
              {/* Matrix Table */}
              <div className="min-w-[420px] flex flex-col gap-1">
                {/* Header labels */}
                <div className="flex items-center gap-1">
                  <div className="w-20 text-[9px] font-bold text-slate-500 uppercase text-right pr-2">Actual \ Pred</div>
                  {metricsEmotions.map(emo => (
                    <div key={`header-${emo}`} className="w-12 text-[9px] font-bold text-slate-500 uppercase text-center truncate">
                      {emo.substring(0, 4)}
                    </div>
                  ))}
                </div>
                
                {/* Matrix Rows */}
                {confusionMatrix.map((row, rowIdx) => {
                  const rowSum = row.reduce((s, v) => s + v, 0);
                  return (
                    <div key={`row-${rowIdx}`} className="flex items-center gap-1">
                      <div className="w-20 text-[9px] font-bold text-slate-400 uppercase text-right pr-2 truncate">
                        {metricsEmotions[rowIdx]}
                      </div>
                      {row.map((val, colIdx) => {
                        const cellPercent = rowSum > 0 ? (val / rowSum) : 0;
                        // Map percentages to opacity colors
                        const opacityStyle = cellPercent > 0.05 ? { backgroundColor: `rgba(59, 130, 246, ${Math.max(0.1, cellPercent)})` } : {};
                        return (
                          <div
                            key={`cell-${rowIdx}-${colIdx}`}
                            style={opacityStyle}
                            className={`w-12 h-10 flex flex-col justify-center items-center rounded border border-slate-800/40 text-[10px] font-bold font-mono text-slate-200 transition-colors hover:border-slate-600 ${
                              cellPercent > 0.4 ? 'text-white' : 'text-slate-400'
                            } ${cellPercent === 0 ? 'bg-slate-900/30' : ''}`}
                            title={`Actual: ${metricsEmotions[rowIdx]}, Predicted: ${metricsEmotions[colIdx]} | ${val} images (${(cellPercent * 100).toFixed(0)}%)`}
                          >
                            <span>{val}</span>
                            <span className="text-[8px] font-normal opacity-70">{(cellPercent * 100).toFixed(0)}%</span>
                          </div>
                        );
                      })}
                    </div>
                  );
                })}
              </div>
            </div>
          ) : (
            <div className="flex items-center justify-center p-12 text-xs text-slate-650">
              No confusion matrix available. Complete a model training run to generate accuracy reports.
            </div>
          )}
        </div>

        {/* Scrollable Session logs */}
        <div className="glass-panel rounded-2xl p-5 flex flex-col h-[320px]">
          <div className="flex justify-between items-center mb-4">
            <div className="flex items-center gap-2">
              <History className="w-5 h-5 text-slate-400" />
              <h3 className="font-semibold text-sm text-slate-200">Session Log</h3>
            </div>
            {session_history?.length > 0 && (
              <button 
                onClick={handleClearHistory}
                className="text-slate-500 hover:text-rose-450 p-1 rounded hover:bg-slate-800/40 transition-colors"
                title="Clear Session History"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            )}
          </div>
          
          <div className="flex-1 overflow-y-auto pr-1 flex flex-col gap-2.5 text-xs">
            {session_history && session_history.length > 0 ? (
              [...session_history].reverse().map((item, idx) => (
                <div key={`log-${idx}`} className="flex justify-between items-center p-2.5 rounded-lg bg-slate-950/40 border border-slate-850 hover:border-slate-800 transition-colors">
                  <div className="flex flex-col gap-0.5">
                    <span className="font-medium text-slate-350 text-[10px]">
                      {new Date(item.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                    </span>
                    <span className="font-semibold text-slate-100 uppercase text-[11px] tracking-wider" style={{ color: EMOTION_COLORS[item.emotion] }}>
                      {item.emotion}
                    </span>
                  </div>
                  <div className="text-right">
                    <span className="font-mono font-bold text-brand-400 bg-brand-950/30 border border-brand-900/30 px-2 py-0.5 rounded text-[10px]">
                      {Math.round(item.confidence * 100)}%
                    </span>
                  </div>
                </div>
              ))
            ) : (
              <div className="h-full flex items-center justify-center text-slate-650">Empty log</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
