import React, { useState, useEffect } from 'react';
import WebcamViewer from './WebcamViewer';
import MediaUploader from './MediaUploader';
import AnalyticsPanel from './AnalyticsPanel';
import Explainability from './Explainability';
import ModelMonitor from './ModelMonitor';
import ResearchPanel from './ResearchPanel';
import EvaluationTab from './EvaluationTab';
import SystemArchitecture from './SystemArchitecture';
import { Camera, Upload, BarChart2, Cpu, Activity, ShieldCheck, ShieldAlert, Target, Network, Database, BookOpen, Download } from 'lucide-react';
import { jsPDF } from 'jspdf';

const BACKEND_URL = typeof window !== 'undefined' && (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
  ? 'http://localhost:8000'
  : (typeof window !== 'undefined' ? `http://${window.location.hostname}:8000` : 'http://localhost:8000');
const EMOJIS = {
  angry: '😡',
  disgust: '🤢',
  fear: '😨',
  happy: '😊',
  sad: '😢',
  surprise: '😲',
  neutral: '😐'
};

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState('analysis'); // analysis, analytics, evaluation, architecture, training
  const [inputMode, setInputMode] = useState('webcam'); // webcam, upload
  const [currentAnalysis, setCurrentAnalysis] = useState(null);
  const [backendConnected, setBackendConnected] = useState(false);
  const [researchMode, setResearchMode] = useState(false);
  const [showLandmarks, setShowLandmarks] = useState(true);
  const [localHistory, setLocalHistory] = useState([]); // Cap at 100 frames
  const [showDatasetInfo, setShowDatasetInfo] = useState(false);

  const [analyticsData, setAnalyticsData] = useState({
    session_history: [],
    session_distribution: {},
    session_avg_confidence: 0,
    trained_metrics: {},
    model_performance: null
  });

  // Check backend health
  const checkBackendHealth = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/analytics`);
      if (res.ok) {
        setBackendConnected(true);
      } else {
        setBackendConnected(false);
      }
    } catch (err) {
      setBackendConnected(false);
    }
  };

  // Fetch analytics metrics
  const fetchAnalytics = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/analytics`);
      if (res.ok) {
        const data = await res.json();
        setAnalyticsData(data);
      }
    } catch (err) {
      console.error("Failed to fetch analytics:", err);
    }
  };

  // Handle analysis update from camera or file uploader
  const handleAnalysisUpdate = (data) => {
    setCurrentAnalysis(data);
    if (data.face_detected) {
      // Add to local history (limit to last 100 entries)
      setLocalHistory(prev => {
        const updated = [{
          timestamp: data.timestamp,
          emotion: data.predicted_emotion,
          confidence: data.confidence,
          pose: data.pose
        }, ...prev];
        return updated.slice(0, 100);
      });
      fetchAnalytics();
    }
  };

  // Clear session locally
  const handleClearSession = () => {
    setCurrentAnalysis(null);
    setLocalHistory([]);
    fetchAnalytics();
  };

  // CSV Export
  const exportCSVHistory = () => {
    if (localHistory.length === 0) return;
    const headers = "Timestamp,Emotion,Confidence,Yaw,Pitch,Roll\n";
    const rows = localHistory.map(item => 
      `"${item.timestamp}","${item.emotion}",${item.confidence},${item.pose?.yaw.toFixed(2) || 0},${item.pose?.pitch.toFixed(2) || 0},${item.pose?.roll.toFixed(2) || 0}`
    ).join("\n");
    
    const blob = new Blob([headers + rows], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `FER_Prediction_History_${Date.now()}.csv`;
    link.click();
  };

  // JSON Export
  const exportJSONHistory = () => {
    if (localHistory.length === 0) return;
    const blob = new Blob([JSON.stringify(localHistory, null, 2)], { type: 'application/json' });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `FER_Prediction_History_${Date.now()}.json`;
    link.click();
  };

  // PDF Report Export
  const exportPDFReport = () => {
    if (!currentAnalysis) return;
    const doc = new jsPDF();
    
    // Header styling
    doc.setFont("Helvetica", "bold");
    doc.setFontSize(22);
    doc.setTextColor(15, 23, 42); // slate-900
    doc.text("AI FACIAL EXPRESSION RECOGNITION SYSTEM", 15, 25);
    doc.setFontSize(10);
    doc.setFont("Helvetica", "normal");
    doc.setTextColor(100, 116, 139); // slate-500
    doc.text("Semester CV Project Evaluation & Research Report", 15, 30);
    
    // Divider
    doc.setDrawColor(226, 232, 240); // slate-200
    doc.setLineWidth(0.5);
    doc.line(15, 35, 195, 35);
    
    // Summary Card Box
    doc.setFillColor(248, 250, 252); // slate-50
    doc.rect(15, 42, 180, 52, 'F');
    
    doc.setFont("Helvetica", "bold");
    doc.setFontSize(12);
    doc.setTextColor(30, 41, 59); // slate-800
    doc.text("1. CLASSIFICATION & POSE ESTIMATION SUMMARY", 20, 49);
    
    doc.setFont("Helvetica", "normal");
    doc.setFontSize(10);
    doc.text(`Detected Emotion: ${currentAnalysis.predicted_emotion.toUpperCase()}`, 22, 57);
    doc.text(`Model Confidence: ${(currentAnalysis.confidence * 100).toFixed(1)}%`, 22, 63);
    doc.text(`Primary Grad-CAM Region: ${currentAnalysis.primary_attention_region.toUpperCase()}`, 22, 69);
    doc.text(`Head Yaw Rotation: ${currentAnalysis.pose?.yaw.toFixed(1)}°`, 22, 75);
    doc.text(`Head Pitch Rotation: ${currentAnalysis.pose?.pitch.toFixed(1)}°`, 22, 81);
    doc.text(`Head Roll Tilt: ${currentAnalysis.pose?.roll.toFixed(1)}°`, 22, 87);
    
    // Action Units description
    doc.setFont("Helvetica", "bold");
    doc.setFontSize(12);
    doc.text("2. FACS ACTION UNIT DETAILED METRICS", 20, 110);
    doc.setFont("Helvetica", "normal");
    doc.setFontSize(10);
    doc.text("Facial Action Coding System (FACS) parameters mapped geometrically:", 20, 117);
    
    doc.text("- AU1 (Inner Brow Raiser): calculated via medial eyebrows raise displacement", 25, 125);
    doc.text("- AU2 (Outer Brow Raiser): calculated via lateral eyebrows raise displacement", 25, 131);
    doc.text("- AU4 (Brow Lowerer): calculated via eyebrows draw-down furrowing", 25, 137);
    doc.text("- AU12 (Lip Corner Puller): calculated via outer mouth curvature (smile scale)", 25, 143);
    doc.text("- AU25 (Lips Part) / AU26 (Jaw Drop): calculated via jaw vertical separation", 25, 149);
    
    // System metadata
    doc.line(15, 160, 195, 160);
    doc.setFontSize(8);
    doc.setTextColor(148, 163, 184); // slate-400
    doc.text(`Timestamp: ${new Date(currentAnalysis.timestamp).toLocaleString()}`, 15, 168);
    doc.text("System architecture utilizes a custom 3-stage EmotionCNN with 256 conv layers trained on FER-2013.", 15, 173);
    doc.text("Grad-CAM overlays and landmarks connections drawn locally on Canvas elements.", 15, 178);

    doc.save(`CV_FER_Evaluation_Report_${currentAnalysis.predicted_emotion}.pdf`);
  };

  // Heartbeat loop
  useEffect(() => {
    checkBackendHealth();
    fetchAnalytics();
    const interval = setInterval(checkBackendHealth, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-dark-950 flex flex-col">
      {/* Top Header Bar */}
      <header className="glass-panel border-b border-slate-800/80 px-6 py-4 flex items-center justify-between sticky top-0 z-50">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-brand-600 flex items-center justify-center text-white shadow-lg shadow-brand-500/20 animate-pulse-glow">
            <Activity className="w-5 h-5" />
          </div>
          <div className="flex flex-col">
            <h1 className="text-sm font-bold tracking-wider text-slate-100 uppercase font-mono">Facial Expression Recognition</h1>
            <span className="text-[10px] text-slate-400 font-medium">B.Tech Final Semester Evaluation Demonstration</span>
          </div>
        </div>

        {/* Server Connection & Controls */}
        <div className="flex items-center gap-4">
          <button 
            onClick={() => setShowDatasetInfo(!showDatasetInfo)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-900 border border-slate-800 hover:bg-slate-850 text-slate-300 transition-colors"
          >
            <Database className="w-3.5 h-3.5" />
            Dataset Info
          </button>

          <div className={`flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium border ${
            backendConnected 
              ? 'bg-emerald-950/20 text-emerald-450 border-emerald-900/30' 
              : 'bg-rose-950/20 text-rose-450 border-rose-900/30'
          }`}>
            {backendConnected ? (
              <>
                <ShieldCheck className="w-4 h-4 text-emerald-400" />
                Backend Connected
              </>
            ) : (
              <>
                <ShieldAlert className="w-4 h-4 text-rose-500 animate-bounce" />
                Server Offline
              </>
            )}
          </div>
        </div>
      </header>

      {/* Dataset Info Modal Overlay */}
      {showDatasetInfo && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-dark-950/80 backdrop-blur-md p-4">
          <div className="glass-panel max-w-lg w-full rounded-2xl p-6 relative">
            <h3 className="font-bold text-sm text-slate-200 uppercase tracking-wider mb-4 flex items-center gap-2">
              <Database className="w-5 h-5 text-brand-500" /> FER-2013 Dataset Information
            </h3>
            
            <div className="flex flex-col gap-3.5 text-xs text-slate-400 mb-6">
              <div className="flex justify-between border-b border-slate-900 pb-1.5">
                <span className="font-medium text-slate-500">Dataset Name:</span>
                <span className="font-mono text-slate-300">Facial Expression Recognition (FER-2013)</span>
              </div>
              <div className="flex justify-between border-b border-slate-900 pb-1.5">
                <span className="font-medium text-slate-500">Origin:</span>
                <span className="text-slate-300">ICML 2013 Challenge (hosted on Kaggle)</span>
              </div>
              <div className="flex justify-between border-b border-slate-900 pb-1.5">
                <span className="font-medium text-slate-500">Image Resolution:</span>
                <span className="font-mono text-slate-300">48 x 48 Pixels (Grayscale)</span>
              </div>
              <div className="flex justify-between border-b border-slate-900 pb-1.5">
                <span className="font-medium text-slate-500">Emotion Classes:</span>
                <span className="font-mono text-slate-300">7 classes (Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral)</span>
              </div>
              <div className="flex justify-between border-b border-slate-900 pb-1.5">
                <span className="font-medium text-slate-500">Training Samples:</span>
                <span className="font-mono text-slate-300">28,709 images</span>
              </div>
              <div className="flex justify-between border-b border-slate-900 pb-1.5">
                <span className="font-medium text-slate-500">Validation / Test Samples:</span>
                <span className="font-mono text-slate-300">3,589 validation / 3,589 test images</span>
              </div>
            </div>

            <button
              onClick={() => setShowDatasetInfo(false)}
              className="w-full py-2.5 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 rounded-xl text-xs font-semibold transition-colors"
            >
              Close Panel
            </button>
          </div>
        </div>
      )}

      {/* Main Tab Navigation */}
      <div className="max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 flex-1 flex flex-col gap-6">
        <div className="flex border-b border-slate-800/60 pb-px">
          <nav className="flex gap-6 text-sm font-semibold">
            <button
              onClick={() => setActiveTab('analysis')}
              className={`pb-4 px-1 border-b-2 transition-all flex items-center gap-2 ${
                activeTab === 'analysis'
                  ? 'border-brand-500 text-brand-400'
                  : 'border-transparent text-slate-400 hover:text-slate-200'
              }`}
            >
              <Camera className="w-4 h-4" />
              Active Analysis
            </button>
            <button
              onClick={() => setActiveTab('analytics')}
              className={`pb-4 px-1 border-b-2 transition-all flex items-center gap-2 ${
                activeTab === 'analytics'
                  ? 'border-brand-500 text-brand-400'
                  : 'border-transparent text-slate-400 hover:text-slate-200'
              }`}
            >
              <BarChart2 className="w-4 h-4" />
              Analytics Dashboard
            </button>
            <button
              onClick={() => setActiveTab('evaluation')}
              className={`pb-4 px-1 border-b-2 transition-all flex items-center gap-2 ${
                activeTab === 'evaluation'
                  ? 'border-brand-500 text-brand-400'
                  : 'border-transparent text-slate-400 hover:text-slate-200'
              }`}
            >
              <Target className="w-4 h-4" />
              Model Performance
            </button>
            <button
              onClick={() => setActiveTab('architecture')}
              className={`pb-4 px-1 border-b-2 transition-all flex items-center gap-2 ${
                activeTab === 'architecture'
                  ? 'border-brand-500 text-brand-400'
                  : 'border-transparent text-slate-400 hover:text-slate-200'
              }`}
            >
              <Network className="w-4 h-4" />
              System Architecture
            </button>
            <button
              onClick={() => setActiveTab('training')}
              className={`pb-4 px-1 border-b-2 transition-all flex items-center gap-2 ${
                activeTab === 'training'
                  ? 'border-brand-500 text-brand-400'
                  : 'border-transparent text-slate-400 hover:text-slate-200'
              }`}
            >
              <Cpu className="w-4 h-4" />
              Training Monitor
            </button>
          </nav>
        </div>

        {/* Tab Panel Content */}
        <div className="flex-1">
          {activeTab === 'analysis' && (
            <div className="flex flex-col gap-8">
              {/* Toggles bar */}
              <div className="glass-panel rounded-2xl p-4 flex flex-col sm:flex-row justify-between items-center gap-4">
                <div className="flex items-center gap-4">
                  <div className="flex items-center gap-2">
                    <input 
                      type="checkbox"
                      id="toggleLandmarks"
                      checked={showLandmarks}
                      onChange={(e) => setShowLandmarks(e.target.checked)}
                      className="w-4 h-4 rounded border-slate-800 text-brand-600 bg-slate-950 focus:ring-brand-500"
                    />
                    <label htmlFor="toggleLandmarks" className="text-xs font-semibold text-slate-350 cursor-pointer">
                      Show Mesh Overlays
                    </label>
                  </div>
                  
                  <div className="flex items-center gap-2">
                    <input 
                      type="checkbox"
                      id="toggleResearch"
                      checked={researchMode}
                      onChange={(e) => setResearchMode(e.target.checked)}
                      className="w-4 h-4 rounded border-slate-800 text-emerald-600 bg-slate-950 focus:ring-emerald-500"
                    />
                    <label htmlFor="toggleResearch" className="text-xs font-semibold text-slate-350 cursor-pointer flex items-center gap-1">
                      <BookOpen className="w-3.5 h-3.5 text-emerald-500" /> Research Mode
                    </label>
                  </div>
                </div>

                {/* Exporters for prediction history */}
                {localHistory.length > 0 && (
                  <div className="flex items-center gap-2">
                    <button 
                      onClick={exportCSVHistory}
                      className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-[10px] font-semibold bg-slate-900 border border-slate-800 hover:bg-slate-850 text-slate-300 transition-colors"
                    >
                      <Download className="w-3 h-3" /> CSV History
                    </button>
                    <button 
                      onClick={exportJSONHistory}
                      className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-[10px] font-semibold bg-slate-900 border border-slate-800 hover:bg-slate-850 text-slate-300 transition-colors"
                    >
                      <Download className="w-3 h-3" /> JSON History
                    </button>
                    {currentAnalysis && (
                      <button 
                        onClick={exportPDFReport}
                        className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-[10px] font-semibold bg-brand-950 border border-brand-900 hover:bg-brand-900/40 text-brand-400 transition-colors"
                      >
                        <Download className="w-3 h-3" /> PDF Report
                      </button>
                    )}
                  </div>
                )}
              </div>

              {/* Side-by-side feed grid */}
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
                {/* Left Column - Input Source */}
                <div className="lg:col-span-7 flex flex-col gap-6">
                  {/* Real-time emotion prediction panel */}
                  {currentAnalysis && currentAnalysis.face_detected && (
                    <div className="glass-panel rounded-2xl p-5 flex items-center justify-between border-brand-500/20 animate-pulse-glow">
                      <div className="flex items-center gap-4">
                        <span className="text-4xl leading-none">{EMOJIS[currentAnalysis.predicted_emotion] || '😐'}</span>
                        <div className="flex flex-col">
                          <span className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider">Current Emotion</span>
                          <span className="text-2xl font-black uppercase text-slate-100 tracking-wide font-mono">
                            {currentAnalysis.predicted_emotion}
                          </span>
                        </div>
                      </div>
                      <div className="flex gap-8 text-right">
                        <div className="flex flex-col">
                          <span className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider">Confidence</span>
                          <span className="text-xl font-bold font-mono text-brand-400">
                            {(currentAnalysis.confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="flex flex-col">
                          <span className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider">Time</span>
                          <span className="text-xl font-bold font-mono text-slate-350">
                            {new Date(currentAnalysis.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                          </span>
                        </div>
                      </div>
                    </div>
                  )}

                  <div className="glass-panel rounded-2xl p-5">
                    <div className="flex justify-between items-center mb-6">
                      <h2 className="font-semibold text-sm text-slate-200 uppercase tracking-wider">Input Feed Stream</h2>
                      <div className="flex bg-slate-900 border border-slate-800 rounded-lg p-0.5">
                        <button
                          onClick={() => { setInputMode('webcam'); setCurrentAnalysis(null); }}
                          className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded-md text-xs font-semibold transition-all ${
                            inputMode === 'webcam'
                              ? 'bg-brand-600 text-white shadow-md'
                              : 'text-slate-400 hover:text-slate-200'
                          }`}
                        >
                          <Camera className="w-3.5 h-3.5" />
                          Webcam
                        </button>
                        <button
                          onClick={() => { setInputMode('upload'); setCurrentAnalysis(null); }}
                          className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded-md text-xs font-semibold transition-all ${
                            inputMode === 'upload'
                              ? 'bg-brand-600 text-white shadow-md'
                              : 'text-slate-400 hover:text-slate-200'
                          }`}
                        >
                          <Upload className="w-3.5 h-3.5" />
                          Upload
                        </button>
                      </div>
                    </div>

                    {inputMode === 'webcam' ? (
                      <WebcamViewer onFrameAnalyzed={handleAnalysisUpdate} backendUrl={BACKEND_URL} showLandmarks={showLandmarks} />
                    ) : (
                      <MediaUploader onImageAnalyzed={handleAnalysisUpdate} backendUrl={BACKEND_URL} showLandmarks={showLandmarks} />
                    )}
                  </div>
                </div>

                {/* Right Column - Explainability */}
                <div className="lg:col-span-5 flex flex-col h-full justify-between">
                  <div className="h-full">
                    <Explainability data={currentAnalysis} />
                  </div>
                </div>
              </div>

              {/* Research Mode Panel (Full Width Below) */}
              {researchMode && (
                <div className="w-full">
                  <ResearchPanel data={currentAnalysis} />
                </div>
              )}
            </div>
          )}

          {activeTab === 'analytics' && (
            <AnalyticsPanel
              analyticsData={analyticsData}
              onClearSession={handleClearSession}
              backendUrl={BACKEND_URL}
            />
          )}

          {activeTab === 'evaluation' && (
            <EvaluationTab analyticsData={analyticsData} />
          )}

          {activeTab === 'architecture' && (
            <SystemArchitecture />
          )}

          {activeTab === 'training' && (
            <ModelMonitor 
              backendUrl={BACKEND_URL} 
              onTrainingComplete={handleClearSession} 
            />
          )}
        </div>
      </div>
    </div>
  );
}
