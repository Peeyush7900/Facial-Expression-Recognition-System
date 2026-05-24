import React, { useState, useRef, useEffect } from 'react';
import { Upload, Image as ImageIcon, Video, Play, Pause, RefreshCw, AlertCircle } from 'lucide-react';

export default function MediaUploader({ onImageAnalyzed, backendUrl, showLandmarks }) {
  const [file, setFile] = useState(null);
  const [fileType, setFileType] = useState(null); // 'image' or 'video'
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [lastAnalysis, setLastAnalysis] = useState(null);
  
  const videoRef = useRef(null);
  const imgRef = useRef(null);
  const canvasRef = useRef(null);
  const analysisInterval = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  const processFile = (selectedFile) => {
    setError(null);
    stopVideoAnalysis();
    clearCanvas();
    setLastAnalysis(null);
    
    const type = selectedFile.type.split('/')[0];
    if (type !== 'image' && type !== 'video') {
      setError("Please upload a valid image or video file.");
      return;
    }
    
    setFile(selectedFile);
    setFileType(type);
    
    const url = URL.createObjectURL(selectedFile);
    setPreviewUrl(url);
    setIsPlaying(false);
    
    // Auto-analyze images
    if (type === 'image') {
      analyzeImageFile(selectedFile);
    }
  };

  // Convert File to base64
  const fileToBase64 = (fileObj) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(fileObj);
      reader.onload = () => resolve(reader.result);
      reader.onerror = (err) => reject(err);
    });
  };

  // Analyze single static image
  const analyzeImageFile = async (imageFile) => {
    setLoading(true);
    try {
      const b64 = await fileToBase64(imageFile);
      const res = await fetch(`${backendUrl}/api/analyze-image`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: b64 })
      });
      
      if (!res.ok) throw new Error("Backend response error");
      const data = await res.json();
      setLastAnalysis(data);
      onImageAnalyzed(data);
      
      if (data.face_detected) {
        // Delay drawing slightly to make sure the image is rendered and sized in the layout
        setTimeout(() => drawOverlay(data), 150);
      } else {
        clearCanvas();
      }
    } catch (err) {
      console.error(err);
      setError("Failed to analyze image. Please ensure the backend is running.");
    } finally {
      setLoading(false);
    }
  };

  // Extract a single frame from video and send to backend
  const captureVideoFrame = async () => {
    const video = videoRef.current;
    if (!video || video.paused || video.ended || video.videoWidth === 0) return;

    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    const base64Img = canvas.toDataURL('image/jpeg', 0.65);
    
    try {
      const res = await fetch(`${backendUrl}/api/analyze-image`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: base64Img })
      });
      
      if (res.ok) {
        const data = await res.json();
        setLastAnalysis(data);
        onImageAnalyzed(data);
        if (data.face_detected) {
          drawOverlay(data);
        } else {
          clearCanvas();
        }
      }
    } catch (err) {
      console.error("Frame processing error:", err);
    }
  };

  const startVideoAnalysis = () => {
    if (!videoRef.current) return;
    videoRef.current.play();
    setIsPlaying(true);
    // Poll frames every 400ms for analysis
    analysisInterval.current = setInterval(captureVideoFrame, 400);
  };

  const stopVideoAnalysis = () => {
    if (videoRef.current) {
      videoRef.current.pause();
    }
    setIsPlaying(false);
    if (analysisInterval.current) {
      clearInterval(analysisInterval.current);
      analysisInterval.current = null;
    }
  };

  const toggleVideoPlayback = () => {
    if (isPlaying) {
      stopVideoAnalysis();
    } else {
      startVideoAnalysis();
    }
  };

  const clearCanvas = () => {
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
  };

  const clearMedia = () => {
    stopVideoAnalysis();
    setFile(null);
    setFileType(null);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setPreviewUrl(null);
    setError(null);
    setLastAnalysis(null);
    clearCanvas();
  };

  // Draw landmarks and bounding box on canvas
  const drawOverlay = (data) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!data || !data.face_detected) return;

    let nativeWidth = 0;
    let nativeHeight = 0;

    if (fileType === 'image') {
      const img = imgRef.current;
      if (img) {
        nativeWidth = img.naturalWidth;
        nativeHeight = img.naturalHeight;
      }
    } else {
      const video = videoRef.current;
      if (video) {
        nativeWidth = video.videoWidth;
        nativeHeight = video.videoHeight;
      }
    }

    if (nativeWidth > 0) {
      canvas.width = nativeWidth;
      canvas.height = nativeHeight;
    } else {
      return; // Wait for media dimensions to be available
    }

    const [xmin, ymin, w, h] = data.bbox;

    // Draw Face Bounding Box
    ctx.strokeStyle = '#10b981'; // Emerald 500
    ctx.lineWidth = Math.max(2, Math.round(nativeWidth / 300));
    ctx.shadowBlur = 10;
    ctx.shadowColor = '#10b981';
    ctx.strokeRect(xmin, ymin, w, h);
    
    // Bounding Box Label
    ctx.fillStyle = 'rgba(16, 185, 129, 0.9)';
    ctx.shadowBlur = 0;
    const text = `${data.predicted_emotion.toUpperCase()} (${(data.confidence * 100).toFixed(0)}%)`;
    const labelFontSize = Math.max(12, Math.round(nativeWidth / 50));
    ctx.font = `bold ${labelFontSize}px monospace`;
    const textWidth = ctx.measureText(text).width;
    const padding = labelFontSize * 0.8;
    ctx.fillRect(xmin, ymin - padding - 6, textWidth + 16, padding + 6);
    
    ctx.fillStyle = '#ffffff';
    ctx.fillText(text, xmin + 8, ymin - 8);

    // Feature drawing helper
    const drawFeaturePath = (points, color, isClosed = false) => {
      if (!points || points.length === 0) return;
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = Math.max(1.5, Math.round(nativeWidth / 400));
      ctx.shadowBlur = 4;
      ctx.shadowColor = color;
      
      ctx.beginPath();
      ctx.moveTo(points[0].px, points[0].py);
      for (let i = 1; i < points.length; i++) {
        ctx.lineTo(points[i].px, points[i].py);
      }
      if (isClosed) {
        ctx.closePath();
      }
      ctx.stroke();

      // Draw minor dots
      ctx.shadowBlur = 0;
      const dotRadius = Math.max(1, Math.round(nativeWidth / 600));
      points.forEach(pt => {
        ctx.beginPath();
        ctx.arc(pt.px, pt.py, dotRadius, 0, 2 * Math.PI);
        ctx.fill();
      });
    };

    // Draw landmark features in custom neon colors if enabled
    if (showLandmarks && data.landmarks) {
      const landmarks = data.landmarks;
      drawFeaturePath(landmarks.left_eye, '#3b82f6', true);       // Blue
      drawFeaturePath(landmarks.right_eye, '#3b82f6', true);      // Blue
      drawFeaturePath(landmarks.left_eyebrow, '#60a5fa', false);  // Light Blue
      drawFeaturePath(landmarks.right_eyebrow, '#60a5fa', false); // Light Blue
      drawFeaturePath(landmarks.lips, '#f43f5e', true);          // Rose/Red
      drawFeaturePath(landmarks.jawline, '#eab308', false);       // Yellow
    }
  };

  // Redraw when landmark visibility changes
  useEffect(() => {
    if (lastAnalysis) {
      drawOverlay(lastAnalysis);
    }
  }, [showLandmarks]);

  // Clean up timers
  useEffect(() => {
    return () => {
      if (analysisInterval.current) {
        clearInterval(analysisInterval.current);
      }
    };
  }, []);

  return (
    <div className="flex flex-col items-center w-full">
      {/* Upload Zone / Preview Area */}
      <div 
        className={`relative w-full max-w-2xl aspect-[4/3] rounded-2xl border-2 border-dashed flex flex-col items-center justify-center transition-all overflow-hidden ${
          previewUrl 
            ? 'border-slate-800 bg-slate-950/40' 
            : dragActive
              ? 'border-brand-500 bg-brand-500/5'
              : 'border-slate-800 hover:border-slate-700 bg-slate-950/20'
        }`}
        onDragEnter={handleDrag}
        onDragOver={handleDrag}
        onDragLeave={handleDrag}
        onDrop={handleDrop}
      >
        {!previewUrl ? (
          <label className="flex flex-col items-center justify-center p-6 text-center cursor-pointer w-full h-full">
            <input 
              type="file" 
              className="hidden" 
              accept="image/*,video/*"
              onChange={handleFileInput}
            />
            <div className="w-16 h-16 rounded-2xl bg-slate-900 border border-slate-800 flex items-center justify-center text-brand-400 mb-4 shadow-xl">
              <Upload className="w-8 h-8" />
            </div>
            <p className="text-lg font-medium text-slate-200">Drag & Drop file here</p>
            <p className="text-sm text-slate-500 mt-1">or browse files from your computer</p>
            <div className="flex gap-4 mt-6 text-xs font-semibold text-slate-400">
              <span className="flex items-center gap-1"><ImageIcon className="w-4 h-4 text-brand-500" /> JPG, PNG, WEBP</span>
              <span className="flex items-center gap-1"><Video className="w-4 h-4 text-emerald-500" /> MP4, WEBM</span>
            </div>
          </label>
        ) : (
          <div className="relative max-w-full max-h-full flex items-center justify-center bg-dark-950">
            <div className="relative">
              {fileType === 'image' ? (
                <img 
                  ref={imgRef}
                  src={previewUrl} 
                  alt="Preview" 
                  className="max-w-full max-h-[50vh] object-contain rounded-lg"
                  onLoad={() => {
                    if (lastAnalysis) {
                      drawOverlay(lastAnalysis);
                    }
                  }}
                />
              ) : (
                <video
                  ref={videoRef}
                  src={previewUrl}
                  loop
                  playsInline
                  className="max-w-full max-h-[50vh] object-contain rounded-lg"
                  onClick={toggleVideoPlayback}
                  onLoadedMetadata={() => {
                    if (lastAnalysis) {
                      drawOverlay(lastAnalysis);
                    }
                  }}
                />
              )}
              
              {/* Canvas Overlay for bounding box and landmarks */}
              <canvas
                ref={canvasRef}
                className="absolute top-0 left-0 w-full h-full pointer-events-none"
              />
            </div>
            
            {/* Image loading indicator */}
            {loading && (
              <div className="absolute inset-0 bg-slate-950/70 backdrop-blur-sm flex flex-col items-center justify-center rounded-lg">
                <RefreshCw className="w-12 h-12 text-brand-500 animate-spin mb-4" />
                <p className="text-sm font-medium text-slate-350">Analyzing facial features...</p>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Media controls */}
      {previewUrl && (
        <div className="flex items-center gap-4 mt-6">
          {fileType === 'video' && (
            <button
              onClick={toggleVideoPlayback}
              className={`flex items-center gap-2 px-5 py-2.5 rounded-xl font-medium transition-all shadow-md ${
                isPlaying 
                  ? 'bg-amber-600 hover:bg-amber-700 text-white shadow-amber-600/10' 
                  : 'bg-emerald-600 hover:bg-emerald-700 text-white shadow-emerald-600/10'
              }`}
            >
              {isPlaying ? (
                <>
                  <Pause className="w-4 h-4 fill-white" />
                  Pause Analysis
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 fill-white" />
                  Run Video Analysis
                </>
              )}
            </button>
          )}

          <button
            onClick={clearMedia}
            className="flex items-center gap-2 px-5 py-2.5 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 rounded-xl font-medium transition-all"
          >
            Clear Media
          </button>
        </div>
      )}

      {error && (
        <div className="flex items-center gap-2 mt-4 text-rose-450 text-sm font-medium bg-rose-950/20 border border-rose-900/30 px-4 py-2 rounded-lg">
          <AlertCircle className="w-4 h-4" />
          <span>{error}</span>
        </div>
      )}
    </div>
  );
}
