import React, { useRef, useEffect, useState } from 'react';
import { Camera, CameraOff, AlertCircle } from 'lucide-react';

export default function WebcamViewer({ onFrameAnalyzed, backendUrl, showLandmarks }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [isActive, setIsActive] = useState(false);
  const [error, setError] = useState(null);
  const [fps, setFps] = useState(0);
  
  let frameIntervalId = useRef(null);
  let lastFrameTime = useRef(0);

  // Toggle webcam
  const toggleWebcam = async () => {
    if (isActive) {
      stopWebcam();
    } else {
      await startWebcam();
    }
  };

  const startWebcam = async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'user' },
        audio: false
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsActive(true);
      }
    } catch (err) {
      console.error("Webcam access error:", err);
      setError("Unable to access camera. Please check permissions.");
    }
  };

  const stopWebcam = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks();
      tracks.forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    setIsActive(false);
    clearCanvas();
    if (frameIntervalId.current) {
      clearInterval(frameIntervalId.current);
    }
  };

  // Clear overlay canvas
  const clearCanvas = () => {
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
  };

  // Capture frame and send to backend
  const captureAndAnalyze = async () => {
    if (!videoRef.current || !canvasRef.current || !isActive || videoRef.current.videoWidth === 0) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    // Sync canvas sizes with video stream size
    if (canvas.width !== video.videoWidth) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }

    // Create offscreen canvas for resizing and base64 export
    const offscreen = document.createElement('canvas');
    offscreen.width = video.videoWidth;
    offscreen.height = video.videoHeight;
    const oCtx = offscreen.getContext('2d');
    oCtx.drawImage(video, 0, 0, offscreen.width, offscreen.height);
    
    // Get base64 string
    const base64Img = offscreen.toDataURL('image/jpeg', 0.7);

    try {
      const startTime = performance.now();
      const res = await fetch(`${backendUrl}/api/analyze-image`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: base64Img })
      });
      
      if (!res.ok) throw new Error("API analysis failed");
      const data = await res.json();
      
      const endTime = performance.now();
      const currentFps = Math.round(1000 / (endTime - lastFrameTime.current));
      lastFrameTime.current = endTime;
      setFps(currentFps > 60 ? 60 : currentFps);
      
      if (data.face_detected) {
        drawOverlay(data);
        onFrameAnalyzed(data);
      } else {
        clearCanvas();
        onFrameAnalyzed({ face_detected: false });
      }
    } catch (err) {
      console.error("Frame analysis error:", err);
    }
  };

  // Draw landmarks and bounding box
  const drawOverlay = (data) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const [xmin, ymin, w, h] = data.bbox;

    // Draw Face Bounding Box (glowing green/blue)
    ctx.strokeStyle = '#10b981'; // Emerald 500
    ctx.lineWidth = 3;
    ctx.shadowBlur = 10;
    ctx.shadowColor = '#10b981';
    ctx.strokeRect(xmin, ymin, w, h);
    
    // Bounding Box Label
    ctx.fillStyle = 'rgba(16, 185, 129, 0.9)';
    ctx.shadowBlur = 0;
    const text = `${data.predicted_emotion.toUpperCase()} (${(data.confidence * 100).toFixed(0)}%)`;
    ctx.font = 'bold 14px monospace';
    const textWidth = ctx.measureText(text).width;
    ctx.fillRect(xmin, ymin - 25, textWidth + 16, 25);
    
    ctx.fillStyle = '#ffffff';
    ctx.fillText(text, xmin + 8, ymin - 7);

    // Feature drawing helper
    const drawFeaturePath = (points, color, isClosed = false) => {
      if (!points || points.length === 0) return;
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = 2;
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
      points.forEach(pt => {
        ctx.beginPath();
        ctx.arc(pt.px, pt.py, 1.5, 0, 2 * Math.PI);
        ctx.fill();
      });
    };

    // Draw landmark features in custom neon colors if enabled
    if (showLandmarks) {
      const landmarks = data.landmarks;
      drawFeaturePath(landmarks.left_eye, '#3b82f6', true);       // Blue
      drawFeaturePath(landmarks.right_eye, '#3b82f6', true);      // Blue
      drawFeaturePath(landmarks.left_eyebrow, '#60a5fa', false);  // Light Blue
      drawFeaturePath(landmarks.right_eyebrow, '#60a5fa', false); // Light Blue
      drawFeaturePath(landmarks.lips, '#f43f5e', true);          // Rose/Red
      drawFeaturePath(landmarks.jawline, '#eab308', false);       // Yellow
    }
  };

  useEffect(() => {
    if (isActive) {
      lastFrameTime.current = performance.now();
      frameIntervalId.current = setInterval(captureAndAnalyze, 150); // Polling ~6.6 frames per second
    } else {
      if (frameIntervalId.current) {
        clearInterval(frameIntervalId.current);
      }
    }
    return () => {
      if (frameIntervalId.current) {
        clearInterval(frameIntervalId.current);
      }
    };
  }, [isActive]);

  // Clean up on unmount
  useEffect(() => {
    return () => stopWebcam();
  }, []);

  return (
    <div className="relative flex flex-col items-center w-full h-full">
      <div className="relative w-full max-w-2xl aspect-[4/3] rounded-2xl overflow-hidden bg-slate-950 border border-slate-800 shadow-2xl">
        {/* Video feed */}
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className={`w-full h-full object-cover transform -scale-x-100 ${isActive ? 'block' : 'hidden'}`}
        />
        
        {/* Landmark Canvas Overlay */}
        <canvas
          ref={canvasRef}
          className="absolute top-0 left-0 w-full h-full pointer-events-none transform -scale-x-100"
        />

        {/* Video Scanner effect */}
        {isActive && <div className="scanner-line pointer-events-none" />}

        {/* Camera Inactive Placeholder */}
        {!isActive && (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-500 bg-slate-950/90 z-10 p-6 text-center">
            <CameraOff className="w-16 h-16 mb-4 text-slate-700 animate-pulse" />
            <p className="text-lg font-medium text-slate-350">Camera Stream Offline</p>
            <p className="text-sm text-slate-500 mt-1 max-w-md">
              Start the camera stream to analyze your expressions in real-time and interpret non-manual facial cues.
            </p>
          </div>
        )}

        {/* Floating FPS/Status indicator */}
        {isActive && (
          <div className="absolute top-4 right-4 bg-slate-900/80 backdrop-blur border border-slate-750 px-3 py-1 rounded-full text-xs font-mono text-emerald-400 flex items-center gap-1.5 z-20">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
            LIVE | {fps} FPS
          </div>
        )}
      </div>

      {/* Control Buttons */}
      <div className="flex flex-col sm:flex-row items-center gap-4 mt-6 w-full max-w-md justify-center">
        <button
          onClick={toggleWebcam}
          className={`flex items-center gap-2 px-6 py-3 rounded-xl font-medium transition-all shadow-lg ${
            isActive
              ? 'bg-rose-600 hover:bg-rose-700 text-white shadow-rose-600/10'
              : 'bg-brand-600 hover:bg-brand-700 text-white shadow-brand-600/10'
          }`}
        >
          {isActive ? (
            <>
              <CameraOff className="w-5 h-5" />
              Stop Camera
            </>
          ) : (
            <>
              <Camera className="w-5 h-5" />
              Start Live Camera
            </>
          )}
        </button>
      </div>

      {error && (
        <div className="flex items-center gap-2 mt-4 text-rose-400 text-sm font-medium">
          <AlertCircle className="w-4 h-4" />
          <span>{error}</span>
        </div>
      )}
    </div>
  );
}
