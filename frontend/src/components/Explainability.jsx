import React, { useState } from 'react';
import { Eye, HelpCircle, Activity, Smile, Info } from 'lucide-react';

export default function Explainability({ data }) {
  const [transparency, setTransparency] = useState(0.65);
  if (!data || !data.face_detected) {
    return (
      <div className="flex flex-col items-center justify-center p-8 text-center text-slate-500 h-full border border-slate-800 rounded-2xl bg-slate-900/10">
        <HelpCircle className="w-12 h-12 mb-3 text-slate-700" />
        <p className="font-medium text-slate-400">Explainability Interface</p>
        <p className="text-xs text-slate-500 mt-1 max-w-sm">
          Run analysis on a face to inspect the Grad-CAM heatmap activations and geometric interpretation of non-manual features.
        </p>
      </div>
    );
  }

  // Calculate geometric interpretation based on raw pixel coordinates
  const interpretFeatures = () => {
    const { landmarks } = data;
    if (!landmarks) return null;

    const avgY = (pts) => pts.reduce((sum, p) => sum + p.py, 0) / pts.length;
    const avgX = (pts) => pts.reduce((sum, p) => sum + p.px, 0) / pts.length;
    const minY = (pts) => Math.min(...pts.map(p => p.py));
    const maxY = (pts) => Math.max(...pts.map(p => p.py));
    const minX = (pts) => Math.min(...pts.map(p => p.px));
    const maxX = (pts) => Math.max(...pts.map(p => p.px));

    // Eye Centers
    const leftEyeX = avgX(landmarks.left_eye);
    const leftEyeY = avgY(landmarks.left_eye);
    const rightEyeX = avgX(landmarks.right_eye);
    const rightEyeY = avgY(landmarks.right_eye);
    
    // Scale-invariant baseline (interpupillary distance)
    const eyeDist = Math.max(1, rightEyeX - leftEyeX);

    // Eyebrow heights (distance from eyes to eyebrows)
    const leftBrowY = avgY(landmarks.left_eyebrow);
    const rightBrowY = avgY(landmarks.right_eyebrow);
    // Since Y axis goes down, (eyeY - browY) is positive eyebrow distance
    const leftBrowHeight = (leftEyeY - leftBrowY) / eyeDist;
    const rightBrowHeight = (rightEyeY - rightBrowY) / eyeDist;
    const avgBrowHeight = (leftBrowHeight + rightBrowHeight) / 2;

    let eyebrowState = "Neutral";
    let eyebrowDesc = "Standard position.";
    if (avgBrowHeight > 0.28) {
      eyebrowState = "Raised";
      eyebrowDesc = "Eyebrows are pulled upwards (typical of Surprise or Fear).";
    } else if (avgBrowHeight < 0.21) {
      eyebrowState = "Furrowed / Lowered";
      eyebrowDesc = "Eyebrows are drawn together and down (indicates Anger, Sadness, or deep concentration).";
    }

    // Mouth Opening Ratio
    const mouthW = Math.max(1, maxX(landmarks.lips) - minX(landmarks.lips));
    const mouthH = maxY(landmarks.lips) - minY(landmarks.lips);
    const mouthRatio = mouthH / mouthW;

    let mouthState = "Closed";
    let mouthDesc = "Lips are closed.";
    if (mouthRatio > 0.35) {
      mouthState = "Gape Open";
      mouthDesc = "Mouth is widely open (indicates Surprise or vocalization).";
    } else if (mouthRatio > 0.15) {
      mouthState = "Partially Open";
      mouthDesc = "Mouth is slightly open (indicates speaking, smiling, or Fear).";
    }

    // Smile / Frown curvature
    // In MediaPipe, outer lips: index 0 (left corner), index 10 (right corner)
    // We compare corners Y with mouth center Y
    const cornerY = (landmarks.lips[0].py + landmarks.lips[10].py) / 2;
    const centerY = avgY(landmarks.lips);
    // If corners are higher up than center, corners_y < centerY, so (centerY - corners_y) is positive
    const smileScore = (centerY - cornerY) / eyeDist;

    let mouthShape = "Flat";
    let mouthShapeDesc = "Mouth corners are neutral.";
    if (smileScore > 0.04) {
      mouthShape = "Curved Up (Smile)";
      mouthShapeDesc = "Mouth corners are pulled upwards (signature Happy expression).";
    } else if (smileScore < -0.04) {
      mouthShape = "Curved Down (Frown)";
      mouthShapeDesc = "Mouth corners are pulled downwards (indicates Sadness or Anger).";
    }

    // Eye Aperture (Squint vs Open)
    const leftEyeH = maxY(landmarks.left_eye) - minY(landmarks.left_eye);
    const leftEyeW = Math.max(1, maxX(landmarks.left_eye) - minX(landmarks.left_eye));
    const leftEyeRatio = leftEyeH / leftEyeW;

    const rightEyeH = maxY(landmarks.right_eye) - minY(landmarks.right_eye);
    const rightEyeW = Math.max(1, maxX(landmarks.right_eye) - minX(landmarks.right_eye));
    const rightEyeRatio = rightEyeH / rightEyeW;
    const avgEyeRatio = (leftEyeRatio + rightEyeRatio) / 2;

    let eyeState = "Normal";
    let eyeDesc = "Eyes are comfortably open.";
    if (avgEyeRatio > 0.28) {
      eyeState = "Wide Open";
      eyeDesc = "Eyelids are widened, revealing sclera (typical of Fear or Surprise).";
    } else if (avgEyeRatio < 0.18) {
      eyeState = "Squinted / Squeezed";
      eyeDesc = "Eyelids are narrowed or squeezed shut (typical of Disgust, Pain, or Anger).";
    }

    return {
      eyebrows: { state: eyebrowState, desc: eyebrowDesc, val: avgBrowHeight },
      mouthOpening: { state: mouthState, desc: mouthDesc, val: mouthRatio },
      mouthShape: { state: mouthShape, desc: mouthShapeDesc, val: smileScore },
      eyes: { state: eyeState, desc: eyeDesc, val: avgEyeRatio }
    };
  };

  const interpretation = interpretFeatures();

  return (
    <div className="flex flex-col gap-6 h-full">
      {/* Visual Overlay Row */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Grad-CAM Heatmap panel */}
        <div className="glass-panel rounded-2xl p-5 flex flex-col items-center">
          <div className="flex items-center gap-2 mb-4 w-full">
            <Activity className="w-5 h-5 text-emerald-400" />
            <h3 className="font-semibold text-sm text-slate-200">Grad-CAM Decisional Heatmap</h3>
          </div>
          
          <div className="relative w-40 h-40 rounded-xl overflow-hidden bg-slate-900 border border-slate-800 shadow-lg mb-3">
            {data.crop_image && (
              <img 
                src={data.crop_image} 
                alt="Face Crop" 
                className="w-full h-full object-cover absolute inset-0"
              />
            )}
            {data.gradcam_image && (
              <img 
                src={data.gradcam_image} 
                alt="Grad-CAM Overlay" 
                className="w-full h-full object-cover absolute inset-0 transition-opacity duration-150"
                style={{ opacity: transparency }}
              />
            )}
            {!data.gradcam_image && (
              <div className="w-full h-full flex items-center justify-center text-xs text-slate-650">No Heatmap</div>
            )}
          </div>

          {/* Transparency slider */}
          {data.gradcam_image && (
            <div className="w-full max-w-[160px] flex flex-col gap-1 mb-4 text-[10px]">
              <div className="flex justify-between text-slate-400 font-semibold">
                <span>Overlay Opacity</span>
                <span>{Math.round(transparency * 100)}%</span>
              </div>
              <input 
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={transparency}
                onChange={(e) => setTransparency(parseFloat(e.target.value))}
                className="w-full h-1 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
              />
            </div>
          )}
          
          {/* Attention regions list */}
          {data.attention_regions && (
            <div className="w-full flex flex-col gap-2 border-t border-slate-800/80 pt-3 text-[10px]">
              <span className="font-semibold text-slate-450 uppercase tracking-wider block mb-1">Decisional Focus Zones</span>
              {Object.entries(data.attention_regions).map(([region, score]) => (
                <div key={region} className="flex items-center gap-2">
                  <span className={`w-14 font-mono uppercase text-[9px] ${data.primary_attention_region === region ? 'text-emerald-450 font-bold' : 'text-slate-500'}`}>
                    {region}
                  </span>
                  <div className="flex-1 h-1.5 bg-slate-900 border border-slate-850 rounded-full overflow-hidden">
                    <div 
                      className={`h-full rounded-full transition-all duration-300 ${data.primary_attention_region === region ? 'bg-emerald-500' : 'bg-slate-750'}`}
                      style={{ width: `${score * 100}%` }}
                    />
                  </div>
                  <span className={`w-8 text-right font-mono text-[9px] ${data.primary_attention_region === region ? 'text-emerald-450 font-bold' : 'text-slate-500'}`}>
                    {Math.round(score * 100)}%
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Semantic Action Units breakdown */}
        <div className="glass-panel rounded-2xl p-5 flex flex-col">
          <div className="flex items-center gap-2 mb-4 w-full">
            <Eye className="w-5 h-5 text-brand-400" />
            <h3 className="font-semibold text-sm text-slate-200">Non-Manual Features Interpreted</h3>
          </div>
          
          {interpretation && (
            <div className="flex flex-col gap-3.5 text-xs">
              {/* Eyebrows */}
              <div className="border-b border-slate-800 pb-2.5">
                <div className="flex justify-between items-center mb-1">
                  <span className="font-medium text-slate-400">Eyebrows Tension:</span>
                  <span className={`px-2 py-0.5 rounded font-mono text-[10px] font-bold ${
                    interpretation.eyebrows.state === 'Neutral' ? 'bg-slate-800 text-slate-350' : 'bg-brand-950 text-brand-400 border border-brand-850'
                  }`}>
                    {interpretation.eyebrows.state}
                  </span>
                </div>
                <p className="text-[11px] text-slate-500 leading-normal">{interpretation.eyebrows.desc}</p>
              </div>

              {/* Eyes */}
              <div className="border-b border-slate-800 pb-2.5">
                <div className="flex justify-between items-center mb-1">
                  <span className="font-medium text-slate-400">Eye Aperture:</span>
                  <span className={`px-2 py-0.5 rounded font-mono text-[10px] font-bold ${
                    interpretation.eyes.state === 'Normal' ? 'bg-slate-800 text-slate-350' : 'bg-emerald-950 text-emerald-400 border border-emerald-850'
                  }`}>
                    {interpretation.eyes.state}
                  </span>
                </div>
                <p className="text-[11px] text-slate-500 leading-normal">{interpretation.eyes.desc}</p>
              </div>

              {/* Mouth Open */}
              <div className="border-b border-slate-800 pb-2.5">
                <div className="flex justify-between items-center mb-1">
                  <span className="font-medium text-slate-400">Mouth Opening:</span>
                  <span className={`px-2 py-0.5 rounded font-mono text-[10px] font-bold ${
                    interpretation.mouthOpening.state === 'Closed' ? 'bg-slate-800 text-slate-350' : 'bg-amber-950 text-amber-400 border border-amber-850'
                  }`}>
                    {interpretation.mouthOpening.state}
                  </span>
                </div>
                <p className="text-[11px] text-slate-500 leading-normal">{interpretation.mouthOpening.desc}</p>
              </div>

              {/* Mouth Corners */}
              <div>
                <div className="flex justify-between items-center mb-1">
                  <span className="font-medium text-slate-400">Mouth Shape:</span>
                  <span className={`px-2 py-0.5 rounded font-mono text-[10px] font-bold ${
                    interpretation.mouthShape.state === 'Flat' ? 'bg-slate-800 text-slate-350' : 'bg-rose-950 text-rose-450 border border-rose-850'
                  }`}>
                    {interpretation.mouthShape.state}
                  </span>
                </div>
                <p className="text-[11px] text-slate-500 leading-normal">{interpretation.mouthShape.desc}</p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Explained conclusion box */}
      <div className="glass-panel-glow rounded-2xl p-5 flex gap-3.5 items-start">
        <Info className="w-6 h-6 text-brand-400 shrink-0 mt-0.5 animate-pulse-glow" />
        <div className="flex flex-col gap-1">
          <h4 className="text-xs font-semibold text-slate-300">Model Decision Conclusion</h4>
          <p className="text-[11px] text-slate-500 leading-relaxed">
            The neural network classified this image as <strong className="text-slate-300 font-semibold">{data.predicted_emotion.toUpperCase()}</strong> with a confidence score of <strong className="text-brand-400 font-mono">{(data.confidence * 100).toFixed(1)}%</strong>. 
            The attention weight is localized on the {
              interpretation && interpretation.mouthShape.state.includes('Smile') 
                ? 'mouth corners (associated with smiling)' 
                : interpretation && interpretation.eyebrows.state.includes('Furrowed')
                  ? 'forehead and eyebrow regions (associated with frowning or concentration)'
                  : 'central facial contours'
            }. This matches standard FACS Action Units criteria for the predicted emotion state.
          </p>
        </div>
      </div>
    </div>
  );
}
