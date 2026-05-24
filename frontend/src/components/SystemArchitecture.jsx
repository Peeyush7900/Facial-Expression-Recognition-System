import React from 'react';
import { Camera, Eye, Scan, Cpu, Activity, LayoutDashboard, ChevronRight } from 'lucide-react';

const STAGES = [
  {
    num: '01',
    title: 'Input Capture Stream',
    icon: Camera,
    color: 'text-brand-400 border-brand-900 bg-brand-950/20',
    desc: 'Receives base64-encoded frames from HTML5 Webcam API polling or drag-and-drop file inputs (images/videos).'
  },
  {
    num: '02',
    title: 'MediaPipe Landmark Tracking',
    icon: Eye,
    color: 'text-emerald-400 border-emerald-900 bg-emerald-950/20',
    desc: 'Extracts 468 3D landmarks. Groups them into eyes, eyebrows, mouth, and jawline paths, and computes Head Pose orientation (yaw, pitch, roll).'
  },
  {
    num: '03',
    title: 'Crop & Normalization',
    icon: Scan,
    color: 'text-amber-400 border-amber-900 bg-amber-950/20',
    desc: 'Calculates the face bounding box with 15% padding, crops the region, converts to grayscale, resizes to 48x48 pixels, and normalizes values to [-1, 1].'
  },
  {
    num: '04',
    title: 'PyTorch CNN Classification',
    icon: Cpu,
    color: 'text-rose-450 border-rose-900 bg-rose-950/20',
    desc: 'Propagates the 48x48 tensor through a 3-stage custom CNN (6 Conv layers, BatchNorm, MaxPool, Dropouts, 512-Dense layer) to output softmax emotion logits.'
  },
  {
    num: '05',
    title: 'Grad-CAM Saliency Maps',
    icon: Activity,
    color: 'text-purple-400 border-purple-900 bg-purple-950/20',
    desc: 'Calculates gradients of the predicted logit back to the conv6 layer. Blends the upscaled JET heatmap over the cropped face to justify prediction focus.'
  },
  {
    num: '06',
    title: 'FACS & Dashboard Presentation',
    icon: LayoutDashboard,
    color: 'text-blue-400 border-blue-900 bg-blue-950/20',
    desc: 'Translates landmarks metrics into Action Unit intensities (AU1, AU2, AU4, AU12, AU25, AU26), renders canvas overlays, and updates Recharts graphs.'
  }
];

export default function SystemArchitecture() {
  return (
    <div className="flex flex-col gap-8 w-full">
      {/* Top Diagram Row */}
      <div className="glass-panel rounded-2xl p-6 flex flex-col items-center">
        <h3 className="font-semibold text-sm text-slate-200 mb-6 uppercase tracking-wider">End-to-End System Pipeline Flow</h3>
        
        {/* Responsive flowchart grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-6 gap-4 w-full relative">
          {STAGES.map((stage, idx) => {
            const Icon = stage.icon;
            return (
              <React.Fragment key={stage.num}>
                <div className={`border p-4 rounded-xl flex flex-col items-center text-center transition-all hover:scale-[1.02] ${stage.color}`}>
                  <span className="font-mono text-[10px] font-bold opacity-40 mb-2">{stage.num}</span>
                  <div className="w-10 h-10 rounded-lg bg-slate-900 border border-slate-800 flex items-center justify-center mb-3">
                    <Icon className="w-5 h-5" />
                  </div>
                  <h4 className="font-bold text-xs text-slate-200 mb-1 leading-tight">{stage.title}</h4>
                  <p className="text-[10px] text-slate-500 leading-normal mt-1.5">{stage.desc}</p>
                </div>
              </React.Fragment>
            );
          })}
        </div>
      </div>

      {/* Low-Level Architecture Text Card */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div className="glass-panel rounded-2xl p-5 flex flex-col gap-3">
          <h4 className="text-xs font-bold text-slate-300 uppercase tracking-wider">Explainable AI & Saliency Mapping</h4>
          <p className="text-[11px] text-slate-550 leading-relaxed">
            The system incorporates **Grad-CAM (Gradient-weighted Class Activation Mapping)** to visualize which regions of the face (e.g. eye width, mouth corners) were the primary drivers for a specific emotion prediction. The gradients of the winning logit score are backpropagated to the last convolutional layer maps ($6 \times 6 \times 256$), global pooled to form channel importance weights, and summed to generate a 2D focus map. This map is upsampled to $48 \times 48$, passed through a ReLU activation, and color-coded.
          </p>
        </div>
        <div className="glass-panel rounded-2xl p-5 flex flex-col gap-3">
          <h4 className="text-xs font-bold text-slate-300 uppercase tracking-wider">FACS Landmark Interpretation Engine</h4>
          <p className="text-[11px] text-slate-550 leading-relaxed">
            Unlike manual feature annotation, our system interprets landmarks dynamically by utilizing scale-invariant geometric relationships. By measuring **interpupillary eye distance** as a relative baseline, we calculate eyebrow furrow intensity, lip corner curvature (smile), eye squint levels, and jaw opening ratios independently of face scale or camera distance. These ratios are then converted to **FACS Action Units (AUs)** to explain the model's prediction semantically.
          </p>
        </div>
      </div>
    </div>
  );
}
