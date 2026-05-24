import React, { useState } from 'react';
import { ShieldAlert, Compass, Table, BarChart } from 'lucide-react';

export default function ResearchPanel({ data }) {
  const [searchQuery, setSearchQuery] = useState('');
  
  if (!data || !data.face_detected) {
    return (
      <div className="flex flex-col items-center justify-center p-8 text-center text-slate-500 h-full border border-slate-800 rounded-2xl bg-slate-900/10">
        <ShieldAlert className="w-12 h-12 mb-3 text-slate-700 animate-pulse" />
        <p className="font-semibold text-slate-400">Research Mode Inactive</p>
        <p className="text-xs text-slate-500 mt-1 max-w-sm">
          Run real-time webcam streaming or upload a face media crop to inspect raw CV variables and Action Unit measurements.
        </p>
      </div>
    );
  }

  const { landmarks, emotion_confidences, pose } = data;

  // FACS Action Units calculations using landmark metrics
  const calculateActionUnits = () => {
    if (!landmarks) return [];

    const avgY = (pts) => pts.reduce((sum, p) => sum + p.py, 0) / pts.length;
    const avgX = (pts) => pts.reduce((sum, p) => sum + p.px, 0) / pts.length;
    const minY = (pts) => Math.min(...pts.map(p => p.py));
    const maxY = (pts) => Math.max(...pts.map(p => p.py));
    const minX = (pts) => Math.min(...pts.map(p => p.px));
    const maxX = (pts) => Math.max(...pts.map(p => p.px));

    const leftEyeX = avgX(landmarks.left_eye);
    const leftEyeY = avgY(landmarks.left_eye);
    const rightEyeX = avgX(landmarks.right_eye);
    const rightEyeY = avgY(landmarks.right_eye);
    const eyeDist = Math.max(1, rightEyeX - leftEyeX);

    const leftBrowY = avgY(landmarks.left_eyebrow);
    const rightBrowY = avgY(landmarks.right_eyebrow);
    const avgBrowHeight = ((leftEyeY - leftBrowY) + (rightEyeY - rightBrowY)) / (2 * eyeDist);

    const mouthW = Math.max(1, maxX(landmarks.lips) - minX(landmarks.lips));
    const mouthH = maxY(landmarks.lips) - minY(landmarks.lips);
    const mouthRatio = mouthH / mouthW;

    const cornerY = (landmarks.lips[0].py + landmarks.lips[10].py) / 2;
    const centerY = avgY(landmarks.lips);
    const smileScore = (centerY - cornerY) / eyeDist;

    // Action Units formulas (scaled 0..100)
    const au1 = Math.max(0, Math.min(100, Math.round((avgBrowHeight - 0.22) * 550))); // Inner Brow Raiser
    const au2 = Math.max(0, Math.min(100, Math.round((avgBrowHeight - 0.24) * 450))); // Outer Brow Raiser
    const au4 = Math.max(0, Math.min(100, Math.round((0.22 - avgBrowHeight) * 650))); // Brow Lowerer
    const au12 = Math.max(0, Math.min(100, Math.round(smileScore * 1300)));          // Lip Corner Puller (Smile)
    const au25 = Math.max(0, Math.min(100, Math.round(mouthRatio * 320)));           // Lips Part
    const au26 = Math.max(0, Math.min(100, Math.round((mouthRatio - 0.18) * 220)));    // Jaw Drop

    return [
      { id: 'AU1', name: 'Inner Brow Raiser', val: au1, desc: 'Frontalis (medial) - Raises inner eyebrows' },
      { id: 'AU2', name: 'Outer Brow Raiser', val: au2, desc: 'Frontalis (lateral) - Raises outer eyebrows' },
      { id: 'AU4', name: 'Brow Lowerer', val: au4, desc: 'Corrugator supercilii - Furrows eyebrows' },
      { id: 'AU12', name: 'Lip Corner Puller', val: au12, desc: 'Zygomaticus major - SMILE activation' },
      { id: 'AU25', name: 'Lips Part', val: au25, desc: 'Depressor labii / Orbicularis oris - Mouth opening' },
      { id: 'AU26', name: 'Jaw Drop', val: au26, desc: 'Masseter relaxation - Gaping jaw drop' }
    ];
  };

  const actionUnits = calculateActionUnits();

  // Format landmarks table rows
  const getLandmarksTableRows = () => {
    const rows = [];
    if (!landmarks) return [];
    
    Object.entries(landmarks).forEach(([groupName, pts]) => {
      pts.forEach((pt, index) => {
        rows.push({
          group: groupName.toUpperCase(),
          index: index + 1,
          x: pt.x.toFixed(4),
          y: pt.y.toFixed(4),
          z: pt.z.toFixed(4),
          px: pt.px,
          py: pt.py
        });
      });
    });
    
    return rows;
  };

  const tableRows = getLandmarksTableRows();
  const filteredRows = tableRows.filter(row => 
    row.group.toLowerCase().includes(searchQuery.toLowerCase()) ||
    row.index.toString() === searchQuery
  );

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 w-full">
      {/* Col 1: Action Units & Raw Probabilities */}
      <div className="flex flex-col gap-8">
        {/* FACS Action Units Card */}
        <div className="glass-panel rounded-2xl p-5 flex flex-col">
          <div className="flex items-center gap-2 mb-5">
            <BarChart className="w-5 h-5 text-brand-400" />
            <h3 className="font-semibold text-sm text-slate-200">FACS Action Unit (AU) Intensities</h3>
          </div>
          <div className="flex flex-col gap-4 text-xs">
            {actionUnits.map(au => (
              <div key={au.id} className="flex flex-col gap-1.5">
                <div className="flex justify-between items-center text-[11px]">
                  <span className="font-semibold text-slate-350">{au.id} - {au.name}</span>
                  <span className="font-mono text-brand-400 font-bold">{au.val}%</span>
                </div>
                <div className="w-full h-2 bg-slate-900 border border-slate-800 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-gradient-to-r from-brand-600 to-brand-400 rounded-full transition-all duration-300"
                    style={{ width: `${au.val}%` }}
                  />
                </div>
                <span className="text-[10px] text-slate-500 italic leading-normal">{au.desc}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Softmax Probabilities Card */}
        <div className="glass-panel rounded-2xl p-5 flex flex-col">
          <div className="flex items-center gap-2 mb-5">
            <BarChart className="w-5 h-5 text-emerald-400" />
            <h3 className="font-semibold text-sm text-slate-200">Raw Softmax Probabilities</h3>
          </div>
          <div className="flex flex-col gap-3 text-xs">
            {Object.entries(emotion_confidences || {}).map(([emotion, confidence]) => (
              <div key={emotion} className="flex items-center gap-3">
                <span className="w-20 uppercase font-mono text-[10px] text-slate-400 text-right shrink-0">{emotion}</span>
                <div className="flex-1 h-3 bg-slate-900 border border-slate-800 rounded-md overflow-hidden relative">
                  <div 
                    className="h-full bg-emerald-600 rounded-md transition-all duration-300"
                    style={{ width: `${confidence * 100}%` }}
                  />
                </div>
                <span className="w-12 font-mono text-emerald-400 text-right text-[10px] font-bold">{(confidence * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Col 2: Pose Orientation & Landmark Coordinates Table */}
      <div className="flex flex-col gap-8">
        {/* Head Pose Orientation Card */}
        <div className="glass-panel rounded-2xl p-5 flex flex-col">
          <div className="flex items-center gap-2 mb-5">
            <Compass className="w-5 h-5 text-amber-400" />
            <h3 className="font-semibold text-sm text-slate-200">Head Pose & Orientation Estimation</h3>
          </div>
          {pose && (
            <div className="grid grid-cols-3 gap-4 text-center">
              <div className="bg-slate-950/40 border border-slate-850 p-3 rounded-xl">
                <span className="text-[10px] font-medium text-slate-500 uppercase tracking-wider block mb-1">Yaw</span>
                <span className={`text-lg font-bold font-mono ${Math.abs(pose.yaw) > 15 ? 'text-amber-400' : 'text-slate-200'}`}>
                  {pose.yaw.toFixed(1)}°
                </span>
                <span className="text-[9px] text-slate-500 block mt-1">
                  {pose.yaw > 15 ? 'Turned Right' : pose.yaw < -15 ? 'Turned Left' : 'Facing Straight'}
                </span>
              </div>
              <div className="bg-slate-950/40 border border-slate-850 p-3 rounded-xl">
                <span className="text-[10px] font-medium text-slate-500 uppercase tracking-wider block mb-1">Pitch</span>
                <span className={`text-lg font-bold font-mono ${Math.abs(pose.pitch) > 12 ? 'text-amber-400' : 'text-slate-200'}`}>
                  {pose.pitch.toFixed(1)}°
                </span>
                <span className="text-[9px] text-slate-500 block mt-1">
                  {pose.pitch > 12 ? 'Looking Down' : pose.pitch < -12 ? 'Looking Up' : 'Level Horizon'}
                </span>
              </div>
              <div className="bg-slate-950/40 border border-slate-850 p-3 rounded-xl">
                <span className="text-[10px] font-medium text-slate-500 uppercase tracking-wider block mb-1">Roll</span>
                <span className={`text-lg font-bold font-mono ${Math.abs(pose.roll) > 10 ? 'text-amber-400' : 'text-slate-200'}`}>
                  {pose.roll.toFixed(1)}°
                </span>
                <span className="text-[9px] text-slate-500 block mt-1">
                  {pose.roll > 10 ? 'Tilted Right' : pose.roll < -10 ? 'Tilted Left' : 'Tilted Straight'}
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Landmark Coordinates Table Card */}
        <div className="glass-panel rounded-2xl p-5 flex flex-col h-[380px]">
          <div className="flex justify-between items-center mb-4">
            <div className="flex items-center gap-2">
              <Table className="w-5 h-5 text-slate-400" />
              <h3 className="font-semibold text-sm text-slate-200">Landmarks Coordinates [X, Y, Z]</h3>
            </div>
            <input 
              type="text"
              placeholder="Search group (e.g. lips)..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="px-2.5 py-1 bg-slate-950 border border-slate-850 rounded-lg text-[10px] focus:outline-none focus:border-brand-500 w-44"
            />
          </div>
          
          <div className="flex-1 overflow-auto pr-1 text-[10px] border border-slate-900 rounded-xl bg-slate-950/20">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="bg-slate-950 text-slate-450 border-b border-slate-850 font-semibold sticky top-0 uppercase tracking-wider text-[8px] z-10">
                  <th className="py-2.5 px-3">Group</th>
                  <th className="py-2.5 px-2">Idx</th>
                  <th className="py-2.5 px-2">Normalized X, Y, Z</th>
                  <th className="py-2.5 px-3 text-right">Pixel Px, Py</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-900 font-mono text-slate-350">
                {filteredRows.length > 0 ? (
                  filteredRows.map((row, idx) => (
                    <tr key={idx} className="hover:bg-slate-900/40 transition-colors">
                      <td className="py-2 px-3 font-semibold text-slate-400">{row.group}</td>
                      <td className="py-2 px-2 text-slate-500">{row.index}</td>
                      <td className="py-2 px-2 text-slate-300">[{row.x}, {row.y}, {row.z}]</td>
                      <td className="py-2 px-3 text-right text-brand-400">({row.px}, {row.py})</td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan="4" className="py-12 text-center text-slate-600">No matching coordinates found</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
