import React from "react";

export default function Feedback() {
  return (
    <div className="feedback">
      <div className="placeholder">
        <p className="big" style={{ textAlign: 'center' }}>Ready to rep? Your real-time feedback will show up here.</p>
        <ul>
          <li>Reps: —</li>
          <li>Depth: —</li>
          <li>Back alignment: —</li>
          <li>Tempo: —</li>
        </ul>
      </div>

      <div className="tips">
        <h3 className="big" style={{ textAlign: 'center' }}>Pro Tips</h3>
        <ol>
          <li>Keep a straight line from head to heels.</li>
          <li>Drive through your palms, not your wrists.</li>
          <li>Control the descent — no bounce.</li>
        </ol>
      </div>
    </div>
  );
}
