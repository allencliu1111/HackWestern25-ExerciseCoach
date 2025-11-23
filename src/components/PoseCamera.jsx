import { useEffect, useRef, useState } from "react";
import * as poseDetection from "@tensorflow-models/pose-detection";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";

export default function PoseCamera() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const detectorRef = useRef(null);

  const [repCount, setRepCount] = useState(0);
  const [feedback, setFeedback] = useState(
    "Start in a straight high plank and begin your first pushup."
  );
  const [score, setScore] = useState(100);
  const [videoSize, setVideoSize] = useState({ width: 960, height: 540 });
  const formStatsRef = useRef({
    angles: [],
    stacks: [],
    tilts: [],
    lastFeedbackTime: Date.now()
  });

  const repStateRef = useRef({
    phase: "top",
    repCount: 0
  });

  useEffect(() => {
    async function init() {
      await setupCamera();
      await tf.setBackend("webgl");
      await tf.ready();
      await loadMoveNet();

      startPoseLoop();
    }
    init();
  }, []);

  async function setupCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    videoRef.current.srcObject = stream;

    await new Promise((resolve) => {
      videoRef.current.onloadedmetadata = () => {
        videoRef.current.play();
        const width = videoRef.current.videoWidth || 960;
        const height = videoRef.current.videoHeight || 540;
        setVideoSize({ width, height });
        if (canvasRef.current) {
          canvasRef.current.width = width;
          canvasRef.current.height = height;
        }
        resolve();
      };
    });
  }

  async function loadMoveNet() {
    detectorRef.current = await poseDetection.createDetector(
      poseDetection.SupportedModels.MoveNet,
      {
        modelType: "SinglePose.Thunder"
      }
    );
  }

  function getPoint(pose, name) {
    return pose.keypoints.find((kp) => kp.name === name && kp.score > 0.4);
  }

  function calculateAngle(a, b, c) {
    const ab = { x: a.x - b.x, y: a.y - b.y };
    const cb = { x: c.x - b.x, y: c.y - b.y };
    const dot = ab.x * cb.x + ab.y * cb.y;
    const magAB = Math.hypot(ab.x, ab.y);
    const magCB = Math.hypot(cb.x, cb.y);
    if (!magAB || !magCB) return 180;
    const cosine = Math.min(Math.max(dot / (magAB * magCB), -1), 1);
    return (Math.acos(cosine) * 180) / Math.PI;
  }

  function interpretForm(averageElbowAngle, averageStack, shoulderTilt) {
    let message = "Solid pushup - keep elbows tracking toward your ribs.";
    let newScore = 100;

    if (averageElbowAngle < 75) {
      message = "Don't dive too deep - aim for about 90 deg at the elbows.";
      newScore -= 20;
    } else if (averageElbowAngle > 150) {
      message = "Lower more; get elbows to roughly 90 deg for full range.";
      newScore -= 15;
    } else {
      message = "Nice elbow bend - controlled 90 deg range.";
    }

    if (averageStack > 90) {
      message = "Stack wrists under shoulders to avoid flaring the elbows.";
      newScore -= 20;
    }

    if (shoulderTilt > 40) {
      message = "Keep shoulders level; avoid twisting on the way up.";
      newScore -= 15;
    }

    newScore = Math.max(60, Math.min(100, newScore));
    return { message, score: newScore };
  }

  function drawSkeleton(pose) {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const keypoints = pose.keypoints;
    keypoints.forEach((kp) => {
      if (kp.score > 0.4) {
        ctx.beginPath();
        ctx.arc(kp.x, kp.y, 5, 0, 2 * Math.PI);
        ctx.fillStyle = "#7cf8ff";
        ctx.fill();
      }
    });

    const adjacentPairs = [
      ["leftShoulder", "rightShoulder"],
      ["leftShoulder", "leftElbow"],
      ["leftElbow", "leftWrist"],
      ["rightShoulder", "rightElbow"],
      ["rightElbow", "rightWrist"],
      ["leftShoulder", "leftHip"],
      ["rightShoulder", "rightHip"],
      ["leftHip", "rightHip"],
      ["leftHip", "leftKnee"],
      ["leftKnee", "leftAnkle"],
      ["rightHip", "rightKnee"],
      ["rightKnee", "rightAnkle"]
    ];

    function getKP(name) {
      return keypoints.find((kp) => kp.name === name);
    }

    ctx.strokeStyle = "#6cf5b5";
    ctx.lineWidth = 3.5;

    adjacentPairs.forEach(([a, b]) => {
      const kpA = getKP(a);
      const kpB = getKP(b);
      if (kpA?.score > 0.4 && kpB?.score > 0.4) {
        ctx.beginPath();
        ctx.moveTo(kpA.x, kpA.y);
        ctx.lineTo(kpB.x, kpB.y);
        ctx.stroke();
      }
    });
  }

  function processPose(pose) {
    const leftShoulder = getPoint(pose, "left_shoulder");
    const rightShoulder = getPoint(pose, "right_shoulder");

    if (!leftShoulder || !rightShoulder) return;

    const shoulderY = (leftShoulder.y + rightShoulder.y) / 2;
    const state = repStateRef.current;
    const downThreshold = 52;     // how far you must travel downward to register the bottom
    const upThreshold = 40;       // how far you must travel upward to register returning to top
    const minimumRange = 70;      // total shoulder travel required to count a rep

    if (state.topY === undefined) {
      state.topY = shoulderY;
      state.bottomY = shoulderY;
    }

    if (state.phase === "top") {
      state.topY = Math.min(state.topY, shoulderY);
      if (shoulderY - state.topY > downThreshold) {
        state.phase = "bottom";
        state.bottomY = shoulderY;
      }
    } else if (state.phase === "bottom") {
      state.bottomY = Math.max(state.bottomY, shoulderY);
      if (state.bottomY - shoulderY > upThreshold && state.bottomY - state.topY > minimumRange) {
        state.phase = "top";
        state.topY = shoulderY;
        state.repCount += 1;
        setRepCount(state.repCount);
      }
    }

    const leftElbow = getPoint(pose, "left_elbow");
    const rightElbow = getPoint(pose, "right_elbow");
    const leftWrist = getPoint(pose, "left_wrist");
    const rightWrist = getPoint(pose, "right_wrist");

    if (leftShoulder && rightShoulder && leftElbow && rightElbow && leftWrist && rightWrist) {
      const leftElbowAngle = calculateAngle(leftShoulder, leftElbow, leftWrist);
      const rightElbowAngle = calculateAngle(rightShoulder, rightElbow, rightWrist);
      const averageElbowAngle = (leftElbowAngle + rightElbowAngle) / 2;
      const wristStackLeft = Math.abs(leftShoulder.x - leftWrist.x);
      const wristStackRight = Math.abs(rightShoulder.x - rightWrist.x);
      const averageStack = (wristStackLeft + wristStackRight) / 2;
      const shoulderTilt = Math.abs(leftShoulder.y - rightShoulder.y);

      formStatsRef.current.angles.push(averageElbowAngle);
      formStatsRef.current.stacks.push(averageStack);
      formStatsRef.current.tilts.push(shoulderTilt);
    }

    const now = Date.now();
    const elapsed = now - formStatsRef.current.lastFeedbackTime;
    if (
      elapsed >= 3000 &&
      formStatsRef.current.angles.length > 0 &&
      formStatsRef.current.stacks.length > 0 &&
      formStatsRef.current.tilts.length > 0
    ) {
      const avgAngle =
        formStatsRef.current.angles.reduce((sum, val) => sum + val, 0) /
        formStatsRef.current.angles.length;
      const avgStack =
        formStatsRef.current.stacks.reduce((sum, val) => sum + val, 0) /
        formStatsRef.current.stacks.length;
      const avgTilt =
        formStatsRef.current.tilts.reduce((sum, val) => sum + val, 0) /
        formStatsRef.current.tilts.length;

      const form = interpretForm(avgAngle, avgStack, avgTilt);
      setFeedback(form.message);
      setScore(form.score);

      formStatsRef.current.angles = [];
      formStatsRef.current.stacks = [];
      formStatsRef.current.tilts = [];
      formStatsRef.current.lastFeedbackTime = now;
    }
  }

  async function startPoseLoop() {
    async function frame() {
      if (detectorRef.current && videoRef.current) {
        try {
          const poses = await detectorRef.current.estimatePoses(videoRef.current);
          if (poses && poses.length > 0) {
            drawSkeleton(poses[0]);
            processPose(poses[0]);
          }
        } catch (err) {
          console.error("Pose estimation error:", err);
        }
      }
      requestAnimationFrame(frame);
    }

    frame();
  }

  return (
    <div className="poseLayout">
      <div className="videoCard">
        <div
          className="videoFrame"
          style={{ aspectRatio: `${videoSize.width}/${videoSize.height}` }}
        >
          <video
            ref={videoRef}
            width={videoSize.width}
            height={videoSize.height}
            className="cameraFeed"
          />
          <canvas
            ref={canvasRef}
            width={videoSize.width}
            height={videoSize.height}
            className="poseCanvas"
          />
          <div className="overlayTop">
            <div className="pill solid">Reps <span>{repCount}</span></div>
            <div className="pill subtle">Form score <span>{score}</span></div>
          </div>
          <div className="overlayBottom">
            <div className="statusText">{feedback}</div>
          </div>
        </div>
      </div>

      <div className="feedbackCard">
        <div className="cardHeader">Feedback</div>
        <p className="feedbackCopy">{feedback}</p>
        <div className="statGrid">
          <div className="stat">
            <span className="label">Reps</span>
            <span className="value">{repCount}</span>
          </div>
          <div className="stat">
            <span className="label">Form score</span>
            <span className="value">{score}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
