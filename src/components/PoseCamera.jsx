import { useEffect, useRef, useState } from "react";
import * as poseDetection from "@tensorflow-models/pose-detection";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";
import { GoogleGenAI } from "@google/genai";

const GEMINI_API_KEY = import.meta.env.VITE_GEMINI_KEY;
const ai = new GoogleGenAI({ apiKey: GEMINI_API_KEY });

async function askGemini(message) {
  const response = await ai.models.generateContent({
    model: "gemini-2.5-flash",
    contents: message
  });
  return response.text;
}

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
        setFeedback("Strong rep! Keep that tempo steady.");
      }
    }

    setScore(100);
  }

  let lastGeminiCall = 0;

  async function sendToGemini(keypoints) {
    const now = Date.now();

    if (now - lastGeminiCall < 1000) return; // limit to once per second
    lastGeminiCall = now;

    try {
      const input = JSON.stringify(keypoints);
      const result = await askGemini(
        "Here are MoveNet keypoints. Do not analyze pushup form. Just summarize what the user is doing: " + input
      );
      console.log("Gemini result:", result);
    } catch (err) {
      console.error("Gemini pose error:", err);
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
            sendToGemini(poses[0].keypoints);
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
