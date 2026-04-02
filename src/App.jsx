import React, { useRef, useEffect, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as faceLandmarksDetection from '@tensorflow-models/face-landmarks-detection';
import * as poseDetection from '@tensorflow-models/pose-detection';

export default function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  // UI State
  const [blinkCount, setBlinkCount] = useState(0);
  const [bpm, setBpm] = useState(0);
  const [liveRatioRight, setLiveRatioRight] = useState(0);
  const [liveRatioLeft, setLiveRatioLeft] = useState(0);
  const [coordsRight, setCoordsRight] = useState({ x: 0, y: 0 });
  const [coordsLeft, setCoordsLeft] = useState({ x: 0, y: 0 });
  const [status, setStatus] = useState('Initializing System...');
  const [isFatigued, setIsFatigued] = useState(false);
  const [isResting, setIsResting] = useState(false);

  // Logic Refs
  const blinkDatesRef = useRef([]);
  const totalBlinksRef = useRef(0);
  const blinkStartTimeRef = useRef(null);
  const appStartTimeRef = useRef(Date.now());
  const audioRef = useRef(new Audio('/alert.mp3'));
  const isRestingRef = useRef(false);
  const lastFaceDetectedRef = useRef(Date.now());
  const earBufferRef = useRef([]);
  const earBaselineSamplesRef = useRef([]);
  const adaptiveThresholdRef = useRef(0.27);
  const workerRef = useRef(null);
  const streamRef = useRef(null);
  const [isShutdown, setIsShutdown] = useState(false);
  const isShutdownRef = useRef(false);

  // Effect Refs
  const wasBlinkingRef = useRef(false);
  const scrambleUntilRef = useRef(0);

  const handleTakeBreak = () => {
    isRestingRef.current = true;
    setIsResting(true);
    setIsFatigued(false);
  };

  const handleIAmBack = () => {
    isRestingRef.current = false;
    setIsResting(false);
    setIsFatigued(false);
    blinkDatesRef.current = [];
    totalBlinksRef.current = 0;
    setBlinkCount(0);
    setBpm(0);
    appStartTimeRef.current = Date.now();
    lastFaceDetectedRef.current = Date.now();
    earBufferRef.current = [];
    earBaselineSamplesRef.current = [];
    adaptiveThresholdRef.current = 0.27;
    setStatus('System Active - Tracking HUD');
  };

  // Smoothed XY coords and positions for floating labels
  const smoothedRightRef = useRef({ x: 0, y: 0 });
  const smoothedLeftRef = useRef({ x: 0, y: 0 });
  const smoothedRightEdgeRef = useRef({ x: 0, y: 0 });
  const smoothedLeftEdgeRef = useRef({ x: 0, y: 0 });

  useEffect(() => {
    let faceDetector;
    let poseDetector;
    let worker;
    let isProcessingFrame = false;

    const init = async () => {
      try {
        setStatus('Loading AI Models...');
        await tf.ready();

        faceDetector = await faceLandmarksDetection.createDetector(
          faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh,
          { runtime: 'tfjs', refineLandmarks: false },
        );
        poseDetector = await poseDetection.createDetector(
          poseDetection.SupportedModels.MoveNet,
        );

        setStatus('Accessing Video Stream...');
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480 },
        });

        if (videoRef.current) {
          streamRef.current = stream;
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = () => {
            videoRef.current.play();
            appStartTimeRef.current = Date.now();
            setStatus('System Active - Tracking HUD');
            startWorker();
          };
        }
      } catch (err) {
        setStatus('System Error: ' + err.message);
      }
    };

    const startWorker = () => {
      const workerCode = `
        let timer = null;
        self.onmessage = function(e) {
          if (e.data === 'start') {
            timer = setInterval(() => self.postMessage('tick'), 33);
          } else if (e.data === 'stop') {
            clearInterval(timer);
          }
        };
      `;
      const blob = new Blob([workerCode], { type: 'application/javascript' });
      worker = new Worker(URL.createObjectURL(blob));
      workerRef.current = worker;
      worker.onmessage = () => {
        if (!isProcessingFrame) renderFrame();
      };
      worker.postMessage('start');
    };

    const handleMessage = (e) => {
      if (e.data?.type === 'SHUTDOWN') {
        isShutdownRef.current = true;
        setIsShutdown(true);
        workerRef.current?.postMessage('stop');
        streamRef.current?.getTracks().forEach((t) => t.stop());
        if (videoRef.current) videoRef.current.srcObject = null;
      } else if (e.data?.type === 'RESUME') {
        isShutdownRef.current = false;
        setIsShutdown(false);
        navigator.mediaDevices
          .getUserMedia({ video: { width: 640, height: 480 } })
          .then((stream) => {
            streamRef.current = stream;
            if (videoRef.current) {
              videoRef.current.srcObject = stream;
              videoRef.current.play();
            }
            workerRef.current?.postMessage('start');
          });
      }
    };
    window.addEventListener('message', handleMessage);

    // --- REFINED HUD BOX DRAWING (No text, just geometry) ---
    const drawEyeBox = (ctx, inPt, outPt, topPt, botPt, isBlinking) => {
      const padX = 2;
      const padY = 3;

      const minX = Math.min(inPt.x, outPt.x) - padX;
      const maxX = Math.max(inPt.x, outPt.x) + padX;
      const minY = Math.min(topPt.y, botPt.y) - padY;
      const maxY = Math.max(topPt.y, botPt.y) + padY;
      const w = maxX - minX;
      const h = maxY - minY;

      ctx.strokeStyle = isBlinking
        ? 'rgba(255, 255, 255, 1)'
        : 'rgba(255, 255, 255, 0.4)';
      ctx.lineWidth = 1;
      ctx.strokeRect(minX, minY, w, h);

      if (isBlinking) {
        ctx.beginPath();
        ctx.moveTo(minX, minY);
        ctx.lineTo(maxX, maxY);
        ctx.moveTo(maxX, minY);
        ctx.lineTo(minX, maxY);
        ctx.stroke();
      }

      // Return bounds so we can anchor the text perfectly
      return { minX, maxX, minY, maxY, w, h };
    };

    const renderFrame = async () => {
      if (isRestingRef.current || isShutdownRef.current) return;
      if (
        !videoRef.current ||
        !faceDetector ||
        !poseDetector ||
        !canvasRef.current
      )
        return;
      if (videoRef.current.readyState < 2) return;

      isProcessingFrame = true;

      const videoWidth = videoRef.current.videoWidth;
      const videoHeight = videoRef.current.videoHeight;

      videoRef.current.width = videoWidth;
      videoRef.current.height = videoHeight;
      canvasRef.current.width = videoWidth;
      canvasRef.current.height = videoHeight;

      const ctx = canvasRef.current.getContext('2d');
      ctx.clearRect(0, 0, videoWidth, videoHeight);
      ctx.globalCompositeOperation = 'source-over';

      try {
        const faces = await faceDetector.estimateFaces(videoRef.current, {
          flipHorizontal: false,
        });
        const poses = await poseDetector.estimatePoses(videoRef.current, {
          flipHorizontal: false,
        });
        const now = Date.now();

        // 1. Draw Arm Skeleton
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
        ctx.lineWidth = 1;

        if (poses && poses.length > 0) {
          const pts = poses[0].keypoints;
          const drawLine = (a, b) => {
            if (pts[a].score > 0.3 && pts[b].score > 0.3) {
              ctx.beginPath();
              ctx.moveTo(pts[a].x, pts[a].y);
              ctx.lineTo(pts[b].x, pts[b].y);
              ctx.stroke();
            }
          };

          drawLine(5, 6);
          drawLine(5, 7);
          drawLine(7, 9);
          drawLine(6, 8);
          drawLine(8, 10);

          const activeJoints = [5, 6, 7, 8, 9, 10];
          activeJoints.forEach((i) => {
            if (pts[i].score > 0.3) {
              ctx.beginPath();
              ctx.arc(pts[i].x, pts[i].y, 2, 0, 2 * Math.PI);
              ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
              ctx.fill();
            }
          });
        }

        // 2. Face Tracking & HUD
        if (faces && faces.length > 0) {
          lastFaceDetectedRef.current = now;
          const keypoints = faces[0].keypoints;

          const rTop = keypoints[159];
          const rBot = keypoints[145];
          const rIn = keypoints[133];
          const rOut = keypoints[33];

          const lTop = keypoints[386];
          const lBot = keypoints[374];
          const lIn = keypoints[362];
          const lOut = keypoints[263];

          const lCenterX = (lIn.x + lOut.x) / 2;
          const lCenterY = (lTop.y + lBot.y) / 2;
          const rCenterX = (rIn.x + rOut.x) / 2;
          const rCenterY = (rTop.y + rBot.y) / 2;

          const lerpFactor = 0.06;
          smoothedRightRef.current.x += lerpFactor * (lCenterX - smoothedRightRef.current.x);
          smoothedRightRef.current.y += lerpFactor * (lCenterY - smoothedRightRef.current.y);
          smoothedLeftRef.current.x += lerpFactor * (rCenterX - smoothedLeftRef.current.x);
          smoothedLeftRef.current.y += lerpFactor * (rCenterY - smoothedLeftRef.current.y);

          const eyeDistPx = Math.hypot(
            lCenterX - rCenterX,
            lCenterY - rCenterY,
          );
          const mmPerPx = eyeDistPx > 0 ? 63 / eyeDistPx : 0;

          const vDist = Math.abs(rTop.y - rBot.y);
          const hDist = Math.abs(rIn.x - rOut.x);
          const ear = hDist > 0 ? vDist / hDist : 0;
          setLiveRatioRight(ear.toFixed(3));

          const lVDist = Math.abs(lTop.y - lBot.y);
          const lHDist = Math.abs(lIn.x - lOut.x);
          const lEAR = lHDist > 0 ? lVDist / lHDist : 0;
          setLiveRatioLeft(lEAR.toFixed(3));

          setCoordsRight({ x: Math.round(rCenterX), y: Math.round(rCenterY) });
          setCoordsLeft({ x: Math.round(lCenterX), y: Math.round(lCenterY) });

          const avgEAR = (ear + lEAR) / 2;

          // Smooth EAR over last 3 frames to reduce landmark jitter
          earBufferRef.current.push(avgEAR);
          if (earBufferRef.current.length > 3) earBufferRef.current.shift();
          const smoothedEAR = earBufferRef.current.reduce((a, b) => a + b, 0) / earBufferRef.current.length;

          // Calibration: collect 180 open-eye samples (~6s at 30fps) then lock threshold
          if (earBaselineSamplesRef.current.length < 180 && smoothedEAR > 0.3) {
            earBaselineSamplesRef.current.push(smoothedEAR);
            if (earBaselineSamplesRef.current.length === 180) {
              const baseline = earBaselineSamplesRef.current.reduce((a, b) => a + b, 0) / 180;
              adaptiveThresholdRef.current = baseline * 0.65;
            }
          }

          const isBlinkingNow = smoothedEAR < adaptiveThresholdRef.current;

          // MATRIX SCRAMBLE TRIGGER
          if (isBlinkingNow && !wasBlinkingRef.current) {
            scrambleUntilRef.current = now + 150;
          }
          wasBlinkingRef.current = isBlinkingNow;
          const isScrambling = now < scrambleUntilRef.current;

          if (isBlinkingNow) {
            if (!blinkStartTimeRef.current) blinkStartTimeRef.current = now;
          } else {
            if (blinkStartTimeRef.current) {
              const durationClosed = now - blinkStartTimeRef.current;
              if (durationClosed > 60 && durationClosed < 600) {
                totalBlinksRef.current += 1;
                setBlinkCount(totalBlinksRef.current);
                blinkDatesRef.current.push(now);
              }
              blinkStartTimeRef.current = null;
            }
          }

          // 3. Render Eye Boxes and retrieve exact bounds
          // rBox is Viewer's Left Eye | lBox is Viewer's Right Eye
          const rBox = drawEyeBox(ctx, rIn, rOut, rTop, rBot, isBlinkingNow);
          const lBox = drawEyeBox(ctx, lIn, lOut, lTop, lBot, isBlinkingNow);

          // Smooth box edge positions for label anchors
          smoothedRightEdgeRef.current.x += lerpFactor * (lBox.minX - smoothedRightEdgeRef.current.x);
          smoothedRightEdgeRef.current.y += lerpFactor * (lBox.minY - smoothedRightEdgeRef.current.y);
          smoothedLeftEdgeRef.current.x += lerpFactor * (rBox.maxX - smoothedLeftEdgeRef.current.x);
          smoothedLeftEdgeRef.current.y += lerpFactor * (rBox.minY - smoothedLeftEdgeRef.current.y);

          // 4. Custom Permanent Text Layout
          if (mmPerPx > 0) {
            const areaR = Math.round(rBox.w * rBox.h * (mmPerPx * mmPerPx));
            const areaL = Math.round(lBox.w * lBox.h * (mmPerPx * mmPerPx));

            let textLeftEyeMsg = `${areaR}`; // Area of left eye (rBox)
            let textRightEyeMsg = `${areaL}`; // Area of right eye (lBox)

            if (isScrambling) {
              const chars = '0123456789!@#$%&*+';
              textLeftEyeMsg = Array.from(
                { length: 4 },
                () => chars[Math.floor(Math.random() * chars.length)],
              ).join('');
              textRightEyeMsg = Array.from(
                { length: 4 },
                () => chars[Math.floor(Math.random() * chars.length)],
              ).join('');
            }

            ctx.save();
            ctx.scale(-1, 1); // Un-mirror for text readability

            ctx.fillStyle = isBlinkingNow
              ? 'rgba(255, 255, 255, 1)'
              : 'rgba(255, 255, 255, 0.6)';
            ctx.textBaseline = 'middle';

            // --- The "Overload" Logic (Anchoring to Viewer's Right Eye / lBox) ---
            // Calculate center and right edge of Viewer's Right Eye in un-mirrored space
            const vRightBoxFloatOver = -(lBox.minX - 25); // Over right eye box right edge
            const vRightBoxFloatFar = -(lBox.minX - 70); // Floating far right, near head
            const vRightBoxY = lBox.minY + lBox.h / 2;

            ctx.font = '12px monospace';

            // Left eye area: over the right eye box
            ctx.textAlign = 'center';
            ctx.fillText(textLeftEyeMsg, vRightBoxFloatOver, vRightBoxY);

            // Right eye area: floating right, next to head
            ctx.textAlign = 'center';
            ctx.fillText(textRightEyeMsg, vRightBoxFloatFar, vRightBoxY);

            // --- XY Coordinates floating around the head ---
            ctx.font = '10px monospace';
            ctx.fillStyle = isBlinkingNow
              ? 'rgba(255, 255, 255, 1)'
              : 'rgba(255, 255, 255, 0.4)';

            // Viewer's right eye coords — further above and to the right of the head
            ctx.textAlign = 'left';
            ctx.fillText(
              `X:${Math.round(smoothedRightRef.current.x)}`,
              -(smoothedRightEdgeRef.current.x - 55),
              smoothedRightEdgeRef.current.y - 90,
            );
            ctx.fillText(
              `Y:${Math.round(smoothedRightRef.current.y)}`,
              -(smoothedRightEdgeRef.current.x - 55),
              smoothedRightEdgeRef.current.y - 77,
            );

            // Viewer's left eye coords — further above and to the left of the head
            ctx.textAlign = 'right';
            ctx.fillText(
              `X:${Math.round(smoothedLeftRef.current.x)}`,
              -(smoothedLeftEdgeRef.current.x + 55),
              smoothedLeftEdgeRef.current.y - 90,
            );
            ctx.fillText(
              `Y:${Math.round(smoothedLeftRef.current.y)}`,
              -(smoothedLeftEdgeRef.current.x + 55),
              smoothedLeftEdgeRef.current.y - 77,
            );

            ctx.restore();
          }
        } else {
          setLiveRatioRight(0);
          setLiveRatioLeft(0);
          blinkStartTimeRef.current = null;
        }

        const tenMinutes = 10 * 60 * 1000;
        const oneMinute = 60 * 1000;
        const eyesLost = now - lastFaceDetectedRef.current > 15000;

        blinkDatesRef.current = blinkDatesRef.current.filter(
          (t) => t > now - tenMinutes,
        );
        setBpm(blinkDatesRef.current.filter((t) => t > now - oneMinute).length);

        if (eyesLost) {
          setStatus('No Face Detected...');
        } else {
          const msRunning = Math.min(now - appStartTimeRef.current, tenMinutes);
          const minutesRunning = msRunning / oneMinute;

          if (minutesRunning >= 2) {
            const averageBpm = blinkDatesRef.current.length / minutesRunning;
            if (averageBpm < 8) {
              setIsFatigued(true);
              setStatus(`⚠️ FATIGUE ALERT (Avg: ${averageBpm.toFixed(1)} BPM)`);
            } else {
              setIsFatigued(false);
              setStatus(`Eyes Healthy (Avg: ${averageBpm.toFixed(1)} BPM)`);
            }
          }
        }
      } catch (error) {
        console.error('System Error during Scanning:', error);
      }

      isProcessingFrame = false;
    };

    init();

    return () => {
      window.removeEventListener('message', handleMessage);
      if (worker) {
        worker.postMessage('stop');
        worker.terminate();
      }
      if (videoRef.current && videoRef.current.srcObject) {
        videoRef.current.srcObject.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  return (
    <div style={containerStyle}>
      <video ref={videoRef} style={videoStyle} muted playsInline />
      <canvas ref={canvasRef} style={canvasStyle} />

      {/* HUD Container (Top Left) */}
      <div style={leftHudStyle}>
        <div
          style={{
            ...counterStyle,
            borderColor: isFatigued ? '#FF3B30' : 'rgba(255,255,255,0.2)',
          }}
        >
          <div
            style={{
              fontSize: 'clamp(9px, 1.5vw, 12px)',
              color: isFatigued ? '#FF3B30' : 'rgba(255,255,255,0.5)',
              marginBottom: '8px',
              fontWeight: 'bold',
            }}
          >
            [ {status.toUpperCase()} ]
          </div>
          <div style={{ fontSize: 'clamp(14px, 3vw, 28px)', color: '#fff' }}>
            SYS_BLINKS:{' '}
            <span style={{ color: 'rgba(255,255,255,0.7)' }}>{blinkCount}</span>
          </div>
          <div
            style={{ fontSize: 'clamp(14px, 3vw, 28px)', color: bpm < 8 ? '#FF3B30' : '#fff' }}
          >
            SYS_BPM: {bpm}
          </div>
        </div>

        {bpm > 0 && (
          <div style={liveDataOverlayStyle}>
            <div
              style={{
                ...liveDataBoxStyle,
                borderLeft:
                  liveRatioRight < 0.25
                    ? '2px solid #fff'
                    : '1px solid rgba(255,255,255,0.2)',
              }}
            >
              [ RIGHT EYE EAR:{liveRatioRight} ]<br />[ COORDS X:{coordsRight.x}{' '}
              Y:{coordsRight.y} ]
            </div>
            <div
              style={{
                ...liveDataBoxStyle,
                borderLeft:
                  liveRatioLeft < 0.25
                    ? '2px solid #fff'
                    : '1px solid rgba(255,255,255,0.2)',
              }}
            >
              [ LEFT EYE EAR:{liveRatioLeft} ]<br />[ COORDS X:{coordsLeft.x} Y:
              {coordsLeft.y} ]
            </div>
          </div>
        )}
      </div>

      {isFatigued && !isResting && (
        <div style={warningOverlayStyle}>
          <h1 style={{ fontSize: 'clamp(18px, 4vw, 32px)', margin: '0 0 12px 0' }}>⚠️ CRITICAL FATIGUE DETECTED</h1>
          <p style={{ fontSize: 'clamp(11px, 2vw, 16px)', margin: '0 0 6px 0' }}>Average blink rate dropped below healthy levels.</p>
          <p style={{ fontSize: 'clamp(11px, 2vw, 16px)', margin: '0 0 20px 0' }}>Implement the 20-20-20 rule immediately.</p>
          <button onClick={handleTakeBreak} style={breakButtonStyle}>
            Okay, I'll take a break
          </button>
        </div>
      )}

      {isShutdown && (
        <div style={{ position: 'absolute', inset: 0, background: '#000', zIndex: 30 }} />
      )}

      {isResting && (
        <div style={restOverlayStyle}>
          <p style={{ fontSize: 'clamp(11px, 2vw, 14px)', margin: '0 0 6px 0', color: 'rgba(255,255,255,0.5)' }}>REST MODE ACTIVE</p>
          <p style={{ fontSize: 'clamp(13px, 2.5vw, 18px)', margin: '0 0 20px 0' }}>Look away from the screen.</p>
          <button onClick={handleIAmBack} style={backButtonStyle}>
            I am back
          </button>
        </div>
      )}
    </div>
  );
}

const containerStyle = {
  position: 'relative',
  width: '100vw',
  height: '100vh',
  background: '#000',
  overflow: 'hidden',
};
const videoStyle = {
  width: '100%',
  height: '100%',
  objectFit: 'cover',
  transform: 'scaleX(-1)',
  filter: 'brightness(0.6)',
};
const canvasStyle = {
  position: 'absolute',
  top: 0,
  left: 0,
  width: '100%',
  height: '100%',
  objectFit: 'cover',
  transform: 'scaleX(-1)',
  zIndex: 1,
};

const leftHudStyle = {
  position: 'absolute',
  top: 'clamp(15px, 3vh, 30px)',
  left: 'clamp(15px, 3vw, 30px)',
  zIndex: 10,
  display: 'flex',
  flexDirection: 'column',
  gap: 'clamp(10px, 2vw, 20px)',
};

const counterStyle = {
  color: '#fff',
  fontFamily: '"JetBrains Mono", monospace',
  fontWeight: '400',
  background: 'rgba(0,0,0,0.6)',
  padding: 'clamp(12px, 2vw, 20px)',
  borderRadius: '4px',
  border: '1px solid rgba(255,255,255,0.2)',
};

const liveDataOverlayStyle = {
  color: '#fff',
  fontFamily: '"JetBrains Mono", monospace',
  textTransform: 'uppercase',
  pointerEvents: 'none',
  display: 'flex',
  flexDirection: 'column',
  gap: 'clamp(5px, 1vw, 8px)',
};
const liveDataBoxStyle = {
  background: 'rgba(0,0,0,0.6)',
  padding: 'clamp(8px, 1.5vw, 12px) clamp(10px, 2vw, 15px)',
  borderRadius: '4px',
  borderLeft: '1px solid rgba(255,255,255,0.2)',
  fontSize: 'clamp(10px, 1.5vw, 13px)',
  color: 'rgba(255,255,255,0.8)',
};

const warningOverlayStyle = {
  position: 'absolute',
  top: '50%',
  left: '50%',
  transform: 'translate(-50%, -50%)',
  backgroundColor: 'rgba(255, 59, 48, 0.9)',
  color: 'white',
  padding: 'clamp(20px, 4vw, 40px)',
  fontFamily: '"JetBrains Mono", monospace',
  textAlign: 'center',
  border: '1px solid white',
  zIndex: 20,
  borderRadius: '4px',
};

const breakButtonStyle = {
  fontFamily: '"JetBrains Mono", monospace',
  fontSize: 'clamp(11px, 1.8vw, 14px)',
  background: 'rgba(255,255,255,0.15)',
  color: '#fff',
  border: '1px solid rgba(255,255,255,0.6)',
  borderRadius: '4px',
  padding: '10px 20px',
  cursor: 'pointer',
  letterSpacing: '0.05em',
};

const restOverlayStyle = {
  position: 'absolute',
  top: '50%',
  left: '50%',
  transform: 'translate(-50%, -50%)',
  backgroundColor: 'rgba(0,0,0,0.85)',
  color: 'white',
  padding: 'clamp(20px, 4vw, 40px)',
  fontFamily: '"JetBrains Mono", monospace',
  textAlign: 'center',
  border: '1px solid rgba(255,255,255,0.2)',
  zIndex: 20,
  borderRadius: '4px',
};

const backButtonStyle = {
  fontFamily: '"JetBrains Mono", monospace',
  fontSize: 'clamp(11px, 1.8vw, 14px)',
  background: 'rgba(255,255,255,0.1)',
  color: '#fff',
  border: '1px solid rgba(255,255,255,0.4)',
  borderRadius: '4px',
  padding: '10px 24px',
  cursor: 'pointer',
  letterSpacing: '0.05em',
};
