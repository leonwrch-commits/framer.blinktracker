import React, { useRef, useEffect, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as faceLandmarksDetection from '@tensorflow-models/face-landmarks-detection';

export default function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  // UI State
  const [blinkCount, setBlinkCount] = useState(0);
  const [bpm, setBpm] = useState(0);
  const [liveRatio, setLiveRatio] = useState(0);
  const [status, setStatus] = useState('Loading AI...');
  const [isFatigued, setIsFatigued] = useState(false);

  // Logic Refs
  const blinkDatesRef = useRef([]);
  const totalBlinksRef = useRef(0);
  const blinkStartTimeRef = useRef(null);
  const appStartTimeRef = useRef(Date.now());
  const audioRef = useRef(new Audio('/alert.mp3'));

  useEffect(() => {
    let detector;
    let worker;
    let isProcessingFrame = false; // Prevents the worker from piling up frames

    const init = async () => {
      try {
        await tf.ready();
        detector = await faceLandmarksDetection.createDetector(
          faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh,
          { runtime: 'tfjs', refineLandmarks: false },
        );
        setStatus('AI Ready. Accessing Camera...');

        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480 },
        });

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = () => {
            videoRef.current.play();
            appStartTimeRef.current = Date.now();
            setStatus('Tracking Active - Gathering Baseline...');
            startWorker(); // Start the background heartbeat
          };
        }
      } catch (err) {
        setStatus('Error: ' + err.message);
      }
    };

    // --- THE BACKGROUND WEB WORKER ---
    const startWorker = () => {
      // 1. Write the worker code as a string
      const workerCode = `
        let timer = null;
        self.onmessage = function(e) {
          if (e.data === 'start') {
            // Send a "tick" 30 times a second (~33ms)
            timer = setInterval(() => self.postMessage('tick'), 33);
          } else if (e.data === 'stop') {
            clearInterval(timer);
          }
        };
      `;

      // 2. Turn it into a URL so the browser can run it
      const blob = new Blob([workerCode], { type: 'application/javascript' });
      worker = new Worker(URL.createObjectURL(blob));

      // 3. Listen for the heartbeat
      worker.onmessage = () => {
        if (!isProcessingFrame) {
          renderFrame();
        }
      };

      // 4. Start the pacemaker
      worker.postMessage('start');
    };

    // The actual AI scanning function (No longer uses requestAnimationFrame)
    const renderFrame = async () => {
      if (!videoRef.current || !detector || !canvasRef.current) return;
      if (videoRef.current.readyState < 2) return;

      isProcessingFrame = true; // Lock the frame so the worker doesn't overwhelm the CPU

      const videoWidth = videoRef.current.videoWidth;
      const videoHeight = videoRef.current.videoHeight;

      videoRef.current.width = videoWidth;
      videoRef.current.height = videoHeight;
      canvasRef.current.width = videoWidth;
      canvasRef.current.height = videoHeight;

      try {
        const faces = await detector.estimateFaces(videoRef.current, {
          flipHorizontal: false,
        });
        const ctx = canvasRef.current.getContext('2d');
        ctx.clearRect(0, 0, videoWidth, videoHeight);

        const now = Date.now();

        if (faces && faces.length > 0) {
          const keypoints = faces[0].keypoints;

          const topLid = keypoints[159];
          const bottomLid = keypoints[145];
          const innerCorner = keypoints[133];
          const outerCorner = keypoints[33];

          const vDist = Math.abs(topLid.y - bottomLid.y);
          const hDist = Math.abs(innerCorner.x - outerCorner.x);
          const ear = hDist > 0 ? vDist / hDist : 0;

          setLiveRatio(ear.toFixed(3));

          const threshold = 0.25;
          const isBlinkingNow = ear < threshold;

          if (isBlinkingNow) {
            if (!blinkStartTimeRef.current) {
              blinkStartTimeRef.current = now;
            }
            ctx.strokeStyle = '#FF3B30';
          } else {
            if (blinkStartTimeRef.current) {
              const durationClosed = now - blinkStartTimeRef.current;

              if (durationClosed > 50 && durationClosed < 600) {
                totalBlinksRef.current += 1;
                setBlinkCount(totalBlinksRef.current);
                blinkDatesRef.current.push(now);
              }
              blinkStartTimeRef.current = null;
            }
            ctx.strokeStyle = '#00FF96';
          }

          ctx.lineWidth = 3;
          [keypoints[145], keypoints[374]].forEach((pt) => {
            ctx.beginPath();
            ctx.arc(pt.x, pt.y, isBlinkingNow ? 15 : 8, 0, 2 * Math.PI);
            ctx.stroke();
          });
        } else {
          setLiveRatio(0);
          blinkStartTimeRef.current = null;
        }

        // --- TREND ANALYSIS ---
        const tenMinutes = 10 * 60 * 1000;
        const oneMinute = 60 * 1000;

        blinkDatesRef.current = blinkDatesRef.current.filter(
          (t) => t > now - tenMinutes,
        );

        const blinksLastMinute = blinkDatesRef.current.filter(
          (t) => t > now - oneMinute,
        ).length;
        setBpm(blinksLastMinute);

        const msRunning = Math.min(now - appStartTimeRef.current, tenMinutes);
        const minutesRunning = msRunning / oneMinute;

        if (minutesRunning >= 2) {
          const averageBpm = blinkDatesRef.current.length / minutesRunning;

          if (averageBpm < 8) {
            setIsFatigued(true);
            setStatus(
              `⚠️ FATIGUE DETECTED (Avg: ${averageBpm.toFixed(1)} BPM)`,
            );
            audioRef.current
              .play()
              .catch((e) => console.log('Audio blocked.', e));
          } else {
            setIsFatigued(false);
            setStatus(`Eyes Healthy (Avg: ${averageBpm.toFixed(1)} BPM)`);
          }
        } else {
          setStatus(
            `Gathering baseline... (${Math.floor(minutesRunning * 60)}s / 120s)`,
          );
        }
      } catch (error) {
        console.error('AI Scanning Error:', error);
      }

      isProcessingFrame = false; // Unlock so the next worker tick can process
    };

    init();

    return () => {
      // Clean up everything when component unmounts
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

      <div
        style={{
          ...counterStyle,
          borderColor: isFatigued ? '#FF3B30' : '#333',
        }}
      >
        <div
          style={{
            fontSize: '14px',
            color: isFatigued ? '#FF3B30' : '#aaa',
            marginBottom: '10px',
            fontWeight: 'bold',
          }}
        >
          {status}
        </div>
        <div style={{ fontSize: '32px', marginBottom: '5px' }}>
          BLINKS: <span style={{ color: '#fff' }}>{blinkCount}</span>
        </div>
        <div
          style={{ fontSize: '32px', color: bpm < 8 ? '#FF3B30' : '#00FF96' }}
        >
          LIVE BPM: {bpm}
        </div>
        <div
          style={{
            marginTop: '20px',
            padding: '10px',
            border: '1px solid #444',
            backgroundColor: 'rgba(0,0,0,0.5)',
          }}
        >
          <div style={{ color: 'yellow', fontSize: '18px' }}>
            EAR RATIO: {liveRatio}
          </div>
          <div style={{ fontSize: '12px', color: '#888' }}>
            Calibrated Threshold: 0.25
          </div>
        </div>
      </div>

      {isFatigued && (
        <div style={warningOverlayStyle}>
          <h1>CRITICAL EYE STRAIN DETECTED</h1>
          <p>Your average blink rate has dropped below healthy levels.</p>
          <p>Look at something 20 feet away for 20 seconds.</p>
        </div>
      )}
    </div>
  );
}

// --- Styles ---
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
};
const canvasStyle = {
  position: 'absolute',
  top: 0,
  left: 0,
  width: '100%',
  height: '100%',
  objectFit: 'cover',
  transform: 'scaleX(-1)',
};
const counterStyle = {
  position: 'absolute',
  top: 30,
  left: 30,
  color: '#00FF96',
  fontFamily: 'monospace',
  fontWeight: 'bold',
  background: 'rgba(0,0,0,0.85)',
  padding: '25px',
  borderRadius: '12px',
  border: '2px solid #333',
  zIndex: 10,
  transition: 'border-color 0.3s ease',
};
const warningOverlayStyle = {
  position: 'absolute',
  top: '50%',
  left: '50%',
  transform: 'translate(-50%, -50%)',
  backgroundColor: 'rgba(255, 59, 48, 0.9)',
  color: 'white',
  padding: '40px',
  fontFamily: 'monospace',
  textAlign: 'center',
  border: '4px solid white',
  zIndex: 20,
  boxShadow: '0 0 100px rgba(255,0,0,0.8)',
  borderRadius: '12px',
};
