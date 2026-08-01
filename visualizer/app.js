import * as THREE from "three";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/examples/jsm/postprocessing/OutputPass.js";

const vertexShader = `
  uniform float u_time;
  uniform float u_frequency;
  uniform float u_speechHz;

  varying vec3 vNormal;
  varying vec3 vViewPosition;
  varying vec3 vPosition;

  // Stable, lightweight trigonometric noise for vertex displacement
  float trigNoise(vec3 p) {
      float n = sin(p.x) * cos(p.y) * sin(p.z);
      n += 0.5 * sin(p.x * 2.1 + p.y) * cos(p.z * 1.9);
      n += 0.25 * sin(p.y * 4.3) * cos(p.x * 3.7 + p.z);
      return n / 1.75;
  }

  void main() {
      // Wave ripple speed scales extremely gently and slowly with audio frequency (preventing volatility)
      float speed = 0.08 + (u_frequency / 255.0) * 0.12;
      float activeTime = u_time * speed;
      vec3 coord = position * 0.85 + vec3(0.0, activeTime, 0.0);
      float noise = trigNoise(coord);
      
      // Compute normal, raw model position, and camera-relative view direction
      vec3 normCam = normalize(normalMatrix * normal);
      vec4 mvPosRaw = modelViewMatrix * vec4(position, 1.0);
      vec3 viewDir = normalize(-mvPosRaw.xyz);
      float dotNV = dot(normCam, viewDir);
      
      // Fresnel-like multiplier: 1.0 at borders (rim), 0.0 at center core
      float borderFactor = pow(1.0 - max(dotNV, 0.0), 1.8);

      // Fresnel Border Jelly Motion intensity depends completely on the real-time Frequency Deformation Index
      // Amplified relative to a lower range (0 to 80) to make wiggles highly visible
      float fdi = clamp(u_frequency / 80.0, 0.0, 1.0);
      float displacement = fdi * 1.8 * noise * borderFactor;
      vec3 newPosition = position + normal * displacement;
      
      vNormal = normalize(normalMatrix * normal);
      vec4 mvPosition = modelViewMatrix * vec4(newPosition, 1.0);
      vViewPosition = -mvPosition.xyz;
      vPosition = newPosition; // Pass deformed position to fragment shader
      
      // Set perspective point size
      gl_PointSize = 110.0 / -mvPosition.z;
      
      gl_Position = projectionMatrix * mvPosition;
  }
`;

const fragmentShader = `
  uniform float u_time;
  uniform float u_frequency;
  uniform vec3 u_colorCenter;
  uniform vec3 u_colorCrescent;
  uniform vec3 u_colorRim;

  varying vec3 vNormal;
  varying vec3 vViewPosition;
  varying vec3 vPosition;

  void main() {
      // Shape particles as soft glowing circles
      vec2 pc = gl_PointCoord - vec2(0.5);
      float dist = length(pc);
      if (dist > 0.5) discard;
      
      float particleGlow = smoothstep(0.5, 0.05, dist);

      vec3 normal = normalize(vNormal);
      vec3 viewDir = normalize(vViewPosition);
      
      float dotNV = dot(normal, viewDir);
      
      // Fresnel effect for outer glass rim reflection (thin glowing edge)
      float fresnel = pow(1.0 - max(dotNV, 0.0), 3.0);
      
      // Gradient mapping: vPosition.y ranges from -3.5 to 3.5.
      // Transition center is shifted down to -1.0 so that 70% of the height is u_colorCenter.
      float colorMix = smoothstep(-2.2, -0.5, vPosition.y + vPosition.x * 0.4);
      vec3 colGrid = mix(u_colorCrescent, u_colorCenter, colorMix);
      
      // High-contrast, bright neon blue outer rim highlight (further amplified for maximum edge contrast)
      vec3 colRim = u_colorRim * fresnel * 4.2;
      
      // Blend layers: core stays soft (0.68) to prevent blowout, rim has extreme contrast (4.2)
      vec3 finalColor = (colGrid * 0.68 + colRim) * particleGlow;
      
      // Extremely gentle audio frequency scaling to prevent contrast spike/blowout
      float amp = 1.0 + u_frequency * 0.006;
      finalColor *= amp;
      
      // Translucency setup:
      float alpha = mix(0.32, 0.95, fresnel) * particleGlow;
      
      gl_FragColor = vec4(finalColor, alpha);
  }
`;

// --- UI / Audio Variables ---
let audioContext = null;
let analyser = null;
let dataArray = null;
let isAudioActive = false;
let currentState = "idle";
let hasSpeechStarted = false;

// --- WebGL Setup ---
const container = document.getElementById("canvas-container");
let W = window.innerWidth;
let H = window.innerHeight;

let mouseX = 0;
let mouseY = 0;

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(W, H);
renderer.outputColorSpace = THREE.SRGBColorSpace;
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, W / H, 0.1, 1000);
camera.position.set(0, -2, 22);
camera.lookAt(0, 0, 0);

// --- Post-processing ---
const renderScene = new RenderPass(scene, camera);
const bloomPass = new UnrealBloomPass(new THREE.Vector2(W, H));
bloomPass.threshold = 0.35;
bloomPass.strength = 0.28; // Toned down from 0.65 to 0.28 to completely prevent blowout and irritation
bloomPass.radius = 0.32;

const bloomComposer = new EffectComposer(renderer);
bloomComposer.addPass(renderScene);
bloomComposer.addPass(bloomPass);

const outputPass = new OutputPass();
bloomComposer.addPass(outputPass);

// --- Dynamic Color Configurations by State ---
const stateColors = {
  idle: {
    center: new THREE.Color(0.02, 0.08, 0.3),     // Deep navy
    crescent: new THREE.Color(0.0, 0.3, 0.8),      // Royal blue
    rim: new THREE.Color(0.0, 0.5, 1.0)           // Glowing cyan-blue
  },
  listening: {
    center: new THREE.Color(0.0, 0.25, 0.15),     // Deep emerald
    crescent: new THREE.Color(0.0, 0.75, 0.45),    // Bright green
    rim: new THREE.Color(0.2, 0.9, 0.6)           // Glowing mint-teal
  },
  speaking: {
    center: new THREE.Color(0.25, 0.02, 0.15),     // Deep plum
    crescent: new THREE.Color(0.85, 0.0, 0.45),    // Vibrant rose
    rim: new THREE.Color(1.0, 0.3, 0.7)           // Glowing neon pink
  },
  thinking: {
    center: new THREE.Color(0.12, 0.02, 0.25),     // Deep indigo/purple
    crescent: new THREE.Color(0.9, 0.4, 0.0),      // Amber orange
    rim: new THREE.Color(1.0, 0.65, 0.0)          // Glowing electric gold
  }
};

const targetColors = {
  center: stateColors.idle.center.clone(),
  crescent: stateColors.idle.crescent.clone(),
  rim: stateColors.idle.rim.clone()
};

// --- Uniforms & Material ---
const uniforms = {
  u_time: { value: 0.0 },
  u_frequency: { value: 0.0 },
  u_speechHz: { value: 0.0 },
  u_colorCenter: { value: stateColors.idle.center.clone() },
  u_colorCrescent: { value: stateColors.idle.crescent.clone() },
  u_colorRim: { value: stateColors.idle.rim.clone() }
};

const mat = new THREE.ShaderMaterial({
  uniforms,
  vertexShader: vertexShader,
  fragmentShader: fragmentShader,
  transparent: true,
  depthWrite: false,
});

const geo = new THREE.SphereGeometry(3.5, 110, 55); // Sphere geometry mapped to concentric horizontal rings of dots
const mesh = new THREE.Points(geo, mat); // Rendered as Point Cloud
scene.add(mesh);

// --- Event Listeners ---
const onMouseMove = (e) => {
  const halfX = window.innerWidth / 2;
  const halfY = window.innerHeight / 2;
  mouseX = (e.clientX - halfX) / 100;
  mouseY = (e.clientY - halfY) / 100;
};
document.addEventListener("mousemove", onMouseMove);

const onResize = () => {
  W = window.innerWidth;
  H = window.innerHeight;
  camera.aspect = W / H;
  camera.updateProjectionMatrix();
  renderer.setSize(W, H);
  bloomComposer.setSize(W, H);
};
window.addEventListener("resize", onResize);

// --- State Manager ---
const setVisualizerState = (state) => {
  const validStates = ["idle", "listening", "speaking", "thinking"];
  if (!validStates.includes(state)) return;

  currentState = state;

  // Keep colors uniform across all states (all states use idle/ideal colors)
  targetColors.center.copy(stateColors.idle.center);
  targetColors.crescent.copy(stateColors.idle.crescent);
  targetColors.rim.copy(stateColors.idle.rim);

  // Reset standard transformations
  mesh.rotation.set(0, 0, 0);

  // Update button elements active state in the HUD
  document.querySelectorAll(".state-btn").forEach((btn) => {
    if (btn.getAttribute("data-state") === state) {
      btn.classList.add("active");
    } else {
      btn.classList.remove("active");
    }
  });

  // Update dynamic background shadow gradient layer visibility
  document.querySelectorAll(".bg-layer").forEach((layer) => {
    if (layer.classList.contains(`bg-${state}`)) {
      layer.classList.add("active");
    } else {
      layer.classList.remove("active");
    }
  });


  // Update State readout in HUD
  const stateValEl = document.getElementById("system-state");
  if (stateValEl) {
    stateValEl.innerText = state.toUpperCase();
    if ((state === "listening" || state === "speaking") && isAudioActive) {
      stateValEl.innerText = state === "listening" ? "STREAMING (IN)" : "STREAMING (OUT)";
    }
  }
};
window.setVisualizerState = setVisualizerState;

// Visualizer state changes react purely to real-time agent signals

// --- Voice WebRTC Dual-DataChannel & Playback Logic ---
const interruptBtn = document.getElementById("interrupt-btn");
const clearBtn = document.getElementById("clear-btn");
const captionsPanel = document.getElementById("captions");
const statusDot = document.getElementById("status-dot");
const statusText = document.getElementById("status-text");

let peerConnection = null;
let mediaChannel = null;
let updatesChannel = null;
let micStream = null;
let scriptProcessor = null;
let activeSources = [];
let nextStartTime = 0;
let isConnected = false;
let lastRole = null;
let lastCaptionTextSpan = null;

const initAudioEngine = () => {
  try {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    
    // Master visualizer analyser
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 64;
    dataArray = new Uint8Array(analyser.frequencyBinCount);
    isAudioActive = true;

    // Connect analyser to a zero-gain path to prevent Chrome/Edge optimizations from pruning it
    const silence = audioContext.createGain();
    silence.gain.setValueAtTime(0, audioContext.currentTime);
    analyser.connect(silence);
    silence.connect(audioContext.destination);
  } catch (err) {
    console.error("Audio system activation failure:", err);
  }
};

// Engage core overlay button
const startBtn = document.getElementById("start-btn");
if (startBtn) {
  startBtn.addEventListener("click", async () => {
    const startOverlay = document.getElementById("start-overlay");
    if (startOverlay) {
      startOverlay.classList.remove("overlay-active");
    }
    if (!audioContext) {
      initAudioEngine();
    }
    await connect();
  });
}

const micBtn = document.getElementById("mic-btn");
if (micBtn) {
  micBtn.addEventListener("click", async () => {
    if (!audioContext) {
      initAudioEngine();
    }
    if (!isConnected) {
      await connect();
    } else {
      disconnect();
    }
  });
}

if (interruptBtn) {
  interruptBtn.addEventListener("click", () => {
    if (mediaChannel && mediaChannel.readyState === "open") {
      mediaChannel.send(JSON.stringify({ type: "interrupt" }));
    }
    interruptPlayback();
  });
}

if (clearBtn) {
  clearBtn.addEventListener("click", () => {
    captionsPanel.innerHTML = "";
    lastRole = null;
    lastCaptionTextSpan = null;
  });
}

// Chat text-input listeners
const chatInput = document.getElementById("chat-input");
const chatSendBtn = document.getElementById("chat-send-btn");

const sendTextMessage = () => {
  if (!chatInput) return;
  const text = chatInput.value.trim();
  if (!text) return;

  if (mediaChannel && mediaChannel.readyState === "open") {
    mediaChannel.send(JSON.stringify({ type: "text", message: text }));
    interruptPlayback(); // Instantly silence client output
    appendCaption("You", text, "user");
    chatInput.value = "";
  } else {
    appendCaption("System", "Directive offline. Connect first by engaging core.", "assistant");
  }
};

if (chatSendBtn) {
  chatSendBtn.addEventListener("click", sendTextMessage);
}
if (chatInput) {
  chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      sendTextMessage();
    }
  });
}

async function connect() {
  if (audioContext.state === "suspended") {
    await audioContext.resume();
  }

  statusText.innerText = "CONNECTING";
  statusDot.className = "status-dot offline";
  
  const sessionId = "live-session-" + Math.random().toString(36).substring(2, 9);
  const sessionIdValEl = document.getElementById("system-session-id");
  if (sessionIdValEl) {
    sessionIdValEl.innerText = sessionId;
  }

  try {
    peerConnection = new RTCPeerConnection({
      iceServers: []
    });

    // Create 2 WebRTC DataChannels:
    // Channel 1: media_channel (Audio, Data & Artifacts)
    // Channel 2: live_updates (SSE Event Stream in real-time)
    mediaChannel = peerConnection.createDataChannel("media_channel");
    updatesChannel = peerConnection.createDataChannel("live_updates");

    mediaChannel.binaryType = "arraybuffer";

    mediaChannel.onopen = async () => {
      isConnected = true;
      if (interruptBtn) interruptBtn.disabled = false;
      statusText.innerText = "ONLINE";
      statusDot.className = "status-dot online";

      const micStatusEl = document.getElementById("system-mic-status");
      if (micStatusEl) micStatusEl.innerText = "RECORDING";

      appendCaption("System", "WebRTC PeerConnection established (Dual DataChannels: media & live_updates).", "assistant");
      setVisualizerState("listening");

      try {
        await startRecording();
      } catch (e) {
        appendCaption("System", `Microphone error: ${e.message}`, "assistant");
        disconnect();
      }
    };

    mediaChannel.onmessage = (event) => {
      let payload;
      if (typeof event.data === "string") {
        try { payload = JSON.parse(event.data); } catch(e) { return; }
      } else {
        return;
      }

      if (payload.type === "audio") {
        setVisualizerState("speaking");
        const base64Audio = payload.data;
        const binaryString = atob(base64Audio);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
          bytes[i] = binaryString.charCodeAt(i);
        }
        const dataView = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
        const float32Data = pcm16ToFloat32(dataView);
        playAudioChunk(float32Data);
      }
    };

    // Data Channel 2 (live_updates): Real-time SSE event updates
    updatesChannel.onmessage = (event) => {
      let payload;
      try { payload = JSON.parse(event.data); } catch(e) { return; }

      if (payload.type === "caption") {
        appendCaption(payload.role === "user" ? "You" : "Assistant", payload.text, payload.role);
      }
      else if (payload.type === "thinking") {
        setVisualizerState("thinking");
      }
      else if (payload.type === "interrupted") {
        interruptPlayback();
      }
      else if (payload.type === "tool_start") {
        createToolCard(payload.call_id, payload.name, payload.args);
      }
      else if (payload.type === "tool_complete") {
        updateToolCard(payload.call_id, payload.output);
      }
      else if (payload.type === "error") {
        appendCaption("Error", payload.message, "user");
      }
    };

    peerConnection.oniceconnectionstatechange = () => {
      if (peerConnection.iceConnectionState === "failed" || peerConnection.iceConnectionState === "closed") {
        disconnect();
      }
    };

    // Negotiate WebRTC SDP Offer / Answer
    const offer = await peerConnection.createOffer();
    await peerConnection.setLocalDescription(offer);

    const res = await fetch("/webrtc/offer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sdp: offer.sdp, type: offer.type, session_id: sessionId })
    });

    if (!res.ok) {
      throw new Error(`WebRTC offer failed with HTTP ${res.status}`);
    }

    const answer = await res.json();
    await peerConnection.setRemoteDescription(new RTCSessionDescription(answer));

  } catch (e) {
    appendCaption("System", `WebRTC connection error: ${e.message}`, "assistant");
    disconnect();
  }
}

function disconnect() {
  isConnected = false;
  if (interruptBtn) interruptBtn.disabled = true;
  statusText.innerText = "OFFLINE";
  statusDot.className = "status-dot offline";
  
  const micStatusEl = document.getElementById("system-mic-status");
  if (micStatusEl) micStatusEl.innerText = "DISCONNECTED";
  
  appendCaption("System", "Connection closed. Session consolidated.", "assistant");
  setVisualizerState("idle");
  hasSpeechStarted = false;

  stopRecording();
  interruptPlayback();

  lastRole = null;
  lastCaptionTextSpan = null;

  if (mediaChannel) {
    try { mediaChannel.close(); } catch(e){}
    mediaChannel = null;
  }
  if (updatesChannel) {
    try { updatesChannel.close(); } catch(e){}
    updatesChannel = null;
  }
  if (peerConnection) {
    try { peerConnection.close(); } catch(e){}
    peerConnection = null;
  }
}

// --- Audio Streams Capture ---
async function startRecording() {
  micStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true
    }
  });

  const source = audioContext.createMediaStreamSource(micStream);
  source.connect(analyser);

  scriptProcessor = audioContext.createScriptProcessor(1024, 1, 1);
  source.connect(scriptProcessor);
  
  // Connect to a zero-gain node to process audio without hearing it back
  const silence = audioContext.createGain();
  silence.gain.setValueAtTime(0, audioContext.currentTime);
  scriptProcessor.connect(silence);
  silence.connect(audioContext.destination);

  scriptProcessor.onaudioprocess = (e) => {
    if (!isConnected || !mediaChannel || mediaChannel.readyState !== "open") return;

    const inputBuffer = e.inputBuffer.getChannelData(0);

    // Local VAD volume tracking
    let sum = 0;
    for (let i = 0; i < inputBuffer.length; i++) {
      sum += inputBuffer[i] * inputBuffer[i];
    }
    const rms = Math.sqrt(sum / inputBuffer.length);
    if (rms > 0.015 && activeSources.length > 0) {
      console.log("[ClientVAD] Speech threshold exceeded! Interrupting playback. RMS:", rms);
      interruptPlayback();
      mediaChannel.send(JSON.stringify({ type: "interrupt" }));
    }

    const downsampled = downsampleBuffer(inputBuffer, audioContext.sampleRate, 16000);
    const pcmBuffer = new ArrayBuffer(downsampled.length * 2);
    const view = new DataView(pcmBuffer);
    floatTo16BitPCM(view, 0, downsampled);

    mediaChannel.send(pcmBuffer);
  };
}

function stopRecording() {
  if (micStream) {
    micStream.getTracks().forEach(t => t.stop());
    micStream = null;
  }
  if (scriptProcessor) {
    scriptProcessor.disconnect();
    scriptProcessor = null;
  }
}

// --- Scheduled Playback ---
function playAudioChunk(float32Data) {
  if (audioContext.state === "suspended") {
    audioContext.resume();
  }
  const buffer = audioContext.createBuffer(1, float32Data.length, 24000);
  buffer.getChannelData(0).set(float32Data);

  const source = audioContext.createBufferSource();
  source.buffer = buffer;
  
  source.connect(analyser);
  source.connect(audioContext.destination);

  let startTime = nextStartTime;
  const currentTime = audioContext.currentTime;
  if (startTime < currentTime) {
    startTime = currentTime + 0.04;
  }

  source.start(startTime);
  nextStartTime = startTime + buffer.duration;

  activeSources.push(source);
  source.onended = () => {
    activeSources = activeSources.filter(s => s !== source);
    if (activeSources.length === 0 && currentState === "speaking") {
      setVisualizerState("idle");
    }
  };
}

function interruptPlayback() {
  activeSources.forEach(src => {
    try { src.stop(); } catch(e) {}
  });
  activeSources = [];
  nextStartTime = 0;
  setVisualizerState("idle");
}

// --- Helpers ---
function appendCaption(role, text, type) {
  if (!captionsPanel) return;
  // For consecutive updates from the same role, append streamed chunks
  // into the same line. This handles both assistant response streaming
  // and word-by-word user speech transcription from Gemini Live.
  if (lastRole === role && lastCaptionTextSpan) {
    const lastText = lastCaptionTextSpan.innerText;
    const lastChar = lastText.slice(-1);
    const firstChar = text.charAt(0);
    
    // Inject a space if appending a word chunk directly onto another word chunk without spaces/punctuation
    const needsSpace = lastChar && 
                       !/\s/.test(lastChar) && 
                       !/\s/.test(firstChar) && 
                       !/[.,\/#!$%\^&\*;:{}=\-_`~()?]/.test(firstChar) &&
                       !/[.,\/#!$%\^&\*;:{}=\-_`~()?]/.test(lastChar);
    
    if (needsSpace) {
      lastCaptionTextSpan.innerText += " ";
    }
    lastCaptionTextSpan.innerText += text;
  } else {
    const div = document.createElement("div");
    div.className = `caption-line ${type || "assistant"}`;
    div.style.animation = "fadeIn 0.3s ease";
    
    const roleSpan = document.createElement("span");
    roleSpan.className = "role";
    roleSpan.style.fontFamily = "var(--font-mono)";
    roleSpan.style.fontSize = "0.65rem";
    roleSpan.style.fontWeight = "600";
    roleSpan.style.textTransform = "uppercase";
    roleSpan.style.display = "block";
    roleSpan.style.marginBottom = "2px";
    roleSpan.style.color = type === "user" ? "#a855f7" : "#0088ff";
    roleSpan.innerText = role;

    const textSpan = document.createElement("span");
    textSpan.className = "text";
    textSpan.style.color = "#f0f4f8";
    
    div.appendChild(roleSpan);
    div.appendChild(textSpan);
    captionsPanel.appendChild(div);
    
    lastCaptionTextSpan = textSpan;
    lastCaptionTextSpan.innerText = text;
    lastRole = role;
  }
  captionsPanel.scrollTop = captionsPanel.scrollHeight;
}

function pcm16ToFloat32(dataView) {
  const floatArray = new Float32Array(dataView.byteLength / 2);
  for (let i = 0; i < floatArray.length; i++) {
    const int16 = dataView.getInt16(i * 2, true);
    floatArray[i] = int16 / 32768.0;
  }
  return floatArray;
}

function downsampleBuffer(buffer, inputSampleRate, outputSampleRate) {
  if (inputSampleRate === outputSampleRate) return buffer;
  const ratio = inputSampleRate / outputSampleRate;
  const newLength = Math.round(buffer.length / ratio);
  const result = new Float32Array(newLength);
  let offsetResult = 0;
  let offsetBuffer = 0;
  while (offsetResult < result.length) {
    const nextOffset = Math.round((offsetResult + 1) * ratio);
    let accum = 0, count = 0;
    for (let i = offsetBuffer; i < nextOffset && i < buffer.length; i++) {
      accum += buffer[i];
      count++;
    }
    result[offsetResult] = count > 0 ? accum / count : 0;
    offsetResult++;
    offsetBuffer = nextOffset;
  }
  return result;
}

function floatTo16BitPCM(output, offset, input) {
  for (let i = 0; i < input.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, input[i]));
    output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }
}

let lastTime = performance.now();
let accumulatedTime = 0;
let baselineRotationY = 0;

// --- Animation Loop ---
const animate = () => {
  const now = performance.now();
  const delta = (now - lastTime) * 0.001;
  lastTime = now;

  let speedMultiplier = 1.0;
  let targetFrequency = 0;
  let micVolume = 0;
  let peakValue = 0;

  const progressFill = document.getElementById("frequency-fill");
  const freqValEl = document.getElementById("frequency-val");

  // Query analyser if connected
  if (isConnected && isAudioActive && analyser) {
    analyser.getByteFrequencyData(dataArray);
    let sum = 0;
    let maxVal = 0;
    for (let i = 0; i < dataArray.length; i++) {
      const val = dataArray[i];
      sum += val;
      if (val > maxVal) maxVal = val;
    }
    const avg = sum / dataArray.length;
    micVolume = avg * 0.45 + maxVal * 0.55;
    peakValue = maxVal;
  }

  // Real-time State Transition Heuristics (0-100 FDI model)
  if (!isConnected) {
    setVisualizerState("idle");
    hasSpeechStarted = false;
  } else {
    // Speech active threshold set to 16 to prevent room hum/static noise triggers from wiggling
    if (peakValue > 16) {
      if (activeSources.length > 0) {
        setVisualizerState("speaking");
      } else {
        setVisualizerState("listening");
      }
    } else {
      // Silence/Waiting: return immediately to calm idle state
      setVisualizerState("idle");
    }
  }

  // State-specific rendering and updates
  if (currentState === "idle") {
    speedMultiplier = 0.0;
    accumulatedTime += delta * speedMultiplier;
    
    targetFrequency = 0.0;

    if (progressFill) progressFill.style.width = "4%";
    if (freqValEl) freqValEl.innerText = "0.00 Hz";

    // Static: freeze rotation and scale
    mesh.scale.set(1.0, 1.0, 1.0);

    camera.position.x += (mouseX - camera.position.x) * 0.03;
    camera.position.y += (-mouseY - camera.position.y) * 0.03;
    camera.position.z += (22.0 - camera.position.z) * 0.03;
    camera.lookAt(scene.position);

  } else {
    // All active states (listening, speaking, thinking) behave uniformly
    speedMultiplier = 1.0;
    accumulatedTime += delta * speedMultiplier;

    // Determine targetFrequency and update progress bar & Hz readout based on state
    if (currentState === "thinking") {
      targetFrequency = 0.0;
      if (progressFill) {
        progressFill.style.width = `${35 + Math.sin(accumulatedTime * 6.0) * 15}%`;
      }
      if (freqValEl) {
        const thinkingFreq = Math.round(380 + Math.sin(accumulatedTime * 12.0) * 80);
        freqValEl.innerText = `${thinkingFreq} Hz`;
      }
      uniforms.u_speechHz.value += (0.0 - uniforms.u_speechHz.value) * 0.15;

    } else if (currentState === "listening") {
      targetFrequency = micVolume > 15 ? micVolume : 0.0;
      const normalizedFreq = targetFrequency / 255;
      if (progressFill) progressFill.style.width = `${Math.min(normalizedFreq * 100, 100)}%`;

      let maxIndex = 0;
      let maxPeak = 0;
      for (let i = 0; i < dataArray.length; i++) {
        if (dataArray[i] > maxPeak) {
          maxPeak = dataArray[i];
          maxIndex = i;
        }
      }
      const sampleRate = audioContext.sampleRate;
      const fftSize = analyser.fftSize;
      const peakHz = Math.round(maxIndex * (sampleRate / fftSize));
      const currentHz = maxPeak > 15 ? peakHz : 0.0;
      if (freqValEl) freqValEl.innerText = maxPeak > 15 ? `${peakHz} Hz` : "0.00 Hz";
      uniforms.u_speechHz.value += (currentHz - uniforms.u_speechHz.value) * 0.15;

    } else if (currentState === "speaking") {
      if (micVolume > 15) {
        targetFrequency = micVolume;
        const normalizedFreq = targetFrequency / 255;
        if (progressFill) progressFill.style.width = `${Math.min(normalizedFreq * 100, 100)}%`;

        let maxIndex = 0;
        let maxPeak = 0;
        for (let i = 0; i < dataArray.length; i++) {
          if (dataArray[i] > maxPeak) {
            maxPeak = dataArray[i];
            maxIndex = i;
          }
        }
        const sampleRate = audioContext.sampleRate;
        const fftSize = analyser.fftSize;
        const peakHz = Math.round(maxIndex * (sampleRate / fftSize));
        const currentHz = maxPeak > 15 ? peakHz : 0.0;
        if (freqValEl) freqValEl.innerText = maxPeak > 15 ? `${peakHz} Hz` : "0.00 Hz";
        uniforms.u_speechHz.value += (currentHz - uniforms.u_speechHz.value) * 0.15;
      } else {
        targetFrequency = 0.0;
        if (progressFill) progressFill.style.width = "4%";
        if (freqValEl) {
          freqValEl.innerText = "0.00 Hz";
          uniforms.u_speechHz.value += (0.0 - uniforms.u_speechHz.value) * 0.15;
        }
      }
    }

    // Stationary (no rotation) and stable scale
    mesh.scale.set(1.0, 1.0, 1.0);

    // Uniform camera tracking
    const isVoiceActive = (currentState === "speaking" && (micVolume > 15 || targetFrequency > 4.0));
    const vibration = isVoiceActive ? Math.sin(accumulatedTime * 65.0) * 0.02 : 0;
    
    camera.position.x += (mouseX - camera.position.x) * 0.05;
    camera.position.y += (-mouseY + vibration - camera.position.y) * 0.05;
    camera.position.z += (22.0 - camera.position.z) * 0.05;
    camera.lookAt(scene.position);
  }

  uniforms.u_time.value = accumulatedTime;
  uniforms.u_frequency.value += (targetFrequency - uniforms.u_frequency.value) * 0.26;

  // Smoothen and transition target colors based on the state machine
  uniforms.u_colorCenter.value.lerp(targetColors.center, 0.05);
  uniforms.u_colorCrescent.value.lerp(targetColors.crescent, 0.05);
  uniforms.u_colorRim.value.lerp(targetColors.rim, 0.05);

  bloomComposer.render();
  updateVisualizerThreads(); // Keep SVG lines updated on resize or frame updates
  requestAnimationFrame(animate);
};


// --- Dynamic Tool Cards & SVG Threads Manager ---
const activeToolCards = new Map(); // call_id -> { element, sectorIndex, pathElement, x, y, isDragging }
const occupiedSectors = new Set(); // track sector indices that are in use
let recentCallId = null; // Track the most recently created tool card ID

// Dynamic logger details generator
const getMockLogs = (toolName, args) => {
  const logs = [];
  if (toolName === "retrieve_rag_context") {
    logs.push(`[SYSTEM] Connecting to Chroma Vector DB...`);
    logs.push(`[RAG] Query: "${args.query || ''}"`);
    logs.push(`[RAG] Scanning 3D sentence embeddings...`);
    logs.push(`[RAG] Comparing cosine similarity scores...`);
    logs.push(`[SYSTEM] Retrieving matched documents...`);
  } else if (toolName === "search_web") {
    logs.push(`[SYSTEM] Launching DuckDuckGo scraper...`);
    logs.push(`[SCRAPER] Query: "${args.query || ''}"`);
    logs.push(`[SCRAPER] HTTP GET status 200 returned.`);
    logs.push(`[AGENT] Summarizing search results...`);
  } else if (toolName === "get_live_weather") {
    logs.push(`[API] Resolving location coordinates...`);
    logs.push(`[METEO] Querying Open-Meteo endpoint...`);
    logs.push(`[SYSTEM] Parsing temperature metadata...`);
  } else {
    logs.push(`[SYSTEM] Invoking agent tool: ${toolName}...`);
    logs.push(`[SYSTEM] Arguments parsed and verified.`);
  }
  return logs;
};

// Dynamically scale card classes based on the number of active cards
const updateCardSizes = () => {
  const count = activeToolCards.size;
  activeToolCards.forEach((card) => {
    const el = card.element;
    el.classList.remove("size-compact", "size-super-compact");
    if (count >= 2 && count <= 4) {
      el.classList.add("size-compact");
      card.width = 250;
      card.height = 160;
    } else if (count > 4) {
      el.classList.add("size-super-compact");
      card.width = 190;
      card.height = 120;
    } else {
      card.width = 330;
      card.height = 250;
    }
  });
  updateThreadPaths();
};

// Predefined available grid sectors surrounding the visualizer with collision checks
const getAvailableSector = () => {
  const W = window.innerWidth;
  const H = window.innerHeight;
  const centerX = W / 2;
  const centerY = H / 2;
  
  // Measure Live Transcript panel dynamically from the DOM (supports media queries)
  const transcriptEl = document.querySelector(".left-panel");
  const transcriptRect = transcriptEl ? transcriptEl.getBoundingClientRect() : null;

  // Bounding box of the Live Transcript panel on the far left with a 15px safety gap
  const transcriptBox = {
    left: 30,
    right: transcriptRect ? transcriptRect.right + 15 : 385,
    top: transcriptRect ? transcriptRect.top : 120,
    bottom: transcriptRect ? transcriptRect.bottom + 15 : H - 45
  };

  // Helper to check if two boxes overlap
  const checkOverlap = (box1, box2) => {
    return !(box1.right < box2.left || 
             box1.left > box2.right || 
             box1.bottom < box2.top || 
             box1.top > box2.bottom);
  };

  const count = activeToolCards.size + 1; // estimate count including this new card
  const cardHeight = count > 4 ? 120 : (count >= 2 ? 160 : 250);
  const cardWidth = count > 4 ? 190 : (count >= 2 ? 250 : 330);

  // Generate candidates dynamically in an elliptical ring surrounding the visualizer
  const candidates = [];
  const isLarge = (W > 1400 && H > 800);
  
  // Symmetric angles configuration surrounding the central visualizer sphere
  const slotsConfig = [
    { angle: -55 * Math.PI / 180, rx: isLarge ? 410 : 360, ry: isLarge ? 270 : 220 },  // 0. Top Right
    { angle: -18 * Math.PI / 180, rx: isLarge ? 410 : 360, ry: isLarge ? 270 : 220 },  // 1. Upper Right
    { angle:  18 * Math.PI / 180, rx: isLarge ? 410 : 360, ry: isLarge ? 270 : 220 },  // 2. Lower Right
    { angle:  55 * Math.PI / 180, rx: isLarge ? 410 : 360, ry: isLarge ? 270 : 220 },  // 3. Bottom Right
    { angle: -125 * Math.PI / 180, rx: isLarge ? 410 : 360, ry: isLarge ? 270 : 220 }, // 4. Top Left
    { angle: -162 * Math.PI / 180, rx: isLarge ? 410 : 360, ry: isLarge ? 270 : 220 }, // 5. Upper Left
    { angle:  162 * Math.PI / 180, rx: isLarge ? 410 : 360, ry: isLarge ? 270 : 220 }, // 6. Lower Left
    { angle:  125 * Math.PI / 180, rx: isLarge ? 410 : 360, ry: isLarge ? 270 : 220 }, // 7. Bottom Left
    { angle: -90 * Math.PI / 180, rx: isLarge ? 410 : 360, ry: isLarge ? 340 : 290 },  // 8. Top Center (Wider vertical depth)
    { angle:  90 * Math.PI / 180, rx: isLarge ? 410 : 360, ry: isLarge ? 340 : 290 }   // 9. Bottom Center (Wider vertical depth)
  ];

  slotsConfig.forEach((slot) => {
    // Dynamic orbit spacing: push cards further away to screen boundaries if there are few cards
    const orbitMultiplier = count <= 2 ? 1.25 : 1.0;
    const cardCenterX = centerX + slot.rx * orbitMultiplier * Math.cos(slot.angle);
    const cardCenterY = centerY + slot.ry * orbitMultiplier * Math.sin(slot.angle);
    
    // Left-side cards are clamped to stay to the right of the transcript panel
    const isLeft = Math.cos(slot.angle) < 0;
    const minX = isLeft ? transcriptBox.right : 30;
    const maxX = W - cardWidth - 30;

    candidates.push({
      x: Math.max(minX, Math.min(maxX, cardCenterX - cardWidth / 2)),
      y: Math.max(120, Math.min(H - cardHeight - 130, cardCenterY - cardHeight / 2))
    });
  });

  // Find the first candidate that doesn't overlap the visualizer, the transcript, or active cards
  for (let i = 0; i < candidates.length; i++) {
    // 0. Double-reservation check
    if (occupiedSectors.has(i)) continue;

    const pos = candidates[i];
    const cardBox = {
      left: pos.x,
      right: pos.x + cardWidth,
      top: pos.y,
      bottom: pos.y + cardHeight
    };

    // 1. Check closest distance from visualizer center to card boundary (AABB-to-circle collision check)
    const closestX = Math.max(cardBox.left, Math.min(centerX, cardBox.right));
    const closestY = Math.max(cardBox.top, Math.min(centerY, cardBox.bottom));
    const distToVisualizer = Math.hypot(centerX - closestX, centerY - closestY);
    
    // Scale visualizer safety buffer dynamically depending on card size
    const minDistance = cardWidth > 250 ? 175 : (cardWidth > 190 ? 160 : 152);
    if (distToVisualizer < minDistance) continue;

    // 2. Check overlap with Live Transcript
    if (checkOverlap(cardBox, transcriptBox)) continue;

    // 3. Check overlap with other active cards (using deterministic logical dimensions)
    let overlapsActiveCard = false;
    activeToolCards.forEach((otherCard) => {
      const otherWidth = otherCard.width || cardWidth;
      const otherHeight = otherCard.height || cardHeight;
      const otherBox = {
        left: otherCard.x,
        right: otherCard.x + otherWidth,
        top: otherCard.y,
        bottom: otherCard.y + otherHeight
      };
      if (checkOverlap(cardBox, otherBox)) {
        overlapsActiveCard = true;
      }
    });

    if (overlapsActiveCard) continue;

    // Safe, non-overlapping slot found!
    occupiedSectors.add(i);
    return { coords: pos, index: i };
  }

  // Fallback: stack in neat columns from right-to-left with spacing matching cardHeight to prevent overlap
  const stepY = cardHeight + 20;
  const columnHeight = H - 250; // Total height workspace for fallback stacking
  const maxCardsPerColumn = Math.max(1, Math.floor(columnHeight / stepY));
  
  const columnIndex = Math.floor(activeToolCards.size / maxCardsPerColumn);
  const cardIndexInColumn = activeToolCards.size % maxCardsPerColumn;

  const fallbackX = W - cardWidth - 50 - columnIndex * (cardWidth + 20);
  const fallbackY = 120 + cardIndexInColumn * stepY;

  return { 
    coords: { x: fallbackX, y: fallbackY }, 
    index: -1 
  };
};

// Create a new SVG path element for the flexible thread
const createThreadPath = (call_id) => {
  const svg = document.getElementById("threads-svg");
  if (!svg) return null;
  const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
  path.setAttribute("id", `thread-${call_id}`);
  path.setAttribute("class", "thread-line");
  svg.appendChild(path);
  return path;
};

// Draw Bezier curves connecting the visualizer's circumference to the left edge of each card
const updateThreadPaths = () => {
  const svg = document.getElementById("threads-svg");
  if (!svg) return;

  const W = window.innerWidth;
  const H = window.innerHeight;

  // Dynamically configure SVG viewBox and dimensions to match screen pixels 1-to-1
  svg.setAttribute("width", W);
  svg.setAttribute("height", H);
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);

  const centerX = W / 2;
  const centerY = H / 2;

  // Estimate visualizer radius in screen pixels based on camera distance (22.0) and FOV (45)
  const sphereRadius3D = 3.5;
  const frustumHeight = 2 * Math.tan((45 * Math.PI) / 360) * 22; // ~18.22
  const radiusPx = H * (sphereRadius3D / frustumHeight);

  activeToolCards.forEach((card, call_id) => {
    if (!card.element || !card.pathElement) return;

    let renderX = card.x;
    let renderY = card.y;

    // Apply smooth drifting floating motion when the card is not being dragged
    if (!card.isDragging) {
      const hash = call_id.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
      const phase = hash % 10;
      const bobY = Math.sin(accumulatedTime * 1.5 + phase) * 8; // gentle 8px bobbing
      const bobX = Math.cos(accumulatedTime * 0.8 + phase) * 4; // gentle 4px drift
      renderX += bobX;
      renderY += bobY;

      card.element.style.transform = `translate(${bobX}px, ${bobY}px)`;
    } else {
      card.element.style.transform = `none`;
    }

    // Connect to right edge if card is left of the central visualizer sphere, otherwise connect to left edge
    const cardWidthActual = card.element.offsetWidth || 330;
    const cardHeightActual = card.element.offsetHeight || 160;
    const isLeftOfSphere = (renderX + cardWidthActual / 2) < centerX;
    const cardX = isLeftOfSphere ? (renderX + cardWidthActual) : renderX;
    const cardY = renderY + cardHeightActual / 2;

    // Vector from visualizer center to card connection point
    const dx = cardX - centerX;
    const dy = cardY - centerY;
    const angle = Math.atan2(dy, dx);

    // Connection coordinates precisely on the sphere's border/circumference
    const borderX = centerX + Math.cos(angle) * radiusPx;
    const borderY = centerY + Math.sin(angle) * radiusPx;

    // Calculate control points for a smooth organic S-curve
    const cp1x = borderX + (cardX - borderX) * 0.45;
    const cp1y = borderY;
    const cp2x = borderX + (cardX - borderX) * 0.55;
    const cp2y = cardY;

    const d = `M ${borderX.toFixed(2)} ${borderY.toFixed(2)} C ${cp1x.toFixed(2)} ${cp1y.toFixed(2)}, ${cp2x.toFixed(2)} ${cp2y.toFixed(2)}, ${cardX.toFixed(2)} ${cardY.toFixed(2)}`;
    card.pathElement.setAttribute("d", d);
  });
};

// Set up resize listener to update center coordinate
window.addEventListener("resize", () => {
  updateThreadPaths();
});

// Update SVG paths at 60fps in the animation loop
const updateVisualizerThreads = () => {
  updateThreadPaths();
};

const createToolCard = (callId, toolName, args) => {
  const container = document.getElementById("tool-cards-container");
  if (!container) return;

  // Clean up any existing card for safety
  removeToolCard(callId);

  const sector = getAvailableSector();
  
  // Remove recent-highlight class from the previous recent card
  if (recentCallId) {
    const prevCardData = activeToolCards.get(recentCallId);
    if (prevCardData && prevCardData.element) {
      prevCardData.element.classList.remove("recent-highlight");
    }
  }
  recentCallId = callId;

  const cardEl = document.createElement("div");
  cardEl.className = "tool-card executing recent-highlight";
  cardEl.setAttribute("id", `card-${callId}`);
  
  // Position absolutely
  cardEl.style.left = `${sector.coords.x}px`;
  cardEl.style.top = `${sector.coords.y}px`;

  const mockLogs = getMockLogs(toolName, args);
  const logsHtml = mockLogs.map(log => `<div>${log}</div>`).join('');

  // HTML content of the card with close icon
  cardEl.innerHTML = `
    <div class="tool-card-header" style="display: flex; justify-content: space-between; align-items: center; cursor: move;">
      <span class="tool-card-title">${toolName}</span>
      <div style="display: flex; gap: 8px; align-items: center;">
        <span class="tool-card-status status-running">EXECUTING</span>
        <span class="tool-card-close" title="Close Card">&times;</span>
      </div>
    </div>
    <div class="tool-card-body">
      <div>
        <div class="tool-section-title">Execution Logs</div>
        <div class="tool-args tool-logs-container" style="font-family: var(--font-mono); font-size: 0.65rem; color: #a5b4fc; max-height: 100px; overflow-y: auto; display: flex; flex-direction: column; gap: 4px;">
          ${logsHtml}
        </div>
      </div>
      <div class="result-section" style="display:none; margin-top: 4px;">
        <div class="tool-section-title">Output</div>
        <div class="tool-result"></div>
      </div>
    </div>
  `;

  container.appendChild(cardEl);

  // Trigger scale entry animation on next frame
  requestAnimationFrame(() => {
    cardEl.classList.add("active");
  });

  const pathEl = createThreadPath(callId);
  if (pathEl) {
    pathEl.setAttribute("class", "thread-line executing");
  }

  const countForSize = activeToolCards.size + 1;
  const initHeight = countForSize > 4 ? 120 : (countForSize > 2 ? 160 : 250);
  const initWidth = countForSize > 4 ? 190 : (countForSize > 2 ? 250 : 330);

  const cardData = {
    element: cardEl,
    sectorIndex: sector.index,
    pathElement: pathEl,
    x: sector.coords.x,
    y: sector.coords.y,
    width: initWidth,
    height: initHeight,
    isDragging: false
  };

  activeToolCards.set(callId, cardData);
  makeCardDraggable(callId, cardData);

  // Bind close action
  const closeBtn = cardEl.querySelector(".tool-card-close");
  if (closeBtn) {
    closeBtn.addEventListener("click", (e) => {
      e.stopPropagation(); // prevent maximize event
      removeToolCard(callId);
    });
  }

  // Bind click maximize action
  cardEl.addEventListener("click", (e) => {
    if (e.target.closest(".tool-card-close, .tool-args, .tool-result") || cardData.isDragging) return;
    cardEl.classList.toggle("maximized");
    setTimeout(() => {
      updateThreadPaths();
    }, 50);
  });

  // Recalculate layout sizes dynamically and update SVG thread connections
  updateCardSizes();
  updateThreadPaths();
};

const updateToolCard = (callId, output) => {
  const cardData = activeToolCards.get(callId);
  if (!cardData) return;

  const cardEl = cardData.element;
  cardEl.classList.remove("executing"); // Clear active glowing border highlight
  
  const statusEl = cardEl.querySelector(".tool-card-status");
  const resultSec = cardEl.querySelector(".result-section");
  const resultEl = cardEl.querySelector(".tool-result");
  const logContainer = cardEl.querySelector(".tool-logs-container");

  statusEl.className = "tool-card-status"; // Clear

  if (output.status === "success") {
    statusEl.classList.add("status-success");
    statusEl.innerText = "SUCCESS";
    if (cardData.pathElement) cardData.pathElement.setAttribute("class", "thread-line success");
    resultEl.innerText = typeof output.result === "object" ? JSON.stringify(output.result, null, 2) : output.result;
    if (logContainer) {
      logContainer.innerHTML += `<div style="color: #4ade80;">[SUCCESS] Context retrieved successfully.</div>`;
    }
  } else {
    statusEl.classList.add("status-error");
    statusEl.innerText = "FAILED";
    if (cardData.pathElement) cardData.pathElement.setAttribute("class", "thread-line error");
    resultEl.innerText = output.error || "Execution failed";
    if (logContainer) {
      logContainer.innerHTML += `<div style="color: #f87171;">[ERROR] ${output.error || 'Execution failed.'}</div>`;
    }
  }

  resultSec.style.display = "block";

  // Recalculate paths to adjust for card height change
  setTimeout(() => {
    updateThreadPaths();
  }, 50);
};

const removeToolCard = (callId) => {
  const cardData = activeToolCards.get(callId);
  if (!cardData) return;

  // Fade out DOM elements
  cardData.element.classList.remove("active");
  cardData.element.style.opacity = "0";
  cardData.element.style.transform = "scale(0.85)";

  if (cardData.pathElement) {
    cardData.pathElement.style.opacity = "0";
    cardData.pathElement.style.transition = "opacity 0.5s";
  }

  // Free up sector index and active registries immediately
  if (cardData.sectorIndex >= 0) {
    occupiedSectors.delete(cardData.sectorIndex);
  }
  if (recentCallId === callId) {
    recentCallId = null;
  }
  activeToolCards.delete(callId);

  // Recalculate layout scales of remaining cards instantly
  updateCardSizes();

  // Delay DOM deletion for transition completion
  setTimeout(() => {
    if (cardData.element.parentNode) {
      cardData.element.parentNode.removeChild(cardData.element);
    }
    if (cardData.pathElement && cardData.pathElement.parentNode) {
      cardData.pathElement.parentNode.removeChild(cardData.pathElement);
    }
  }, 500);
};

// Make the cards draggable and stretchy!
const makeCardDraggable = (callId, cardData) => {
  const card = cardData.element;

  const onStart = (clientX, clientY) => {
    let hasMoved = false;
    const startX = clientX - cardData.x;
    const startY = clientY - cardData.y;

    const onMove = (moveX, moveY) => {
      cardData.isDragging = true;
      hasMoved = true;
      const newX = moveX - startX;
      const newY = moveY - startY;

      // Keep within viewport boundaries
      cardData.x = Math.max(10, Math.min(window.innerWidth - card.offsetWidth - 10, newX));
      cardData.y = Math.max(10, Math.min(window.innerHeight - card.offsetHeight - 10, newY));

      card.style.left = `${cardData.x}px`;
      card.style.top = `${cardData.y}px`;

      updateThreadPaths();
    };

    const onEnd = () => {
      document.removeEventListener("mousemove", onMouseMove);
      document.removeEventListener("mouseup", onMouseUp);
      document.removeEventListener("touchmove", onTouchMove);
      document.removeEventListener("touchend", onTouchEnd);
      // Wait a frame so the click listener can see the true drag state
      setTimeout(() => {
        cardData.isDragging = false;
      }, 50);
    };

    const onMouseMove = (e) => onMove(e.clientX, e.clientY);
    const onMouseUp = () => onEnd();
    const onTouchMove = (e) => {
      if (e.touches.length > 0) {
        onMove(e.touches[0].clientX, e.touches[0].clientY);
      }
    };
    const onTouchEnd = () => onEnd();

    document.addEventListener("mousemove", onMouseMove);
    document.addEventListener("mouseup", onMouseUp);
    document.addEventListener("touchmove", onTouchMove, { passive: true });
    document.addEventListener("touchend", onTouchEnd);
  };

  card.addEventListener("mousedown", (e) => {
    if (e.target.closest(".tool-card-close, .tool-args, .tool-result")) return;
    cardData.isDragging = false; // Reset drag state initially
    onStart(e.clientX, e.clientY);
  });

  card.addEventListener("touchstart", (e) => {
    if (e.target.closest(".tool-card-close, .tool-args, .tool-result")) return;
    cardData.isDragging = false;
    if (e.touches.length > 0) {
      onStart(e.touches[0].clientX, e.touches[0].clientY);
    }
  }, { passive: true });
};

animate();
