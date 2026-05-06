type AudioWindow = Window & typeof globalThis & {
  webkitAudioContext?: typeof AudioContext;
};

export function delay(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

export async function waitForVideoReady(video: HTMLVideoElement, timeoutMs = 5000) {
  if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && video.videoWidth > 0 && video.videoHeight > 0) {
    return;
  }

  await new Promise<void>((resolve, reject) => {
    const timeout = window.setTimeout(() => {
      cleanup();
      reject(new Error("Camera preview did not become ready in time."));
    }, timeoutMs);

    const onReady = () => {
      if (video.videoWidth > 0 && video.videoHeight > 0) {
        cleanup();
        resolve();
      }
    };

    const cleanup = () => {
      window.clearTimeout(timeout);
      video.removeEventListener("loadedmetadata", onReady);
      video.removeEventListener("canplay", onReady);
      video.removeEventListener("playing", onReady);
    };

    video.addEventListener("loadedmetadata", onReady);
    video.addEventListener("canplay", onReady);
    video.addEventListener("playing", onReady);
    onReady();
  });
}

export async function captureVideoFrame(video: HTMLVideoElement): Promise<Blob> {
  await waitForVideoReady(video);
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  const context = canvas.getContext("2d");
  if (!context) {
    throw new Error("Could not prepare camera frame.");
  }
  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (!blob) {
        reject(new Error("Could not capture camera frame."));
        return;
      }
      resolve(blob);
    }, "image/jpeg", 0.88);
  });
}

export async function captureVideoFrames(
  video: HTMLVideoElement,
  count = 5,
  intervalMs = 700
): Promise<Blob[]> {
  const frames: Blob[] = [];
  for (let index = 0; index < count; index += 1) {
    frames.push(await captureVideoFrame(video));
    if (index < count - 1) {
      await delay(intervalMs);
    }
  }
  return frames;
}

export function recordAudioSample(stream: MediaStream, durationMs = 3000): Promise<Blob> {
  return new Promise((resolve, reject) => {
    const audioTracks = stream.getAudioTracks();
    if (audioTracks.length === 0) {
      reject(new Error("No microphone track is available for this session."));
      return;
    }

    const audioWindow = window as AudioWindow;
    const AudioContextClass = audioWindow.AudioContext || audioWindow.webkitAudioContext;
    if (!AudioContextClass) {
      reject(new Error("Audio recording is not supported in this browser."));
      return;
    }
    const audioContext = new AudioContextClass();
    const audioStream = new MediaStream(audioTracks);
    const source = audioContext.createMediaStreamSource(audioStream);
    const processor = audioContext.createScriptProcessor(4096, 1, 1);
    const silence = audioContext.createGain();
    const chunks: Float32Array[] = [];
    silence.gain.value = 0;

    processor.onaudioprocess = (event) => {
      chunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
    };

    source.connect(processor);
    processor.connect(silence);
    silence.connect(audioContext.destination);

    audioContext.resume().catch(() => undefined);

    window.setTimeout(() => {
      processor.disconnect();
      source.disconnect();
      silence.disconnect();
      const samples = flattenAudioChunks(chunks);
      const minimumSamples = Math.floor(audioContext.sampleRate * Math.min(durationMs / 1000, 2));
      audioContext.close().catch(() => undefined);
      if (samples.length < minimumSamples) {
        reject(new Error("The microphone sample was too short. Please try again and allow the scan to finish."));
        return;
      }
      resolve(encodeWav(samples, audioContext.sampleRate));
    }, durationMs);
  });
}

function flattenAudioChunks(chunks: Float32Array[]) {
  const totalLength = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const result = new Float32Array(totalLength);
  let offset = 0;
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.length;
  }
  return result;
}

function encodeWav(samples: Float32Array, sampleRate: number) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(view, 8, "WAVE");
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, "data");
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (const sample of samples) {
    const clamped = Math.max(-1, Math.min(1, sample));
    view.setInt16(offset, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true);
    offset += 2;
  }
  return new Blob([buffer], { type: "audio/wav" });
}

function writeString(view: DataView, offset: number, value: string) {
  for (let index = 0; index < value.length; index += 1) {
    view.setUint8(offset + index, value.charCodeAt(index));
  }
}
