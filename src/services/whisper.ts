import { isLoopbackUrl } from "../core/offline/types";
import { whisperFileName } from "./storage";

export function whisperEndpoints(baseUrl: string) {
  const base = baseUrl.replace(/\/$/, "");
  return {
    inference: `${base}/inference`,
    openaiCompatible: `${base}/v1/audio/transcriptions`,
  };
}

export function assertLocalWhisper(baseUrl: string, wanBlocked: boolean): void {
  if (wanBlocked && !isLoopbackUrl(baseUrl)) {
    throw new Error("Whisper URL is not local and WAN is blocked");
  }
}

export function missingWhisperHint(modelId: string, modelsDir: string): string {
  return `Place ${whisperFileName(modelId)} in ${modelsDir} or start whisper.cpp on localhost:8090.`;
}
