export const LOCAL_PATHS = {
  models: "models",
  whisper: "models/whisper",
  llm: "models/llm",
  cache: "cache",
  offline: "offline",
} as const;

export function whisperFileName(modelId: string): string {
  return `ggml-${modelId}.bin`;
}

export function describeLocalLayout(dataDir: string): string[] {
  const root = dataDir.replace(/[/\\]$/, "");
  return [
    `${root}/${LOCAL_PATHS.whisper}`,
    `${root}/${LOCAL_PATHS.llm}`,
    `${root}/${LOCAL_PATHS.cache}`,
    `${root}/${LOCAL_PATHS.offline}`,
  ];
}
