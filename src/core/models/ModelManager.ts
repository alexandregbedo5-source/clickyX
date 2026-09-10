import type { LocalModel, ModelInventory, ModelKind } from "../offline/types";

const CATALOG: LocalModel[] = [
  entry("llama3.2", "Llama 3.2 (Ollama)", "llm", "ollama", "ollama pull llama3.2"),
  entry("llama3.2:1b", "Llama 3.2 1B (Ollama)", "llm", "ollama", "Smallest useful local chat model"),
  entry("mistral", "Mistral 7B (Ollama)", "llm", "ollama", "ollama pull mistral"),
  entry("qwen2.5:3b", "Qwen 2.5 3B (Ollama)", "llm", "ollama", "Multilingual local default"),
  entry("tiny", "Whisper tiny (ggml)", "whisper", "whisper-local", "ggml-tiny.bin"),
  entry("base", "Whisper base (ggml)", "whisper", "whisper-local", "ggml-base.bin"),
  entry("small", "Whisper small (ggml)", "whisper", "whisper-local", "ggml-small.bin"),
];

function entry(
  id: string,
  display_name: string,
  kind: ModelKind,
  provider: string,
  notes: string,
): LocalModel {
  return {
    id,
    display_name,
    kind,
    provider,
    installed: false,
    path: null,
    size_bytes: 0,
    download_url:
      kind === "whisper"
        ? `https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-${id}.bin`
        : null,
    notes,
  };
}

export class ModelManager {
  private installed = new Map<string, LocalModel>();

  catalog(): LocalModel[] {
    return CATALOG.map((model) => {
      const extra = this.installed.get(model.id);
      return extra ? { ...model, ...extra, installed: true } : { ...model };
    });
  }

  markInstalled(model: LocalModel): void {
    this.installed.set(model.id, { ...model, installed: true });
  }

  inventory(): ModelInventory {
    const extras = [...this.installed.values()].filter(
      (model) => !CATALOG.some((item) => item.id === model.id),
    );
    return {
      models: [...this.catalog(), ...extras],
      data_dir: "",
    };
  }

  listInstalled(): LocalModel[] {
    return this.inventory().models.filter((model) => model.installed);
  }

  find(id: string): LocalModel | undefined {
    return this.inventory().models.find((model) => model.id === id);
  }

  canDownload(id: string, wanAllowed: boolean): { ok: boolean; reason?: string } {
    const model = this.find(id);
    if (!model) return { ok: false, reason: `Unknown model '${id}'` };
    if (model.installed) return { ok: true };
    if (!wanAllowed) {
      return {
        ok: false,
        reason: `Cannot download ${id} while offline. Import the file locally first.`,
      };
    }
    return { ok: true };
  }
}

let singleton: ModelManager | null = null;

export function getModelManager(): ModelManager {
  if (!singleton) singleton = new ModelManager();
  return singleton;
}

export function resetModelManagerForTests(): void {
  singleton = null;
}
