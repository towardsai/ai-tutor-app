export type TutorSource = {
  key: string;
  label: string;
  group: "courses" | "docs";
  selectedByDefault: boolean;
  version?: string | null;
  indexedAt?: string | null;
};

export type TutorConfigurableTool = {
  kind: "configurable";
  key: "retrieval";
  label: string;
  active: boolean;
  sources: TutorSource[];
};

export type TutorToggleTool = {
  kind: "toggle";
  key: string;
  label: string;
  active: boolean;
};

export type TutorTool = TutorConfigurableTool | TutorToggleTool;

export type AvailableModel = {
  id: string;
  label: string;
};

export type ToolsResponse = {
  model: string;
  availableModels: AvailableModel[];
  tools: TutorTool[];
};

export type SourcePartData = {
  docId: string;
  title: string;
  url: string;
  sourceKey: string;
  sourceLabel: string;
  score: number;
  group: string;
};

const FALLBACK_API_BASE_URL = "http://127.0.0.1:8000";

export function getApiBaseUrl() {
  const configured =
    process.env.NEXT_PUBLIC_AI_TUTOR_API_BASE_URL?.replace(/\/+$/, "");
  if (configured) return configured;
  if (typeof window !== "undefined") return window.location.origin;
  return FALLBACK_API_BASE_URL;
}

export async function fetchTools(
  signal?: AbortSignal,
  model?: string,
): Promise<ToolsResponse> {
  const url = new URL(`${getApiBaseUrl()}/api/tools`);
  if (model) {
    url.searchParams.set("model", model);
  }
  const response = await fetch(url.toString(), {
    method: "GET",
    signal,
  });

  if (!response.ok) {
    throw new Error(`Unable to load tools (${response.status})`);
  }

  return (await response.json()) as ToolsResponse;
}
