import type { UIMessage } from "ai";
import type { SourcePartData } from "./api";

export type TutorMessage = UIMessage;

export type TutorMessagePart = {
  type: string;
  text?: string;
  state?: string;
  data?: unknown;
  toolCallId?: string;
  input?: unknown;
  output?: unknown;
  errorText?: string;
  sourceId?: string;
  url?: string;
  title?: string;
  mediaType?: string;
};

export type TutorMessageBlock = {
  key: string;
  kind: "text" | "activity";
  parts: TutorMessagePart[];
};

export type ActivityItem =
  | { kind: "reasoning"; key: string; text: string }
  | { kind: "tool"; key: string; part: TutorMessagePart };

export type CitationKind = "web" | "course" | "doc";

export type MessageCitation = {
  label: string;
  url: string;
  kind: CitationKind;
  sublabel?: string;
};

export function getTextParts(message: TutorMessage) {
  return message.parts
    .filter((part) => "type" in part && part.type === "text")
    .map((part) => part as TutorMessagePart);
}

export function getMessageTextContent(message: TutorMessage) {
  return getTextParts(message)
    .map((part) => (part.text ?? "").trim())
    .filter(Boolean)
    .join("\n\n");
}

export function getReasoningParts(message: TutorMessage) {
  return message.parts
    .filter((part) => "type" in part && part.type === "reasoning")
    .map((part) => part as TutorMessagePart);
}

export function getToolParts(message: TutorMessage) {
  return message.parts
    .filter((part) => "type" in part && String(part.type).startsWith("tool-"))
    .map((part) => part as TutorMessagePart);
}

export function getOrderedMessageBlocks(message: TutorMessage): TutorMessageBlock[] {
  const blocks: TutorMessageBlock[] = [];
  let currentKind: TutorMessageBlock["kind"] | null = null;
  let currentParts: TutorMessagePart[] = [];

  for (const part of message.parts) {
    if (!("type" in part)) {
      continue;
    }

    const typedPart = part as TutorMessagePart;
    const nextKind = classifyMessagePart(typedPart);

    if (!nextKind) {
      continue;
    }

    if (currentKind && currentKind !== nextKind) {
      blocks.push({
        key: `${message.id}-${currentKind}-${blocks.length}`,
        kind: currentKind,
        parts: currentParts,
      });
      currentParts = [];
    }

    currentKind = nextKind;
    currentParts.push(typedPart);
  }

  if (currentKind && currentParts.length > 0) {
    blocks.push({
      key: `${message.id}-${currentKind}-${blocks.length}`,
      kind: currentKind,
      parts: currentParts,
    });
  }

  return blocks;
}

export function hasRenderableContent(message: TutorMessage): boolean {
  for (const part of message.parts) {
    if (!("type" in part)) {
      continue;
    }
    const type = String(part.type);
    if (type === "text" || type === "reasoning" || type.startsWith("tool-")) {
      return true;
    }
  }
  return false;
}

export function getLastAssistantMessage(messages: TutorMessage[]) {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index]?.role === "assistant") {
      return messages[index];
    }
  }

  return undefined;
}

function classifySource(group: string): CitationKind {
  if (group === "web") {
    return "web";
  }
  if (group === "courses") {
    return "course";
  }
  return "doc";
}

export function getMessageCitations(message: TutorMessage): MessageCitation[] {
  const seen = new Set<string>();
  const citations: MessageCitation[] = [];

  // Sources are resolved server-side (deduped, in citation order) and arrive as
  // `data-source` parts. The frontend only maps them to display props — it does
  // NOT re-parse the answer text for links, which previously produced duplicate
  // chips (a corpus URL added as kind "doc"/"course" AND again as kind "web").
  for (const part of message.parts) {
    if (!("type" in part) || part.type !== "data-source") {
      continue;
    }
    const data = (part as TutorMessagePart).data as SourcePartData | undefined;
    if (!data) {
      continue;
    }
    const url = data.url?.trim();
    if (!url) {
      continue;
    }
    const key = url.replace(/#.*$/, "").replace(/\/+$/, ""); // mirror server normalize_url
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);

    const kind = classifySource(data.group);
    const title = cleanTitle(data.title);
    const label = (
      kind === "web"
        ? title || hostnameFromUrl(url) || "Source"
        : title || data.sourceLabel || "Source"
    ).slice(0, 80);
    const sublabel =
      kind === "web"
        ? hostnameFromUrl(url)
        : data.sourceLabel?.trim() || undefined;

    citations.push({ label, url, kind, sublabel });
  }

  return citations;
}

function cleanTitle(raw: string | undefined) {
  const trimmed = raw?.trim();
  if (!trimmed) {
    return "";
  }
  if (/^[-–—\s]+$/.test(trimmed)) {
    return "";
  }
  return trimmed;
}

function hostnameFromUrl(url: string) {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return "";
  }
}

const TOOL_DISPLAY_NAMES: Record<string, string> = {
  run_kb_command: "KB shell",
  retrieve_tutor_context: "Hybrid search",
  web_search: "Web search",
  google_search: "Web search",
  url_context: "Fetch URL",
  web_fetch: "Fetch URL",
};

export function prettifyToolName(type: string) {
  const raw = type.replace(/^tool-/, "");
  if (TOOL_DISPLAY_NAMES[raw]) {
    return TOOL_DISPLAY_NAMES[raw];
  }
  return raw
    .replaceAll("_", " ")
    .replace(/\b\w/g, (match) => match.toUpperCase());
}

export function toolInputSummary(input: unknown) {
  if (!input || typeof input !== "object") {
    return "";
  }

  const query = "query" in input ? String(input.query ?? "").trim() : "";
  if (query) {
    return query;
  }

  const text = "text" in input ? String(input.text ?? "").trim() : "";
  if (text) {
    return text;
  }

  const serialized = JSON.stringify(input);
  // Some providers announce a tool call before its args are parsed; an empty
  // object is a streaming placeholder, not meaningful input.
  return serialized === "{}" ? "" : serialized;
}

function classifyMessagePart(
  part: TutorMessagePart,
): TutorMessageBlock["kind"] | null {
  if (part.type === "text") {
    return "text";
  }

  if (
    part.type === "reasoning" ||
    String(part.type).startsWith("tool-")
  ) {
    return "activity";
  }

  return null;
}

// Build a flat list of activity items from a parts array, joining consecutive
// reasoning fragments into one entry and keeping each tool call as its own row.
export function buildActivityItems(parts: TutorMessagePart[]): ActivityItem[] {
  const items: ActivityItem[] = [];
  let pendingReasoning: string[] = [];

  const flushReasoning = () => {
    if (pendingReasoning.length === 0) {
      return;
    }
    const text = pendingReasoning.map((entry) => entry.trim()).filter(Boolean).join("\n\n");
    if (text) {
      items.push({ kind: "reasoning", key: `r-${items.length}`, text });
    }
    pendingReasoning = [];
  };

  for (const part of parts) {
    if (part.type === "reasoning") {
      pendingReasoning.push(part.text ?? "");
      continue;
    }
    if (String(part.type).startsWith("tool-")) {
      flushReasoning();
      const id = part.toolCallId ?? "";
      items.push({ kind: "tool", key: `t-${items.length}-${id}`, part });
    }
  }
  flushReasoning();

  return items;
}
