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
  kind: "text" | "reasoning" | "tool";
  parts: TutorMessagePart[];
};

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

export function getMessageSources(message: TutorMessage): SourcePartData[] {
  const seen = new Set<string>();
  const sources: SourcePartData[] = [];

  for (const part of message.parts) {
    if (!("type" in part) || part.type !== "data-source") {
      continue;
    }
    const data = (part as TutorMessagePart).data as SourcePartData | undefined;
    if (!data) {
      continue;
    }
    const key = data.docId || data.url;
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    sources.push(data);
  }

  return sources.sort((left, right) => right.score - left.score);
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

  for (const source of getMessageSources(message)) {
    const url = source.url?.trim();
    if (!url) {
      continue;
    }
    const kind = classifySource(source.group);
    const title = cleanTitle(source.title);
    const label = (
      kind === "web"
        ? title || hostnameFromUrl(url) || "Source"
        : title || source.sourceLabel || "Source"
    ).slice(0, 80);
    const sublabel =
      kind === "web"
        ? hostnameFromUrl(url)
        : source.sourceLabel?.trim() || undefined;
    const key = `${kind}::${url}`;
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    citations.push({ label, url, kind, sublabel });
  }

  for (const part of getTextParts(message)) {
    const text = part.text ?? "";
    for (const citation of extractMarkdownLinkCitations(text)) {
      const key = `web::${citation.url}`;
      if (seen.has(key)) {
        continue;
      }
      seen.add(key);
      citations.push({
        label: citation.label,
        url: citation.url,
        kind: "web",
        sublabel: hostnameFromUrl(citation.url) || undefined,
      });
    }
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

export function prettifyToolName(type: string) {
  return type
    .replace(/^tool-/, "")
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

  return JSON.stringify(input);
}

function classifyMessagePart(
  part: TutorMessagePart,
): TutorMessageBlock["kind"] | null {
  if (part.type === "text") {
    return "text";
  }

  if (part.type === "reasoning") {
    return "reasoning";
  }

  if (String(part.type).startsWith("tool-")) {
    return "tool";
  }

  return null;
}

type RawLinkCitation = { label: string; url: string };

function extractMarkdownLinkCitations(text: string): RawLinkCitation[] {
  const citations: RawLinkCitation[] = [];
  const markdownLinkPattern = /\[([^\]]+)\]\((https?:\/\/[^\s)]+(?:\([^\s)]+\)[^\s)]*)*)\)/g;
  const bareUrlPattern = /(?<!\()https?:\/\/[^\s<>()]+/g;
  const urlsInMarkdown = new Set<string>();

  for (const match of text.matchAll(markdownLinkPattern)) {
    const label = (match[1] ?? "").trim();
    const url = (match[2] ?? "").trim();
    if (!label || !url) {
      continue;
    }
    urlsInMarkdown.add(url);
    citations.push({ label, url });
  }

  for (const match of text.matchAll(bareUrlPattern)) {
    const url = (match[0] ?? "").trim();
    if (!url || urlsInMarkdown.has(url)) {
      continue;
    }
    citations.push({ label: "Source", url });
  }

  return citations;
}
