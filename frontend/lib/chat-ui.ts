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

function normalizeCitationUrl(url: string) {
  // Mirror the server's normalize_url (drop fragment and trailing slash),
  // then canonicalize percent-encoding: react-markdown renders hrefs through
  // micromark's normalizeUri, so the rendered href may be an encoded variant
  // of the raw answer text. Decode-then-encode maps both to one form (it is
  // idempotent across that encoding); URLs that fail to decode stay as-is.
  const base = url.replace(/#.*$/, "").replace(/\/+$/, "");
  try {
    return encodeURI(decodeURI(base));
  } catch {
    return base;
  }
}

// Drop fenced code blocks and inline code spans before scanning for links:
// rendered code never produces anchors, so a markdown link inside code must
// not consume a citation number. The $ alternates keep a streaming,
// not-yet-closed fence from leaking back in.
function stripCodeSegments(text: string) {
  return text
    .replace(/```[\s\S]*?(?:```|$)/g, " ")
    .replace(/~~~[\s\S]*?(?:~~~|$)/g, " ")
    .replace(/`[^`\n]*`/g, " ");
}

// micromark parses balanced parens inside a link destination, so a regex that
// stops at the first ")" truncates URLs like .../Spring_(framework). Scan the
// destination the way the parser does: <...> form, or up to whitespace / the
// first unbalanced closing paren. The destination only renders as an anchor
// when it is followed (after optional whitespace) by the closing ")" or a
// link title; a citation the model closed with "]" instead of ")" stays plain
// text in the rendered markdown, so it must not consume a citation number
// (which would leave a gap like 1, 3 in the displayed chips).
function scanLinkDestination(text: string, start: number): string {
  let index = start;
  while (index < text.length && /\s/.test(text[index])) {
    index += 1;
  }
  let url = "";
  let end = index;
  if (text[index] === "<") {
    const close = text.indexOf(">", index + 1);
    if (close === -1) {
      return "";
    }
    url = text.slice(index + 1, close);
    end = close + 1;
  } else {
    let depth = 0;
    while (end < text.length) {
      const char = text[end];
      if (/\s/.test(char)) {
        break;
      }
      if (char === "(") {
        depth += 1;
      } else if (char === ")") {
        if (depth === 0) {
          break;
        }
        depth -= 1;
      }
      end += 1;
    }
    url = text.slice(index, end);
  }
  while (end < text.length && /\s/.test(text[end])) {
    end += 1;
  }
  const terminator = text[end];
  const rendersAsLink =
    terminator === ")" ||
    terminator === '"' ||
    terminator === "'" ||
    terminator === "(";
  return rendersAsLink ? url : "";
}

// Link text may contain one level of balanced brackets (CommonMark-legal,
// e.g. [LoRA [PEFT docs]](url)); micromark renders such links as anchors, so
// the scanner must find them too or their citations never get a number. The
// alternatives are disjoint on the first character, so matching stays linear.
const LINK_OPENER_PATTERN = /(!?)\[(?:[^\[\]]|\[[^\]]*\])*\]\(/g;

function extractMarkdownLinkUrls(text: string): string[] {
  const urls: string[] = [];
  const scannable = stripCodeSegments(text);
  for (const match of scannable.matchAll(LINK_OPENER_PATTERN)) {
    if (match[1] === "!") {
      continue;
    }
    const url = scanLinkDestination(scannable, match.index + match[0].length);
    if (url) {
      urls.push(url);
    }
  }
  return urls;
}

export function isHttpUrl(url: string | undefined): boolean {
  return Boolean(url && /^https?:\/\//i.test(url.trim()));
}

// The system prompt lets the model cite KB-browsed files as kb://doc/<id> or
// KB-root paths like raw/docs/peft/lora.md; normalize the path variants the
// model may produce to the KB-root-relative form the server sends.
function normalizeKbPath(ref: string) {
  let value = ref.trim();
  if (value.startsWith("./")) {
    value = value.slice(2);
  }
  if (value.startsWith("data/kb/")) {
    value = value.slice("data/kb/".length);
  }
  return value;
}

// Map inline KB references (kb://doc/<id>, raw/... paths) to the resolved
// https URL of the matching data-source part, so inline citations of
// KB-browsed files link to the real page instead of rendering as a dead
// kb:// or relative href.
export function buildCitationResolutions(
  message: TutorMessage,
): Map<string, string> {
  const resolutions = new Map<string, string>();
  for (const part of message.parts) {
    if (!("type" in part) || part.type !== "data-source") {
      continue;
    }
    const data = (part as TutorMessagePart).data as SourcePartData | undefined;
    const url = data?.url?.trim();
    if (!data || !isHttpUrl(url)) {
      continue;
    }
    if (data.docId) {
      resolutions.set(`kb://doc/${data.docId}`, url as string);
    }
    const path = normalizeKbPath(data.path ?? "");
    if (path) {
      resolutions.set(path, url as string);
    }
  }
  return resolutions;
}

export function resolveCitationHref(
  href: string | undefined,
  resolutions: Map<string, string> | undefined,
): string | undefined {
  if (!href || !resolutions || resolutions.size === 0) {
    return undefined;
  }
  return resolutions.get(href.trim()) ?? resolutions.get(normalizeKbPath(href));
}

// Number the citation links in a message's answer text by order of first
// appearance, deduped by normalized URL. KB references are keyed under their
// resolved https URL so the inline chip and the source card share a number;
// references that resolve to nothing navigable get no number (the renderer
// shows them as plain text instead of a broken chip). The same map drives the
// inline citation chips and the numbers on the sources row.
export function buildCitationNumbers(
  message: TutorMessage,
  resolutions?: Map<string, string>,
): Map<string, number> {
  const numbers = new Map<string, number>();
  for (const part of message.parts) {
    if (!("type" in part) || part.type !== "text") {
      continue;
    }
    const text = (part as TutorMessagePart).text ?? "";
    for (const url of extractMarkdownLinkUrls(text)) {
      const resolved = resolveCitationHref(url, resolutions) ?? url;
      if (!isHttpUrl(resolved)) {
        continue;
      }
      const key = normalizeCitationUrl(resolved);
      if (key && !numbers.has(key)) {
        numbers.set(key, numbers.size + 1);
      }
    }
  }
  return numbers;
}

export function citationNumberFor(
  numbers: Map<string, number> | undefined,
  url: string | undefined,
) {
  if (!numbers || !url) {
    return undefined;
  }
  return numbers.get(normalizeCitationUrl(url));
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
