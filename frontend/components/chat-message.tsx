"use client";

import clsx from "clsx";
import {
  BookOpen,
  Check,
  ChevronDown,
  ChevronRight,
  Copy,
  Database,
  ExternalLink,
  Globe,
  GraduationCap,
  LibraryBig,
  Loader2,
  Pencil,
  RefreshCw,
  Sparkles,
  Terminal,
  Wrench,
} from "lucide-react";
import type { ComponentType, SVGProps } from "react";
import {
  useEffect,
  useRef,
  useState,
  type ReactNode,
  type RefObject,
} from "react";
import { MarkdownBlock } from "@/components/markdown-block";
import type {
  ActivityItem,
  MessageCitation,
  TutorMessage,
  TutorMessageBlock,
  TutorMessagePart,
} from "@/lib/chat-ui";
import {
  buildActivityItems,
  getMessageCitations,
  getOrderedMessageBlocks,
  prettifyToolName,
  toolInputSummary,
} from "@/lib/chat-ui";

type ChatMessageProps = {
  message: TutorMessage;
  editDraft?: string;
  isEditing?: boolean;
  isStreaming?: boolean;
  onAssistantCopy?: (message: TutorMessage) => void;
  onAssistantRedo?: (messageId: string) => void;
  onEditCancel?: () => void;
  onEditChange?: (value: string) => void;
  onEditSave?: (messageId: string) => void;
  onUserEdit?: (messageId: string) => void;
  actionDisabled?: boolean;
  copied?: boolean;
  showAssistantActions?: boolean;
};

export function ChatMessage({
  message,
  editDraft = "",
  isEditing = false,
  isStreaming = false,
  onAssistantCopy,
  onAssistantRedo,
  onEditCancel,
  onEditChange,
  onEditSave,
  onUserEdit,
  actionDisabled = false,
  copied = false,
  showAssistantActions = true,
}: ChatMessageProps) {
  const isAssistant = message.role === "assistant";
  const contentBlocks = getOrderedMessageBlocks(message);
  const citations = isAssistant && !isStreaming ? getMessageCitations(message) : [];
  const editTextareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (!isEditing || isAssistant) {
      return;
    }

    const textarea = editTextareaRef.current;
    if (!textarea) {
      return;
    }

    textarea.focus();
    const length = textarea.value.length;
    textarea.setSelectionRange(length, length);
  }, [isAssistant, isEditing]);

  useEffect(() => {
    if (!isEditing || isAssistant) {
      return;
    }

    const textarea = editTextareaRef.current;
    if (!textarea) {
      return;
    }

    textarea.style.height = "0px";
    textarea.style.height = `${Math.min(textarea.scrollHeight, 240)}px`;
  }, [editDraft, isAssistant, isEditing]);

  return (
    <div
      data-message-id={message.id}
      data-role={message.role}
      className={clsx(
        "animate-rise-in flex flex-col",
        isAssistant ? "w-full items-start" : "items-end",
      )}
    >
      <article
        className={clsx(
          "rounded-[1.8rem] border px-4 py-3 text-left shadow-[0_20px_50px_rgba(18,42,204,0.08)] transition outline-none sm:px-5",
          isAssistant
            ? "w-full border-[var(--line)] bg-[var(--surface)]"
            : "ml-auto border-[var(--accent)] bg-[linear-gradient(135deg,rgba(11,136,238,0.16),rgba(193,235,255,0.35))]",
        )}
      >
        {isEditing && !isAssistant ? (
          <InlineEditor
            disabled={actionDisabled}
            textareaRef={editTextareaRef}
            value={editDraft}
            onCancel={onEditCancel}
            onChange={onEditChange}
            onSave={() => onEditSave?.(message.id)}
          />
        ) : (
          <div className="space-y-3">
            {contentBlocks.map((block, index) => {
              const isLastBlock = index === contentBlocks.length - 1;
              const isActive =
                isStreaming && isLastBlock && block.kind !== "text";
              return (
                <ContentBlock
                  key={block.key}
                  block={block}
                  isActive={isActive}
                />
              );
            })}
            {citations.length > 0 ? <CitationRow citations={citations} /> : null}
          </div>
        )}
      </article>

      {isAssistant && showAssistantActions ? (
        <div className="mt-1.5 flex items-center gap-1.5 pl-2">
          <MessageActionButton
            label="Redo"
            onClick={() => onAssistantRedo?.(message.id)}
            disabled={actionDisabled}
          >
            <RefreshCw className="h-3.5 w-3.5" />
          </MessageActionButton>
          <MessageActionButton
            label={copied ? "Copied" : "Copy"}
            onClick={() => onAssistantCopy?.(message)}
          >
            {copied ? (
              <Check className="h-3.5 w-3.5" />
            ) : (
              <Copy className="h-3.5 w-3.5" />
            )}
          </MessageActionButton>
        </div>
      ) : !isAssistant ? (
        !isEditing ? (
          <div className="mt-1.5 flex items-center gap-1 pr-3">
            <MessageActionButton
              label="Edit"
              onClick={() => onUserEdit?.(message.id)}
              disabled={actionDisabled}
            >
              <Pencil className="h-3.5 w-3.5" />
            </MessageActionButton>
          </div>
        ) : null
      ) : null}
    </div>
  );
}

function InlineEditor({
  disabled,
  onCancel,
  onChange,
  onSave,
  textareaRef,
  value,
}: {
  disabled: boolean;
  onCancel?: () => void;
  onChange?: (value: string) => void;
  onSave?: () => void;
  textareaRef: RefObject<HTMLTextAreaElement | null>;
  value: string;
}) {
  const trimmed = value.trim();

  return (
    <div className="space-y-3">
      <textarea
        ref={textareaRef}
        rows={1}
        value={value}
        onChange={(event) => onChange?.(event.target.value)}
        onKeyDown={(event) => {
          if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
            event.preventDefault();
            if (!disabled && trimmed) {
              onSave?.();
            }
            return;
          }

          if (event.key === "Escape") {
            event.preventDefault();
            onCancel?.();
          }
        }}
        className="max-h-60 min-h-20 w-full resize-none overflow-y-auto rounded-[1.15rem] border border-[var(--accent)]/20 bg-[var(--surface-subtle)] px-3 py-3 text-[15px] leading-7 text-[var(--ink)] outline-none placeholder:text-[var(--muted)]"
      />
      <div className="flex items-center justify-end gap-2">
        <button
          type="button"
          onClick={onCancel}
          disabled={disabled}
          className="rounded-full border border-[var(--accent)]/20 bg-[var(--surface-soft)] px-3 py-1.5 text-sm font-medium text-[var(--ink)] transition enabled:hover:border-[var(--accent)] enabled:hover:text-[var(--accent)] disabled:cursor-not-allowed disabled:opacity-40"
        >
          Cancel
        </button>
        <button
          type="button"
          onClick={onSave}
          disabled={disabled || !trimmed}
          className="rounded-full bg-[var(--accent)] px-3 py-1.5 text-sm font-semibold text-white transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-40"
        >
          Save
        </button>
      </div>
    </div>
  );
}

function MessageActionButton({
  children,
  label,
  onClick,
  disabled = false,
}: {
  children: ReactNode;
  label: string;
  onClick?: () => void;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-label={label}
      title={label}
      className="inline-flex h-7 w-7 items-center justify-center rounded-full border border-[var(--line)] bg-[var(--surface)] text-[var(--muted)] shadow-[0_8px_24px_rgba(18,42,204,0.06)] transition enabled:hover:border-[var(--accent)] enabled:hover:text-[var(--accent)] disabled:cursor-not-allowed disabled:opacity-40"
    >
      {children}
    </button>
  );
}

function ContentBlock({
  block,
  isActive = false,
}: {
  block: TutorMessageBlock;
  isActive?: boolean;
}) {
  if (block.kind === "activity") {
    return <ActivityPanel parts={block.parts} isActive={isActive} />;
  }
  return <TextBlock parts={block.parts} />;
}

function ActivityPanel({
  parts,
  isActive = false,
}: {
  parts: TutorMessagePart[];
  isActive?: boolean;
}) {
  // Open by default while streaming; closed when done. A user toggle pins it
  // to that explicit state.
  const [override, setOverride] = useState<boolean | null>(null);
  const isOpen = override ?? isActive;

  const items = buildActivityItems(parts);
  const toolItems = items.filter(
    (item): item is Extract<ActivityItem, { kind: "tool" }> => item.kind === "tool",
  );
  const reasoningItems = items.filter(
    (item): item is Extract<ActivityItem, { kind: "reasoning" }> =>
      item.kind === "reasoning",
  );
  const sourceCount = getToolPartSourceCount(parts);

  const liveTool = isActive
    ? toolItems.find((item) => item.part.state !== "output-available")
      ?? toolItems[toolItems.length - 1]
    : undefined;

  const summary = buildActivitySummary({
    toolCount: toolItems.length,
    reasoningCount: reasoningItems.length,
    sourceCount,
  });

  return (
    <section className="relative overflow-hidden rounded-[1.35rem] border border-[var(--line)] bg-[var(--surface)]">
      {isActive ? (
        <span
          aria-hidden="true"
          className="activity-sheen pointer-events-none absolute inset-0"
        />
      ) : null}
      <button
        type="button"
        onClick={(event) => {
          event.stopPropagation();
          setOverride(!isOpen);
        }}
        className="relative z-[1] flex w-full items-center justify-between gap-3 px-4 py-2.5 text-left"
        aria-expanded={isOpen}
      >
        <div className="flex min-w-0 items-center gap-2 text-[12.5px] font-semibold tracking-[-0.01em] text-[var(--ink)]">
          {isActive ? (
            <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin text-[var(--accent)]" />
          ) : (
            <Wrench className="h-3.5 w-3.5 shrink-0 text-[var(--accent)]" />
          )}
          {isActive && liveTool ? (
            <span className="truncate text-[var(--muted)]">
              <span className="text-[var(--ink)]">Running</span>{" "}
              <code className="rounded bg-[var(--paper)]/70 px-1 py-0.5 font-mono text-[11.5px]">
                {prettifyToolName(liveTool.part.type)}
              </code>{" "}
              {toolInputSummary(liveTool.part.input)}
            </span>
          ) : (
            <span className="truncate">Activity{summary ? ` · ${summary}` : ""}</span>
          )}
        </div>
        <ChevronDown
          className={clsx(
            "h-4 w-4 shrink-0 text-[var(--muted)] transition-transform",
            isOpen && "rotate-180",
          )}
        />
      </button>
      {isOpen ? (
        <ol className="relative z-[1] border-t border-[var(--line)]/70">
          {items.map((item) =>
            item.kind === "reasoning" ? (
              <ReasoningLine key={item.key} text={item.text} />
            ) : (
              <ToolRow key={item.key} part={item.part} />
            ),
          )}
        </ol>
      ) : null}
    </section>
  );
}

function buildActivitySummary({
  toolCount,
  reasoningCount,
  sourceCount,
}: {
  toolCount: number;
  reasoningCount: number;
  sourceCount: number;
}) {
  const fragments: string[] = [];
  if (toolCount > 0) {
    fragments.push(`${toolCount} tool${toolCount === 1 ? "" : "s"}`);
  }
  if (reasoningCount > 0) {
    fragments.push(
      `${reasoningCount} thought${reasoningCount === 1 ? "" : "s"}`,
    );
  }
  if (sourceCount > 0) {
    fragments.push(
      `${sourceCount} source${sourceCount === 1 ? "" : "s"}`,
    );
  }
  return fragments.join(" · ");
}

function ReasoningLine({ text }: { text: string }) {
  return (
    <li className="border-b border-[var(--line)]/40 px-4 py-2.5 last:border-b-0">
      <div className="flex gap-2">
        <Sparkles className="mt-1 h-3 w-3 shrink-0 text-[var(--muted)]" />
        <div className="min-w-0 flex-1 italic text-[13px] leading-[1.55] text-[var(--muted)]">
          <MarkdownBlock className="markdown-block-muted">
            {text}
          </MarkdownBlock>
        </div>
      </div>
    </li>
  );
}

function ToolRow({ part }: { part: TutorMessagePart }) {
  const [isOpen, setIsOpen] = useState(false);
  const name = prettifyToolName(part.type);
  const inputSummary = toolInputSummary(part.input);
  const outputObject =
    part.output && typeof part.output === "object"
      ? (part.output as { text?: string; matches?: unknown[] })
      : undefined;
  const outputText = (outputObject?.text ?? "").trim();
  const matchCount = Array.isArray(outputObject?.matches)
    ? outputObject.matches.length
    : 0;
  const resultSummary = formatToolResultSummary({
    outputText,
    matchCount,
    state: part.state,
    errorText: part.errorText,
  });
  const stateBadge = formatToolStateBadge(part.state, part.errorText);
  const canExpand = Boolean(outputText) || Boolean(part.errorText);

  return (
    <li className="border-b border-[var(--line)]/40 last:border-b-0">
      <button
        type="button"
        onClick={(event) => {
          event.stopPropagation();
          if (canExpand) {
            setIsOpen((current) => !current);
          }
        }}
        disabled={!canExpand}
        className={clsx(
          "flex w-full items-center gap-2 px-4 py-2 text-left text-[12.5px] leading-[1.5]",
          canExpand && "hover:bg-[var(--paper)]/40",
        )}
        aria-expanded={canExpand ? isOpen : undefined}
      >
        <ToolKindIcon type={part.type} className="h-3.5 w-3.5 shrink-0 text-[var(--muted)]" />
        <span className="shrink-0 text-[11px] uppercase tracking-[0.06em] text-[var(--muted)]">
          {name}
        </span>
        {inputSummary ? (
          <code className="min-w-0 flex-1 truncate font-mono text-[12px] text-[var(--ink)]">
            {inputSummary}
          </code>
        ) : (
          <span className="flex-1" />
        )}
        {resultSummary ? (
          <span className="shrink-0 text-[11px] text-[var(--muted)]">{resultSummary}</span>
        ) : null}
        {stateBadge ? (
          <span
            className={clsx(
              "shrink-0 rounded-full px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-[0.08em]",
              stateBadge.tone === "error"
                ? "bg-red-100 text-red-700"
                : "bg-[var(--paper)]/60 text-[var(--muted)]",
            )}
          >
            {stateBadge.label}
          </span>
        ) : null}
        {canExpand ? (
          <ChevronRight
            className={clsx(
              "h-3 w-3 shrink-0 text-[var(--muted)]/70 transition-transform",
              isOpen && "rotate-90",
            )}
          />
        ) : null}
      </button>
      {isOpen && canExpand ? (
        <div className="border-t border-[var(--line)]/30 bg-[var(--paper)]/40 px-4 py-2">
          {part.errorText ? (
            <pre className="whitespace-pre-wrap break-words text-[11.5px] leading-[1.55] text-red-700">
              {part.errorText}
            </pre>
          ) : (
            <pre className="max-h-72 overflow-y-auto whitespace-pre-wrap break-words font-mono text-[11.5px] leading-[1.55] text-[var(--ink)]/80">
              {outputText}
            </pre>
          )}
        </div>
      ) : null}
    </li>
  );
}

function ToolKindIcon({ type, className }: { type: string; className?: string }) {
  const name = type.replace(/^tool-/, "");
  if (name.includes("kb_command") || name.includes("shell") || name.includes("command")) {
    return <Terminal className={className} />;
  }
  if (name.includes("retrieve") || name.includes("search")) {
    return <Database className={className} />;
  }
  if (name.includes("web") || name.includes("url") || name.includes("google")) {
    return <Globe className={className} />;
  }
  return <Wrench className={className} />;
}

function formatToolStateBadge(
  state: string | undefined,
  errorText: string | undefined,
): { label: string; tone: "default" | "error" } | null {
  if (errorText) {
    return { label: "error", tone: "error" };
  }
  if (!state || state === "output-available") {
    return null;
  }
  if (state === "input-streaming" || state === "input-available") {
    return { label: "running", tone: "default" };
  }
  return { label: state.replaceAll("-", " "), tone: "default" };
}

function formatToolResultSummary({
  outputText,
  matchCount,
  state,
  errorText,
}: {
  outputText: string;
  matchCount: number;
  state?: string;
  errorText?: string;
}) {
  if (errorText) {
    return "";
  }
  if (matchCount > 0) {
    return `${matchCount} match${matchCount === 1 ? "" : "es"}`;
  }
  if (outputText) {
    const lineCount = outputText.split("\n").length;
    const charCount = outputText.length;
    if (lineCount > 3) {
      return `${lineCount} lines`;
    }
    return charCount > 1000 ? `${(charCount / 1000).toFixed(1)}k chars` : `${charCount} chars`;
  }
  if (state && state !== "output-available") {
    return "";
  }
  return "";
}

function TextBlock({ parts }: { parts: TutorMessagePart[] }) {
  return (
    <div className="space-y-3 text-[15px] leading-[1.72] tracking-[-0.012em] text-[var(--ink)]">
      {parts.map((part, index) => (
        <MarkdownBlock
          key={`text-${index}`}
          className="text-[15px] leading-[1.72] tracking-[-0.012em] text-[var(--ink)]"
        >
          {part.text ?? ""}
        </MarkdownBlock>
      ))}
    </div>
  );
}

function getToolPartSourceCount(parts: TutorMessagePart[]) {
  const seen = new Set<string>();

  for (const part of parts) {
    const output =
      part.output && typeof part.output === "object"
        ? (part.output as { matches?: unknown[] })
        : undefined;

    for (const match of Array.isArray(output?.matches) ? output.matches : []) {
      if (!match || typeof match !== "object") {
        continue;
      }

      const source = match as { docId?: string; url?: string };
      const key = source.docId || source.url;
      if (!key) {
        continue;
      }
      seen.add(key);
    }
  }

  return seen.size;
}

const CITATION_KIND_META: Record<
  MessageCitation["kind"],
  { icon: ComponentType<SVGProps<SVGSVGElement>>; label: string }
> = {
  web: { icon: Globe, label: "Web" },
  course: { icon: GraduationCap, label: "Course" },
  doc: { icon: BookOpen, label: "Docs" },
};

function CitationRow({ citations }: { citations: MessageCitation[] }) {
  return (
    <section className="mt-2 space-y-2 border-t border-[var(--line)] pt-3">
      <div className="inline-flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-[0.12em] text-[var(--muted)]">
        <LibraryBig className="h-3.5 w-3.5 text-[var(--accent)]" />
        <span>Sources</span>
        <span className="rounded-full bg-[var(--accent-faint)] px-1.5 py-0.5 text-[10px] font-semibold text-[var(--accent)]">
          {citations.length}
        </span>
      </div>
      <div className="flex flex-wrap gap-2">
        {citations.map((citation, index) => {
          const meta = CITATION_KIND_META[citation.kind];
          const Icon = meta.icon;
          return (
            <a
              key={`${citation.kind}-${citation.url}-${index}`}
              href={citation.url}
              target="_blank"
              rel="noreferrer"
              title={`${meta.label}${citation.sublabel ? ` · ${citation.sublabel}` : ""}`}
              className="group inline-flex max-w-full items-center gap-1.5 rounded-full border border-[var(--line-strong)] bg-[var(--accent-faint)] px-2.5 py-1.5 text-xs font-medium text-[var(--accent)] transition hover:border-[var(--accent)] hover:bg-[var(--surface-strong)]"
            >
              <Icon className="h-3.5 w-3.5 shrink-0" />
              <span className="truncate">{citation.label}</span>
              <ExternalLink className="h-3 w-3 shrink-0 opacity-60 transition group-hover:opacity-100" />
            </a>
          );
        })}
      </div>
    </section>
  );
}
