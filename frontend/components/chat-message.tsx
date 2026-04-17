"use client";

import clsx from "clsx";
import {
  BookOpen,
  Check,
  ChevronDown,
  Copy,
  ExternalLink,
  Globe,
  GraduationCap,
  LibraryBig,
  Pencil,
  RefreshCw,
  SearchCheck,
  Sparkles,
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
  TutorMessage,
  TutorMessageBlock,
  MessageCitation,
  TutorMessagePart,
} from "@/lib/chat-ui";
import {
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
  if (block.kind === "reasoning") {
    return <ReasoningBlock parts={block.parts} isActive={isActive} />;
  }

  if (block.kind === "tool") {
    return <ToolActivityBlock parts={block.parts} isActive={isActive} />;
  }

  return <TextBlock parts={block.parts} />;
}

function ReasoningBlock({
  parts,
  isActive = false,
}: {
  parts: TutorMessagePart[];
  isActive?: boolean;
}) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <section className="relative overflow-hidden rounded-[1.35rem] border border-[var(--line)] bg-[var(--paper)]/80 px-4 py-3">
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
          setIsOpen((current) => !current);
        }}
        className="relative z-[1] flex w-full items-center justify-between gap-3 text-left"
        aria-expanded={isOpen}
      >
        <div className="inline-flex items-center gap-2 text-[13px] font-semibold tracking-[-0.012em] text-[var(--ink)]">
          <Sparkles
            className={clsx(
              "h-4 w-4 text-[var(--accent)]",
              isActive && "animate-pulse",
            )}
          />
          <span>{isActive ? "Thinking" : "Reasoning trace"}</span>
          {isActive ? (
            <span
              aria-hidden="true"
              className="processing-button__pulse ml-0.5 h-1.5 w-1.5 rounded-full text-[var(--accent)]"
            />
          ) : null}
        </div>
        <ChevronDown
          className={clsx(
            "h-4 w-4 text-[var(--muted)] transition-transform",
            isOpen && "rotate-180",
          )}
        />
      </button>
      {isOpen ? (
        <div className="relative z-[1] mt-3 space-y-3 text-[14px] leading-[1.65] text-[var(--muted)]">
          {parts.map((part, index) => (
            <MarkdownBlock
              key={`reasoning-${index}`}
              className="markdown-block-muted"
            >
              {part.text ?? ""}
            </MarkdownBlock>
          ))}
        </div>
      ) : null}
    </section>
  );
}

function ToolActivityBlock({
  parts,
  isActive = false,
}: {
  parts: TutorMessagePart[];
  isActive?: boolean;
}) {
  const [isOpen, setIsOpen] = useState(false);

  const sourceCount = getToolPartSourceCount(parts);
  const sourceCountLabel =
    sourceCount > 0
      ? `${sourceCount} source${sourceCount === 1 ? "" : "s"}`
      : "";

  return (
    <section className="relative overflow-hidden rounded-[1.35rem] border border-[var(--line)] bg-[var(--surface)] px-4 py-3">
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
          setIsOpen((current) => !current);
        }}
        className="relative z-[1] flex w-full items-center justify-between gap-3 text-left"
        aria-expanded={isOpen}
      >
        <div className="inline-flex items-center gap-2 text-[13px] font-semibold tracking-[-0.012em] text-[var(--ink)]">
          <Wrench
            className={clsx(
              "h-4 w-4 text-[var(--accent)]",
              isActive && "animate-pulse",
            )}
          />
          <span>{isActive ? "Running tools" : "Tool activity"}</span>
          {isActive ? (
            <span
              aria-hidden="true"
              className="processing-button__pulse ml-0.5 h-1.5 w-1.5 rounded-full text-[var(--accent)]"
            />
          ) : null}
        </div>
        <div className="flex items-center gap-3">
          {sourceCountLabel ? (
            <span className="rounded-full bg-[var(--surface)] px-2.5 py-1 text-[10px] font-semibold uppercase leading-none tracking-[0.1em] text-[var(--muted)]">
              {sourceCountLabel}
            </span>
          ) : null}
          <ChevronDown
            className={clsx(
              "h-4 w-4 text-[var(--muted)] transition-transform",
              isOpen && "rotate-180",
            )}
          />
        </div>
      </button>
      {isOpen ? (
        <div className="relative z-[1] mt-3 space-y-3">
          {parts.map((part, index) => (
            <ToolCard key={`tool-${index}`} part={part} />
          ))}
        </div>
      ) : null}
    </section>
  );
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

function ToolCard({ part }: { part: TutorMessagePart }) {
  const summary = toolInputSummary(part.input);
  const output =
    part.output && typeof part.output === "object"
      ? (part.output as { text?: string; matches?: unknown[] })
      : undefined;
  const matchCount = Array.isArray(output?.matches) ? output.matches.length : 0;
  const resultSummary = describeToolResult(output?.text, matchCount);

  return (
    <div className="rounded-[1.25rem] border border-[var(--line)] bg-[var(--surface)] px-4 py-3 shadow-[0_10px_30px_rgba(18,42,204,0.05)]">
      <div className="flex items-center justify-between gap-3">
        <div>
          <p className="text-[10px] font-semibold uppercase tracking-[0.11em] text-[var(--muted)]">
            Tool call
          </p>
          <p className="mt-1 text-[13px] font-semibold tracking-[-0.012em] text-[var(--ink)]">
            {prettifyToolName(part.type)}
          </p>
        </div>
        <span className="rounded-full bg-[var(--surface)] px-2 py-1 text-[10px] font-medium uppercase tracking-[0.1em] text-[var(--muted)]">
          {formatToolState(part.state)}
        </span>
      </div>

      {summary ? (
        <div className="mt-3 rounded-[1rem] border border-[var(--line)] bg-[var(--paper)]/65 px-3 py-3">
          <div className="inline-flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.11em] text-[var(--muted)]">
            <SearchCheck className="h-3.5 w-3.5 text-[var(--accent)]" />
            Tool input
          </div>
          <p className="mt-2 text-[13px] leading-[1.65] text-[var(--ink)]">{summary}</p>
        </div>
      ) : null}

      {resultSummary ? (
        <div className="mt-3 rounded-[1rem] border border-[var(--line)] bg-[rgba(11,136,238,0.08)] px-3 py-3">
          <p className="text-[10px] font-semibold uppercase tracking-[0.11em] text-[var(--muted)]">
            Tool result
          </p>
          <p className="mt-2 text-[13px] leading-[1.65] text-[var(--ink)]">
            {resultSummary}
          </p>
        </div>
      ) : null}

      {part.errorText ? (
        <p className="mt-3 text-xs leading-5 text-red-700">{part.errorText}</p>
      ) : null}
    </div>
  );
}

function formatToolState(state?: string) {
  if (!state) {
    return "running";
  }

  return state.replaceAll("-", " ");
}

function describeToolResult(outputText?: string, matchCount = 0) {
  if (matchCount > 0) {
    return `${matchCount} source match${matchCount === 1 ? "" : "es"} captured for the final answer.`;
  }

  if ((outputText ?? "").trim()) {
    return "Tool completed.";
  }

  return "";
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
              title={`${meta.label}${citation.sublabel ? ` — ${citation.sublabel}` : ""}`}
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
