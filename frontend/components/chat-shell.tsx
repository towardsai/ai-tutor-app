"use client";

import clsx from "clsx";
import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import { Check, ChevronDown, Lock, Send, Square, WandSparkles } from "lucide-react";
import {
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  useTransition,
} from "react";
import { ChatMessage } from "@/components/chat-message";
import { SourceSidebar } from "@/components/source-sidebar";
import {
  fetchTools,
  getApiBaseUrl,
  type AvailableModel,
  type TutorTool,
} from "@/lib/api";
import {
  getMessageTextContent,
  hasRenderableContent,
  type TutorMessage,
} from "@/lib/chat-ui";

type ThreadDataPart = {
  type: "data-thread";
  data: {
    threadId: string;
  };
};

export function ChatShell() {
  const [transport] = useState(
    () => new DefaultChatTransport({ api: `${getApiBaseUrl()}/api/chat` }),
  );
  const [tools, setTools] = useState<TutorTool[]>([]);
  const [availableModels, setAvailableModels] = useState<AvailableModel[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [selectedSourceKeys, setSelectedSourceKeys] = useState<string[]>([]);
  const [sourceError, setSourceError] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [threadId, setThreadId] = useState("");
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);
  const [editingMessageId, setEditingMessageId] = useState<string | null>(null);
  const [editingText, setEditingText] = useState("");
  const [, startTransition] = useTransition();
  const threadViewportRef = useRef<HTMLDivElement>(null);
  const composerInputRef = useRef<HTMLTextAreaElement>(null);
  const pendingScrollRef = useRef(false);
  const [spacerHeight, setSpacerHeight] = useState(0);

  const {
    messages,
    sendMessage,
    setMessages,
    regenerate,
    stop,
    status,
    error,
    clearError,
  } = useChat({
    transport,
    onData: (part) => {
      if (part.type === "data-thread") {
        const nextThreadId = (part as ThreadDataPart).data.threadId;
        if (nextThreadId) {
          startTransition(() => setThreadId(nextThreadId));
        }
      }
    },
    experimental_throttle: 50,
  });

  const initialFetchDoneRef = useRef(false);

  useEffect(() => {
    const controller = new AbortController();

    async function loadTools() {
      try {
        const {
          tools: loadedTools,
          availableModels: models,
          model,
        } = await fetchTools(controller.signal);
        setTools(loadedTools);
        setAvailableModels(models ?? []);
        setSelectedModel(model);
        const retrieval = loadedTools.find(
          (tool) => tool.kind === "configurable",
        );
        if (retrieval) {
          setSelectedSourceKeys(
            retrieval.sources
              .filter((source) => source.selectedByDefault)
              .map((source) => source.key),
          );
        }
        setSourceError(null);
      } catch (loadError) {
        if (controller.signal.aborted) {
          return;
        }
        setSourceError(
          loadError instanceof Error
            ? loadError.message
            : "Unable to load tool registry.",
        );
      }
    }

    void loadTools();
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!selectedModel) {
      return;
    }
    if (!initialFetchDoneRef.current) {
      initialFetchDoneRef.current = true;
      return;
    }

    const controller = new AbortController();

    async function refetchTools() {
      try {
        const { tools: loadedTools, availableModels: models } =
          await fetchTools(controller.signal, selectedModel);
        setTools(loadedTools);
        setAvailableModels(models ?? []);
      } catch (loadError) {
        if (controller.signal.aborted) {
          return;
        }
        setSourceError(
          loadError instanceof Error
            ? loadError.message
            : "Unable to load tool registry.",
        );
      }
    }

    void refetchTools();
    return () => controller.abort();
  }, [selectedModel]);

  const isStreaming = status === "submitted" || status === "streaming";
  const isReady = status === "ready";
  const typedMessages = messages as TutorMessage[];
  const latestMessage = typedMessages[typedMessages.length - 1];
  const streamingAssistantId =
    isStreaming && latestMessage?.role === "assistant"
      ? latestMessage.id
      : null;
  const visibleMessages = typedMessages.filter((message) => {
    if (message.id !== streamingAssistantId) {
      return true;
    }
    return hasRenderableContent(message);
  });
  const chatColumnClass =
    "mx-auto w-full max-w-[1040px] px-3 sm:px-5 lg:px-8 xl:px-10";

  useEffect(() => {
    const viewport = threadViewportRef.current;
    if (!viewport) {
      return;
    }

    const userElements = viewport.querySelectorAll<HTMLElement>(
      '[data-role="user"]',
    );
    const lastUserEl = userElements[userElements.length - 1];

    let needed = 0;
    if (lastUserEl && (isStreaming || pendingScrollRef.current)) {
      const container = lastUserEl.parentElement;
      const children = container
        ? (Array.from(container.children) as HTMLElement[])
        : [];
      const lastNonSpacer = children
        .filter((child) => child.dataset.spacer !== "true")
        .pop();
      if (lastNonSpacer) {
        const spaceFromLastUserTop =
          lastNonSpacer.getBoundingClientRect().bottom -
          lastUserEl.getBoundingClientRect().top;
        needed = Math.max(
          0,
          viewport.clientHeight - 24 - spaceFromLastUserTop,
        );
      }
    }

    setSpacerHeight(needed);
  }, [messages, isStreaming]);

  useLayoutEffect(() => {
    if (!pendingScrollRef.current) {
      return;
    }
    const viewport = threadViewportRef.current;
    if (!viewport) {
      return;
    }
    pendingScrollRef.current = false;

    const users = viewport.querySelectorAll<HTMLElement>(
      '[data-role="user"]',
    );
    const lastUserEl = users[users.length - 1];
    if (!lastUserEl) {
      return;
    }

    const delta =
      lastUserEl.getBoundingClientRect().top -
      viewport.getBoundingClientRect().top;
    viewport.scrollTo({
      top: Math.max(0, viewport.scrollTop + delta - 24),
      behavior: "instant",
    });
  }, [spacerHeight]);

  useEffect(() => {
    const textarea = composerInputRef.current;
    if (!textarea) {
      return;
    }

    textarea.style.height = "0px";
    textarea.style.height = `${Math.min(textarea.scrollHeight, 176)}px`;
  }, [input]);

  async function handleSubmit() {
    const trimmed = input.trim();
    if (!trimmed || isStreaming) {
      return;
    }

    clearError();
    setCopiedMessageId(null);
    setEditingMessageId(null);
    setEditingText("");
    setInput("");
    pendingScrollRef.current = true;
    await sendMessage(
      { text: trimmed },
      {
        body: {
          sourceKeys: selectedSourceKeys,
          includeReasoning: true,
          threadId,
          model: selectedModel,
        },
      },
    );
  }

  async function handleRedo(messageId?: string) {
    clearError();
    setCopiedMessageId(null);
    await regenerate({
      messageId,
      body: {
        sourceKeys: selectedSourceKeys,
        includeReasoning: true,
        threadId,
        model: selectedModel,
      },
    });
  }

  function handleEdit(messageId: string) {
    const messageIndex = messages.findIndex((message) => message.id === messageId);
    if (messageIndex === -1) {
      return;
    }

    const message = messages[messageIndex] as TutorMessage;
    const draft = getMessageTextContent(message);

    clearError();
    setCopiedMessageId(null);
    setEditingMessageId(messageId);
    setEditingText(draft);
  }

  async function handleCopy(message: TutorMessage) {
    const text = getMessageTextContent(message);
    if (!text) {
      return;
    }

    try {
      await navigator.clipboard.writeText(text);
      setCopiedMessageId(message.id);
      window.setTimeout(() => {
        setCopiedMessageId((current) =>
          current === message.id ? null : current,
        );
      }, 1400);
    } catch {
      setCopiedMessageId(null);
    }
  }

  function handleEditCancel() {
    setEditingMessageId(null);
    setEditingText("");
  }

  async function handleEditSave(messageId: string) {
    const trimmed = editingText.trim();
    if (!trimmed || isStreaming) {
      return;
    }

    clearError();
    setCopiedMessageId(null);
    setEditingMessageId(null);
    setEditingText("");
    pendingScrollRef.current = true;
    await sendMessage(
      { text: trimmed, messageId },
      {
        body: {
          sourceKeys: selectedSourceKeys,
          includeReasoning: true,
          threadId,
          model: selectedModel,
        },
      },
    );
  }

  function handleNewChat() {
    if (isStreaming) {
      stop();
    }
    clearError();
    setMessages([]);
    setThreadId("");
    setInput("");
    setEditingMessageId(null);
    setEditingText("");
    setCopiedMessageId(null);
  }

  function toggleSource(sourceKey: string) {
    setSelectedSourceKeys((current) =>
      current.includes(sourceKey)
        ? current.filter((key) => key !== sourceKey)
        : [...current, sourceKey],
    );
  }

  return (
    <main className="min-h-screen p-2 lg:h-screen lg:overflow-hidden">
      <div className="flex min-h-[calc(100vh-1rem)] w-full flex-col gap-2 lg:h-[calc(100vh-1rem)] lg:min-h-0 lg:grid lg:grid-cols-[248px_minmax(0,1fr)]">
        <SourceSidebar
          onNewChat={handleNewChat}
          onToggleSource={toggleSource}
          selectedSourceKeys={selectedSourceKeys}
          sourceError={sourceError}
          tools={tools}
        />

        <section className="glass-panel flex min-h-0 flex-col overflow-hidden rounded-[1.5rem] p-2 sm:p-2.5 lg:max-h-[calc(100vh-1rem)] lg:min-h-[calc(100vh-1rem)]">
          <div
            ref={threadViewportRef}
            className="scrollbar-thin min-h-0 flex-1 overflow-y-auto"
          >
            {typedMessages.length === 0 ? (
              <div className={clsx(chatColumnClass, "flex min-h-full")}>
                <EmptyConversation
                  onSelect={(prompt) => {
                    setInput(prompt);
                    composerInputRef.current?.focus();
                  }}
                />
              </div>
            ) : (
              <div className={clsx(chatColumnClass, "flex flex-col gap-3 pt-4 pb-3")}>
                {visibleMessages.map((message) => (
                  <ChatMessage
                    key={message.id}
                    message={message}
                    actionDisabled={isStreaming}
                    copied={copiedMessageId === message.id}
                    editDraft={editingMessageId === message.id ? editingText : ""}
                    isEditing={editingMessageId === message.id}
                    isStreaming={message.id === streamingAssistantId}
                    onAssistantCopy={handleCopy}
                    onAssistantRedo={(messageId) => void handleRedo(messageId)}
                    onEditCancel={handleEditCancel}
                    onEditChange={setEditingText}
                    onEditSave={(messageId) => void handleEditSave(messageId)}
                    onUserEdit={handleEdit}
                    showAssistantActions={message.id !== streamingAssistantId}
                  />
                ))}
                <div
                  aria-hidden
                  data-spacer="true"
                  className="shrink-0"
                  style={{ height: spacerHeight }}
                />
              </div>
            )}
          </div>

          <footer className="sticky bottom-0 z-10 mt-2">
            <div className={chatColumnClass}>
              <div className="rounded-[1.1rem] border border-[var(--line)] bg-[var(--panel-strong)] px-2.5 py-2 shadow-[0_10px_30px_rgba(18,42,204,0.08)] backdrop-blur-xl">
                <textarea
                  ref={composerInputRef}
                  rows={1}
                  value={input}
                  onChange={(event) => setInput(event.target.value)}
                  onKeyDown={(event) => {
                    if (
                      event.key === "Enter" &&
                      !event.shiftKey &&
                      !event.altKey &&
                      !event.ctrlKey &&
                      !event.metaKey &&
                      !event.nativeEvent.isComposing
                    ) {
                      event.preventDefault();
                      void handleSubmit();
                    }
                  }}
                  placeholder="Ask about RAG, LangGraph, PEFT, or any of your selected sources…"
                  className="max-h-44 min-h-10 w-full resize-none overflow-y-auto bg-transparent px-0.5 py-2 text-[15px] leading-[1.6] tracking-[-0.012em] text-[var(--ink)] outline-none placeholder:text-[var(--muted)]"
                />

                <div className="mt-1.5 flex items-center justify-between gap-2">
                  <ModelPicker
                    availableModels={availableModels}
                    locked={typedMessages.length > 0}
                    onSelect={setSelectedModel}
                    selectedModel={selectedModel}
                  />
                  <ComposerActionButton
                    disabled={!isStreaming && (!input.trim() || !isReady)}
                    isStreaming={isStreaming}
                    onClick={
                      isStreaming ? () => stop() : () => void handleSubmit()
                    }
                  />
                </div>
              </div>

              {error ? (
                <p className="mt-2 rounded-[0.9rem] border border-red-300/70 bg-red-50 px-3 py-2 text-sm text-red-800">
                  {error.message}
                </p>
              ) : null}
            </div>
          </footer>
        </section>

      </div>
    </main>
  );
}

function formatModelName(raw: string): string {
  const name = raw.includes(":") ? raw.split(":").slice(1).join(":") : raw;
  return name
    .replace(/-latest$/, "")
    .replace(/-preview$/, " preview")
    .replace(/-/g, " ")
    .replace(/\bgpt\b/gi, "GPT")
    .replace(/\b\w/g, (c) => c.toUpperCase())
    .trim();
}

function ModelPicker({
  availableModels,
  locked,
  onSelect,
  selectedModel,
}: {
  availableModels: AvailableModel[];
  locked: boolean;
  onSelect: (modelId: string) => void;
  selectedModel: string;
}) {
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) {
      return;
    }
    function handlePointerDown(event: MouseEvent) {
      if (!containerRef.current?.contains(event.target as Node)) {
        setOpen(false);
      }
    }
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", handlePointerDown);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("mousedown", handlePointerDown);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [open]);

  if (!selectedModel) {
    return <span />;
  }

  const label =
    availableModels.find((model) => model.id === selectedModel)?.label ??
    formatModelName(selectedModel);
  const hasOptions = availableModels.length > 0;

  return (
    <div ref={containerRef} className="group relative">
      {locked ? (
        <span
          role="tooltip"
          className="pointer-events-none absolute bottom-full left-0 z-20 mb-2 whitespace-nowrap rounded-[0.6rem] border border-[var(--line)] bg-[var(--panel-strong)] px-2.5 py-1.5 opacity-0 shadow-[0_8px_20px_rgba(18,42,204,0.12)] backdrop-blur-xl transition-opacity duration-150 group-hover:opacity-100"
        >
          <span className="text-[10.5px] font-medium tracking-[-0.005em] text-[var(--ink)]">
            Start a new chat to change models
          </span>
        </span>
      ) : null}
      <button
        type="button"
        onClick={() => {
          if (!locked && hasOptions) {
            setOpen((current) => !current);
          }
        }}
        disabled={locked || !hasOptions}
        aria-haspopup="listbox"
        aria-expanded={open}
        title={
          locked
            ? undefined
            : hasOptions
              ? "Change model"
              : selectedModel
        }
        className={clsx(
          "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 transition",
          locked
            ? "cursor-not-allowed border-[var(--line)] bg-[var(--surface-soft)] text-[var(--muted)] opacity-75"
            : "border-[var(--line)] bg-[var(--surface-soft)] text-[var(--muted)] hover:border-[var(--accent)] hover:bg-[var(--accent-faint)] hover:text-[var(--accent)]",
        )}
      >
        <span className="h-1.5 w-1.5 rounded-full bg-[var(--accent)]" />
        <span className="text-[10.5px] font-medium tracking-[-0.005em]">
          {label}
        </span>
        {locked ? (
          <Lock className="h-2.5 w-2.5" aria-hidden="true" />
        ) : (
          <ChevronDown
            className={clsx(
              "h-3 w-3 transition-transform",
              open && "rotate-180",
            )}
            aria-hidden="true"
          />
        )}
      </button>

      {open ? (
        <div
          role="listbox"
          aria-label="Select model"
          className="absolute bottom-full left-0 z-20 mb-2 w-64 overflow-hidden rounded-[0.9rem] border border-[var(--line)] bg-[var(--panel-strong)] p-1 shadow-[0_16px_40px_rgba(18,42,204,0.18)] backdrop-blur-xl"
        >
          {availableModels.map((model) => {
            const isActive = model.id === selectedModel;
            return (
              <button
                key={model.id}
                type="button"
                role="option"
                aria-selected={isActive}
                onClick={() => {
                  onSelect(model.id);
                  setOpen(false);
                }}
                className={clsx(
                  "flex w-full items-center justify-between gap-2 rounded-[0.65rem] px-2.5 py-1.5 text-left transition",
                  isActive
                    ? "bg-[var(--accent-faint)] text-[var(--accent)]"
                    : "text-[var(--ink)] hover:bg-[var(--surface-hover)]",
                )}
              >
                <span className="truncate text-[12.5px] font-medium tracking-[-0.01em]">
                  {model.label}
                </span>
                {isActive ? (
                  <Check className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
                ) : null}
              </button>
            );
          })}
        </div>
      ) : null}
    </div>
  );
}

function ComposerActionButton({
  disabled,
  isStreaming,
  onClick,
}: {
  disabled: boolean;
  isStreaming: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-label={isStreaming ? "Stop generating" : "Send message"}
      title={isStreaming ? "Stop generating" : "Send message"}
      className={clsx(
        "relative inline-flex h-10 min-w-[8.75rem] items-center justify-center overflow-hidden rounded-full px-4 text-sm font-semibold transition disabled:cursor-not-allowed disabled:opacity-50",
        isStreaming
          ? "border border-[var(--line-strong)] bg-[rgba(11,136,238,0.12)] text-[var(--accent)] shadow-[0_12px_30px_rgba(11,136,238,0.14)] hover:border-[var(--accent)] hover:bg-[rgba(11,136,238,0.15)]"
          : "bg-[var(--accent)] text-white hover:brightness-110",
      )}
    >
      {isStreaming ? (
        <>
          <span
            aria-hidden="true"
            className="processing-button__sheen absolute inset-0 rounded-full"
          />
          <span
            aria-hidden="true"
            className="processing-button__orb absolute left-2 top-1/2 h-6 w-6 -translate-y-1/2 rounded-full"
          />
          <span className="relative z-10 inline-flex items-center gap-2.5">
            <span
              aria-hidden="true"
              className="processing-button__pulse h-2.5 w-2.5 rounded-full"
            />
            <span>Streaming</span>
            <span className="inline-flex h-6 w-6 items-center justify-center rounded-full bg-white/92 text-[var(--accent)] shadow-[0_6px_16px_rgba(11,136,238,0.18)]">
              <Square className="h-3 w-3 fill-current" />
            </span>
          </span>
        </>
      ) : (
        <span className="relative z-10 inline-flex items-center gap-2">
          <Send className="h-4 w-4" />
          <span>Send</span>
        </span>
      )}
    </button>
  );
}

const SUGGESTIONS: ReadonlyArray<{ title: string; prompt: string }> = [
  {
    title: "RAG vs fine-tuning",
    prompt:
      "When should I use retrieval-augmented generation instead of fine-tuning a model?",
  },
  {
    title: "LoRA with PEFT",
    prompt:
      "Walk me through fine-tuning a model with LoRA using the PEFT library.",
  },
  {
    title: "Build a LangGraph agent",
    prompt:
      "How do I build a tool-calling agent with LangGraph, step by step?",
  },
  {
    title: "Evaluate a RAG pipeline",
    prompt:
      "What are practical ways to evaluate the quality of a RAG pipeline?",
  },
];

function EmptyConversation({
  onSelect,
}: {
  onSelect: (prompt: string) => void;
}) {
  return (
    <div className="flex min-h-full flex-1 flex-col items-center justify-center px-4 py-10 text-center">
      <div className="inline-flex h-11 w-11 items-center justify-center rounded-2xl bg-[var(--accent-faint)] text-[var(--accent)]">
        <WandSparkles className="h-5 w-5" />
      </div>
      <h2 className="display-font mt-4 text-[1.95rem] leading-[1] text-[var(--ink)] sm:text-[2.2rem]">
        Ask your AI tutor
      </h2>
      <p className="mt-2 max-w-md text-[13.5px] leading-[1.6] text-[var(--muted)]">
        Your companion for{" "}
        <a
          href="https://academy.towardsai.net/"
          target="_blank"
          rel="noreferrer"
          className="font-semibold text-[var(--ink)] underline decoration-[var(--accent)]/40 underline-offset-2 transition hover:text-[var(--accent)] hover:decoration-[var(--accent)]"
        >
          Towards AI Academy
        </a>{" "}
        courses. Answers are grounded in the sources you select — try one of
        these to start:
      </p>
      <div className="mt-6 grid w-full max-w-[640px] grid-cols-1 gap-2 sm:grid-cols-2">
        {SUGGESTIONS.map((suggestion) => (
          <button
            key={suggestion.title}
            type="button"
            onClick={() => onSelect(suggestion.prompt)}
            className="group flex flex-col items-start gap-1 rounded-[0.95rem] border border-[var(--line)] bg-[var(--surface)] px-3.5 py-3 text-left transition hover:-translate-y-0.5 hover:border-[var(--accent)] hover:bg-[var(--surface-strong)] hover:shadow-[0_8px_20px_rgba(11,136,238,0.08)]"
          >
            <span className="text-[12.5px] font-semibold tracking-[-0.01em] text-[var(--ink)]">
              {suggestion.title}
            </span>
            <span className="text-[12px] leading-[1.45] text-[var(--muted)]">
              {suggestion.prompt}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}
