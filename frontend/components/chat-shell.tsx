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
} from "react";
import { ChatMessage } from "@/components/chat-message";
import { SourceSidebar } from "@/components/source-sidebar";
import {
  fetchTools,
  getApiBaseUrl,
  type AvailableModel,
  type TutorTool,
} from "@/lib/api";
import { prepareTutorChatRequest } from "@/lib/chat-transport";
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

const STREAMING_WORDS = [
  "Tutoring",
  "Pondering",
  "Scribbling",
  "Annotating",
  "Consulting",
  "Mulling",
  "Dissertating",
  "Leafing",
  "Researching",
  "Outlining",
  "Skimming",
  "Unpacking",
  "Diagramming",
  "Curating",
  "Deliberating",
  "Summarizing",
  "Highlighting",
  "Chalkboarding",
];

function pickStreamingWordForKey(key: string) {
  let hash = 0;
  for (let index = 0; index < key.length; index += 1) {
    hash = (hash * 31 + key.charCodeAt(index)) | 0;
  }
  return STREAMING_WORDS[Math.abs(hash) % STREAMING_WORDS.length];
}

export function ChatShell() {
  const [transport] = useState(
    () =>
      new DefaultChatTransport({
        api: `${getApiBaseUrl()}/api/chat`,
        prepareSendMessagesRequest: prepareTutorChatRequest,
      }),
  );
  const [tools, setTools] = useState<TutorTool[]>([]);
  const [availableModels, setAvailableModels] = useState<AvailableModel[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [selectedSourceKeys, setSelectedSourceKeys] = useState<string[]>([]);
  const [enabledToolKeys, setEnabledToolKeys] = useState<string[]>([]);
  const [sourceError, setSourceError] = useState<string | null>(null);
  const [toolRegistryReady, setToolRegistryReady] = useState(false);
  const [input, setInput] = useState("");
  const [threadId, setThreadId] = useState("");
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);
  const [editingMessageId, setEditingMessageId] = useState<string | null>(null);
  const [editingText, setEditingText] = useState("");
  const threadViewportRef = useRef<HTMLDivElement>(null);
  const composerInputRef = useRef<HTMLTextAreaElement>(null);
  const pendingScrollRef = useRef(false);
  const spacerRef = useRef<HTMLDivElement>(null);
  const acceptThreadDataRef = useRef(false);

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
      if (part.type === "data-thread" && acceptThreadDataRef.current) {
        const nextThreadId = (part as ThreadDataPart).data.threadId;
        if (nextThreadId) {
          // Urgent update on purpose: deferring it (startTransition) lets a
          // concurrent "New chat" reset finish first and then be overwritten
          // by the stale thread id.
          setThreadId(nextThreadId);
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
        setEnabledToolKeys(
          loadedTools
            .filter((tool) => tool.kind === "toggle" && tool.active)
            .map((tool) => tool.key),
        );
        setSourceError(null);
        setToolRegistryReady(true);
      } catch (loadError) {
        if (controller.signal.aborted) {
          return;
        }
        setSourceError(
          loadError instanceof Error
            ? loadError.message
            : "Unable to load tool registry.",
        );
        setToolRegistryReady(false);
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
        setEnabledToolKeys(
          loadedTools
            .filter((tool) => tool.kind === "toggle" && tool.active)
            .map((tool) => tool.key),
        );
        setSourceError(null);
        setToolRegistryReady(true);
      } catch (loadError) {
        if (controller.signal.aborted) {
          return;
        }
        setSourceError(
          loadError instanceof Error
            ? loadError.message
            : "Unable to load tool registry.",
        );
        setToolRegistryReady(false);
      }
    }

    void refetchTools();
    return () => controller.abort();
  }, [selectedModel]);

  const isStreaming = status === "submitted" || status === "streaming";
  // After a stream error the SDK parks status on "error" until clearError()
  // or the next request; handleSubmit clears it, so the error state must
  // stay sendable or the Send button locks out mouse users for good.
  const canSend =
    toolRegistryReady && (status === "ready" || status === "error");
  const typedMessages = messages as TutorMessage[];
  const latestMessage = typedMessages[typedMessages.length - 1];
  const streamingAssistantId =
    isStreaming && latestMessage?.role === "assistant"
      ? latestMessage.id
      : null;
  const streamingWord = streamingAssistantId
    ? pickStreamingWordForKey(streamingAssistantId)
    : STREAMING_WORDS[0];
  const visibleMessages = typedMessages.filter((message) => {
    if (message.id !== streamingAssistantId) {
      return true;
    }
    return hasRenderableContent(message);
  });
  const chatColumnClass =
    "mx-auto w-full max-w-[1040px] px-3 sm:px-5 lg:px-8 xl:px-10";

  useLayoutEffect(() => {
    const viewport = threadViewportRef.current;
    const spacer = spacerRef.current;
    if (!viewport || !spacer) {
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

    spacer.style.height = `${needed}px`;

    if (!pendingScrollRef.current) {
      return;
    }
    pendingScrollRef.current = false;

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
  }, [messages, isStreaming]);

  useEffect(() => {
    const textarea = composerInputRef.current;
    if (!textarea) {
      return;
    }

    textarea.style.height = "0px";
    textarea.style.height = `${Math.min(textarea.scrollHeight, 176)}px`;
  }, [input]);

  async function handleSubmit(override?: string) {
    const trimmed = (override ?? input).trim();
    if (!trimmed || !canSend) {
      return;
    }

    clearError();
    acceptThreadDataRef.current = true;
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
          enabledTools: enabledToolKeys,
          includeReasoning: true,
          threadId,
          model: selectedModel,
        },
      },
    );
  }

  async function handleRedo(messageId?: string) {
    if (!canSend) {
      return;
    }
    clearError();
    acceptThreadDataRef.current = true;
    setCopiedMessageId(null);
    await regenerate({
      messageId,
      body: {
        sourceKeys: selectedSourceKeys,
        enabledTools: enabledToolKeys,
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
    if (!trimmed || !canSend) {
      return;
    }

    clearError();
    acceptThreadDataRef.current = true;
    setCopiedMessageId(null);
    setEditingMessageId(null);
    setEditingText("");
    pendingScrollRef.current = true;
    await sendMessage(
      { text: trimmed, messageId },
      {
        body: {
          sourceKeys: selectedSourceKeys,
          enabledTools: enabledToolKeys,
          includeReasoning: true,
          threadId,
          model: selectedModel,
        },
      },
    );
  }

  function handleNewChat() {
    acceptThreadDataRef.current = false;
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

  function toggleTool(toolKey: string) {
    setEnabledToolKeys((current) =>
      current.includes(toolKey)
        ? current.filter((key) => key !== toolKey)
        : [...current, toolKey],
    );
  }

  return (
    <main className="min-h-screen p-2 lg:h-screen lg:overflow-hidden">
      <div className="flex min-h-[calc(100vh-1rem)] w-full flex-col gap-2 lg:h-[calc(100vh-1rem)] lg:min-h-0 lg:grid lg:grid-cols-[248px_minmax(0,1fr)]">
        <SourceSidebar
          onNewChat={handleNewChat}
          onToggleSource={toggleSource}
          onToggleTool={toggleTool}
          selectedSourceKeys={selectedSourceKeys}
          enabledToolKeys={enabledToolKeys}
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
                    void handleSubmit(prompt);
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
                  ref={spacerRef}
                  aria-hidden
                  data-spacer="true"
                  className="shrink-0"
                  style={{ height: 0 }}
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
                  placeholder="Ask about RAG, agents, or Python, paste an error to debug your project, or get a course concept explained…"
                  className="max-h-44 min-h-10 w-full resize-none overflow-y-auto bg-transparent px-0.5 py-2 text-[15px] leading-[1.6] tracking-[-0.012em] text-[var(--ink)] outline-none placeholder:text-[var(--muted)]"
                />

                <div className="mt-1.5 flex items-center justify-between gap-2">
                  <ModelPicker
                    availableModels={availableModels}
                    locked={typedMessages.length > 0}
                    onSelect={(modelId) => {
                      if (modelId === selectedModel) {
                        return;
                      }
                      setToolRegistryReady(false);
                      setSelectedModel(modelId);
                    }}
                    selectedModel={selectedModel}
                  />
                  <ComposerActionButton
                    disabled={!isStreaming && (!input.trim() || !canSend)}
                    isStreaming={isStreaming}
                    streamingLabel={streamingWord}
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
  streamingLabel,
  onClick,
}: {
  disabled: boolean;
  isStreaming: boolean;
  streamingLabel: string;
  onClick: () => void;
}) {
  const highlightRef = useRef<HTMLSpanElement>(null);

  function handlePointerMove(event: React.PointerEvent<HTMLButtonElement>) {
    const rect = event.currentTarget.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * 100;
    const y = ((event.clientY - rect.top) / rect.height) * 100;
    event.currentTarget.style.setProperty("--glass-x", `${x}%`);
    event.currentTarget.style.setProperty("--glass-y", `${y}%`);
  }

  function handlePointerEnter() {
    if (highlightRef.current) highlightRef.current.style.opacity = "1";
  }

  function handlePointerLeave() {
    if (highlightRef.current) highlightRef.current.style.opacity = "0";
  }

  return (
    <button
      type="button"
      onClick={onClick}
      onPointerMove={handlePointerMove}
      onPointerEnter={handlePointerEnter}
      onPointerLeave={handlePointerLeave}
      disabled={disabled}
      aria-label={isStreaming ? "Stop generating" : "Send message"}
      title={isStreaming ? "Stop generating" : "Send message"}
      style={{ "--glass-x": "50%", "--glass-y": "50%" } as React.CSSProperties}
      className={clsx(
        "relative inline-flex h-10 min-w-[8.75rem] items-center justify-center overflow-hidden rounded-full px-4 text-sm font-semibold shadow-[inset_0_1px_0_rgba(255,255,255,0.28),inset_0_-1px_0_rgba(0,0,0,0.08)] transition disabled:cursor-not-allowed disabled:opacity-50",
        isStreaming
          ? "border border-[var(--line-strong)] bg-[rgba(11,136,238,0.12)] text-[var(--accent)] hover:border-[var(--accent)] hover:bg-[rgba(11,136,238,0.15)]"
          : "bg-[var(--accent)] text-white hover:brightness-110",
      )}
    >
      {!disabled ? (
        <span
          ref={highlightRef}
          aria-hidden="true"
          className="pointer-events-none absolute inset-0 rounded-full"
          style={{
            opacity: 0,
            transition: "opacity 220ms ease",
            mixBlendMode: "screen",
            background:
              "radial-gradient(200px 130px at var(--glass-x) var(--glass-y), rgba(255,255,255,0.55) 0%, rgba(255,255,255,0.18) 28%, rgba(255,255,255,0) 60%)",
            zIndex: 1,
          }}
        />
      ) : null}
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
            <span>{streamingLabel}</span>
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

type Suggestion = {
  title: string;
  prompt: string;
  kind: "technical" | "accessible";
};

const SUGGESTION_POOL: ReadonlyArray<Suggestion> = [
  // RAG and retrieval (Full Stack AI Engineering)
  {
    title: "RAG with LlamaIndex",
    prompt:
      "How do I build a basic RAG pipeline over my own data with LlamaIndex?",
    kind: "technical",
  },
  {
    title: "How should I chunk?",
    prompt:
      "How should I chunk my documents for RAG, and how does chunk size affect the answers I get?",
    kind: "technical",
  },
  {
    title: "Pick an embedding model",
    prompt:
      "How do I choose the right embedding model for my use case?",
    kind: "technical",
  },
  {
    title: "Re-ranking and top K",
    prompt:
      "How does re-ranking work in a RAG pipeline, and how do I pick the right value of K?",
    kind: "technical",
  },
  {
    title: "Vector indexing methods",
    prompt:
      "What are the main indexing methods for vector retrieval, and how do I choose one for a production system?",
    kind: "technical",
  },
  {
    title: "Evaluate retrieval",
    prompt:
      "Beyond hit rate and MRR, what metrics should I use to evaluate retrieval in my RAG system?",
    kind: "technical",
  },
  {
    title: "Context caching vs RAG",
    prompt:
      "When should I use a long-context model with context caching instead of RAG?",
    kind: "technical",
  },

  // Building LLM apps (Full Stack AI Engineering) — concepts + debugging
  {
    title: "Structured JSON outputs",
    prompt:
      "How do I get reliable structured JSON output from an LLM using a Pydantic model?",
    kind: "technical",
  },
  {
    title: "Pydantic output error",
    prompt:
      "I get an error when I use a Pydantic BaseModel for structured outputs. How do I debug and fix it?",
    kind: "technical",
  },
  {
    title: "Routing and validation",
    prompt:
      "How do I add question validation and routing so my app sends each query to the right model or tool?",
    kind: "technical",
  },
  {
    title: "Cut API token costs",
    prompt:
      "What are practical ways to reduce the input token costs when calling an LLM API?",
    kind: "technical",
  },
  {
    title: "Prompt injection",
    prompt:
      "How do I protect my app's system prompt from prompt injection and hacking?",
    kind: "technical",
  },
  {
    title: "Evaluate my prompts",
    prompt:
      "How do I evaluate and iterate on my prompts instead of just eyeballing the outputs?",
    kind: "technical",
  },

  // Agents (Agent Engineering: Building Multi-Agent Systems)
  {
    title: "Define a tool for an agent",
    prompt:
      "Show me a minimal example of defining a tool that an LLM agent can call.",
    kind: "technical",
  },
  {
    title: "How LangGraph flows",
    prompt:
      "Can you walk me through how a LangGraph agent's flow actually works under the hood?",
    kind: "technical",
  },
  {
    title: "MCP and A2A protocols",
    prompt:
      "What are the MCP and A2A protocols, and when would I use each one?",
    kind: "technical",
  },
  {
    title: "What is ReAct?",
    prompt:
      "What is the ReAct pattern, and how is it different from a plain agentic loop?",
    kind: "technical",
  },
  {
    title: "Pick an agent framework",
    prompt:
      "How do I choose between LangGraph, CrewAI, and other agent frameworks for my use case?",
    kind: "technical",
  },
  {
    title: "Context engineering",
    prompt:
      "What is context engineering, and how is it different from prompt engineering?",
    kind: "technical",
  },

  // Setup and debugging (across all courses)
  {
    title: "Module not found",
    prompt:
      "I'm getting a ModuleNotFoundError when I run a course notebook. How do I fix it?",
    kind: "technical",
  },
  {
    title: "Dependency version clash",
    prompt:
      "I have a version conflict between two libraries when installing the course requirements. How do I resolve it?",
    kind: "technical",
  },
  {
    title: "Clone the course repo",
    prompt:
      "I'm having trouble cloning the course GitHub repository. What am I doing wrong?",
    kind: "accessible",
  },
  {
    title: "Store API keys safely",
    prompt:
      "Where should I store my API keys safely on my machine so I don't leak them?",
    kind: "accessible",
  },
  {
    title: "Set up my environment",
    prompt:
      "How do I set up a clean Python environment for the course exercises on my machine?",
    kind: "accessible",
  },

  // Getting started (Beginner Python for AI Engineering)
  {
    title: "New to Python",
    prompt:
      "I'm new to Python. What's the minimum I need to know to start the AI engineering course?",
    kind: "accessible",
  },
  {
    title: "List comprehensions",
    prompt:
      "Can you explain Python list comprehensions in simple terms, with a small example?",
    kind: "accessible",
  },
  {
    title: "Do I need to install anything?",
    prompt:
      "Do I need to install anything on my computer to do the course exercises, or can I use Google Colab?",
    kind: "accessible",
  },
  {
    title: "ChatGPT vs Claude vs Gemini",
    prompt:
      "I'm just starting out. How do ChatGPT, Claude, and Gemini differ, and which should I use?",
    kind: "accessible",
  },
  {
    title: "New to LLMs",
    prompt:
      "I'm new to LLMs. What core concepts should I understand before I start building anything?",
    kind: "accessible",
  },
];

const INITIAL_SUGGESTION_COUNT = 4;

function shuffle<T>(items: ReadonlyArray<T>): T[] {
  const copy = [...items];
  for (let index = copy.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(Math.random() * (index + 1));
    [copy[index], copy[swapIndex]] = [copy[swapIndex], copy[index]];
  }
  return copy;
}

function pickRandomSuggestions(count: number): Suggestion[] {
  if (count < 2) {
    return shuffle(SUGGESTION_POOL).slice(0, count);
  }

  const technical = shuffle(
    SUGGESTION_POOL.filter((item) => item.kind === "technical"),
  );
  const accessible = shuffle(
    SUGGESTION_POOL.filter((item) => item.kind === "accessible"),
  );

  const picked: Suggestion[] = [];
  if (technical[0]) picked.push(technical[0]);
  if (accessible[0]) picked.push(accessible[0]);

  const pickedTitles = new Set(picked.map((item) => item.title));
  const remaining = shuffle(
    SUGGESTION_POOL.filter((item) => !pickedTitles.has(item.title)),
  );
  picked.push(...remaining.slice(0, count - picked.length));

  return shuffle(picked).slice(0, count);
}

function EmptyConversation({
  onSelect,
}: {
  onSelect: (prompt: string) => void;
}) {
  const [suggestions, setSuggestions] = useState<Suggestion[]>(() =>
    SUGGESTION_POOL.slice(0, INITIAL_SUGGESTION_COUNT),
  );

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- intentional client-only enhancement after hydration
    setSuggestions(pickRandomSuggestions(INITIAL_SUGGESTION_COUNT));
  }, []);

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
        courses. Answers are grounded in the sources you select. Try one of
        these to start:
      </p>
      <div className="mt-6 grid w-full max-w-[640px] grid-cols-1 gap-2 sm:grid-cols-2">
        {suggestions.map((suggestion) => (
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
