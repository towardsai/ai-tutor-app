"use client";

import clsx from "clsx";
import {
  BookOpen,
  ChevronDown,
  ExternalLink,
  Globe,
  GraduationCap,
  Info,
  Library,
  Link as LinkIcon,
  SquarePen,
  Wrench,
} from "lucide-react";
import {
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type ComponentType,
  type SVGProps,
} from "react";
import { createPortal } from "react-dom";
import type { TutorSource, TutorTool } from "@/lib/api";

type SourceSidebarProps = {
  onNewChat: () => void;
  onToggleSource: (sourceKey: string) => void;
  onToggleTool: (toolKey: string) => void;
  selectedSourceKeys: string[];
  enabledToolKeys: string[];
  sourceError: string | null;
  tools: TutorTool[];
};

type ToggleToolMeta = {
  icon: ComponentType<SVGProps<SVGSVGElement>>;
  description?: string;
};

const POPOVER_GAP = 6;

const TOGGLE_TOOL_META: Record<string, ToggleToolMeta> = {
  web_search: {
    icon: Globe,
    description:
      "Live web search for recent events or facts outside the knowledge base.",
  },
  url_context: {
    icon: LinkIcon,
    description:
      "Reads a specific URL you paste in the chat so the tutor can answer from its content.",
  },
  web_fetch: {
    icon: LinkIcon,
    description:
      "Reads a specific URL you paste in the chat so the tutor can answer from its content.",
  },
};

export function SourceSidebar({
  onNewChat,
  onToggleSource,
  onToggleTool,
  selectedSourceKeys,
  enabledToolKeys,
  sourceError,
  tools,
}: SourceSidebarProps) {
  const retrievalTool = tools.find(
    (tool): tool is Extract<TutorTool, { kind: "configurable" }> =>
      tool.kind === "configurable",
  );
  const toggleTools = tools.filter(
    (tool): tool is Extract<TutorTool, { kind: "toggle" }> =>
      tool.kind === "toggle",
  );
  const activeCount =
    (retrievalTool && selectedSourceKeys.length > 0 ? 1 : 0) +
    toggleTools.filter((tool) => enabledToolKeys.includes(tool.key)).length;
  const totalCount = (retrievalTool ? 1 : 0) + toggleTools.length;
  const [openToolInfoKey, setOpenToolInfoKey] = useState<string | null>(null);
  const [isRetrievalOpen, setIsRetrievalOpen] = useState(true);
  const toggleToolsRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (openToolInfoKey === null) return;
    function onDocMouseDown(event: MouseEvent) {
      if (
        toggleToolsRef.current &&
        !toggleToolsRef.current.contains(event.target as Node)
      ) {
        setOpenToolInfoKey(null);
      }
    }
    function onKey(event: KeyboardEvent) {
      if (event.key === "Escape") setOpenToolInfoKey(null);
    }
    document.addEventListener("mousedown", onDocMouseDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDocMouseDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [openToolInfoKey]);

  return (
    <aside className="glass-panel relative overflow-hidden rounded-[1.5rem] p-2.5 lg:flex lg:min-h-0 lg:max-h-[calc(100vh-1rem)] lg:min-h-[calc(100vh-1rem)] lg:flex-col">
      <div className="grain-mask absolute inset-0" />
      <div className="relative flex flex-col gap-2.5 lg:min-h-0 lg:flex-1">
        <div className="flex items-center gap-2.5 px-1 pt-0.5">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src="/towardsai-logo.png"
            alt="Towards AI"
            width={48}
            height={48}
            className="shrink-0 rounded-full shadow-[0_4px_12px_rgba(11,136,238,0.18)]"
          />
          <h1 className="text-[1.35rem] font-semibold leading-none tracking-[-0.03em] text-[var(--accent)]">
            AI Tutor
          </h1>
        </div>

        <div className="border-t border-[var(--line)] px-1 pt-2">
          <button
            type="button"
            onClick={onNewChat}
            className="group flex w-full items-center gap-2 rounded-[0.9rem] border border-[var(--line-strong)] bg-[var(--accent-faint)] px-2.5 py-2 text-left shadow-[0_4px_12px_rgba(11,136,238,0.08)] transition hover:-translate-y-0.5 hover:border-[var(--accent)] hover:bg-[var(--accent-soft)] hover:shadow-[0_8px_20px_rgba(11,136,238,0.16)]"
          >
            <SquarePen className="h-3.5 w-3.5 shrink-0 text-[var(--accent)]" />
            <span className="min-w-0 flex-1 truncate text-[12.5px] font-semibold tracking-[-0.01em] text-[var(--accent)]">
              New chat
            </span>
          </button>
        </div>

        <div className="space-y-1 border-t border-[var(--line)] px-1 pt-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-1.5">
              <Wrench className="h-3.5 w-3.5 text-[var(--accent)]" />
              <span className="text-[10.5px] font-semibold uppercase tracking-[0.12em] text-[var(--muted)]">
                Tools
              </span>
            </div>
            <span className="rounded-full bg-[var(--accent-faint)] px-2 py-0.5 text-[10.5px] font-semibold tracking-[-0.01em] text-[var(--accent)]">
              {activeCount}/{totalCount}
            </span>
          </div>
          <p className="text-[11px] leading-[1.4] text-[var(--muted)]">
            Toggle tools, pick sources.
          </p>
        </div>

        <div className="scrollbar-thin space-y-2 pr-0.5 lg:min-h-0 lg:flex-1 lg:overflow-y-auto">
          {retrievalTool ? (
            <RetrievalTool
              tool={retrievalTool}
              isOpen={isRetrievalOpen}
              onOpenChange={setIsRetrievalOpen}
              selectedSourceKeys={selectedSourceKeys}
              onToggleSource={onToggleSource}
            />
          ) : null}
          <div ref={toggleToolsRef} className="relative z-10 space-y-2">
            {toggleTools.map((tool) => (
              <ToggleToolRow
                key={tool.key}
                tool={tool}
                enabled={enabledToolKeys.includes(tool.key)}
                onToggle={() => onToggleTool(tool.key)}
                infoOpen={openToolInfoKey === tool.key}
                onInfoToggle={() =>
                  setOpenToolInfoKey((current) =>
                    current === tool.key ? null : tool.key,
                  )
                }
              />
            ))}
          </div>
        </div>

        {sourceError ? (
          <p className="rounded-[0.75rem] border border-amber-300/60 bg-amber-50 px-2.5 py-1.5 text-[12px] text-amber-800">
            {sourceError}
          </p>
        ) : null}

        <a
          href="https://academy.towardsai.net/"
          target="_blank"
          rel="noreferrer"
          className="group mt-auto inline-flex items-center justify-between gap-2 border-t border-[var(--line)] px-1 pt-2 text-[10.5px] text-[var(--muted)] transition hover:text-[var(--accent)]"
        >
          <span className="tracking-[-0.005em]">
            Powered by{" "}
            <span className="font-semibold text-[var(--ink)] transition group-hover:text-[var(--accent)]">
              Towards AI Academy
            </span>
          </span>
          <ExternalLink className="h-3 w-3 shrink-0 opacity-60 transition group-hover:opacity-100" />
        </a>
      </div>
    </aside>
  );
}

function RetrievalTool({
  isOpen,
  onOpenChange,
  tool,
  selectedSourceKeys,
  onToggleSource,
}: {
  isOpen: boolean;
  onOpenChange: (next: boolean) => void;
  tool: Extract<TutorTool, { kind: "configurable" }>;
  selectedSourceKeys: string[];
  onToggleSource: (sourceKey: string) => void;
}) {
  const courseSources = tool.sources.filter((source) => source.group === "courses");
  const docSources = tool.sources.filter((source) => source.group === "docs");
  const enabled = selectedSourceKeys.length > 0;
  const [infoOpen, setInfoOpen] = useState(false);
  const infoRef = useRef<HTMLDivElement>(null);
  const [dialogPos, setDialogPos] = useState<
    { top: number; left: number; width: number; placement: "above" | "below" } | null
  >(null);

  useEffect(() => {
    if (!infoOpen) return;
    function onDocMouseDown(event: MouseEvent) {
      const target = event.target as Node | null;
      const insideTrigger = !!infoRef.current && !!target && infoRef.current.contains(target);
      const insidePortaledDialog =
        !!target &&
        target instanceof Element &&
        !!target.closest('[data-knowledge-base-popover="true"]');
      if (
        !insideTrigger &&
        !insidePortaledDialog
      ) {
        setInfoOpen(false);
      }
    }
    function onKey(event: KeyboardEvent) {
      if (event.key === "Escape") setInfoOpen(false);
    }
    document.addEventListener("mousedown", onDocMouseDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDocMouseDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [infoOpen]);

  useLayoutEffect(() => {
    if (!infoOpen || !infoRef.current) {
      return;
    }

    function recompute() {
      if (!infoRef.current) return;
      const trigger = infoRef.current.getBoundingClientRect();
      const width = 224;
      const estimatedHeight = 150;
      const spaceBelow = window.innerHeight - trigger.bottom;
      const openAbove = spaceBelow < estimatedHeight + 16 && trigger.top > spaceBelow;
      setDialogPos({
        top: openAbove ? trigger.top : trigger.bottom + POPOVER_GAP,
        left: Math.max(12, trigger.right - width),
        width,
        placement: openAbove ? "above" : "below",
      });
    }

    recompute();
    window.addEventListener("resize", recompute);
    window.addEventListener("scroll", recompute, true);
    return () => {
      window.removeEventListener("resize", recompute);
      window.removeEventListener("scroll", recompute, true);
    };
  }, [infoOpen]);

  return (
    <section className="space-y-1">
      <div
        className={clsx(
          "relative flex w-full items-center gap-1 rounded-[0.9rem] border px-1.5 py-2 transition",
          enabled
            ? "border-[var(--line-strong)] bg-[var(--accent-faint)] hover:border-[var(--accent)]"
            : "border-[var(--line)] bg-[var(--surface-subtle)] hover:border-[var(--line-strong)]",
        )}
      >
        <button
          type="button"
          onClick={() => onOpenChange(!isOpen)}
          aria-expanded={isOpen}
          title="Expand to pick sources. On when at least one source is selected."
          className="flex min-w-0 flex-1 items-center gap-1.5 text-left"
        >
          <Library
            className={clsx(
              "h-3.5 w-3.5 shrink-0",
              enabled ? "text-[var(--accent)]" : "text-[var(--muted)]",
            )}
          />
          <span className="min-w-0 flex-1 truncate text-[12.5px] font-medium tracking-[-0.01em] text-[var(--ink)]">
            {tool.label}
          </span>
          <span
            aria-hidden
            className={clsx(
              "inline-flex items-center rounded-full px-1.25 py-0.5 text-[9.5px] font-semibold uppercase tracking-[0.1em]",
              enabled
                ? "bg-[var(--accent-faint)] text-[var(--accent)]"
                : "bg-[var(--muted)]/15 text-[var(--muted)]",
            )}
          >
            {enabled ? "on" : "off"}
          </span>
          <ChevronDown
            className={clsx(
              "h-3.25 w-3.25 shrink-0 text-[var(--muted)] transition-transform",
              isOpen && "rotate-180",
            )}
          />
        </button>
        <div ref={infoRef} className="relative shrink-0">
          <button
            type="button"
            onClick={() => setInfoOpen((current) => !current)}
            aria-label={`About ${tool.label}`}
            aria-expanded={infoOpen}
            className={clsx(
              "flex h-5 w-5 items-center justify-center rounded-full transition",
              enabled
                ? "text-[var(--accent)] hover:bg-[var(--accent-soft)]"
                : "text-[var(--muted)] hover:bg-[var(--surface-hover)] hover:text-[var(--ink)]",
            )}
          >
            <Info className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>
      {infoOpen && dialogPos && typeof document !== "undefined"
        ? createPortal(
            <div
              role="dialog"
              data-knowledge-base-popover="true"
              style={{
                position: "fixed",
                top: dialogPos.top,
                left: dialogPos.left,
                width: dialogPos.width,
                transform:
                  dialogPos.placement === "above"
                    ? `translateY(calc(-100% - ${POPOVER_GAP}px))`
                    : undefined,
                zIndex: 50,
              }}
              className="rounded-[0.9rem] border border-[var(--line-strong)] bg-[var(--surface-strong)] p-3 shadow-[0_12px_32px_rgba(0,0,0,0.18)] backdrop-blur-md"
            >
              <p className="text-[12px] leading-[1.45] text-[var(--ink)]">
                Grounds answers in the sources you select using hybrid retrieval:
                semantic and keyword (BM25) search, then reranking. The tutor can
                also browse the knowledge base like a filesystem to read full
                documents before answering.
              </p>
            </div>,
            document.body,
          )
        : null}

      {isOpen ? (
        <div className="ml-[11px] space-y-2.5 border-l border-[var(--line-strong)] pl-3">
          {courseSources.length > 0 ? (
            <SourceGroup
              label="Courses"
              icon={GraduationCap}
              sources={courseSources}
              selectedSourceKeys={selectedSourceKeys}
              onToggleSource={onToggleSource}
            />
          ) : null}
          {docSources.length > 0 ? (
            <SourceGroup
              label="Docs & references"
              icon={BookOpen}
              sources={docSources}
              selectedSourceKeys={selectedSourceKeys}
              onToggleSource={onToggleSource}
            />
          ) : null}
        </div>
      ) : null}
    </section>
  );
}

function SourceGroup({
  label,
  icon: Icon,
  sources,
  selectedSourceKeys,
  onToggleSource,
}: {
  label: string;
  icon: ComponentType<SVGProps<SVGSVGElement>>;
  sources: TutorSource[];
  selectedSourceKeys: string[];
  onToggleSource: (sourceKey: string) => void;
}) {
  const [openPopoverKey, setOpenPopoverKey] = useState<string | null>(null);
  const groupRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (openPopoverKey === null) return;
    function onDocMouseDown(event: MouseEvent) {
      const target = event.target as Node | null;
      const insideGroup = !!groupRef.current && !!target && groupRef.current.contains(target);
      const insidePortaledDialog =
        !!target &&
        target instanceof Element &&
        !!target.closest('[data-source-popover="true"]');
      if (!insideGroup && !insidePortaledDialog) {
        setOpenPopoverKey(null);
      }
    }
    function onKey(event: KeyboardEvent) {
      if (event.key === "Escape") setOpenPopoverKey(null);
    }
    document.addEventListener("mousedown", onDocMouseDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDocMouseDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [openPopoverKey]);

  return (
    <div className="space-y-1">
      <div className="flex items-center gap-1.5 px-1">
        <Icon className="h-3 w-3 text-[var(--muted)]/70" />
        <h2 className="text-[10px] font-medium tracking-[0.02em] text-[var(--muted)]/80">
          {label}
        </h2>
      </div>

      <div ref={groupRef} className="space-y-1">
        {sources.map((source) => (
          <SourceRow
            key={source.key}
            source={source}
            selected={selectedSourceKeys.includes(source.key)}
            onToggle={onToggleSource}
            popoverOpen={openPopoverKey === source.key}
            onPopoverToggle={() =>
              setOpenPopoverKey((current) =>
                current === source.key ? null : source.key,
              )
            }
          />
        ))}
      </div>
    </div>
  );
}

type PopoverInfo = {
  description: string;
  linkUrl: string;
  linkLabel: string;
  meta?: string;
};

function formatIndexedMonth(isoDate: string | null | undefined): string | null {
  if (!isoDate) return null;
  const date = new Date(isoDate);
  if (Number.isNaN(date.getTime())) return null;
  return date.toLocaleString("en-US", { month: "short", year: "numeric" });
}

function getPopoverInfo(source: TutorSource): PopoverInfo | undefined {
  // Description and link come from the registry via /api/tools; the frontend
  // renders them verbatim so adding a source needs no UI edit.
  if (!source.description || !source.infoUrl) return undefined;
  if (source.group === "courses") {
    return {
      description: source.description,
      linkUrl: source.infoUrl,
      linkLabel: "View on Academy",
    };
  }
  const indexedMonth = formatIndexedMonth(source.indexedAt);
  const metaParts = [source.version, indexedMonth ? `indexed ${indexedMonth}` : null]
    .filter((part): part is string => Boolean(part));
  return {
    description: source.description,
    linkUrl: source.infoUrl,
    linkLabel: "View docs",
    meta: metaParts.length > 0 ? metaParts.join(" · ") : undefined,
  };
}

function SourceRow({
  source,
  selected,
  onToggle,
  popoverOpen,
  onPopoverToggle,
}: {
  source: TutorSource;
  selected: boolean;
  onToggle: (sourceKey: string) => void;
  popoverOpen: boolean;
  onPopoverToggle: () => void;
}) {
  const isCourse = source.group === "courses";
  const popover = getPopoverInfo(source);
  const rowRef = useRef<HTMLDivElement>(null);
  const [dialogPos, setDialogPos] = useState<
    { top: number; left: number; width: number; placement: "above" | "below" } | null
  >(null);

  useLayoutEffect(() => {
    if (!popoverOpen || !rowRef.current) {
      return;
    }
    function recompute() {
      if (!rowRef.current) return;
      const row = rowRef.current.getBoundingClientRect();
      const estimatedHeight = 140;
      const spaceBelow = window.innerHeight - row.bottom;
      const openAbove = spaceBelow < estimatedHeight + 16 && row.top > spaceBelow;
      setDialogPos({
        top: openAbove ? row.top : row.bottom + POPOVER_GAP,
        left: row.left,
        width: row.width,
        placement: openAbove ? "above" : "below",
      });
    }
    recompute();
    window.addEventListener("resize", recompute);
    return () => window.removeEventListener("resize", recompute);
  }, [popoverOpen]);

  return (
    <div ref={rowRef} className="relative">
      <div
        className={clsx(
          "flex items-center gap-1 rounded-[0.75rem] border py-1.5 pl-2 pr-1 transition",
          selected
            ? isCourse
              ? "border-[var(--accent)] bg-[var(--accent)] text-white shadow-[0_4px_12px_rgba(11,136,238,0.18)]"
              : "border-[var(--accent)] bg-[var(--accent-soft)] text-[var(--ink)]"
            : "border-[var(--line)] bg-[var(--surface-soft)] text-[var(--ink)] hover:border-[var(--line-strong)] hover:bg-[var(--surface-hover)]",
        )}
      >
        <button
          type="button"
          onClick={() => onToggle(source.key)}
          aria-pressed={selected}
          className="flex min-w-0 flex-1 items-center gap-2 text-left"
        >
          <span
            aria-hidden="true"
            className={clsx(
              "flex h-4 w-4 shrink-0 items-center justify-center rounded-full border text-[9px] font-bold",
              selected
                ? isCourse
                  ? "border-white/70 bg-white text-[var(--accent)]"
                  : "border-[var(--accent)] bg-[var(--accent)] text-white"
                : "border-[var(--line-strong)] text-transparent",
            )}
          >
            ✓
          </span>
          <span className="min-w-0 truncate text-[12.5px] font-medium tracking-[-0.01em]">
            {source.shortLabel || source.label}
          </span>
        </button>
        {popover ? (
          <button
            type="button"
            onClick={onPopoverToggle}
            aria-label={`About ${source.label}`}
            aria-expanded={popoverOpen}
            className={clsx(
              "flex h-5 w-5 shrink-0 items-center justify-center rounded-full transition",
              selected
                ? isCourse
                  ? "text-white/80 hover:bg-white/15 hover:text-white"
                  : "text-[var(--accent)] hover:bg-[var(--accent-soft)]"
                : "text-[var(--muted)] hover:bg-[var(--surface-hover)] hover:text-[var(--ink)]",
            )}
          >
            <Info className="h-3.5 w-3.5" />
          </button>
        ) : (
          <span className="h-5 w-5 shrink-0" aria-hidden />
        )}
      </div>
      {popover && popoverOpen && dialogPos && typeof document !== "undefined"
        ? createPortal(
            <div
              role="dialog"
              data-source-popover="true"
              style={{
                position: "fixed",
                top: dialogPos.top,
                left: dialogPos.left,
                width: dialogPos.width,
                transform:
                  dialogPos.placement === "above"
                    ? `translateY(calc(-100% - ${POPOVER_GAP}px))`
                    : undefined,
                zIndex: 50,
              }}
              className="rounded-[0.9rem] border border-[var(--line-strong)] bg-[var(--surface-strong)] p-3 shadow-[0_12px_32px_rgba(0,0,0,0.18)] backdrop-blur-md"
            >
              <p className="text-[12px] leading-[1.45] text-[var(--ink)]">
                {popover.description}
              </p>
              {popover.meta ? (
                <p className="mt-1.5 text-[11px] font-medium tracking-[-0.005em] text-[var(--muted)]">
                  {popover.meta}
                </p>
              ) : null}
              <a
                href={popover.linkUrl}
                target="_blank"
                rel="noreferrer"
                className="mt-2 inline-flex items-center gap-1 text-[11.5px] font-semibold text-[var(--accent)] hover:underline"
              >
                {popover.linkLabel}
                <ExternalLink className="h-3 w-3" />
              </a>
            </div>,
            document.body,
          )
        : null}
    </div>
  );
}

function ToggleToolRow({
  tool,
  enabled,
  onToggle,
  infoOpen,
  onInfoToggle,
}: {
  tool: Extract<TutorTool, { kind: "toggle" }>;
  enabled: boolean;
  onToggle: () => void;
  infoOpen: boolean;
  onInfoToggle: () => void;
}) {
  const meta = TOGGLE_TOOL_META[tool.key];
  const Icon = meta?.icon ?? Globe;
  const description = meta?.description;
  const rowRef = useRef<HTMLDivElement>(null);
  const [dialogPos, setDialogPos] = useState<
    { top: number; left: number; width: number; placement: "above" | "below" } | null
  >(null);

  useLayoutEffect(() => {
    if (!infoOpen || !rowRef.current) {
      return;
    }

    function recompute() {
      if (!rowRef.current) return;
      const row = rowRef.current.getBoundingClientRect();
      const estimatedHeight = 92;
      const spaceAbove = row.top;
      const openAbove = spaceAbove > estimatedHeight + 12;
      setDialogPos({
        top: openAbove ? row.top : row.bottom + POPOVER_GAP,
        left: row.left,
        width: row.width,
        placement: openAbove ? "above" : "below",
      });
    }

    recompute();
    window.addEventListener("resize", recompute);
    window.addEventListener("scroll", recompute, true);
    return () => {
      window.removeEventListener("resize", recompute);
      window.removeEventListener("scroll", recompute, true);
    };
  }, [infoOpen]);

  return (
    <div ref={rowRef} className="relative">
      <div
        className={clsx(
          "flex items-center gap-1 rounded-[0.9rem] border pl-2 pr-1 py-2 transition",
          enabled
            ? "border-[var(--line-strong)] bg-[var(--accent-faint)]"
            : "border-[var(--line)] bg-[var(--surface-subtle)]",
        )}
      >
        <button
          type="button"
          onClick={onToggle}
          aria-pressed={enabled}
          title={enabled ? "Click to disable" : "Click to enable"}
          className="flex min-w-0 flex-1 items-center gap-2 text-left"
        >
          <Icon
            className={clsx(
              "h-3.5 w-3.5 shrink-0",
              enabled ? "text-[var(--accent)]" : "text-[var(--muted)]",
            )}
          />
          <span className="min-w-0 flex-1 truncate text-[12.5px] font-medium tracking-[-0.01em] text-[var(--ink)]">
            {tool.label}
          </span>
          <span
            aria-hidden
            className={clsx(
              "inline-flex items-center rounded-full px-1.5 py-0.5 text-[9.5px] font-semibold uppercase tracking-[0.1em]",
              enabled
                ? "bg-[var(--accent-faint)] text-[var(--accent)]"
                : "bg-[var(--muted)]/15 text-[var(--muted)]",
            )}
          >
            {enabled ? "on" : "off"}
          </span>
        </button>
        {description ? (
          <button
            type="button"
            onClick={onInfoToggle}
            aria-label={`About ${tool.label}`}
            aria-expanded={infoOpen}
            className={clsx(
              "flex h-5 w-5 shrink-0 items-center justify-center rounded-full transition",
              enabled
                ? "text-[var(--accent)] hover:bg-[var(--accent-soft)]"
                : "text-[var(--muted)] hover:bg-[var(--surface-hover)] hover:text-[var(--ink)]",
            )}
          >
            <Info className="h-3.5 w-3.5" />
          </button>
        ) : (
          <span className="h-5 w-5 shrink-0" aria-hidden />
        )}
      </div>
      {description && infoOpen && dialogPos && typeof document !== "undefined"
        ? createPortal(
            <div
              role="dialog"
              data-toggle-tool-popover="true"
              style={{
                position: "fixed",
                top: dialogPos.top,
                left: dialogPos.left,
                width: dialogPos.width,
                transform:
                  dialogPos.placement === "above"
                    ? `translateY(calc(-100% - ${POPOVER_GAP}px))`
                    : undefined,
                zIndex: 80,
              }}
              className="rounded-[0.9rem] border border-[var(--line-strong)] bg-[var(--surface-strong)] p-3 shadow-[0_12px_32px_rgba(0,0,0,0.18)] backdrop-blur-md"
            >
              <p className="text-[12px] leading-[1.45] text-[var(--ink)]">
                {description}
              </p>
            </div>,
            document.body,
          )
        : null}
    </div>
  );
}
