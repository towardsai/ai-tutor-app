"use client";

import clsx from "clsx";
import {
  BookOpen,
  ChevronDown,
  ExternalLink,
  Globe,
  GraduationCap,
  Library,
  Link as LinkIcon,
  SquarePen,
  Wrench,
} from "lucide-react";
import { useState, type ComponentType, type SVGProps } from "react";
import type { TutorSource, TutorTool } from "@/lib/api";

type SourceSidebarProps = {
  onNewChat: () => void;
  onToggleSource: (sourceKey: string) => void;
  selectedSourceKeys: string[];
  sourceError: string | null;
  tools: TutorTool[];
};

type ToggleToolMeta = {
  icon: ComponentType<SVGProps<SVGSVGElement>>;
};

const TOGGLE_TOOL_META: Record<string, ToggleToolMeta> = {
  web_search: { icon: Globe },
  url_context: { icon: LinkIcon },
  web_fetch: { icon: LinkIcon },
};

export function SourceSidebar({
  onNewChat,
  onToggleSource,
  selectedSourceKeys,
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
    toggleTools.filter((tool) => tool.active).length;
  const totalCount = (retrievalTool ? 1 : 0) + toggleTools.length;

  return (
    <aside className="glass-panel relative overflow-hidden rounded-[1.5rem] p-2.5 lg:flex lg:min-h-0 lg:max-h-[calc(100vh-1rem)] lg:min-h-[calc(100vh-1rem)] lg:flex-col">
      <div className="grain-mask absolute inset-0" />
      <div className="relative flex flex-col gap-2.5 lg:min-h-0 lg:flex-1">
        <div className="flex items-center gap-2.5 px-1 pt-0.5">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src="/towardsai-logo.png"
            alt="Towards AI"
            width={36}
            height={36}
            className="shrink-0 rounded-full shadow-[0_4px_12px_rgba(11,136,238,0.18)]"
          />
          <h1 className="text-[1.35rem] font-semibold leading-none tracking-[-0.03em] text-[var(--ink)]">
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
            Choose which sources the tutor searches.
          </p>
        </div>

        <div className="scrollbar-thin space-y-2 pr-0.5 lg:min-h-0 lg:flex-1 lg:overflow-y-auto">
          {retrievalTool ? (
            <RetrievalTool
              tool={retrievalTool}
              selectedSourceKeys={selectedSourceKeys}
              onToggleSource={onToggleSource}
            />
          ) : null}
          {toggleTools.map((tool) => (
            <ToggleToolRow key={tool.key} tool={tool} />
          ))}
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
          className="group inline-flex items-center justify-between gap-2 border-t border-[var(--line)] px-1 pt-2 text-[10.5px] text-[var(--muted)] transition hover:text-[var(--accent)]"
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
  tool,
  selectedSourceKeys,
  onToggleSource,
}: {
  tool: Extract<TutorTool, { kind: "configurable" }>;
  selectedSourceKeys: string[];
  onToggleSource: (sourceKey: string) => void;
}) {
  const [isOpen, setIsOpen] = useState(true);
  const courseSources = tool.sources.filter((source) => source.group === "courses");
  const docSources = tool.sources.filter((source) => source.group === "docs");

  return (
    <section className="space-y-1">
      <button
        type="button"
        onClick={() => setIsOpen((current) => !current)}
        aria-expanded={isOpen}
        className="flex w-full items-center gap-2 rounded-[0.75rem] px-2 py-1.5 text-left transition hover:bg-[var(--surface-soft)]"
      >
        <Library className="h-3.5 w-3.5 shrink-0 text-[var(--accent)]" />
        <span className="min-w-0 flex-1 truncate text-[12.5px] font-semibold tracking-[-0.01em] text-[var(--ink)]">
          {tool.label}
        </span>
        <ChevronDown
          className={clsx(
            "h-3.5 w-3.5 shrink-0 text-[var(--muted)] transition-transform",
            isOpen && "rotate-180",
          )}
        />
      </button>

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
              label="Open-source docs"
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
  return (
    <div className="space-y-1">
      <div className="flex items-center gap-1.5 px-1">
        <Icon className="h-3 w-3 text-[var(--muted)]/70" />
        <h2 className="text-[10px] font-medium tracking-[0.02em] text-[var(--muted)]/80">
          {label}
        </h2>
      </div>

      <div className="space-y-1">
        {sources.map((source) => {
          const selected = selectedSourceKeys.includes(source.key);
          const isCourse = source.group === "courses";
          return (
            <button
              key={source.key}
              type="button"
              onClick={() => onToggleSource(source.key)}
              className={clsx(
                "flex w-full items-center gap-2 rounded-[0.75rem] border px-2 py-1.5 text-left transition",
                selected
                  ? isCourse
                    ? "border-[var(--accent)] bg-[var(--accent)] text-white shadow-[0_4px_12px_rgba(11,136,238,0.18)]"
                    : "border-[var(--accent)] bg-[var(--accent-soft)] text-[var(--ink)]"
                  : "border-[var(--line)] bg-[var(--surface-soft)] text-[var(--ink)] hover:border-[var(--line-strong)] hover:bg-[var(--surface-hover)]",
              )}
            >
              <span
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
                {formatSourceLabel(source.label)}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function ToggleToolRow({
  tool,
}: {
  tool: Extract<TutorTool, { kind: "toggle" }>;
}) {
  const meta = TOGGLE_TOOL_META[tool.key];
  const Icon = meta?.icon ?? Globe;
  return (
    <div
      className="flex items-center gap-2 rounded-[0.9rem] border border-[var(--line)] bg-[var(--surface-subtle)] px-2 py-2"
      title={tool.active ? "Always on for this model" : "Unavailable"}
    >
      <Icon className="h-3.5 w-3.5 shrink-0 text-[var(--accent)]" />
      <span className="min-w-0 flex-1 truncate text-[12.5px] font-medium tracking-[-0.01em] text-[var(--ink)]">
        {tool.label}
      </span>
      <span
        aria-label={tool.active ? "on" : "off"}
        className={clsx(
          "inline-flex items-center rounded-full px-1.5 py-0.5 text-[9.5px] font-semibold uppercase tracking-[0.1em]",
          tool.active
            ? "bg-[var(--accent-faint)] text-[var(--accent)]"
            : "bg-[var(--muted)]/15 text-[var(--muted)]",
        )}
      >
        {tool.active ? "on" : "off"}
      </span>
    </div>
  );
}

function formatSourceLabel(label: string) {
  return label.replace(/\s+Docs$/, "");
}
