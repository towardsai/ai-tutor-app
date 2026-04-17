"use client";

import clsx from "clsx";
import {
  ChevronLeft,
  ChevronRight,
  ExternalLink,
  LibraryBig,
  SearchCode,
} from "lucide-react";
import type { SourcePartData } from "@/lib/api";
import type { TutorMessage } from "@/lib/chat-ui";
import { getMessageSources } from "@/lib/chat-ui";

type SourcePanelProps = {
  isCollapsed: boolean;
  message: TutorMessage | undefined;
  onToggleCollapse: () => void;
};

export function SourcePanel({
  isCollapsed,
  message,
  onToggleCollapse,
}: SourcePanelProps) {
  const sources = message ? getMessageSources(message) : [];

  return (
    <div className="relative min-h-0">
      <button
        type="button"
        onClick={onToggleCollapse}
        aria-label={isCollapsed ? "Expand sources panel" : "Collapse sources panel"}
        aria-pressed={isCollapsed}
        className="absolute left-0 top-5 z-20 hidden h-10 w-10 -translate-x-1/2 items-center justify-center rounded-full border border-[var(--line-strong)] bg-[rgba(247,251,254,0.96)] text-[var(--accent)] shadow-[0_14px_34px_rgba(18,42,204,0.12)] backdrop-blur-xl transition hover:border-[var(--accent)] hover:text-[var(--ink)] lg:flex"
      >
        {isCollapsed ? (
          <ChevronLeft className="h-4 w-4" />
        ) : (
          <ChevronRight className="h-4 w-4" />
        )}
      </button>

      <aside
        className={clsx(
          "glass-panel overflow-hidden rounded-[1.5rem] p-3 transition-[padding] duration-300 lg:flex lg:min-h-0 lg:max-h-[calc(100vh-1rem)] lg:min-h-[calc(100vh-1rem)] lg:rounded-r-none lg:flex-col",
          isCollapsed && "lg:items-center lg:px-2 lg:py-3",
        )}
      >
        <div
          className={clsx(
            "hidden h-full flex-col items-center gap-3 lg:flex",
            !isCollapsed && "lg:hidden",
          )}
        >
          <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-[var(--accent-faint)] text-[var(--accent)]">
            <LibraryBig className="h-[18px] w-[18px]" />
          </div>
          {message ? (
            <span className="rounded-full bg-white/70 px-2 py-1 text-[10px] font-semibold tracking-[0.02em] text-[var(--accent)]">
              {sources.length}
            </span>
          ) : null}
          <span className="[writing-mode:vertical-rl] text-[10px] font-semibold uppercase tracking-[0.16em] text-[var(--muted)]">
            Sources
          </span>
        </div>

        <div className={clsx("lg:flex lg:min-h-0 lg:flex-1 lg:flex-col", isCollapsed && "lg:hidden")}>
          <div className="flex items-center gap-2 text-[13px] font-semibold tracking-[-0.012em] text-[var(--ink)]">
            <LibraryBig className="h-4 w-4 text-[var(--accent)]" />
            Sources
          </div>

          {!message ? (
            <EmptyState
              title="No answer selected"
              body="Send a question or click any assistant answer to inspect the sources and relevance scores behind it."
            />
          ) : sources.length === 0 ? (
            <EmptyState
              title="No sources attached yet"
              body="This assistant turn has not emitted source data yet. During streaming, the cards will populate here automatically."
            />
          ) : (
            <div className="mt-3 space-y-2 lg:flex lg:min-h-0 lg:flex-1 lg:flex-col">
              <div className="rounded-[1.1rem] border border-[var(--line)] bg-white/60 p-3">
                <p className="text-[10px] font-semibold uppercase tracking-[0.12em] text-[var(--muted)]">
                  Current answer
                </p>
                <p className="mt-1.5 text-[13px] leading-[1.45] tracking-[-0.01em] text-[var(--ink)]">
                  {sources.length} supporting reference{sources.length === 1 ? "" : "s"}
                </p>
              </div>

              <div className="scrollbar-thin min-w-0 space-y-2 pr-2 lg:min-h-0 lg:flex-1 lg:overflow-y-auto">
                {sources.map((source) => (
                  <SourceCard key={source.docId || source.url} source={source} />
                ))}
              </div>
            </div>
          )}
        </div>
      </aside>
    </div>
  );
}

function EmptyState({ title, body }: { title: string; body: string }) {
  return (
    <div className="mt-3 rounded-[1.1rem] border border-dashed border-[var(--line-strong)] bg-white/45 p-3">
      <div className="flex items-center gap-2 text-[13px] font-semibold tracking-[-0.012em] text-[var(--ink)]">
        <SearchCode className="h-4 w-4 text-[var(--accent)]" />
        {title}
      </div>
      <p className="mt-1.5 text-[13px] leading-[1.55] text-[var(--muted)]">{body}</p>
    </div>
  );
}

function SourceCard({ source }: { source: SourcePartData }) {
  const percentage = Math.max(0, Math.min(100, Math.round(source.score * 100)));

  return (
    <article className="animate-rise-in min-w-0 rounded-[1.1rem] border border-[var(--line)] bg-white/70 p-3 shadow-[0_18px_50px_rgba(18,42,204,0.08)]">
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <p className="text-[10px] font-semibold uppercase tracking-[0.12em] text-[var(--muted)]">
            {source.sourceLabel}
          </p>
          <h3 className="mt-1.5 line-clamp-3 text-[15px] font-semibold leading-[1.35] tracking-[-0.02em] text-[var(--ink)]">
            {source.title}
          </h3>
        </div>
        <span className="rounded-full bg-[var(--accent-faint)] px-2.5 py-1 text-[11px] font-semibold tracking-[-0.01em] text-[var(--accent)]">
          {percentage}%
        </span>
      </div>

      <div className="mt-2 h-2 overflow-hidden rounded-full bg-[var(--accent-faint)]">
        <div
          className="h-full rounded-full bg-[var(--accent)] transition-[width] duration-500"
          style={{ width: `${percentage}%` }}
        />
      </div>

      <a
        href={source.url}
        target="_blank"
        rel="noreferrer"
        className="mt-3 inline-flex items-center gap-2 text-[13px] font-medium tracking-[-0.01em] text-[var(--accent)] transition hover:text-[var(--ink)]"
      >
        Open source
        <ExternalLink className="h-4 w-4" />
      </a>
    </article>
  );
}
