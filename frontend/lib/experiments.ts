// Curated data for the workshop experiment showcase. Hand-authored from the eval
// writeups (evals.md F-series, evals_graphrag.md, evals_compaction.md,
// evals_slm_compaction.md) so each page stays intentional and uncluttered.
// Numbers are coarse screens (small n, 1 trial); read them as rankings.

export type Bar = {
  label: string;
  pct: number; // 0-100, drives the bar width
  value?: string; // primary metric display, e.g. "12/15 (80%)"
  sub?: string; // secondary metric, e.g. "2.9k tok"
  winner?: boolean;
  tone?: "good" | "bad" | "neutral";
};

export type BarsView = {
  kind: "bars";
  key: string;
  label: string;
  caption?: string;
  metricLabel: string;
  bars: Bar[];
};

export type TableView = {
  kind: "table";
  key: string;
  label: string;
  caption?: string;
  columns: string[];
  rows: string[][];
};

export type View = BarsView | TableView;

export type ResultGroup = {
  title: string;
  intro?: string;
  views: View[]; // rendered as tabs when more than one
};

export type Experiment = {
  slug: string;
  order: number;
  shortTitle: string;
  title: string;
  badge: string;
  accent: string; // hex, for per-experiment accenting
  question: string;
  takeaway: string;
  highlights: { label: string; value: string }[];
  groups: ResultGroup[];
  setup: string[];
  caveats: string[];
  links: { label: string; href: string }[];
};

const REPO = "https://github.com/towardsai/ai-tutor-app";

// ---- helpers to keep the data terse ------------------------------------------

const SLM_AXIS_B_TOKENS: Record<string, string> = {
  full_context: "37.8k tok (truncated to 32.8k)",
  rag: "2.9k tok",
  graphrag: "8.2k tok",
  trim: "4.0k tok",
  summary: "0.5k tok",
  hierarchical_summary: "1.0k tok",
  selective: "4.0k tok",
};

function bar(label: string, pct: number, sub?: string): Bar {
  return { label, pct, value: `${pct}%`, sub };
}

function markWinner(bars: Bar[]): Bar[] {
  const top = Math.max(...bars.map((b) => b.pct));
  return bars.map((b) => ({
    ...b,
    winner: b.pct === top,
    tone: b.pct === top ? "good" : b.pct === 0 ? "bad" : "neutral",
  }));
}

function slmAxisBView(key: string, label: string, pairs: [string, number][]): BarsView {
  const bars = markWinner(
    pairs
      .map(([m, pct]) => bar(m, pct, SLM_AXIS_B_TOKENS[m]))
      .sort((a, b) => b.pct - a.pct),
  );
  return {
    kind: "bars",
    key,
    label,
    metricLabel: "Answer quality (judge pass, n=15)",
    bars,
  };
}

function slmAxisAView(key: string, label: string, scores: Record<string, number>): BarsView {
  const bars = markWinner(
    Object.entries(scores)
      .map(([m, pct]) => bar(m, pct))
      .sort((a, b) => b.pct - a.pct),
  );
  return {
    kind: "bars",
    key,
    label,
    metricLabel: "Answer quality (judge pass, n=15)",
    bars,
  };
}

// ---- experiments -------------------------------------------------------------

export const EXPERIMENTS: Experiment[] = [
  {
    slug: "slm-compaction",
    order: 1,
    shortTitle: "Compaction on small local models",
    title: "Knowledge compaction on small local models (SLMs)",
    badge: "Local SLMs · F24 + F25",
    accent: "#0b88ee",
    question:
      "On a cheap local model with a small context window and no caching, what is the best way to survive a long lesson: keep it, compact it, or retrieve it?",
    takeaway:
      "On a 32k local model the lesson does not fit, so 'keep everything' is not an option. For fetching a document, RAG wins on every model. For compacting a growing chat, no single method wins (it depends on the model), and the model's own capability matters more than the compaction strategy.",
    highlights: [
      { label: "Models", value: "llama3.1:8b · qwen2.5:7b · qwen3:8b" },
      { label: "Hardware", value: "M1 Pro, 16 GB, Ollama, $0" },
      { label: "Window", value: "32k (lesson is ~37.7k tokens)" },
      { label: "Judge", value: "Gemini 2.5 Flash vs the full lesson" },
    ],
    groups: [
      {
        title: "Axis B: how to fit a document into context",
        intro:
          "One long lesson, answered 7 ways. RAG (retrieve the right chunk) ties or beats stuffing the whole document, at a fraction of the tokens and latency. Switch models to compare.",
        views: [
          slmAxisBView("qwen3", "qwen3:8b", [
            ["rag", 100],
            ["graphrag", 100],
            ["full_context", 73],
            ["hierarchical_summary", 73],
            ["selective", 73],
            ["trim", 67],
            ["summary", 67],
          ]),
          slmAxisBView("qwen2.5", "qwen2.5:7b", [
            ["rag", 100],
            ["graphrag", 100],
            ["full_context", 67],
            ["trim", 47],
            ["selective", 47],
            ["summary", 33],
            ["hierarchical_summary", 20],
          ]),
          slmAxisBView("llama3.1", "llama3.1:8b", [
            ["graphrag", 87],
            ["full_context", 80],
            ["rag", 80],
            ["trim", 53],
            ["selective", 47],
            ["hierarchical_summary", 27],
            ["summary", 0],
          ]),
        ],
      },
      {
        title: "Axis A: how to compact a growing conversation",
        intro:
          "The lesson is loaded turn 0, then questions accumulate with retrieval off, and each real memory preset manages the growing context. No single method wins across models, and 'keep everything' never tops the table.",
        views: [
          slmAxisAView("qwen3", "qwen3:8b", {
            summarization_only: 73,
            delta_summarization: 73,
            full_history: 67,
            hierarchical_summarization: 67,
            prompt_compression: 60,
            selective_retention: 60,
            incontext_history_retrieval: 60,
            sliding_window: 40,
          }),
          slmAxisAView("qwen2.5", "qwen2.5:7b", {
            prompt_compression: 60,
            selective_retention: 47,
            delta_summarization: 47,
            incontext_history_retrieval: 33,
            hierarchical_summarization: 33,
            full_history: 27,
            summarization_only: 20,
            sliding_window: 13,
          }),
          slmAxisAView("llama3.1", "llama3.1:8b", {
            hierarchical_summarization: 40,
            sliding_window: 13,
            prompt_compression: 13,
            selective_retention: 13,
            incontext_history_retrieval: 13,
            full_history: 7,
            summarization_only: 0,
            delta_summarization: 0,
          }),
        ],
      },
    ],
    setup: [
      "Largest course lesson (~37.7k tokens), one fixed 15-question set reused across every model for comparability.",
      "Each model runs in Ollama with a num_ctx=32768 variant, so the lesson overflows the window and compaction is forced.",
      "Axis B answers each question statelessly from a built context; Axis A runs the real app middlewares over a multi-turn session with retrieval off.",
      "Quality is judged by Gemini 2.5 Flash reading the full lesson as ground truth (a small model cannot hold it, and a model never grades itself).",
    ],
    caveats: [
      "Coarse n: 15 questions per method, 1 trial. Read the bars as rankings, not exact rates.",
      "LLM-judge variance moves individual cells by about 1-3 of 15, so only sizeable gaps are meaningful.",
      "One lesson, one domain. 'Keep everything' numbers reflect Ollama evicting the oversized turn to fit the window.",
    ],
    links: [
      { label: "Writeup: evals_slm_compaction.md", href: `${REPO}/blob/experiment/slm-compaction/evals_slm_compaction.md` },
      { label: "Pull request #3", href: `${REPO}/pull/3` },
    ],
  },
  {
    slug: "graphrag-vs-rag",
    order: 2,
    shortTitle: "GraphRAG vs classical RAG",
    title: "GraphRAG vs classical hybrid RAG",
    badge: "Retrieval · head to head",
    accent: "#7c4dff",
    question:
      "Does a true Microsoft GraphRAG index beat the production hybrid retriever for grounding the tutor's answers?",
    takeaway:
      "No. GraphRAG ties classical RAG on grounding accuracy, ranks the right lesson slightly worse, and costs about 44% more per turn (plus a one-time index build). Both surface and cite the right source 100% of the time.",
    highlights: [
      { label: "Chat model", value: "Gemini 3.5 Flash (both arms)" },
      { label: "Index build", value: "$44.96 one-time (Gemini 2.5 Flash)" },
      { label: "Per-turn cost", value: "$0.147 classical vs $0.212 GraphRAG" },
      { label: "Verdict", value: "Ties on accuracy, costs more" },
    ],
    groups: [
      {
        title: "Same battery, same chat model, only the retriever changes",
        intro:
          "The retriever is the single variable: production hybrid (dense + BM25, fused, reranked) vs a true GraphRAG local-search index. Everything else is held constant.",
        views: [
          {
            kind: "table",
            key: "headtohead",
            label: "Classical vs GraphRAG",
            caption:
              "Grounding accuracy ties; GraphRAG ranks the right lesson a touch worse (MRR) and uses more tokens, so it costs more.",
            columns: ["Metric", "Classical RAG", "GraphRAG"],
            rows: [
              ["recall@shown source", "100%", "100%"],
              ["recall@shown lesson", "76%", "76%"],
              ["right-lesson rank (MRR)", "0.70", "0.65"],
              ["cited-correct source", "100%", "100%"],
              ["cited-correct lesson", "85%", "85%"],
              ["behavior proxy", "89%", "92%"],
              ["input tokens / turn", "110k", "178k"],
              ["est cost / turn", "$0.147", "$0.212"],
            ],
          },
        ],
      },
    ],
    setup: [
      "GraphRAG is a context provider only: it never runs its own answer synthesis, so Gemini 3.5 Flash writes the answer in both arms.",
      "Scoped to one source (full_stack_ai_engineering, 486k tokens) to keep the index build affordable; a full-corpus index projected to roughly $2,000.",
      "Graded with tool-agnostic retrieval and citation metrics (no LLM judge needed).",
    ],
    caveats: [
      "Single source, single-turn battery, 1 trial.",
      "cited-correct saturates near 100% at this n, so any-tool recall and MRR are the discriminating metrics.",
    ],
    links: [
      { label: "Writeup: evals_graphrag.md", href: `${REPO}/blob/experiment/graphrag-vs-rag/evals_graphrag.md` },
      { label: "Pull request #2", href: `${REPO}/pull/2` },
    ],
  },
  {
    slug: "gemini-compaction",
    order: 3,
    shortTitle: "Compaction on Gemini",
    title: "Keep-all vs compaction vs retrieve, on Gemini 2.5 Flash",
    badge: "Gemini 2.5 · cost story",
    accent: "#12accc",
    question:
      "Given a large model with a big context window and prompt caching, how should a long lesson be put into context: keep it all, compact it, or retrieve it?",
    takeaway:
      "When context is cheap and cached, retrieval (RAG) gives the best answer for the fewest tokens and dollars. Keep-everything ties the better compaction methods on quality but is the most expensive, and heavy summarization both costs more and answers worse.",
    highlights: [
      { label: "Model", value: "Gemini 2.5 Flash (standalone fleet)" },
      { label: "Best value", value: "RAG: 60% at $0.020/turn" },
      { label: "Keep-all", value: "53% at $0.134/turn" },
      { label: "Worst value", value: "hierarchical: 33% at $0.605/turn" },
    ],
    groups: [
      {
        title: "12 strategies, one long lesson, 15 questions",
        intro:
          "Bars show answer quality; the sub-label shows cost per turn. RAG is both the most accurate and by far the cheapest, which is the intuitive cost story large-model caching otherwise hides.",
        views: [
          {
            kind: "bars",
            key: "all",
            label: "All strategies",
            metricLabel: "Answer quality (judge pass, n=15) · cost per turn",
            bars: markWinner([
              bar("rag (retrieve)", 60, "$0.020 · 3.2k tok"),
              bar("incontext_history_retrieval", 60, "$0.097 · 41.7k tok"),
              bar("graphrag (retrieve)", 53, "$0.045 · 9.0k tok"),
              bar("summarization_only", 53, "$0.089 · 27.3k tok"),
              bar("prompt_compression", 53, "$0.134 · 40.9k tok"),
              bar("full_history (keep all)", 53, "$0.134 · 42.9k tok"),
              bar("delta_summarization", 47, "$0.107 · 27.0k tok"),
              bar("selective_retention", 40, "$0.105 · 27.1k tok"),
              bar("prod", 40, "$0.103 · 27.4k tok"),
              bar("aggressive", 33, "$0.075 · 11.1k tok"),
              bar("sliding_window", 33, "$0.058 · 18.5k tok"),
              bar("hierarchical_summarization", 33, "$0.605 · 30.7k tok"),
            ]),
          },
        ],
      },
    ],
    setup: [
      "Same largest lesson and 15-question set as the SLM study, on Gemini 2.5 Flash via the OpenAI-compatible endpoint.",
      "Family A keeps/compacts the lesson in context through the real middlewares; Family B retrieves per question (RAG / GraphRAG).",
      "Standalone fleet: compare arms only within this table, not against the 3.5-Flash runs (different model, pricing, caching).",
    ],
    caveats: [
      "15 questions, 1 trial; coarse ranking.",
      "Costs are post-cache estimates and can rank presets differently from raw tokens.",
    ],
    links: [
      { label: "Writeup: evals_compaction.md", href: `${REPO}/blob/experiment/context-compaction/evals_compaction.md` },
      { label: "Pull request #5", href: `${REPO}/pull/5` },
    ],
  },
];

export function getExperiment(slug: string): Experiment | undefined {
  return EXPERIMENTS.find((e) => e.slug === slug);
}
