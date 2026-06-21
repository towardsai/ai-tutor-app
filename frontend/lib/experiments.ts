// Curated data for the workshop showcase, grouped by model and ending in a
// cross-model comparison. Hand-authored from the eval writeups (evals.md
// F-series incl. DeepSeek F25/F26, evals_graphrag.md, evals_compaction.md,
// evals_slm_compaction.md). Numbers are coarse screens (small n, 1 trial);
// read them as rankings. Different models used different batteries, so quality
// is only compared within a page, never across models (called out per page).

export type Bar = {
  label: string;
  pct: number; // 0-100, drives the bar width
  value?: string;
  sub?: string;
  winner?: boolean;
  tone?: "good" | "bad" | "neutral";
};

export type Finding = { id?: string; title: string; stat?: string; text: string };

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
export type FindingsView = {
  kind: "findings";
  key: string;
  label: string;
  caption?: string;
  findings: Finding[];
};
export type MetricSeries = { key: string; label: string; bars: Bar[] };
export type MetricView = {
  kind: "metric";
  key: string;
  label: string;
  caption?: string;
  series: MetricSeries[];
};
export type View = BarsView | TableView | FindingsView | MetricView;

// Magnitude visual for the home page: how big the things we juggle actually are.
export type ScaleItem = { label: string; value: number; display: string; note: string };
export type ScaleStrip = { title: string; unit: string; items: ScaleItem[] };

export type ResultGroup = { title: string; intro?: string; views: View[] };

export type Experiment = {
  slug: string;
  order: number;
  shortTitle: string;
  title: string;
  badge: string;
  accent: string;
  question: string;
  takeaway: string;
  highlights: { label: string; value: string }[];
  groups: ResultGroup[];
  setup: string[];
  caveats: string[];
  links: { label: string; href: string }[];
};

const REPO = "https://github.com/towardsai/ai-tutor-app";

// ---- helpers -----------------------------------------------------------------

function markHigh(bars: Bar[]): Bar[] {
  const top = Math.max(...bars.map((b) => b.pct));
  return bars
    .map((b) => ({ ...b, winner: b.pct === top, tone: (b.pct === top ? "good" : b.pct === 0 ? "bad" : "neutral") as Bar["tone"] }))
    .sort((a, b) => b.pct - a.pct);
}

// cost bars: lower is better. pct is scaled to the max for visible widths.
function costBars(rows: [string, number, string?][]): Bar[] {
  const max = Math.max(...rows.map((r) => r[1]));
  const min = Math.min(...rows.map((r) => r[1]));
  return rows
    .map(([label, dollars, sub]) => ({
      label,
      pct: Math.round((dollars / max) * 100),
      value: `$${dollars.toFixed(4)}`,
      sub,
      winner: dollars === min,
      tone: (dollars === min ? "good" : dollars === max ? "bad" : "neutral") as Bar["tone"],
    }))
    .sort((a, b) => a.pct - b.pct);
}

function slmA(key: string, label: string, scores: Record<string, number>): BarsView {
  return {
    kind: "bars",
    key,
    label,
    metricLabel: "Answer quality (judge pass, n=15)",
    bars: markHigh(Object.entries(scores).map(([m, p]) => ({ label: m, pct: p, value: `${p}%` }))),
  };
}

function fmtTok(t: number): string {
  return t >= 1000 ? `${(t / 1000).toFixed(1)}k tok` : `${t} tok`;
}
// lower-is-better bars (tokens, latency): scaled to the max for visible widths.
function lowerBars(rows: [string, number, string][]): Bar[] {
  const max = Math.max(...rows.map((r) => r[1]));
  const min = Math.min(...rows.map((r) => r[1]));
  return rows
    .map(([label, v, display]) => ({
      label,
      pct: Math.max(2, Math.round((v / max) * 100)),
      value: display,
      winner: v === min,
      tone: (v === min ? "good" : v === max ? "bad" : "neutral") as Bar["tone"],
    }))
    .sort((a, b) => a.pct - b.pct);
}
// One model's Axis-B result with a Quality / Tokens / Latency metric toggle.
// rows: [method, quality%, input tokens, latency p50 s] from compare.md.
function slmBMetric(key: string, label: string, rows: [string, number, number, number][]): MetricView {
  return {
    kind: "metric",
    key,
    label,
    series: [
      {
        key: "quality",
        label: "Quality",
        bars: markHigh(rows.map(([m, q]) => ({ label: m, pct: q, value: `${q}%` }))),
      },
      {
        key: "tokens",
        label: "Input tokens",
        bars: lowerBars(rows.map(([m, , t]) => [m, t, fmtTok(t)])),
      },
      {
        key: "latency",
        label: "Latency",
        bars: lowerBars(rows.map(([m, , , l]) => [m, l, `${l}s`])),
      },
    ],
  };
}

// ---- pages -------------------------------------------------------------------

export const EXPERIMENTS: Experiment[] = [
  {
    slug: "gemini",
    order: 1,
    shortTitle: "Gemini 3.5 Flash: the full eval program",
    title: "Gemini 3.5 Flash: the full eval program",
    badge: "Cloud · Gemini 3.5 · F1-F23",
    accent: "#0b88ee",
    question:
      "On a large model with a big window and prompt caching, which context strategy gives the best answers for the least cost, tokens, and latency?",
    takeaway:
      "The naive baseline wins. Up to about 13 turns, keeping the full history is cheapest, fastest, AND has the best memory. Compaction saves tokens but often costs more dollars because it breaks the cache, and it drops the oldest facts. Retrieval payloads (not chat history) dominate the token bill, a stored profile wins personalization, and GraphRAG does not beat classical RAG.",
    highlights: [
      { label: "Model", value: "gemini-3.5-flash (every run)" },
      { label: "Scale", value: "1,000+ turns, 0 API errors" },
      { label: "Batteries", value: "single-turn, sessions, personas" },
      { label: "Grading", value: "human-confirmed, judge validated 98%" },
    ],
    groups: [
      {
        title: "Headline findings",
        intro:
          "The same study that motivated the workshop. Each finding is measured, not a vibe-check.",
        views: [
          {
            kind: "findings",
            key: "findings",
            label: "Findings",
            findings: [
              {
                id: "F9",
                title: "Full history is cheapest and fastest up to 13 turns",
                stat: "$0.034/turn",
                text: "Keep-all: $0.034/turn, 17s to first text, 2.8 tool calls, vs $0.051-0.066 and 21-43s for compaction. Raw history lets the agent reuse earlier evidence; summaries force re-retrieval.",
              },
              {
                id: "F10",
                title: "Compaction degrades memory, and it drops old facts",
                stat: "92% vs 38%",
                text: "Session-probe accuracy: full_history 92% vs prod and profile_memory 38%, aggressive 42%. The collapse is entirely turn-0 material (fact recall 100% to 17-25%); recent facts survive.",
              },
              {
                id: "F2",
                title: "Compaction saves tokens but not dollars",
                stat: "44% fewer tokens, 68% more cost",
                text: "Summarization rewrites the prompt prefix and invalidates Gemini's implicit cache. One arm used 44% fewer tokens yet cost 68% more than keep-all.",
              },
              {
                id: "F1",
                title: "Retrieval payloads dominate input tokens",
                stat: "~200k tok/turn",
                text: "Each retrieval can return up to 100k tokens; turns average ~200k input. The tokens are in tool outputs, not the conversation history, which flips where compaction should focus.",
              },
              {
                id: "F8",
                title: "A stored profile wins personalization and cost",
                stat: "94% vs 56-67%",
                text: "profile_memory: 94% personalization vs 56-67% without, at the cheapest persona cost, because the stored profile saves the agent from re-searching for user context.",
              },
              {
                id: "F4 / F12",
                title: "Aggressive compaction is dominated on every axis",
                stat: "worst everywhere",
                text: "18 vs 9.6 LLM calls/turn, 57s vs 39s, key-point coverage 36% vs 72%, session probes 42% vs 92%. No metric favors it.",
              },
            ],
          },
        ],
      },
      {
        title: "Memory under compaction",
        intro:
          "Session-probe accuracy by memory preset: can the tutor still recall facts planted earlier in the conversation?",
        views: [
          {
            kind: "bars",
            key: "probes",
            label: "Session-probe accuracy",
            metricLabel: "Memory probe accuracy (n=24 per preset, human-confirmed)",
            bars: markHigh([
              { label: "full_history", pct: 92, value: "92%" },
              { label: "aggressive", pct: 42, value: "42%" },
              { label: "prod", pct: 38, value: "38%" },
              { label: "profile_memory", pct: 38, value: "38%" },
            ]),
            caption:
              "Keep-all retains old facts; every compaction arm loses the turn-0 material the probes test.",
          },
        ],
      },
      {
        title: "Retrieval: GraphRAG vs classical RAG",
        intro:
          "Does a true Microsoft GraphRAG index beat the production hybrid retriever? Only the retriever changes; Gemini 3.5 writes the answer in both arms.",
        views: [
          {
            kind: "table",
            key: "graphrag",
            label: "Classical vs GraphRAG",
            caption:
              "Grounding accuracy ties; GraphRAG ranks the right lesson slightly worse and uses more tokens, so it costs about 44% more. Index build was a one-time $44.96.",
            columns: ["Metric", "Classical RAG", "GraphRAG"],
            rows: [
              ["recall@shown source", "100%", "100%"],
              ["right-lesson rank (MRR)", "0.70", "0.65"],
              ["cited-correct lesson", "85%", "85%"],
              ["input tokens / turn", "110k", "178k"],
              ["est cost / turn", "$0.147", "$0.212"],
            ],
          },
        ],
      },
    ],
    setup: [
      "Production tutor (same agent, tools, prompts), single variable is the context strategy: 13 arms across memory (Axis A) and retrieval (Axis B).",
      "Batteries: 60 single-turn questions, multi-turn sessions with memory probes, and 10 personas. Part B ran 4 arms x2 trials; Part C screened 11 arms.",
      "Memory probes were human-confirmed; the LLM judge was then validated at 98% agreement before grading the rest.",
    ],
    caveats: [
      "Coarse screen: most arms 1-2 trials; nothing at promotion grade yet.",
      "Sessions tested were short to medium (<=13 turns); the long-horizon question moves to the DeepSeek page.",
    ],
    links: [
      { label: "Writeup: evals.md", href: `${REPO}/blob/main/evals.md` },
      { label: "GraphRAG writeup", href: `${REPO}/blob/experiment/graphrag-vs-rag/evals_graphrag.md` },
      { label: "GraphRAG PR #2", href: `${REPO}/pull/2` },
    ],
  },

  {
    slug: "deepseek",
    order: 2,
    shortTitle: "DeepSeek V4-Flash: cost and long horizon",
    title: "DeepSeek V4-Flash: cost and long-horizon memory",
    badge: "Cloud · DeepSeek · F25 + F26",
    accent: "#2bb673",
    question:
      "Does a cheaper model with stronger caching change the compaction story, even when sessions get long enough that keep-all should finally lose?",
    takeaway:
      "No, it sharpens it. DeepSeek's roughly 50x cache discount makes keeping everything the cheapest arm even at 36 turns, undercutting every compaction method (which break the cache). Keep-all also resolves contradictions perfectly where summarization fails. Per turn it runs about 10-15x cheaper than Gemini.",
    highlights: [
      { label: "Model", value: "DeepSeek-V4-Flash (first-party API)" },
      { label: "Scale", value: "12 arms, 840 turns, 0 errors" },
      { label: "Cache", value: "~97% cache-hit, ~50x discount" },
      { label: "Tiers", value: "contradiction (22t) + long-horizon (36t)" },
    ],
    groups: [
      {
        title: "Cost per turn at long horizon (36-turn sessions)",
        intro:
          "Lower is better. Keeping everything bills the most tokens (1.78M input/turn) but is the cheapest arm, because about 97% of those tokens are cache hits and compaction rewrites break the cache.",
        views: [
          {
            kind: "bars",
            key: "cost",
            label: "Cost per turn",
            metricLabel: "Estimated $/turn at 36 turns (lower is better)",
            bars: costBars([
              ["full_history", 0.0101, "keep all, ~97% cache-hit"],
              ["incontext_history_retrieval", 0.0147, "retrieve old turns"],
              ["prod", 0.0336, "summarization + clearing"],
              ["selective_retention", 0.0357, "constraint-preserving summary"],
            ]),
            caption:
              "DeepSeek's ~50x cache discount makes the large cached prefix cheaper than any compaction's cache-breaking rewrite. The crossover where compaction wins never appears in any length tested.",
          },
        ],
      },
      {
        title: "Memory: contradictions and long-horizon recall",
        intro:
          "Plant a fact, update it, then probe after compaction has evicted the original. This is the one regime where keep-all could lose (it carries both the old and new fact and must pick the current one).",
        views: [
          {
            kind: "table",
            key: "memory",
            label: "Memory probes",
            caption:
              "Keep-all is perfect on both tiers. Summarization (prod) fails contradictions outright. Retrieving old turns matches keep-all but costs more.",
            columns: ["Arm", "Contradiction (3)", "Long-horizon recall (2)"],
            rows: [
              ["full_history", "3/3", "2/2"],
              ["incontext_history_retrieval", "3/3", "2/2"],
              ["profile_memory", "2/3", "-"],
              ["prompt_compression", "-", "2/2"],
              ["prod (summarization)", "0/3", "1/2"],
              ["context_reset", "-", "0/2"],
            ],
          },
        ],
      },
    ],
    setup: [
      "A full Part-C-style v2 matrix re-run on DeepSeek-V4-Flash via the first-party API (verified live prefix caching, non-zero cache reads).",
      "Tier 1: contradiction (3 x 22-turn sessions). Tier 2: long-horizon (2 x 36-turn). Judge-graded via the validated subagent path.",
      "Compaction fired at 100% of probes for every compaction arm and 0% for full_history (the gate that makes the comparison valid).",
    ],
    caveats: [
      "Thin n (3 and 2 per arm, 1 trial); coarse rankings.",
      "Provider-specific: first-party single-endpoint caching. A teammate's OpenRouter run fragmented the cache and changed the cost ranking, so the caching path must be held constant.",
    ],
    links: [
      { label: "Findings F25 / F26 in evals.md", href: `${REPO}/blob/main/evals.md` },
    ],
  },

  {
    slug: "deepseek-vs-gemini",
    order: 3,
    shortTitle: "DeepSeek vs Gemini",
    title: "DeepSeek vs Gemini: the caching and cost story",
    badge: "Cloud · provider comparison",
    accent: "#f0913e",
    question:
      "Same context strategies, two providers: how much does the model and its caching change the conclusion?",
    takeaway:
      "The ranking is the same on both (keep-all wins cost and memory), but the economics differ. DeepSeek's 50x cache discount beats Gemini's roughly 10x, so keep-all's cost lead is larger and holds further out, and per-turn cost is about 10-15x lower. The lesson is provider-independent; it just gets cheaper the stronger the cache.",
    highlights: [
      { label: "Per-turn cost", value: "DeepSeek ~10-15x cheaper" },
      { label: "Cache discount", value: "DeepSeek ~50x vs Gemini ~10x" },
      { label: "Both agree", value: "keep-all wins cost + memory" },
    ],
    groups: [
      {
        title: "Side by side",
        intro:
          "The strategies behave the same way; the cache strength sets how strong keep-all's advantage is and how far it holds.",
        views: [
          {
            kind: "table",
            key: "vs",
            label: "Gemini vs DeepSeek",
            columns: ["", "Gemini Flash", "DeepSeek V4-Flash"],
            rows: [
              ["cache discount", "~10x", "~50x"],
              ["cheapest arm", "full_history", "full_history"],
              ["keep-all cost lead holds to", "crossover may appear sooner", "past 36 turns"],
              ["contradiction (keep-all vs summarize)", "full_history wins", "3/3 vs 0/3"],
              ["per-turn cost (comparable tier)", "baseline", "~10-15x cheaper"],
            ],
            caption:
              "A stronger cache pushes the point where compaction could win further out, so on DeepSeek it never appears in any length tested.",
          },
        ],
      },
      {
        title: "Why keep-all wins on both",
        views: [
          {
            kind: "findings",
            key: "why",
            label: "Why",
            findings: [
              {
                title: "The cache is the whole game",
                text: "Both providers bill a huge cached prefix far below the cache-miss rate. Compaction's value (fewer tokens) is outweighed by the cache it breaks when it rewrites the prefix. The bigger the discount, the more this favors keep-all.",
              },
              {
                title: "Hold the caching path constant",
                text: "The same arm can flip cost ranking on OpenRouter (fallback routing fragments the prefix cache) vs a first-party single endpoint. Provider comparisons are only valid when the caching path is held constant.",
              },
            ],
          },
        ],
      },
    ],
    setup: [
      "Gemini findings come from the Part B/C battery; DeepSeek from the v2 contradiction + long-horizon matrix. Different tiers, so this compares the mechanism and cost ratio, not a single shared score.",
    ],
    caveats: [
      "The Gemini v2 long-horizon tier was only partially run, so the long-horizon comparison leans on the DeepSeek matrix plus Gemini's shorter-session results.",
    ],
    links: [{ label: "Findings in evals.md", href: `${REPO}/blob/main/evals.md` }],
  },

  {
    slug: "slm",
    order: 4,
    shortTitle: "Small local models (SLMs)",
    title: "Small local models: forced to compact",
    badge: "Local · Ollama · F24 + F25",
    accent: "#7c4dff",
    question:
      "On a cheap local model with a small window and no caching, what is the best way to survive a long lesson: keep it, compact it, or retrieve it?",
    takeaway:
      "On a 32k local model the lesson does not fit, so keep-everything is not even an option (the runtime evicts it). For fetching a document, RAG wins on every model. For compacting a growing chat, no single method wins (it depends on the model), and the model's own capability matters more than the strategy.",
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
          "One long lesson, answered 7 ways. Switch the model (tabs), then switch the metric (Quality, Input tokens, Latency) to see the trade-off: RAG ties or beats stuffing the whole document on quality, while full_context costs an order of magnitude more tokens and time.",
        views: [
          // rows: [method, quality %, input tokens, latency p50 s] from compare.md
          slmBMetric("qwen3", "qwen3:8b", [
            ["rag", 100, 3093, 25.9], ["graphrag", 100, 8672, 58.7], ["full_context", 73, 32767, 345.5],
            ["hierarchical_summary", 73, 1796, 10.4], ["selective", 73, 4219, 13.0],
            ["trim", 67, 4222, 26.1], ["summary", 67, 1042, 8.7],
          ]),
          slmBMetric("qwen2.5", "qwen2.5:7b", [
            ["rag", 100, 3085, 20.1], ["graphrag", 100, 8664, 52.2], ["full_context", 67, 32767, 265.6],
            ["trim", 47, 4214, 9.5], ["selective", 47, 4211, 4.4],
            ["summary", 33, 743, 2.7], ["hierarchical_summary", 20, 1161, 2.4],
          ]),
          slmBMetric("llama3.1", "llama3.1:8b", [
            ["graphrag", 87, 8324, 50.8], ["full_context", 80, 32767, 296.3], ["rag", 80, 3019, 19.6],
            ["trim", 53, 4087, 11.6], ["selective", 47, 4086, 9.2],
            ["hierarchical_summary", 27, 598, 4.2], ["summary", 0, 527, 4.1],
          ]),
        ],
      },
      {
        title: "Axis A: how to compact a growing conversation",
        intro:
          "The lesson is loaded turn 0, questions accumulate with retrieval off, and each real memory preset manages the growing context. No single method wins across models, and keep-all never tops the table.",
        views: [
          slmA("qwen3", "qwen3:8b", {
            summarization_only: 73, delta_summarization: 73, full_history: 67,
            hierarchical_summarization: 67, prompt_compression: 60, selective_retention: 60,
            incontext_history_retrieval: 60, sliding_window: 40,
          }),
          slmA("qwen2.5", "qwen2.5:7b", {
            prompt_compression: 60, selective_retention: 47, delta_summarization: 47,
            incontext_history_retrieval: 33, hierarchical_summarization: 33, full_history: 27,
            summarization_only: 20, sliding_window: 13,
          }),
          slmA("llama3.1", "llama3.1:8b", {
            hierarchical_summarization: 40, sliding_window: 13, prompt_compression: 13,
            selective_retention: 13, incontext_history_retrieval: 13, full_history: 7,
            summarization_only: 0, delta_summarization: 0,
          }),
        ],
      },
    ],
    setup: [
      "Largest course lesson (~37.7k tokens), one fixed 15-question set reused across every model.",
      "Each model runs in Ollama with a num_ctx=32768 variant, so the lesson overflows the window and compaction is forced.",
      "Quality is judged by Gemini 2.5 Flash reading the full lesson; a small model cannot hold it, and a model never grades itself.",
    ],
    caveats: [
      "Coarse n: 15 questions per method, 1 trial; LLM-judge variance moves cells by about 1-3 of 15.",
      "Keep-all numbers reflect Ollama evicting the oversized turn to fit the window.",
    ],
    links: [
      { label: "Writeup: evals_slm_compaction.md", href: `${REPO}/blob/experiment/slm-compaction/evals_slm_compaction.md` },
      { label: "SLM PR #3", href: `${REPO}/pull/3` },
    ],
  },

  {
    slug: "comparison",
    order: 5,
    shortTitle: "Cross-model: what to actually do",
    title: "Cross-model: cost, latency, quality, and the call",
    badge: "Synthesis · all models",
    accent: "#12213d",
    question:
      "Across a large cloud model, a cheap API model, and tiny local ones, what is the best context strategy for cost, latency, and quality?",
    takeaway:
      "There are two regimes. With a big window and caching (Gemini, DeepSeek), keep everything: it is cheapest, fastest, and best on memory, and compaction must justify itself. With a small local window (SLMs), you cannot keep everything (the runtime evicts it), so you must compact or retrieve, RAG wins document tasks, and the model's capability matters more than the method. Cost spans three orders of magnitude.",
    highlights: [
      { label: "Local SLM", value: "$0 / turn, must compact or retrieve" },
      { label: "DeepSeek", value: "~$0.01 / turn, keep-all wins" },
      { label: "Gemini", value: "~$0.1-0.2 / turn, keep-all wins to ~13t" },
      { label: "The split", value: "caching vs a hard window" },
    ],
    groups: [
      {
        title: "The two regimes",
        intro:
          "The right answer flips on one thing: can the model cache a growing context, or does it hit a hard window?",
        views: [
          {
            kind: "table",
            key: "regimes",
            label: "By model class",
            columns: ["", "Gemini 3.5", "DeepSeek V4-Flash", "Local SLM (8B, 32k)"],
            rows: [
              ["cost / turn", "~$0.1-0.2", "~$0.01", "$0 (your hardware)"],
              ["context window", "very large", "very large", "small, lesson overflows"],
              ["prompt caching", "~10x discount", "~50x discount", "none"],
              ["keep everything?", "yes, wins to ~13t", "yes, wins past 36t", "impossible, evicted"],
              ["best context strategy", "full history", "full history", "RAG / summarize to fit"],
              ["answer quality", "high", "high", "model-dependent"],
            ],
            caption:
              "Quality is not compared on one shared score here: the models ran different batteries and judge versions. The table compares the strategy decision and the cost order of magnitude.",
          },
        ],
      },
      {
        title: "General outcomes",
        intro: "What carries across every study.",
        views: [
          {
            kind: "findings",
            key: "outcomes",
            label: "General outcomes",
            findings: [
              {
                title: "If you can cache, hoarding wins",
                stat: "keep all",
                text: "On both cloud models, full history is cheapest, fastest, and best on memory up to the lengths tested. Compaction breaks the cache and drops old facts, so it has to earn its place on very long horizons or hard quality needs.",
              },
              {
                title: "If you cannot cache, you must compact, and RAG is the tool",
                stat: "retrieve",
                text: "On a small local model the document does not fit and the runtime silently evicts it. Retrieval (pull the right chunk) wins document tasks at a fraction of the tokens; pure recency dropping (sliding window) is reliably worst.",
              },
              {
                title: "Cheaper models can carry real load",
                stat: "~10-15x",
                text: "DeepSeek matched the qualitative conclusions at roughly 10-15x lower per-turn cost than Gemini, and local 8B models answered retrieval tasks well for free. The expensive model is not always required.",
              },
              {
                title: "Model choice can beat strategy choice",
                stat: "qwen3 > the rest",
                text: "Among the SLMs, the stronger model scored 40-73% across every compaction method while the weakest sat at 0-40%. Picking a better small model bought more than picking the best compaction strategy.",
              },
            ],
          },
        ],
      },
    ],
    setup: [
      "This page synthesizes the four studies; it does not introduce a new run.",
      "Cost figures are per-turn estimates on each study's own battery and tier, so treat them as orders of magnitude, not a single benchmark.",
    ],
    caveats: [
      "Different models used different batteries and (for the one-lesson study) different judge versions, so cross-model quality is directional, not a head-to-head score.",
      "All underlying numbers are coarse screens (small n, 1 trial).",
    ],
    links: [
      { label: "All findings: evals.md", href: `${REPO}/blob/main/evals.md` },
      { label: "Repository", href: REPO },
    ],
  },
];

export function getExperiment(slug: string): Experiment | undefined {
  return EXPERIMENTS.find((e) => e.slug === slug);
}

// The scale of the problem, shown as proportional bars on the home page.
export const SCALES: ScaleStrip[] = [
  {
    title: "Context windows",
    unit: "tokens",
    items: [
      { label: "Local SLM (8B)", value: 32_768, display: "32k", note: "the lesson does not fit" },
      { label: "Gemini / DeepSeek", value: 1_000_000, display: "~1M", note: "the lesson is a rounding error" },
    ],
  },
  {
    title: "What goes into one turn",
    unit: "tokens",
    items: [
      { label: "One course lesson", value: 37_700, display: "37.7k", note: "overflows a 32k window" },
      { label: "Retrieval payload / turn (Gemini)", value: 200_000, display: "~200k", note: "where the tokens actually are (F1)" },
      { label: "Cached prefix at 36 turns (DeepSeek)", value: 1_780_000, display: "1.78M", note: "~97% cache-hit, so it is cheap" },
    ],
  },
  {
    title: "Conversation length tested",
    unit: "turns",
    items: [
      { label: "Short / medium sessions", value: 13, display: "13", note: "keep-all still wins here" },
      { label: "Contradiction tier", value: 22, display: "22", note: "plant a fact, update it, probe" },
      { label: "Long-horizon tier", value: 36, display: "36", note: "keep-all still cheapest (DeepSeek)" },
    ],
  },
];
