# everything context engineering

## Context Engineering

- **Context engineering:** Designing what the model sees at each step: instructions, history, memory, retrieved docs, tool outputs, and skills.
- **Context window:** The fixed token budget available to the model in one call.
- **Context management:** The practical work of keeping the useful context and removing or externalizing the rest.
- **Context rot:** Long contexts can reduce quality, increase latency/cost, and bury important details.

Useful links:

- [Claude compaction docs](https://platform.claude.com/docs/en/build-with-claude/compaction)
- [OpenAI compaction guide](https://developers.openai.com/api/docs/guides/compaction)
- [Factory: evaluating compression](https://factory.ai/news/evaluating-compression)
- [Intro to context management and compaction](https://youtu.be/PQglg4N_jxo)

---

## Context Compaction

- **Compaction:** Umbrella term for reducing context while preserving what still matters.
- **Trimming:** Drop older messages when they are no longer relevant.
- **Sliding window:** Keep the most recent N turns, sometimes with overlap for continuity.
- **Observation truncation:** Cut huge tool outputs to head/tail or important excerpts before adding them to history.
- **Tool-result clearing:** Replace old tool outputs with a placeholder while keeping the tool-call record.
- **Selective retention:** Keep constraints, decisions, open tasks, and state; drop exploration and duplicates.
- **Summarization:** Replace older history with a shorter lossy summary.
- **Hierarchical summarization:** Summarize chunks, then summarize the summaries for long-running sessions.
- **Delta summarization:** Update a running summary only with what changed since the last turn.
- **Context reset:** Start a fresh conversation state seeded with the compacted summary.
- **Prompt compression:** Rewrite prompts/history into fewer tokens while trying to preserve meaning.
- **Offload to files or memory:** Persist details externally and keep only a pointer in context.
- **In-context retrieval:** Pull only the relevant memory/docs back into the prompt when needed.

Helpful explainers:

- [Claude Code compaction explained](https://okhlopkov.com/claude-code-compaction-explained/)
- [Context compaction in Codex, Claude Code, and OpenCode](https://justin3go.com/en/posts/2026/04/09-context-compaction-in-codex-claude-code-and-opencode)

---

## Memory

- **Working memory:** Current-session context: recent turns, scratchpad, tool results, intermediate state.
- **Semantic memory:** Durable facts about the user, company, project, preferences, goals, or entities.
- **Episodic memory:** Past events or examples retrieved as “what happened before.”
- **Procedural memory:** Learned rules, workflows, habits, or skills the agent uses to improve how it works.
- **Profile memory:** One compact document representing current known facts about a user/entity.
- **Collection memory:** Many small fact documents that can be searched, merged, or updated.
- **Entity memory:** Profiles for multiple people, companies, repos, projects, or objects.
- **Memory operations:** Add, update, merge, consolidate, delete, and resolve contradictions.
- **Sleep-time consolidation:** Background process that cleans and compresses memories outside the live turn.
- **Karpathy-style memory system:** Human-readable project/profile/instruction files that agents read and update as durable context.

---

## Retrieval

- **Keyword search:** Exact or fuzzy text lookup; cheap and strong when terms are known.
- **Vector DBs:** Embedding-based semantic search for “similar meaning” retrieval.
- **Hybrid search:** Combines keyword + vector search for better recall.
- **Reranking:** Reorders retrieved chunks by relevance before giving them to the model.
- **RAG:** Retrieve external knowledge, then inject it into the model context.
- **RAG-as-memory:** Store memories as retrievable documents; simple baseline but weak with changing facts.
- **GraphRAG:** Uses a knowledge graph to retrieve entities, relationships, and community summaries.
- **Temporal graph memory:** Tracks facts over time, useful when facts change or contradict each other.
- **In-context retrieval:** Retrieve only what fits the current task, then place it directly into the prompt.

---

## Skills

- **Skills:** Reusable instructions, workflows, tools, or examples loaded only when relevant.
- **Progressive disclosure:** Keep skills small and load detailed instructions only when needed.
- **Lazy prompt loading:** Avoid putting the whole knowledge base/system manual into every call.
- **Skill registry:** A table of contents the agent can search to decide which skill to load.
- **Small focused skills:** Easier to retrieve, update, and compose than one giant skill file.

---

## Tools

- **Tool calls:** Let the model act on external systems: search, files, code, browser, DBs, APIs.
- **Memory write tool:** Lets the agent explicitly save durable facts or summaries.
- **Context editing middleware:** Automatically clears, trims, or rewrites context before the next call.
- **Checkpointer:** Stores conversation state inside a session.
- **Store:** Persists long-term memory across sessions.
- **Token/cost/latency meter:** Makes context growth visible so compaction can be demonstrated.

---

## Agents

- **Single-agent workflow:** One strong agent with good context, tools, skills, and memory.
- **Sub-agent isolation:** Give a side task to another agent so its exploration does not pollute main context.
- **Parallel research agents:** Useful for broad read/research tasks where branches can run independently.
- **Multi-agent systems:** Multiple agents coordinating; powerful but costly and harder for shared-state writing tasks.
- **Context-first agents:** Agents become better less by adding more agents, and more by improving context, memory, tools, and workflows.

---

## Demo / Case Study

- **AI Tutor repo:** [https://github.com/towardsai/ai-tutor-app](https://github.com/towardsai/ai-tutor-app)
- **AI Tutor HF Space:** [https://huggingface.co/spaces/towardsai-tutors/ai-tutor](https://huggingface.co/spaces/towardsai-tutors/ai-tutor)
- **Demo idea:** Show failure first, then fix it with compaction, memory, and lazy skills.
- **Compaction demo:** Context grows from ~4k to ~45k tokens, then drops to ~8k after compaction.
- **Memory demo:** New session recall goes from 0/5 facts to 5/5 facts with profile memory.
- **Skills demo:** System prompt drops from ~4k tokens to ~800 tokens with lazy loading.