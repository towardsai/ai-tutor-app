export type DocMetadata = {
  description: string;
  docsUrl: string;
};

export const DOC_METADATA: Record<string, DocMetadata> = {
  transformers: {
    description:
      "Hugging Face library for state-of-the-art NLP and multimodal models. Load, run, and train pretrained transformers.",
    docsUrl: "https://huggingface.co/docs/transformers",
  },
  peft: {
    description:
      "Parameter-Efficient Fine-Tuning: LoRA, prefix tuning, and other methods for adapting large models with minimal compute.",
    docsUrl: "https://huggingface.co/docs/peft",
  },
  trl: {
    description:
      "Train language models with reinforcement learning. Covers SFT, DPO, PPO, and other alignment techniques.",
    docsUrl: "https://huggingface.co/docs/trl",
  },
  llama_index: {
    description:
      "Framework for building RAG apps: ingestion, indexing, retrievers, and query engines over your own data.",
    docsUrl: "https://docs.llamaindex.ai",
  },
  langchain: {
    description:
      "Framework for building LLM apps: chains, agents, tool-calling, and production observability.",
    docsUrl: "https://docs.langchain.com/oss/python/langchain/overview",
  },
  langgraph: {
    description:
      "Graph-based runtime for reliable, stateful AI agents with persistence, streaming, human review, and deployment patterns.",
    docsUrl: "https://docs.langchain.com/oss/python/langgraph/overview",
  },
  deep_agents: {
    description:
      "LangChain's deep agent harness for planning, delegation, filesystem context, and longer-running agent workflows.",
    docsUrl: "https://docs.langchain.com/oss/python/deepagents/overview",
  },
  openai_cookbooks: {
    description:
      "Example notebooks and recipes from OpenAI covering practical patterns for using their APIs.",
    docsUrl: "https://cookbook.openai.com",
  },
  openai_docs: {
    description:
      "Official OpenAI API, Agents SDK, and Codex documentation from the developer docs Markdown index.",
    docsUrl: "https://developers.openai.com",
  },
  claude_code_docs: {
    description:
      "Official Claude Code and Claude Agent SDK documentation from Anthropic's Markdown index.",
    docsUrl: "https://code.claude.com/docs/en/overview",
  },
};
