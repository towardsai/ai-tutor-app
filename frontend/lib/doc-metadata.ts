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
  openai_cookbooks: {
    description:
      "Example notebooks and recipes from OpenAI covering practical patterns for using their APIs.",
    docsUrl: "https://cookbook.openai.com",
  },
};
