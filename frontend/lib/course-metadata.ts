export type CourseMetadata = {
  description: string;
  academyUrl: string;
};

export const COURSE_METADATA: Record<string, CourseMetadata> = {
  full_stack_ai_engineering: {
    description:
      "Full-stack LLM engineering, covering RAG, fine-tuning, evaluation, and deploying production systems end-to-end. The deepest technical course.",
    academyUrl:
      "https://academy.towardsai.net/courses/beginner-to-advanced-llm-dev",
  },
  beginner_python_for_ai_engineering: {
    description:
      "Python for the LLM era: API integration, using open-source models, and core training/testing workflows. Assumes no prior Python.",
    academyUrl: "https://academy.towardsai.net/courses/python-for-genai",
  },
  master_ai_for_work: {
    description:
      "Non-engineer course on using AI tools (ChatGPT, Claude, etc.) for workplace productivity and rolling them out across a team.",
    academyUrl:
      "https://academy.towardsai.net/courses/ai-business-professionals",
  },
  agentic_ai_engineering: {
    description:
      "Designing, building, evaluating, and deploying production-grade AI agents end-to-end.",
    academyUrl: "https://academy.towardsai.net/courses/agent-engineering",
  },
};
