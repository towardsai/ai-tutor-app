import assert from "node:assert/strict";
import test from "node:test";

import {
  readUIMessageStream,
  type UIMessage,
  type UIMessageChunk,
} from "ai";

import { compactChatMessages } from "./chat-transport.ts";

function streamChunks(chunks: UIMessageChunk[]) {
  return new ReadableStream<UIMessageChunk>({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(chunk);
      }
      controller.close();
    },
  });
}

async function foldChunks(chunks: UIMessageChunk[]) {
  let latest: UIMessage | undefined;
  for await (const message of readUIMessageStream({
    stream: streamChunks(chunks),
  })) {
    latest = message;
  }
  assert.ok(latest, "the stream should produce an assistant message");
  return latest;
}

test("folds reasoning, tools, sources, and text in provider order", async () => {
  const message = await foldChunks([
    { type: "start", messageId: "assistant-1" },
    { type: "start-step" },
    { type: "reasoning-start", id: "thought-1" },
    { type: "reasoning-delta", id: "thought-1", delta: "First thought" },
    { type: "reasoning-end", id: "thought-1" },
    {
      type: "tool-input-start",
      toolCallId: "call-1",
      toolName: "retrieve_tutor_context",
    },
    {
      type: "tool-input-available",
      toolCallId: "call-1",
      toolName: "retrieve_tutor_context",
      input: { query: "LoRA" },
    },
    {
      type: "tool-output-available",
      toolCallId: "call-1",
      output: { text: "retrieved context", matches: [] },
    },
    { type: "reasoning-start", id: "thought-2" },
    { type: "reasoning-delta", id: "thought-2", delta: "Second thought" },
    { type: "reasoning-end", id: "thought-2" },
    {
      type: "source-url",
      sourceId: "source-1",
      url: "https://example.com/lora",
      title: "LoRA guide",
    },
    {
      type: "data-source",
      data: {
        title: "LoRA guide",
        url: "https://example.com/lora",
        source: "docs",
      },
    },
    { type: "text-start", id: "answer-1" },
    {
      type: "text-delta",
      id: "answer-1",
      delta: "  LoRA is parameter-efficient.\n",
    },
    { type: "text-end", id: "answer-1" },
    { type: "finish-step" },
    { type: "finish" },
  ]);

  assert.equal(message.id, "assistant-1");
  assert.equal(message.role, "assistant");
  assert.deepEqual(
    message.parts.map((part) => part.type),
    [
      "step-start",
      "reasoning",
      "tool-retrieve_tutor_context",
      "reasoning",
      "source-url",
      "data-source",
      "text",
    ],
  );
  assert.equal(
    (message.parts[1] as { text: string }).text,
    "First thought",
  );
  assert.equal(
    (message.parts[3] as { text: string }).text,
    "Second thought",
  );
  assert.equal(
    (message.parts[6] as { text: string }).text,
    "  LoRA is parameter-efficient.\n",
  );
});

test("folded tool state stays in the UI while follow-up history is text-only", async () => {
  const largeOutput = "retrieved context ".repeat(20_000);
  const assistant = await foldChunks([
    { type: "start", messageId: "assistant-1" },
    { type: "start-step" },
    {
      type: "tool-input-available",
      toolCallId: "call-1",
      toolName: "retrieve_tutor_context",
      input: { query: "LoRA" },
    },
    {
      type: "tool-output-available",
      toolCallId: "call-1",
      output: { text: largeOutput, matches: [] },
    },
    { type: "text-start", id: "answer-1" },
    { type: "text-delta", id: "answer-1", delta: "Exact answer  \n" },
    { type: "text-end", id: "answer-1" },
    { type: "finish-step" },
    { type: "finish" },
  ]);

  const toolPart = assistant.parts.find((part) => part.type.startsWith("tool-"));
  assert.ok(toolPart);
  assert.equal(
    (toolPart as { output: { text: string } }).output.text,
    largeOutput,
  );

  const compacted = compactChatMessages([
    { id: "user-1", role: "user", parts: [{ type: "text", text: "Question" }] },
    assistant,
    {
      id: "user-2",
      role: "user",
      parts: [{ type: "text", text: "Follow up" }],
    },
  ]);

  assert.deepEqual(compacted[1].parts, [
    { type: "text", text: "Exact answer  \n" },
  ]);
  assert.ok(JSON.stringify(compacted).length < 1_000);
  assert.equal(
    (toolPart as { output: { text: string } }).output.text,
    largeOutput,
    "request compaction must not mutate folded UI state",
  );
});
