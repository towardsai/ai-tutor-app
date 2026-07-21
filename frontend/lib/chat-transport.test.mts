import assert from "node:assert/strict";
import test from "node:test";
import type { UIMessage } from "ai";
import {
  prepareTutorChatRequest,
  toTextOnlyMessages,
} from "./chat-transport.ts";

test("keeps only text parts of large tool turns without changing raw assistant text", () => {
  const largeToolOutput = "retrieved context ".repeat(20_000);
  const messages = [
    {
      id: "user-1",
      role: "user",
      parts: [{ type: "text", text: "What is LoRA?" }],
    },
    {
      id: "assistant-1",
      role: "assistant",
      parts: [
        { type: "reasoning", text: "I should search." },
        { type: "text", text: "  Let me check the sources.\n\n" },
        {
          type: "tool-retrieve_tutor_context",
          toolCallId: "call-1",
          state: "output-available",
          input: { query: "LoRA" },
          output: { text: largeToolOutput },
        },
        { type: "text", text: "LoRA is parameter-efficient.\n" },
        { type: "source-url", sourceId: "source-1", url: "https://example.com" },
        { type: "data-source", data: { title: "LoRA" } },
      ],
    },
    {
      id: "user-2",
      role: "user",
      parts: [{ type: "text", text: "Can you expand on that?" }],
    },
  ] as unknown as UIMessage[];

  const textOnly = toTextOnlyMessages(messages);

  assert.deepEqual(textOnly, [
    {
      id: "user-1",
      role: "user",
      parts: [{ type: "text", text: "What is LoRA?" }],
    },
    {
      id: "assistant-1",
      role: "assistant",
      parts: [
        { type: "text", text: "  Let me check the sources.\n\n" },
        { type: "text", text: "LoRA is parameter-efficient.\n" },
      ],
    },
    {
      id: "user-2",
      role: "user",
      parts: [{ type: "text", text: "Can you expand on that?" }],
    },
  ]);
  assert.ok(JSON.stringify(messages).length > 200_000);
  assert.ok(JSON.stringify(textOnly).length < 1_000);
  assert.equal(
    (messages[1].parts[2] as { output: { text: string } }).output.text,
    largeToolOutput,
    "stripping to text parts must not mutate the UI transcript",
  );
});

test("prepares submit and regenerate requests with text-only messages", async () => {
  const messages = [
    {
      id: "user-1",
      role: "user",
      parts: [{ type: "text", text: "Original question" }],
    },
  ] as UIMessage[];

  for (const [trigger, messageId] of [
    ["submit-message", undefined],
    ["regenerate-message", "assistant-1"],
  ] as const) {
    const prepared = await prepareTutorChatRequest({
      id: "chat-1",
      messages,
      requestMetadata: undefined,
      body: { threadId: "thread-1", model: "deepseek:deepseek-v4-flash" },
      credentials: undefined,
      headers: undefined,
      api: "/api/chat",
      trigger,
      messageId,
    });
    const body = prepared.body as Record<string, unknown>;

    assert.equal(body.id, "chat-1");
    assert.equal(body.threadId, "thread-1");
    assert.equal(body.model, "deepseek:deepseek-v4-flash");
    assert.equal(body.trigger, trigger);
    assert.equal(body.messageId, messageId);
    assert.deepEqual(body.messages, toTextOnlyMessages(messages));
  }
});
