import assert from "node:assert/strict";
import test from "node:test";

import { Chat } from "@ai-sdk/react";
import { DefaultChatTransport, type UIMessage, type UIMessageChunk } from "ai";

import { prepareTutorChatRequest } from "./chat-transport.ts";

function sseResponse(chunks: UIMessageChunk[]) {
  const body = [
    ...chunks.map((chunk) => `data: ${JSON.stringify(chunk)}\n\n`),
    "data: [DONE]\n\n",
  ].join("");
  return new Response(body, {
    status: 200,
    headers: {
      "content-type": "text/event-stream",
      "x-vercel-ai-ui-message-stream": "v1",
    },
  });
}

test("DefaultChatTransport sends compact history and preserves checkpoint identity", async () => {
  const largeToolOutput = "retrieved context ".repeat(20_000);
  const originalAssistantText = "  Exact streamed answer  \n";
  const initialMessages = [
    {
      id: "user-1",
      role: "user",
      parts: [{ type: "text", text: "Question" }],
    },
    {
      id: "assistant-1",
      role: "assistant",
      parts: [
        { type: "reasoning", text: "Private thought" },
        {
          type: "tool-retrieve_tutor_context",
          toolCallId: "call-1",
          state: "output-available",
          input: { query: "Question" },
          output: { text: largeToolOutput, matches: [] },
        },
        { type: "source-url", sourceId: "source-1", url: "https://example.test" },
        { type: "text", text: originalAssistantText },
      ],
    },
  ] as unknown as UIMessage[];

  let requestBody: Record<string, unknown> | undefined;
  const receivedData: unknown[] = [];
  const transport = new DefaultChatTransport<UIMessage>({
    api: "https://api.example.test/api/chat",
    prepareSendMessagesRequest: prepareTutorChatRequest,
    fetch: async (_input, init) => {
      requestBody = JSON.parse(String(init?.body)) as Record<string, unknown>;
      return sseResponse([
        {
          type: "data-thread",
          data: { threadId: "thread-1" },
          transient: true,
        },
        { type: "start", messageId: "assistant-2" },
        { type: "start-step" },
        { type: "text-start", id: "answer-2" },
        { type: "text-delta", id: "answer-2", delta: "Follow-up answer" },
        { type: "text-end", id: "answer-2" },
        { type: "finish-step" },
        { type: "finish" },
      ]);
    },
  });
  const chat = new Chat<UIMessage>({
    id: "chat-1",
    messages: initialMessages,
    transport,
    onData: (part) => receivedData.push(part),
  });

  await chat.sendMessage(
    { text: "Follow up" },
    {
      body: {
        threadId: "thread-1",
        model: "deepseek:deepseek-v4-flash",
      },
    },
  );

  assert.ok(requestBody);
  assert.equal(requestBody.threadId, "thread-1");
  assert.equal(requestBody.model, "deepseek:deepseek-v4-flash");
  assert.deepEqual(requestBody.messages, [
    {
      id: "user-1",
      role: "user",
      parts: [{ type: "text", text: "Question" }],
    },
    {
      id: "assistant-1",
      role: "assistant",
      parts: [{ type: "text", text: originalAssistantText }],
    },
    {
      id: chat.messages[2]?.id,
      role: "user",
      parts: [{ type: "text", text: "Follow up" }],
    },
  ]);
  assert.ok(JSON.stringify(requestBody).length < 2_000);
  assert.deepEqual(receivedData, [
    {
      type: "data-thread",
      data: { threadId: "thread-1" },
      transient: true,
    },
  ]);

  const retainedTool = chat.messages[1]?.parts.find((part) =>
    part.type.startsWith("tool-"),
  );
  assert.equal(
    (retainedTool as { output: { text: string } }).output.text,
    largeToolOutput,
  );
  assert.equal(
    (chat.messages[1]?.parts.at(-1) as { text: string }).text,
    originalAssistantText,
  );
});
