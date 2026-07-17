import assert from "node:assert/strict";
import test from "node:test";

import { Chat } from "@ai-sdk/react";
import type { ChatTransport, UIMessage, UIMessageChunk } from "ai";

type SendOptions = Parameters<ChatTransport<UIMessage>["sendMessages"]>[0];

function assistantStream(messageId: string) {
  const chunks: UIMessageChunk[] = [
    { type: "start", messageId },
    { type: "start-step" },
    { type: "text-start", id: `${messageId}-text` },
    { type: "text-delta", id: `${messageId}-text`, delta: "New answer" },
    { type: "text-end", id: `${messageId}-text` },
    { type: "finish-step" },
    { type: "finish" },
  ];
  return new ReadableStream<UIMessageChunk>({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(chunk);
      controller.close();
    },
  });
}

class CapturingTransport implements ChatTransport<UIMessage> {
  calls: Array<Omit<SendOptions, "messages"> & { messages: UIMessage[] }> = [];

  async sendMessages(options: SendOptions) {
    this.calls.push({
      ...options,
      messages: structuredClone(options.messages),
    });
    return assistantStream(`response-${this.calls.length}`);
  }

  async reconnectToStream() {
    return null;
  }
}

function transcript(): UIMessage[] {
  return [
    { id: "u1", role: "user", parts: [{ type: "text", text: "Question 1" }] },
    { id: "a1", role: "assistant", parts: [{ type: "text", text: "Answer 1" }] },
    { id: "u2", role: "user", parts: [{ type: "text", text: "Question 2" }] },
    { id: "a2", role: "assistant", parts: [{ type: "text", text: "Answer 2" }] },
    { id: "u3", role: "user", parts: [{ type: "text", text: "Question 3" }] },
    { id: "a3", role: "assistant", parts: [{ type: "text", text: "Answer 3" }] },
  ];
}

function messageSummary(messages: UIMessage[]) {
  return messages.map((message) => ({
    id: message.id,
    role: message.role,
    text: message.parts
      .filter((part) => part.type === "text")
      .map((part) => part.text)
      .join(""),
  }));
}

test("editing a user turn replaces it and drops its answer and later turns", async () => {
  const transport = new CapturingTransport();
  const chat = new Chat<UIMessage>({
    id: "chat-edit",
    messages: transcript(),
    transport,
  });

  await chat.sendMessage({ text: "Question 2 edited", messageId: "u2" });

  assert.equal(transport.calls.length, 1);
  assert.equal(transport.calls[0].trigger, "submit-message");
  assert.equal(transport.calls[0].messageId, "u2");
  assert.deepEqual(messageSummary(transport.calls[0].messages), [
    { id: "u1", role: "user", text: "Question 1" },
    { id: "a1", role: "assistant", text: "Answer 1" },
    { id: "u2", role: "user", text: "Question 2 edited" },
  ]);
});

test("regenerating an assistant turn replays its user prompt and drops later turns", async () => {
  const transport = new CapturingTransport();
  const chat = new Chat<UIMessage>({
    id: "chat-regenerate",
    messages: transcript(),
    transport,
  });

  await chat.regenerate({ messageId: "a2" });

  assert.equal(transport.calls.length, 1);
  assert.equal(transport.calls[0].trigger, "regenerate-message");
  assert.equal(transport.calls[0].messageId, "a2");
  assert.deepEqual(messageSummary(transport.calls[0].messages), [
    { id: "u1", role: "user", text: "Question 1" },
    { id: "a1", role: "assistant", text: "Answer 1" },
    { id: "u2", role: "user", text: "Question 2" },
  ]);
});

test("regenerating an earlier answer removes every subsequent turn", async () => {
  const transport = new CapturingTransport();
  const chat = new Chat<UIMessage>({
    id: "chat-regenerate-earlier",
    messages: transcript(),
    transport,
  });

  await chat.regenerate({ messageId: "a1" });

  assert.deepEqual(messageSummary(transport.calls[0].messages), [
    { id: "u1", role: "user", text: "Question 1" },
  ]);
});
