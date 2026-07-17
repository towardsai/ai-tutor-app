import { expect, test, type Page } from "@playwright/test";

const MODEL = "deepseek:deepseek-v4-flash";
const largeToolOutput = "retrieved context ".repeat(20_000);

const toolsResponse = {
  model: MODEL,
  availableModels: [{ id: MODEL, label: "DeepSeek V4 Flash" }],
  tools: [
    {
      kind: "configurable",
      key: "retrieval",
      label: "Knowledge base",
      active: true,
      sources: [
        {
          key: "example-docs",
          label: "Example documentation",
          shortLabel: "Example docs",
          description: "Documentation used by the browser test.",
          infoUrl: "https://example.test/docs",
          group: "docs",
          selectedByDefault: true,
          version: "v1.0",
          indexedAt: "2026-07-01",
        },
      ],
    },
  ],
};

function sse(parts: Array<Record<string, unknown>>) {
  return `${parts.map((part) => `data: ${JSON.stringify(part)}\n\n`).join("")}data: [DONE]\n\n`;
}

async function mockTools(page: Page) {
  await page.route("**/api/tools**", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(toolsResponse),
    }),
  );
}

test("streams activity and sends a compact follow-up on the same thread", async ({
  page,
}) => {
  const requests: Array<Record<string, unknown>> = [];
  await mockTools(page);
  await page.route("**/api/chat", async (route) => {
    requests.push(route.request().postDataJSON() as Record<string, unknown>);
    const firstTurn = requests.length === 1;
    const parts = firstTurn
      ? [
          {
            type: "data-thread",
            data: { threadId: "thread-browser" },
            transient: true,
          },
          { type: "start", messageId: "assistant-1" },
          { type: "start-step" },
          { type: "reasoning-start", id: "thought-1" },
          { type: "reasoning-delta", id: "thought-1", delta: "First thought" },
          { type: "reasoning-end", id: "thought-1" },
          {
            type: "tool-input-available",
            toolCallId: "call-1",
            toolName: "retrieve_tutor_context",
            input: { query: "LoRA" },
          },
          {
            type: "tool-output-available",
            toolCallId: "call-1",
            output: {
              text: largeToolOutput,
              matches: [
                {
                  docId: "lora-doc",
                  title: "LoRA guide",
                  url: "https://example.test/lora",
                  sourceKey: "example-docs",
                  sourceLabel: "Example docs",
                  score: 0.9,
                  group: "docs",
                },
              ],
            },
          },
          { type: "reasoning-start", id: "thought-2" },
          { type: "reasoning-delta", id: "thought-2", delta: "Second thought" },
          { type: "reasoning-end", id: "thought-2" },
          {
            type: "source-url",
            sourceId: "lora-doc",
            url: "https://example.test/lora",
          },
          {
            type: "data-source",
            data: {
              docId: "lora-doc",
              title: "LoRA guide",
              url: "https://example.test/lora",
              sourceKey: "example-docs",
              sourceLabel: "Example docs",
              score: 0.9,
              group: "docs",
            },
          },
          { type: "text-start", id: "answer-1" },
          {
            type: "text-delta",
            id: "answer-1",
            delta: "LoRA is parameter-efficient [1](https://example.test/lora).",
          },
          { type: "text-end", id: "answer-1" },
          { type: "finish-step" },
          { type: "finish" },
        ]
      : [
          { type: "start", messageId: "assistant-2" },
          { type: "start-step" },
          { type: "text-start", id: "answer-2" },
          { type: "text-delta", id: "answer-2", delta: "Follow-up answer" },
          { type: "text-end", id: "answer-2" },
          { type: "finish-step" },
          { type: "finish" },
        ];
    await route.fulfill({
      status: 200,
      headers: {
        "content-type": "text/event-stream",
        "x-vercel-ai-ui-message-stream": "v1",
      },
      body: sse(parts),
    });
  });

  await page.goto("/");
  await expect(page.getByRole("button", { name: "DeepSeek V4 Flash" })).toBeVisible();
  const composer = page.getByRole("textbox");
  await composer.fill("What is LoRA?");
  await page.getByRole("button", { name: "Send message" }).click();

  await expect(page.getByText("LoRA is parameter-efficient")).toBeVisible();
  const activity = page.getByRole("button", {
    name: "Activity · 1 tool · 2 thoughts · 1 source",
  });
  await expect(activity).toBeVisible();
  await activity.click();
  await expect(page.getByText("First thought")).toBeVisible();
  await expect(page.getByText("Second thought")).toBeVisible();

  await composer.fill("Can you expand?");
  await page.getByRole("button", { name: "Send message" }).click();
  await expect(page.getByText("Follow-up answer")).toBeVisible();
  await expect.poll(() => requests.length).toBe(2);

  const followUp = requests[1];
  expect(followUp.threadId).toBe("thread-browser");
  expect(JSON.stringify(followUp)).not.toContain(largeToolOutput.slice(0, 500));
  expect(followUp.messages).toEqual([
    expect.objectContaining({
      role: "user",
      parts: [{ type: "text", text: "What is LoRA?" }],
    }),
    expect.objectContaining({
      role: "assistant",
      parts: [
        {
          type: "text",
          text: "LoRA is parameter-efficient [1](https://example.test/lora).",
        },
      ],
    }),
    expect.objectContaining({
      role: "user",
      parts: [{ type: "text", text: "Can you expand?" }],
    }),
  ]);
});

test("keeps controls and source metadata inside a narrow viewport", async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 812 });
  await mockTools(page);
  await page.goto("/");

  await expect(page.getByRole("button", { name: "Send message" })).toBeVisible();
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth),
  ).toBe(true);

  await page
    .getByRole("button", { name: "About Example documentation" })
    .click();
  const dialog = page.getByRole("dialog");
  await expect(dialog).toContainText("v1.0 · indexed Jul 2026");
  const box = await dialog.boundingBox();
  expect(box).not.toBeNull();
  expect(box!.x).toBeGreaterThanOrEqual(0);
  expect(box!.x + box!.width).toBeLessThanOrEqual(375);
});
