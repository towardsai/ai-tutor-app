import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { ChatMessage } from "@/components/chat-message";
import type { TutorMessage, TutorMessagePart } from "@/lib/chat-ui";

afterEach(cleanup);

function assistantMessage(
  parts: TutorMessagePart[],
  id = "assistant-message",
): TutorMessage {
  return { id, role: "assistant", parts } as unknown as TutorMessage;
}

describe("ChatMessage activity rendering", () => {
  it("preserves reasoning, tool, reasoning order and counts merged thoughts", () => {
    render(
      <ChatMessage
        showAssistantActions={false}
        message={assistantMessage([
          { type: "reasoning", text: "First thought" },
          { type: "reasoning", text: "Continuation of the first thought" },
          {
            type: "tool-retrieve_tutor_context",
            toolCallId: "retrieve-1",
            state: "output-available",
            input: { query: "agent memory" },
            output: {
              matches: [
                { docId: "doc-a", url: "https://example.com/a" },
                { docId: "doc-b", url: "https://example.com/b" },
                // Repeated evidence must not inflate the source count.
                { docId: "doc-a", url: "https://example.com/a#section" },
              ],
            },
          },
          { type: "reasoning", text: "Second thought after retrieval" },
          { type: "text", text: "Final answer" },
        ])}
      />,
    );

    const activityToggle = screen.getByRole("button", {
      name: "Activity · 1 tool · 2 thoughts · 2 sources",
    });
    expect(activityToggle.getAttribute("aria-expanded")).toBe("false");

    fireEvent.click(activityToggle);

    const items = within(screen.getByRole("list")).getAllByRole("listitem");
    expect(items).toHaveLength(3);
    expect(items[0]?.textContent).toContain("First thought");
    expect(items[0]?.textContent).toContain("Continuation of the first thought");
    expect(items[1]?.textContent).toContain("Hybrid search");
    expect(items[1]?.textContent).toContain("agent memory");
    expect(items[2]?.textContent).toContain("Second thought after retrieval");
    expect(screen.getByText("Final answer")).toBeTruthy();
  });

  it("shows Running only while a tool call is genuinely unfinished", () => {
    const { rerender } = render(
      <ChatMessage
        isStreaming
        showAssistantActions={false}
        message={assistantMessage([
          { type: "reasoning", text: "I should search first" },
          {
            type: "tool-retrieve_tutor_context",
            toolCallId: "retrieve-1",
            state: "input-available",
            input: { query: "transformer attention" },
          },
        ])}
      />,
    );

    expect(
      screen.getByRole("button", {
        name: /Running Hybrid search transformer attention/,
      }),
    ).toBeTruthy();

    rerender(
      <ChatMessage
        isStreaming
        showAssistantActions={false}
        message={assistantMessage([
          { type: "reasoning", text: "I should search first" },
          {
            type: "tool-retrieve_tutor_context",
            toolCallId: "retrieve-1",
            state: "output-available",
            input: { query: "transformer attention" },
            output: {
              matches: [
                { docId: "attention", url: "https://example.com/attention" },
              ],
            },
          },
          { type: "reasoning", text: "The result answers the question" },
        ])}
      />,
    );

    expect(
      screen.getByRole("button", {
        name: "Activity · 1 tool · 2 thoughts · 1 source",
      }),
    ).toBeTruthy();
    expect(screen.queryByText("Running")).toBeNull();
    expect(screen.getByText("The result answers the question")).toBeTruthy();
  });

  it("uses the web icon for provider-native search tools", () => {
    render(
      <ChatMessage
        showAssistantActions={false}
        message={assistantMessage([
          {
            type: "tool-google_search",
            toolCallId: "search-1",
            state: "output-available",
            input: { query: "latest LangChain release" },
            output: { text: "Search results" },
          },
        ])}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Activity · 1 tool" }));
    const toolItem = screen.getByRole("listitem");
    expect(toolItem.querySelector(".lucide-globe")).not.toBeNull();
    expect(toolItem.querySelector(".lucide-database")).toBeNull();
  });

  it("bounds expanded tool output by both line and character limits", () => {
    const lineLimitedOutput = "one\ntwo\nthree\nfour\nfive\nsix";
    const characterLimitedOutput = `${"x".repeat(1005)}TAIL`;

    render(
      <ChatMessage
        showAssistantActions={false}
        message={assistantMessage([
          {
            type: "tool-run_kb_command",
            toolCallId: "kb-1",
            state: "output-available",
            input: { text: "head raw/docs/file.md" },
            output: { text: lineLimitedOutput },
          },
          {
            type: "tool-custom_reader",
            toolCallId: "custom-1",
            state: "output-available",
            input: { text: "read the long result" },
            output: { text: characterLimitedOutput },
          },
        ])}
      />,
    );

    fireEvent.click(
      screen.getByRole("button", { name: "Activity · 2 tools" }),
    );
    fireEvent.click(
      screen.getByRole("button", { name: /KB shell.*6 lines/ }),
    );

    const lineTruncation = screen.getByText(/truncated \(4 chars more\)/);
    const linePreview = lineTruncation.closest("pre");
    expect(linePreview?.firstChild?.textContent).toBe(
      "one\ntwo\nthree\nfour\nfive",
    );
    expect(linePreview?.textContent).not.toContain("six");

    fireEvent.click(
      screen.getByRole("button", { name: /Custom Reader.*1\.0k chars/ }),
    );

    const characterTruncation = screen.getByText(/truncated \(9 chars more\)/);
    const characterPreview = characterTruncation.closest("pre");
    expect(characterPreview?.firstChild?.textContent).toHaveLength(1000);
    expect(characterPreview?.textContent).not.toContain("TAIL");
  });
});

describe("ChatMessage citations", () => {
  it("reuses an inline citation number on its source card and keeps unsafe links inert", () => {
    const { container } = render(
      <ChatMessage
        showAssistantActions={false}
        message={assistantMessage([
          {
            type: "text",
            text: [
              "[Trusted](https://example.com/docs#section)",
              "[Again](https://example.com/docs/)",
              "[Unsafe](javascript:alert(1))",
              "[Missing KB file](raw/docs/missing.md)",
            ].join(" "),
          },
          {
            type: "data-source",
            data: {
              docId: "docs-1",
              title: "Docs title",
              url: "https://example.com/docs",
              sourceKey: "example",
              sourceLabel: "Example Docs",
              score: 0.9,
              group: "docs",
            },
          },
        ])}
      />,
    );

    const inlineChips = Array.from(
      container.querySelectorAll<HTMLAnchorElement>("a.citation-chip"),
    );
    expect(inlineChips).toHaveLength(2);
    expect(inlineChips.map((chip) => chip.textContent)).toEqual(["1", "1"]);

    const sourceCard = screen.getByTitle("Docs · Example Docs");
    expect(sourceCard.tagName).toBe("A");
    expect(sourceCard.getAttribute("href")).toBe("https://example.com/docs");
    expect(sourceCard.textContent).toContain("1");
    expect(sourceCard.textContent).toContain("Docs title");

    expect(screen.getByText("Unsafe").closest("a")).toBeNull();
    expect(screen.getByText("Missing KB file").closest("a")).toBeNull();
  });
});
