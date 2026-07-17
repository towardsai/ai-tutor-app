import assert from "node:assert/strict";
import test from "node:test";
import type { SourcePartData } from "./api.ts";
import {
  buildActivityItems,
  buildCitationNumbers,
  buildCitationResolutions,
  citationNumberFor,
  getMessageCitations,
  getMessageTextContent,
  getOrderedMessageBlocks,
  hasRenderableContent,
  resolveCitationHref,
  type TutorMessage,
  type TutorMessagePart,
} from "./chat-ui.ts";

function assistantMessage(
  parts: TutorMessagePart[],
  id = "assistant-1",
): TutorMessage {
  return { id, role: "assistant", parts } as unknown as TutorMessage;
}

function source(overrides: Partial<SourcePartData> = {}): SourcePartData {
  return {
    docId: "doc-1",
    title: "A useful source",
    url: "https://docs.example.test/guide",
    sourceKey: "docs",
    sourceLabel: "Example docs",
    score: 0.9,
    group: "docs",
    ...overrides,
  };
}

test("citation numbers follow rendered-link order and ignore non-rendered links", () => {
  const message = assistantMessage([
    {
      type: "text",
      text: [
        "[Nested [label]](https://docs.example.test/Spring_(framework)#intro)",
        "`[inline code](https://ignored.example.test/inline)`",
        "![image](https://ignored.example.test/image)",
        "```md\n[fenced](https://ignored.example.test/fenced)\n```",
        "[Duplicate](https://docs.example.test/Spring_(framework)/)",
      ].join("\n"),
    },
    {
      type: "text",
      text: [
        "[Encoded](https://docs.example.test/caf%C3%A9)",
        "[Unicode duplicate](https://docs.example.test/café#heading)",
        "[Angle form](<https://other.example.test/a(b)>)",
        "[With title](https://third.example.test/path \"Source title\")",
        "[Malformed](https://ignored.example.test/wrong]",
        "[Relative](raw/docs/example.md)",
      ].join("\n"),
    },
  ]);

  const numbers = buildCitationNumbers(message);

  assert.deepEqual([...numbers.entries()], [
    ["https://docs.example.test/Spring_(framework)", 1],
    ["https://docs.example.test/caf%C3%A9", 2],
    ["https://other.example.test/a(b)", 3],
    ["https://third.example.test/path", 4],
  ]);
  assert.equal(citationNumberFor(numbers, "https://docs.example.test/café/"), 2);
  assert.equal(citationNumberFor(numbers, "https://ignored.example.test/inline"), undefined);
});

test("KB references resolve to one navigable citation number", () => {
  const canonicalUrl = "https://academy.example.test/lessons/lora";
  const kbSource = source({
    docId: "lora-doc",
    path: "raw/docs/peft/lora.md",
    url: canonicalUrl,
  });
  const message = assistantMessage([
    {
      type: "text",
      text: [
        "[By id](kb://doc/lora-doc)",
        "[By relative path](./raw/docs/peft/lora.md)",
        "[By full path](data/kb/raw/docs/peft/lora.md)",
        "[Direct URL](https://academy.example.test/lessons/lora#examples)",
        "[Missing KB file](raw/docs/peft/missing.md)",
      ].join(" "),
    },
    { type: "data-source", data: kbSource },
  ]);

  const resolutions = buildCitationResolutions(message);
  assert.deepEqual([...resolutions.entries()], [
    ["kb://doc/lora-doc", canonicalUrl],
    ["raw/docs/peft/lora.md", canonicalUrl],
  ]);
  assert.equal(
    resolveCitationHref("data/kb/raw/docs/peft/lora.md", resolutions),
    canonicalUrl,
  );
  assert.equal(resolveCitationHref("raw/docs/peft/missing.md", resolutions), undefined);

  const numbers = buildCitationNumbers(message, resolutions);
  assert.deepEqual([...numbers.entries()], [[canonicalUrl, 1]]);
  assert.equal(citationNumberFor(numbers, `${canonicalUrl}/#details`), 1);
});

test("server-provided sources preserve order and deduplicate normalized URLs", () => {
  const longTitle = "T".repeat(90);
  const message = assistantMessage([
    {
      type: "data-source",
      data: source({
        title: longTitle,
        url: "https://docs.example.test/guide/#intro",
        group: "docs",
      }),
    },
    {
      type: "data-source",
      data: source({
        title: "Duplicate",
        url: "https://docs.example.test/guide/",
        group: "docs",
      }),
    },
    {
      type: "data-source",
      data: source({
        title: "   —   ",
        url: "https://www.search.example.test/result",
        sourceLabel: "Search result",
        group: "web",
      }),
    },
    {
      type: "data-source",
      data: source({
        title: "Course lesson",
        url: "https://academy.example.test/course/lesson",
        sourceLabel: "Agent Engineering",
        group: "courses",
      }),
    },
    { type: "data-source", data: source({ url: "   " }) },
  ]);

  assert.deepEqual(getMessageCitations(message), [
    {
      label: longTitle.slice(0, 80),
      url: "https://docs.example.test/guide/#intro",
      kind: "doc",
      sublabel: "Example docs",
    },
    {
      label: "search.example.test",
      url: "https://www.search.example.test/result",
      kind: "web",
      sublabel: "search.example.test",
    },
    {
      label: "Course lesson",
      url: "https://academy.example.test/course/lesson",
      kind: "course",
      sublabel: "Agent Engineering",
    },
  ]);
});

test("ordered blocks retain provider order and ignored parts do not split blocks", () => {
  const reasoningOne = { type: "reasoning", text: "First thought" };
  const reasoningTwo = { type: "reasoning", text: "Second fragment" };
  const tool = {
    type: "tool-retrieve_tutor_context",
    toolCallId: "call-1",
    state: "output-available",
  };
  const reasoningThree = { type: "reasoning", text: "After the tool" };
  const textOne = { type: "text", text: "Answer one" };
  const textTwo = { type: "text", text: "Answer two" };
  const finalTool = {
    type: "tool-web_search",
    toolCallId: "call-2",
    state: "input-available",
  };
  const message = assistantMessage([
    { type: "data-source", data: source() },
    reasoningOne,
    { type: "data-context_stats", data: { tokens: 10 } },
    reasoningTwo,
    tool,
    reasoningThree,
    textOne,
    { type: "data-source", data: source() },
    textTwo,
    finalTool,
  ]);

  assert.deepEqual(getOrderedMessageBlocks(message), [
    {
      key: "assistant-1-activity-0",
      kind: "activity",
      parts: [reasoningOne, reasoningTwo, tool, reasoningThree],
    },
    {
      key: "assistant-1-text-1",
      kind: "text",
      parts: [textOne, textTwo],
    },
    {
      key: "assistant-1-activity-2",
      kind: "activity",
      parts: [finalTool],
    },
  ]);
});

test("activity items merge reasoning fragments but split them at tool calls", () => {
  const tool = {
    type: "tool-run_kb_command",
    toolCallId: "call-1",
    state: "output-available",
  };
  const parts: TutorMessagePart[] = [
    { type: "reasoning", text: "  First fragment  " },
    { type: "data-source", data: source() },
    { type: "reasoning", text: "" },
    { type: "reasoning", text: "Second fragment" },
    tool,
    { type: "data-context_stats", data: {} },
    { type: "reasoning", text: "Final thought" },
  ];

  assert.deepEqual(buildActivityItems(parts), [
    {
      kind: "reasoning",
      key: "r-0",
      text: "First fragment\n\nSecond fragment",
    },
    { kind: "tool", key: "t-1-call-1", part: tool },
    { kind: "reasoning", key: "r-2", text: "Final thought" },
  ]);
});

test("renderability ignores empty stream placeholders but retains real activity", () => {
  assert.equal(hasRenderableContent(assistantMessage([])), false);
  assert.equal(
    hasRenderableContent(assistantMessage([{ type: "data-source", data: source() }])),
    false,
  );
  assert.equal(hasRenderableContent(assistantMessage([{ type: "text", text: "" }])), false);
  assert.equal(
    hasRenderableContent(assistantMessage([{ type: "reasoning", text: "" }])),
    false,
  );
  assert.equal(
    hasRenderableContent(assistantMessage([{ type: "reasoning", text: "Thinking" }])),
    true,
  );
  assert.equal(
    hasRenderableContent(
      assistantMessage([{ type: "tool-web_search", state: "input-streaming" }]),
    ),
    true,
  );
});

test("message text joins trimmed non-empty text parts without reshaping content inside them", () => {
  const message = assistantMessage([
    { type: "text", text: "  First paragraph\n  " },
    { type: "reasoning", text: "Not user-facing" },
    { type: "text", text: "   " },
    { type: "text", text: "Second paragraph\nwith another line" },
  ]);

  assert.equal(
    getMessageTextContent(message),
    "First paragraph\n\nSecond paragraph\nwith another line",
  );
});
