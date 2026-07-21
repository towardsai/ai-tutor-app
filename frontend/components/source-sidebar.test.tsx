import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { SourceSidebar } from "@/components/source-sidebar";
import type { TutorSource, TutorTool } from "@/lib/api";

afterEach(cleanup);

const sources: TutorSource[] = [
  {
    key: "course-a",
    label: "Course A",
    shortLabel: "Course A",
    description: "Learn practical agent engineering.",
    infoUrl: "https://academy.example.com/course-a",
    group: "courses",
    selectedByDefault: true,
  },
  {
    key: "docs-a",
    label: "Documentation A",
    shortLabel: "Docs A",
    description: "Reference documentation for the library.",
    infoUrl: "https://docs.example.com/",
    group: "docs",
    selectedByDefault: false,
    version: "v1.2",
    indexedAt: "2026-07-01",
  },
];

const tools: TutorTool[] = [
  {
    kind: "configurable",
    key: "retrieval",
    label: "Knowledge base",
    active: true,
    sources,
  },
  {
    kind: "toggle",
    key: "web_search",
    label: "Web search",
    active: false,
  },
];

function renderSidebar() {
  const handlers = {
    onNewChat: vi.fn(),
    onToggleSource: vi.fn(),
    onToggleTool: vi.fn(),
  };
  render(
    <SourceSidebar
      {...handlers}
      selectedSourceKeys={["course-a"]}
      enabledToolKeys={["web_search"]}
      sourceError={null}
      tools={tools}
    />,
  );
  return handlers;
}

describe("SourceSidebar", () => {
  it("reports active tools and routes new-chat, source, and tool actions", () => {
    const handlers = renderSidebar();

    expect(screen.getByText("2 of 2 on")).toBeTruthy();
    expect(
      screen.getByRole("button", { name: "Course A" }).getAttribute(
        "aria-pressed",
      ),
    ).toBe("true");
    expect(
      screen.getByRole("button", { name: "Docs A" }).getAttribute(
        "aria-pressed",
      ),
    ).toBe("false");
    expect(
      screen.getByRole("button", { name: "Web search" }).getAttribute(
        "aria-pressed",
      ),
    ).toBe("true");

    fireEvent.click(screen.getByRole("button", { name: "New chat" }));
    fireEvent.click(screen.getByRole("button", { name: "Docs A" }));
    fireEvent.click(screen.getByRole("button", { name: "Web search" }));

    expect(handlers.onNewChat).toHaveBeenCalledOnce();
    expect(handlers.onToggleSource).toHaveBeenCalledWith("docs-a");
    expect(handlers.onToggleTool).toHaveBeenCalledWith("web_search");
  });

  it("shows registry metadata in a portaled source popover and closes it with Escape", () => {
    renderSidebar();

    const infoButton = screen.getByRole("button", {
      name: "About Documentation A",
    });
    expect(infoButton.getAttribute("aria-expanded")).toBe("false");

    fireEvent.click(infoButton);

    const dialog = screen.getByRole("dialog");
    expect(dialog.textContent).toContain(
      "Reference documentation for the library.",
    );
    expect(dialog.textContent).toContain("v1.2 · indexed Jul 2026");
    const docsLink = screen.getByRole("link", { name: /View docs/ });
    expect(docsLink.getAttribute("href")).toBe("https://docs.example.com/");
    expect(docsLink.getAttribute("target")).toBe("_blank");

    fireEvent.keyDown(document, { key: "Escape" });
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(infoButton.getAttribute("aria-expanded")).toBe("false");
  });
});
