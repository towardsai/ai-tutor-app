import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { MarkdownBlock } from "@/components/markdown-block";

afterEach(cleanup);

describe("MarkdownBlock links", () => {
  it("renders short numbered links as chips but retains long claim text", () => {
    const url = "https://example.com/reference";
    const longClaim =
      "This linked sentence is a substantive claim that should stay readable in the answer";

    render(
      <MarkdownBlock citationNumbers={new Map([[url, 1]])}>
        {`[Short](${url}) and [${longClaim}](${url})`}
      </MarkdownBlock>,
    );

    const chips = screen.getAllByRole("link", { name: "1" });
    expect(chips).toHaveLength(2);
    expect(chips.every((chip) => chip.getAttribute("href") === url)).toBe(true);
    expect(screen.queryByText("Short")).toBeNull();
    expect(screen.getByText(longClaim)).toBeTruthy();
  });

  it("allows external HTTP and email links while leaving unsafe and relative links inert", () => {
    render(
      <MarkdownBlock>
        {
          "[Website](https://safe.example/path) [Email](mailto:tutor@example.com) [Unsafe](javascript:alert(1)) [Local](raw/docs/private.md)"
        }
      </MarkdownBlock>,
    );

    const website = screen.getByRole("link", { name: "Website" });
    expect(website.getAttribute("href")).toBe("https://safe.example/path");
    expect(website.getAttribute("target")).toBe("_blank");
    expect(website.getAttribute("rel")).toBe("noreferrer");

    const email = screen.getByRole("link", { name: "Email" });
    expect(email.getAttribute("href")).toBe("mailto:tutor@example.com");
    expect(email.getAttribute("target")).toBe("_blank");
    expect(email.getAttribute("rel")).toBe("noreferrer");

    expect(screen.getByText("Unsafe").closest("a")).toBeNull();
    expect(screen.getByText("Local").closest("a")).toBeNull();
  });

  it("resolves KB references before rendering and gives equivalent references the same number", () => {
    const resolvedUrl = "https://docs.example.com/guide";
    const citationNumbers = new Map([[resolvedUrl, 7]]);
    const citationResolutions = new Map([
      ["raw/docs/guide.md", resolvedUrl],
      ["kb://doc/guide-id", resolvedUrl],
    ]);

    render(
      <MarkdownBlock
        citationNumbers={citationNumbers}
        citationResolutions={citationResolutions}
      >
        {"[Raw path](raw/docs/guide.md) [Document ID](kb://doc/guide-id)"}
      </MarkdownBlock>,
    );

    const chips = screen.getAllByRole("link", { name: "7" });
    expect(chips).toHaveLength(2);
    expect(chips.map((chip) => chip.getAttribute("href"))).toEqual([
      resolvedUrl,
      resolvedUrl,
    ]);
    expect(chips.map((chip) => chip.getAttribute("title"))).toEqual([
      "Raw path",
      "Document ID",
    ]);
  });
});
