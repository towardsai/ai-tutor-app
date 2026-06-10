import type { ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import remarkGfm from "remark-gfm";
import { citationNumberFor } from "@/lib/chat-ui";

type MarkdownBlockProps = {
  children: string;
  className?: string;
  citationNumbers?: Map<string, number>;
};

// Above this length the link text is a claim the model wrapped in a link, not
// a source title; keep the text as prose and append the chip after it.
const CITATION_TEXT_KEEP_THRESHOLD = 48;

function nodeToText(node: ReactNode): string {
  if (typeof node === "string" || typeof node === "number") {
    return String(node);
  }
  if (Array.isArray(node)) {
    return node.map(nodeToText).join("");
  }
  if (node && typeof node === "object" && "props" in node) {
    return nodeToText(
      (node as { props: { children?: ReactNode } }).props.children,
    );
  }
  return "";
}

export function MarkdownBlock({
  children,
  className = "",
  citationNumbers,
}: MarkdownBlockProps) {
  return (
    <div className={`markdown-block ${className}`.trim()}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[[rehypeHighlight, { detect: true, ignoreMissing: true }]]}
        components={{
          a: ({ children: linkChildren, href, ...props }) => {
            const number = citationNumberFor(citationNumbers, href);
            if (number !== undefined) {
              const label = nodeToText(linkChildren).trim();
              const keepText = label.length > CITATION_TEXT_KEEP_THRESHOLD;
              return (
                <>
                  {keepText ? <span>{label}</span> : null}
                  <a
                    href={href}
                    target="_blank"
                    rel="noreferrer"
                    title={label}
                    className="citation-chip"
                  >
                    {number}
                  </a>
                </>
              );
            }
            return (
              <a {...props} href={href} target="_blank" rel="noreferrer">
                {linkChildren}
              </a>
            );
          },
        }}
      >
        {children}
      </ReactMarkdown>
    </div>
  );
}
