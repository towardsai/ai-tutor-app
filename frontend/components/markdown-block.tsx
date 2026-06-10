import type { ReactNode } from "react";
import ReactMarkdown, { defaultUrlTransform } from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import remarkGfm from "remark-gfm";
import {
  citationNumberFor,
  isHttpUrl,
  resolveCitationHref,
} from "@/lib/chat-ui";

type MarkdownBlockProps = {
  children: string;
  className?: string;
  citationNumbers?: Map<string, number>;
  citationResolutions?: Map<string, string>;
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
  citationResolutions,
}: MarkdownBlockProps) {
  return (
    <div className={`markdown-block ${className}`.trim()}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[[rehypeHighlight, { detect: true, ignoreMissing: true }]]}
        // KB references (kb://doc/<id>, raw/... paths) must resolve before the
        // default transform runs: it blanks the kb:// scheme and would leave
        // raw paths as relative hrefs into the app's own origin.
        urlTransform={(url) =>
          resolveCitationHref(url, citationResolutions) ?? defaultUrlTransform(url)
        }
        components={{
          a: ({ children: linkChildren, href, node: _node, ...props }) => {
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
            // An unresolved KB reference arrives here with an empty (kb://
            // blanked by the transform) or relative (raw/...) href; a real
            // anchor would open the app's own origin in a new tab.
            if (!isHttpUrl(href) && !href?.startsWith("mailto:")) {
              return <span>{linkChildren}</span>;
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
