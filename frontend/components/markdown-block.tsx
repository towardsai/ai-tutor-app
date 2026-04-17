import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import remarkGfm from "remark-gfm";

type MarkdownBlockProps = {
  children: string;
  className?: string;
};

export function MarkdownBlock({
  children,
  className = "",
}: MarkdownBlockProps) {
  return (
    <div className={`markdown-block ${className}`.trim()}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[[rehypeHighlight, { detect: true, ignoreMissing: true }]]}
        components={{
          a: ({ ...props }) => (
            <a
              {...props}
              target="_blank"
              rel="noreferrer"
            />
          ),
        }}
      >
        {children}
      </ReactMarkdown>
    </div>
  );
}
