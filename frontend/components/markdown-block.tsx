import ReactMarkdown from "react-markdown";
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
