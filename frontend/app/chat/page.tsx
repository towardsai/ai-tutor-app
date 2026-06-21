import { ChatShell } from "@/components/chat-shell";

// data-chat-root marks this route so globals.css can lock the body to the
// viewport here (full-screen chat) while the showcase pages scroll normally.
// display: contents keeps ChatShell a direct layout child of the body.
export default function ChatPage() {
  return (
    <div data-chat-root style={{ display: "contents" }}>
      <ChatShell />
    </div>
  );
}
