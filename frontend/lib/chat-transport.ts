import type { PrepareSendMessagesRequest, UIMessage } from "ai";

/**
 * Strip each UIMessage down to its text parts before sending the transcript
 * back to the server.
 *
 * The AI SDK stores reasoning, tool outputs, and source cards inside each
 * UIMessage. Sending those parts back on every turn can make a normal follow-up
 * exceed the API's request-size limits, even though the backend deliberately
 * ignores them when it reconstructs visible history. Preserve text parts
 * verbatim so checkpoint/transcript comparison keeps its exact whitespace.
 *
 * This is plain filtering, not context engineering: summarization and
 * tool-output capping happen server-side (see the app/memory_presets.py
 * middlewares), never in the client.
 */
export function toTextOnlyMessages(messages: UIMessage[]): UIMessage[] {
  return messages.map((message) => ({
    id: message.id,
    role: message.role,
    parts: message.parts.flatMap((part) =>
      part.type === "text"
        ? [
            {
              type: "text" as const,
              text: part.text,
            },
          ]
        : [],
    ),
  }));
}

export const prepareTutorChatRequest: PrepareSendMessagesRequest<UIMessage> = ({
  id,
  messages,
  body,
  trigger,
  messageId,
}) => ({
  body: {
    ...body,
    id,
    messages: toTextOnlyMessages(messages),
    trigger,
    messageId,
  },
});
