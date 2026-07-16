import type { PrepareSendMessagesRequest, UIMessage } from "ai";

/**
 * Keep only the transcript data the backend consumes.
 *
 * The AI SDK stores reasoning, tool outputs, and source cards inside each
 * UIMessage. Sending those parts back on every turn can make a normal follow-up
 * exceed the API's request-size limits, even though the backend deliberately
 * ignores them when it reconstructs visible history. Preserve text parts
 * verbatim so checkpoint/transcript comparison keeps its exact whitespace.
 */
export function compactChatMessages(messages: UIMessage[]): UIMessage[] {
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
    messages: compactChatMessages(messages),
    trigger,
    messageId,
  },
});
