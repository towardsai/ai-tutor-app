import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

const fetchToolsMock = vi.hoisted(() => vi.fn());

const chatHarness = vi.hoisted(() => ({
  messages: [] as Array<{
    id: string;
    role: "assistant" | "user";
    parts: Array<{ type: "text"; text: string }>;
  }>,
  status: "ready" as "error" | "ready" | "streaming" | "submitted",
  error: null as Error | null,
  onData: undefined as ((part: unknown) => void) | undefined,
  sendMessage: vi.fn(),
  setMessages: vi.fn(),
  regenerate: vi.fn(),
  stop: vi.fn(),
  clearError: vi.fn(),
  transportOptions: undefined as unknown,
}));

vi.mock("@/lib/api", async (importOriginal) => {
  const original = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...original,
    fetchTools: fetchToolsMock,
    getApiBaseUrl: () => "http://localhost:8000",
  };
});

vi.mock("@ai-sdk/react", async () => {
  const { useReducer } = await import("react");
  return {
    useChat: (options: { onData?: (part: unknown) => void }) => {
      const [, forceRender] = useReducer((version: number) => version + 1, 0);
      chatHarness.onData = options.onData;
      return {
        messages: chatHarness.messages,
        sendMessage: chatHarness.sendMessage,
        setMessages: (next: typeof chatHarness.messages) => {
          chatHarness.setMessages(next);
          forceRender();
        },
        regenerate: chatHarness.regenerate,
        stop: chatHarness.stop,
        status: chatHarness.status,
        error: chatHarness.error,
        clearError: chatHarness.clearError,
      };
    },
  };
});

vi.mock("ai", () => ({
  DefaultChatTransport: class DefaultChatTransport {
    constructor(options: unknown) {
      chatHarness.transportOptions = options;
    }
  },
}));

vi.mock("@/components/source-sidebar", () => ({
  SourceSidebar: ({
    onNewChat,
    onToggleSource,
    onToggleTool,
    sourceError,
  }: {
    onNewChat: () => void;
    onToggleSource: (key: string) => void;
    onToggleTool: (key: string) => void;
    sourceError: string | null;
  }) => (
    <aside>
      <button type="button" onClick={onNewChat}>
        New chat
      </button>
      <button type="button" onClick={() => onToggleSource("course-default")}>
        Toggle default course
      </button>
      <button type="button" onClick={() => onToggleSource("docs-optional")}>
        Toggle optional docs
      </button>
      <button type="button" onClick={() => onToggleTool("web_search")}>
        Toggle web search
      </button>
      {sourceError ? <p role="alert">{sourceError}</p> : null}
    </aside>
  ),
}));

vi.mock("@/components/chat-message", () => ({
  ChatMessage: ({
    message,
    isEditing,
    editDraft,
    onAssistantRedo,
    onEditChange,
    onEditSave,
    onUserEdit,
  }: {
    message: {
      id: string;
      role: "assistant" | "user";
      parts: Array<{ type: string; text?: string }>;
    };
    isEditing: boolean;
    editDraft: string;
    onAssistantRedo: (messageId: string) => void;
    onEditChange: (value: string) => void;
    onEditSave: (messageId: string) => void;
    onUserEdit: (messageId: string) => void;
  }) => {
    const text = message.parts
      .filter((part) => part.type === "text")
      .map((part) => part.text ?? "")
      .join("");

    return (
      <div data-testid={`message-${message.id}`}>
        {isEditing ? (
          <>
            <textarea
              aria-label={`Edit ${message.id}`}
              value={editDraft}
              onChange={(event) => onEditChange(event.target.value)}
            />
            <button type="button" onClick={() => onEditSave(message.id)}>
              Save {message.id}
            </button>
          </>
        ) : (
          <span>{text}</span>
        )}
        {message.role === "user" && !isEditing ? (
          <button type="button" onClick={() => onUserEdit(message.id)}>
            Edit {message.id}
          </button>
        ) : null}
        {message.role === "assistant" ? (
          <button type="button" onClick={() => onAssistantRedo(message.id)}>
            Redo {message.id}
          </button>
        ) : null}
      </div>
    );
  },
}));

import { ChatShell } from "@/components/chat-shell";

const DEFAULT_MODEL = "deepseek:deepseek-v4-flash";
const CLAUDE_MODEL = "anthropic:claude-haiku-4-5";

const toolsResponse = {
  model: DEFAULT_MODEL,
  availableModels: [
    { id: DEFAULT_MODEL, label: "DeepSeek V4 Flash" },
    { id: CLAUDE_MODEL, label: "Claude Haiku 4.5" },
  ],
  tools: [
    {
      kind: "configurable" as const,
      key: "retrieval" as const,
      label: "Knowledge base",
      active: true,
      sources: [
        {
          key: "course-default",
          label: "Default course",
          shortLabel: "Course",
          group: "courses" as const,
          selectedByDefault: true,
        },
        {
          key: "docs-optional",
          label: "Optional docs",
          shortLabel: "Docs",
          group: "docs" as const,
          selectedByDefault: false,
        },
      ],
    },
    {
      kind: "toggle" as const,
      key: "web_search",
      label: "Web search",
      active: true,
    },
    {
      kind: "toggle" as const,
      key: "url_context",
      label: "URL context",
      active: false,
    },
  ],
};

function message(
  id: string,
  role: "assistant" | "user",
  text: string,
) {
  return { id, role, parts: [{ type: "text" as const, text }] };
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, reject, resolve };
}

function sentBody(callIndex = 0) {
  return chatHarness.sendMessage.mock.calls[callIndex]?.[1]?.body;
}

beforeEach(() => {
  chatHarness.messages = [];
  chatHarness.status = "ready";
  chatHarness.error = null;
  chatHarness.onData = undefined;
  chatHarness.transportOptions = undefined;
  chatHarness.sendMessage.mockReset();
  chatHarness.regenerate.mockReset();
  chatHarness.stop.mockReset();
  chatHarness.clearError.mockReset();
  chatHarness.setMessages.mockReset();
  chatHarness.setMessages.mockImplementation(
    (next: typeof chatHarness.messages) => {
      chatHarness.messages = next;
    },
  );
  fetchToolsMock.mockReset();
  fetchToolsMock.mockResolvedValue(toolsResponse);
});

describe("ChatShell request lifecycle", () => {
  it("keeps Send disabled until the initial tool and model defaults are ready", async () => {
    const pendingTools = deferred<typeof toolsResponse>();
    fetchToolsMock.mockReturnValueOnce(pendingTools.promise);
    const user = userEvent.setup();

    render(<ChatShell />);
    await user.type(screen.getByRole("textbox"), "What is RAG?");

    expect(
      (screen.getByRole("button", { name: "Send message" }) as HTMLButtonElement)
        .disabled,
    ).toBe(true);
    await user.keyboard("{Enter}");
    const suggestion = screen
      .getByRole("heading", { name: "Ask your AI tutor" })
      .parentElement?.querySelector("button");
    expect(suggestion).not.toBeNull();
    await user.click(suggestion!);
    expect(chatHarness.sendMessage).not.toHaveBeenCalled();

    await act(async () => pendingTools.resolve(toolsResponse));

    await waitFor(() =>
      expect(
        (screen.getByRole("button", {
          name: "Send message",
        }) as HTMLButtonElement).disabled,
      ).toBe(false),
    );
  });

  it("submits the loaded model, default sources, and active tools", async () => {
    const user = userEvent.setup();
    render(<ChatShell />);
    await screen.findByRole("button", { name: "DeepSeek V4 Flash" });

    await user.type(screen.getByRole("textbox"), "  Explain agents  ");
    await user.click(screen.getByRole("button", { name: "Send message" }));

    expect(chatHarness.sendMessage).toHaveBeenCalledWith(
      { text: "Explain agents" },
      {
        body: {
          sourceKeys: ["course-default"],
          enabledTools: ["web_search"],
          includeReasoning: true,
          threadId: "",
          model: DEFAULT_MODEL,
        },
      },
    );
  });

  it("sends the user's current source and tool selections, including no sources", async () => {
    const user = userEvent.setup();
    render(<ChatShell />);
    await screen.findByRole("button", { name: "DeepSeek V4 Flash" });

    await user.click(screen.getByRole("button", { name: "Toggle optional docs" }));
    await user.click(screen.getByRole("button", { name: "Toggle web search" }));
    await user.type(screen.getByRole("textbox"), "Selected sources");
    await user.click(screen.getByRole("button", { name: "Send message" }));

    expect(sentBody()).toMatchObject({
      sourceKeys: ["course-default", "docs-optional"],
      enabledTools: [],
    });

    await user.click(screen.getByRole("button", { name: "Toggle default course" }));
    await user.click(screen.getByRole("button", { name: "Toggle optional docs" }));
    await user.type(screen.getByRole("textbox"), "No knowledge base");
    await user.click(screen.getByRole("button", { name: "Send message" }));

    expect(sentBody(1)).toMatchObject({ sourceKeys: [], enabledTools: [] });
  });

  it("adopts data-thread immediately and sends it with the next turn", async () => {
    const user = userEvent.setup();
    render(<ChatShell />);
    await screen.findByRole("button", { name: "DeepSeek V4 Flash" });

    await user.type(screen.getByRole("textbox"), "First question");
    await user.click(screen.getByRole("button", { name: "Send message" }));
    act(() => {
      chatHarness.onData?.({
        type: "data-thread",
        data: { threadId: "thread-from-server" },
      });
    });
    await user.type(screen.getByRole("textbox"), "Follow up");
    await user.click(screen.getByRole("button", { name: "Send message" }));

    expect(sentBody(1)).toMatchObject({ threadId: "thread-from-server" });

    act(() => {
      chatHarness.onData?.({
        type: "data-thread",
        data: { threadId: "replacement-thread" },
      });
    });
    await user.type(screen.getByRole("textbox"), "One more turn");
    await user.click(screen.getByRole("button", { name: "Send message" }));

    expect(sentBody(2)).toMatchObject({ threadId: "replacement-thread" });
  });

  it("locks model switching for an existing conversation and unlocks on New chat", async () => {
    chatHarness.messages = [message("u1", "user", "First question")];
    const user = userEvent.setup();
    render(<ChatShell />);

    const modelPicker = await screen.findByRole("button", {
      name: "DeepSeek V4 Flash",
    });
    expect((modelPicker as HTMLButtonElement).disabled).toBe(true);

    await user.click(screen.getByRole("button", { name: "New chat" }));

    expect(chatHarness.setMessages).toHaveBeenCalledWith([]);
    await waitFor(() =>
      expect((modelPicker as HTMLButtonElement).disabled).toBe(false),
    );
  });

  it("clears a failed model-specific tools request after a successful retry", async () => {
    fetchToolsMock
      .mockResolvedValueOnce(toolsResponse)
      .mockRejectedValueOnce(new Error("Unable to load Claude tools"))
      .mockResolvedValueOnce(toolsResponse);
    const user = userEvent.setup();
    render(<ChatShell />);

    await user.click(
      await screen.findByRole("button", { name: "DeepSeek V4 Flash" }),
    );
    await user.click(screen.getByRole("option", { name: "Claude Haiku 4.5" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Unable to load Claude tools",
    );

    await user.click(screen.getByRole("button", { name: "Claude Haiku 4.5" }));
    await user.click(screen.getByRole("option", { name: "DeepSeek V4 Flash" }));

    await waitFor(() => expect(fetchToolsMock).toHaveBeenCalledTimes(3));
    await waitFor(() => expect(screen.queryByRole("alert")).toBeNull());
  });

  it("pauses submissions while model-specific tools are reloading", async () => {
    const pendingClaudeTools = deferred<typeof toolsResponse>();
    fetchToolsMock
      .mockResolvedValueOnce(toolsResponse)
      .mockReturnValueOnce(pendingClaudeTools.promise);
    const user = userEvent.setup();
    render(<ChatShell />);

    await user.click(
      await screen.findByRole("button", { name: "DeepSeek V4 Flash" }),
    );
    await user.click(screen.getByRole("option", { name: "Claude Haiku 4.5" }));
    await waitFor(() => expect(fetchToolsMock).toHaveBeenCalledTimes(2));

    await user.type(screen.getByRole("textbox"), "Compare the providers");
    await waitFor(() =>
      expect(
        (screen.getByRole("button", {
          name: "Send message",
        }) as HTMLButtonElement).disabled,
      ).toBe(true),
    );
    await user.keyboard("{Enter}");
    expect(chatHarness.sendMessage).not.toHaveBeenCalled();

    await act(async () => pendingClaudeTools.resolve(toolsResponse));
    await waitFor(() =>
      expect(
        (screen.getByRole("button", {
          name: "Send message",
        }) as HTMLButtonElement).disabled,
      ).toBe(false),
    );
    await user.click(screen.getByRole("button", { name: "Send message" }));

    expect(sentBody()).toMatchObject({ model: CLAUDE_MODEL });
  });

  it("keeps the registry ready when the active model is selected again", async () => {
    const user = userEvent.setup();
    render(<ChatShell />);

    await user.click(
      await screen.findByRole("button", { name: "DeepSeek V4 Flash" }),
    );
    await user.click(screen.getByRole("option", { name: "DeepSeek V4 Flash" }));
    expect(fetchToolsMock).toHaveBeenCalledOnce();

    await user.type(screen.getByRole("textbox"), "Still ready");
    const sendButton = screen.getByRole("button", { name: "Send message" });
    expect((sendButton as HTMLButtonElement).disabled).toBe(false);
    await user.click(sendButton);
    expect(chatHarness.sendMessage).toHaveBeenCalledOnce();
  });

  it("clears a previous stream error and allows a retry", async () => {
    chatHarness.status = "error";
    chatHarness.error = new Error("The previous request failed");
    const user = userEvent.setup();
    render(<ChatShell />);
    await screen.findByRole("button", { name: "DeepSeek V4 Flash" });

    expect(screen.getByText("The previous request failed")).toBeVisible();
    await user.type(screen.getByRole("textbox"), "Try again");
    await user.click(screen.getByRole("button", { name: "Send message" }));

    expect(chatHarness.clearError).toHaveBeenCalled();
    expect(chatHarness.sendMessage).toHaveBeenCalledWith(
      { text: "Try again" },
      expect.objectContaining({ body: expect.any(Object) }),
    );
  });

  it.each(["submitted", "streaming"] as const)(
    "offers Stop and aborts while the request is %s",
    async (status) => {
      chatHarness.status = status;
      const user = userEvent.setup();
      render(<ChatShell />);

      await user.click(screen.getByRole("button", { name: "Stop generating" }));

      expect(chatHarness.stop).toHaveBeenCalledTimes(1);
      expect(chatHarness.sendMessage).not.toHaveBeenCalled();
    },
  );

  it("passes the selected message boundary to edit and regenerate", async () => {
    chatHarness.messages = [
      message("u1", "user", "Original question"),
      message("a1", "assistant", "First answer"),
      message("u2", "user", "Later question"),
      message("a2", "assistant", "Later answer"),
    ];
    const user = userEvent.setup();
    render(<ChatShell />);
    await screen.findByRole("button", { name: "DeepSeek V4 Flash" });

    await user.click(screen.getByRole("button", { name: "Redo a1" }));
    expect(chatHarness.regenerate).toHaveBeenCalledWith({
      messageId: "a1",
      body: expect.objectContaining({
        model: DEFAULT_MODEL,
        threadId: "",
      }),
    });

    await user.click(screen.getByRole("button", { name: "Edit u1" }));
    const editor = screen.getByRole("textbox", { name: "Edit u1" });
    fireEvent.change(editor, { target: { value: "Rewritten question" } });
    await user.click(screen.getByRole("button", { name: "Save u1" }));

    expect(chatHarness.sendMessage).toHaveBeenCalledWith(
      { text: "Rewritten question", messageId: "u1" },
      {
        body: expect.objectContaining({
          model: DEFAULT_MODEL,
          threadId: "",
        }),
      },
    );
  });

  it("clears an adopted thread when starting a new chat", async () => {
    const user = userEvent.setup();
    render(<ChatShell />);
    await screen.findByRole("button", { name: "DeepSeek V4 Flash" });

    await user.type(screen.getByRole("textbox"), "First question");
    await user.click(screen.getByRole("button", { name: "Send message" }));
    act(() => {
      chatHarness.onData?.({
        type: "data-thread",
        data: { threadId: "old-thread" },
      });
    });
    await user.click(screen.getByRole("button", { name: "New chat" }));
    await user.type(screen.getByRole("textbox"), "Fresh question");
    await user.click(screen.getByRole("button", { name: "Send message" }));

    expect(sentBody(1)).toMatchObject({ threadId: "" });
  });

  it("ignores a late thread event from a stream abandoned by New chat", async () => {
    chatHarness.status = "streaming";
    const user = userEvent.setup();
    render(<ChatShell />);
    await waitFor(() => expect(fetchToolsMock).toHaveBeenCalledOnce());

    await user.click(screen.getByRole("button", { name: "New chat" }));
    expect(chatHarness.stop).toHaveBeenCalledOnce();

    act(() => {
      chatHarness.onData?.({
        type: "data-thread",
        data: { threadId: "late-abandoned-thread" },
      });
    });

    chatHarness.status = "ready";
    await user.type(screen.getByRole("textbox"), "Fresh after abort");
    await user.click(screen.getByRole("button", { name: "Send message" }));

    expect(sentBody()).toMatchObject({ threadId: "" });
  });
});
