import assert from "node:assert/strict";
import test from "node:test";
import { fetchTools, getApiBaseUrl, type ToolsResponse } from "./api.ts";

const ENV_NAME = "NEXT_PUBLIC_AI_TUTOR_API_BASE_URL";

function restoreEnvironment(previous: string | undefined) {
  if (previous === undefined) {
    delete process.env[ENV_NAME];
  } else {
    process.env[ENV_NAME] = previous;
  }
}

test("getApiBaseUrl prefers configured URLs and removes trailing slashes", () => {
  const previous = process.env[ENV_NAME];
  try {
    process.env[ENV_NAME] = "https://api.example.test/tutor///";
    assert.equal(getApiBaseUrl(), "https://api.example.test/tutor");
  } finally {
    restoreEnvironment(previous);
  }
});

test("getApiBaseUrl uses the browser origin, then the server fallback", () => {
  const previousEnv = process.env[ENV_NAME];
  const previousWindow = Object.getOwnPropertyDescriptor(globalThis, "window");
  try {
    delete process.env[ENV_NAME];
    Object.defineProperty(globalThis, "window", {
      configurable: true,
      value: { location: { origin: "https://browser.example.test" } },
    });
    assert.equal(getApiBaseUrl(), "https://browser.example.test");

    delete (globalThis as { window?: unknown }).window;
    assert.equal(getApiBaseUrl(), "http://127.0.0.1:8000");
  } finally {
    restoreEnvironment(previousEnv);
    if (previousWindow) {
      Object.defineProperty(globalThis, "window", previousWindow);
    } else {
      delete (globalThis as { window?: unknown }).window;
    }
  }
});

test("fetchTools sends the model query and AbortSignal and returns parsed JSON", async () => {
  const previousEnv = process.env[ENV_NAME];
  const previousFetch = globalThis.fetch;
  const calls: Array<{ url: string; init: RequestInit | undefined }> = [];
  const expected: ToolsResponse = {
    model: "deepseek:deepseek-v4-flash",
    availableModels: [
      { id: "deepseek:deepseek-v4-flash", label: "DeepSeek V4 Flash" },
    ],
    tools: [],
  };
  const controller = new AbortController();

  try {
    process.env[ENV_NAME] = "https://api.example.test/base/";
    globalThis.fetch = (async (input, init) => {
      calls.push({ url: String(input), init });
      return new Response(JSON.stringify(expected), {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    }) as typeof fetch;

    const actual = await fetchTools(
      controller.signal,
      "deepseek:deepseek-v4-flash",
    );

    assert.deepEqual(actual, expected);
    assert.equal(calls.length, 1);
    assert.equal(
      calls[0]?.url,
      "https://api.example.test/base/api/tools?model=deepseek%3Adeepseek-v4-flash",
    );
    assert.equal(calls[0]?.init?.method, "GET");
    assert.equal(calls[0]?.init?.signal, controller.signal);
  } finally {
    restoreEnvironment(previousEnv);
    globalThis.fetch = previousFetch;
  }
});

test("fetchTools omits an empty model and reports non-success status codes", async () => {
  const previousEnv = process.env[ENV_NAME];
  const previousFetch = globalThis.fetch;
  let requestedUrl = "";

  try {
    process.env[ENV_NAME] = "https://api.example.test";
    globalThis.fetch = (async (input) => {
      requestedUrl = String(input);
      return new Response("unavailable", { status: 503 });
    }) as typeof fetch;

    await assert.rejects(fetchTools(undefined, ""), {
      name: "Error",
      message: "Unable to load tools (503)",
    });
    assert.equal(requestedUrl, "https://api.example.test/api/tools");
  } finally {
    restoreEnvironment(previousEnv);
    globalThis.fetch = previousFetch;
  }
});
