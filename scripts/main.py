from __future__ import annotations

import gradio as gr

from .chat_service import ChatRequest, is_google_genai_model, normalize_history, stream_chat
from .gradio_presenter import GradioPresenterState
from .setup import (
    AVAILABLE_SOURCES_UI,
    DEFAULT_MODEL_NAME,
    DEFAULT_SELECTED_SOURCES_UI,
    SOURCE_UI_TO_KEY,
)


async def generate_completion(
    query: str,
    history,
    sources,
    model,
    show_gemini_thoughts,
    thread_id,
):
    request = ChatRequest(
        query=query,
        history=normalize_history(history),
        source_keys=tuple(
            SOURCE_UI_TO_KEY[source] for source in sources if source in SOURCE_UI_TO_KEY
        ),
        model_name=model,
        include_reasoning=bool(show_gemini_thoughts),
        thread_id=thread_id,
    )
    presenter = GradioPresenterState(
        show_activity=bool(show_gemini_thoughts) and is_google_genai_model(model)
    )
    last_emitted = ""

    async for event in stream_chat(request):
        presenter.apply(event)
        current_output = presenter.render()
        if current_output and current_output != last_emitted:
            last_emitted = current_output
            yield current_output, presenter.thread_id

    final_output = presenter.render()
    if final_output and final_output != last_emitted:
        yield final_output, presenter.thread_id


accordion = gr.Accordion(label="Customize Sources (Click to expand)", open=False)
sources = gr.CheckboxGroup(
    AVAILABLE_SOURCES_UI,
    label="Sources",
    value=DEFAULT_SELECTED_SOURCES_UI,
    interactive=True,
)
model = gr.Textbox(
    label="Model (provider:model)",
    value=DEFAULT_MODEL_NAME,
    interactive=False,
    placeholder="openai:gpt-5.4-mini | anthropic:claude-opus-4-6 | google-genai:gemini-3.1-pro-preview",
)
show_gemini_thoughts = gr.Checkbox(
    label="Show Gemini thinking and tool timeline",
    value=True,
    info="Uses Gemini include_thoughts and shows tool activity when supported.",
)
thread_id = gr.Textbox(
    label="Thread ID",
    value="",
    visible=False,
    container=False,
)

with gr.Blocks(
    title="Towards AI 🤖",
    analytics_enabled=True,
    fill_height=True,
) as demo:

    def reset_thread_id():
        return ""

    chatbot = gr.Chatbot(
        scale=20,
        placeholder="<strong>Towards AI 🤖: A Question-Answering Bot for anything AI-related</strong><br>",
        show_label=False,
        buttons=["copy"],
    )
    chatbot.clear(
        reset_thread_id,
        None,
        [thread_id],
        api_visibility="undocumented",
        queue=False,
    )
    chatbot.undo(
        reset_thread_id,
        None,
        [thread_id],
        api_visibility="undocumented",
        queue=False,
    )
    chatbot.retry(
        reset_thread_id,
        None,
        [thread_id],
        api_visibility="undocumented",
        queue=False,
    )
    chatbot.edit(
        reset_thread_id,
        None,
        [thread_id],
        api_visibility="undocumented",
        queue=False,
    )
    gr.ChatInterface(
        fn=generate_completion,
        chatbot=chatbot,
        additional_inputs=[sources, model, show_gemini_thoughts, thread_id],
        additional_outputs=[thread_id],
        additional_inputs_accordion=accordion,
        api_name="chat",
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, debug=False, share=False)
