import pdb

import gradio as gr
import logfire
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.llms import MessageRole
from llama_index.core.memory import ChatSummaryMemoryBuffer
from llama_index.core.tools import RetrieverTool, ToolMetadata
from llama_index.core.vector_stores import (FilterCondition, FilterOperator,
                                            MetadataFilter, MetadataFilters)
from llama_index.llms.openai import OpenAI

from .custom_retriever import CustomRetriever
from .prompts import system_message_openai_agent
from .setup import (AVAILABLE_SOURCES, AVAILABLE_SOURCES_UI, CONCURRENCY_COUNT,
                    custom_retriever_all_sources)


def update_query_engine_tools(selected_sources) -> list[RetrieverTool]:
    tools = []
    source_mapping: dict[str, tuple[CustomRetriever, str, str]] = {
        "All Sources": (
            custom_retriever_all_sources,
            "all_sources_info",
            """Useful tool that contains general information about the field of AI.""",
        ),
    }

    for source in selected_sources:
        if source in source_mapping:
            custom_retriever, name, description = source_mapping[source]
            tools.append(
                RetrieverTool(
                    retriever=custom_retriever,
                    metadata=ToolMetadata(
                        name=name,
                        description=description,
                    ),
                )
            )

    return tools


def generate_completion(
    query,
    history,
    sources,
    model,
    memory,
):
    llm = OpenAI(temperature=1, model=model, max_tokens=None)
    client = llm._get_client()
    logfire.instrument_openai(client)

    with logfire.span(f"Running query: {query}"):
        logfire.info(f"User chosen sources: {sources}")

        memory_chat_list = memory.get()

        if len(memory_chat_list) != 0:
            user_index_memory = [
                i
                for i, msg in enumerate(memory_chat_list)
                if msg.role == MessageRole.USER
            ]

            user_index_history = [
                i for i, msg in enumerate(history) if msg["role"] == "user"
            ]

            if len(user_index_memory) > len(user_index_history):
                logfire.warn(f"There are more user messages in memory than in history")
                user_index_to_remove = user_index_memory[len(user_index_history)]
                memory_chat_list = memory_chat_list[:user_index_to_remove]
                memory.set(memory_chat_list)

        logfire.info(f"chat_history: {len(memory.get())} {memory.get()}")
        logfire.info(f"gradio_history: {len(history)} {history}")

        query_engine_tools: list[RetrieverTool] = update_query_engine_tools(
            ["All Sources"]
        )

        filter_list = []
        source_mapping = {
            "Transformers Docs": "transformers",
            "PEFT Docs": "peft",
            "TRL Docs": "trl",
            "LlamaIndex Docs": "llama_index",
            "LangChain Docs": "langchain",
            "OpenAI Cookbooks": "openai_cookbooks",
            "Towards AI Blog": "tai_blog",
            "8 Hour Primer": "8-hour_primer",
            "Advanced LLM Developer": "llm_developer",
            "Python Primer": "python_primer",
        }

        for source in sources:
            if source in source_mapping:
                filter_list.append(
                    MetadataFilter(
                        key="source",
                        operator=FilterOperator.EQ,
                        value=source_mapping[source],
                    )
                )

        filters = MetadataFilters(
            filters=filter_list,
            condition=FilterCondition.OR,
        )
        logfire.info(f"Filters: {filters}")
        query_engine_tools[0].retriever._vector_retriever._filters = filters

        # pdb.set_trace()

        agent = OpenAIAgent.from_tools(
            llm=llm,
            memory=memory,
            tools=query_engine_tools,
            system_prompt=system_message_openai_agent,
        )

        completion = agent.stream_chat(query)

    answer_str = ""
    for token in completion.response_gen:
        answer_str += token
        yield answer_str

    for answer_str in add_sources(answer_str, completion):
        yield answer_str


def add_sources(answer_str, completion):
    if completion is None:
        yield answer_str

    formatted_sources = format_sources(completion)
    if formatted_sources == "":
        yield answer_str

    if formatted_sources != "":
        answer_str += "\n\n" + formatted_sources

    yield answer_str


def format_sources(completion) -> str:
    if len(completion.sources) == 0:
        return ""

    # logfire.info(f"Formatting sources: {completion.sources}")

    display_source_to_ui = {
        src: ui for src, ui in zip(AVAILABLE_SOURCES, AVAILABLE_SOURCES_UI)
    }

    documents_answer_template: str = (
        "📝 Here are the sources I used to answer your question:\n{documents}"
    )
    document_template: str = "[🔗 {source}: {title}]({url}), relevance: {score:2.2f}"
    all_documents = []
    for source in completion.sources:  # looping over list[ToolOutput]
        if isinstance(source.raw_output, Exception):
            logfire.error(f"Error in source output: {source.raw_output}")
            # pdb.set_trace()
            continue

        if not isinstance(source.raw_output, list):
            logfire.warn(f"Unexpected source output type: {type(source.raw_output)}")
            continue
        for src in source.raw_output:  # looping over list[NodeWithScore]
            document = document_template.format(
                title=src.metadata["title"],
                score=src.score,
                source=display_source_to_ui.get(
                    src.metadata["source"], src.metadata["source"]
                ),
                url=src.metadata["url"],
            )
            all_documents.append(document)

    if len(all_documents) == 0:
        return ""
    else:
        documents = "\n".join(all_documents)
        return documents_answer_template.format(documents=documents)


def save_completion(completion, history):
    pass


def vote(data: gr.LikeData):
    pass


accordion = gr.Accordion(label="Customize Sources (Click to expand)", open=False)
sources = gr.CheckboxGroup(
    AVAILABLE_SOURCES_UI,
    label="Sources",
    value=[
        "Advanced LLM Developer",
        "8 Hour Primer",
        "Python Primer",
        "Towards AI Blog",
        "Transformers Docs",
        "PEFT Docs",
        "TRL Docs",
        "LlamaIndex Docs",
        "LangChain Docs",
        "OpenAI Cookbooks",
    ],
    interactive=True,
)
model = gr.Dropdown(
    [
        "gpt-4o-mini",
    ],
    label="Model",
    value="gpt-4o-mini",
    interactive=False,
)

with gr.Blocks(
    title="Towards AI 🤖",
    analytics_enabled=True,
    fill_height=True,
) as demo:

    memory = gr.State(
        lambda: ChatSummaryMemoryBuffer.from_defaults(
            token_limit=120000,
        )
    )
    chatbot = gr.Chatbot(
        type="messages",
        scale=20,
        placeholder="<strong>Towards AI 🤖: A Question-Answering Bot for anything AI-related</strong><br>",
        show_label=False,
        show_copy_button=True,
    )
    chatbot.like(vote, None, None)
    gr.ChatInterface(
        fn=generate_completion,
        type="messages",
        chatbot=chatbot,
        additional_inputs=[sources, model, memory],
        additional_inputs_accordion=accordion,
        # fill_height=True,
        # fill_width=True,
        analytics_enabled=True,
    )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=CONCURRENCY_COUNT)
    demo.launch(server_name="0.0.0.0", server_port=7860, debug=False, share=False)
