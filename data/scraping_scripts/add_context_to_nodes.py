import asyncio
import json
import pdb
import pickle
from typing import Dict, List

import instructor
import logfire
import tiktoken
# from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from jinja2 import Template
from llama_index.core import Document
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm.asyncio import tqdm

load_dotenv(".env")

# logfire.configure()


def create_docs(input_file: str) -> List[Document]:
    with open(input_file, "r") as f:
        documents: list[Document] = []
        for line in f:
            data = json.loads(line)
            documents.append(
                Document(
                    doc_id=data["doc_id"],
                    text=data["content"],
                    metadata={  # type: ignore
                        "url": data["url"],
                        "title": data["name"],
                        "tokens": data["tokens"],
                        "retrieve_doc": data["retrieve_doc"],
                        "source": data["source"],
                    },
                    excluded_llm_metadata_keys=[
                        "title",
                        "tokens",
                        "retrieve_doc",
                        "source",
                    ],
                    excluded_embed_metadata_keys=[
                        "url",
                        "tokens",
                        "retrieve_doc",
                        "source",
                    ],
                )
            )
    return documents


class SituatedContext(BaseModel):
    title: str = Field(..., description="The title of the document.")
    context: str = Field(
        ..., description="The context to situate the chunk within the document."
    )


# client = AsyncInstructor(
#     client=AsyncAnthropic(),
#     create=patch(
#         create=AsyncAnthropic().beta.prompt_caching.messages.create,
#         mode=Mode.ANTHROPIC_TOOLS,
#     ),
#     mode=Mode.ANTHROPIC_TOOLS,
# )
aclient = AsyncOpenAI()
# logfire.instrument_openai(aclient)
client: instructor.AsyncInstructor = instructor.from_openai(aclient)


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
async def situate_context(doc: str, chunk: str) -> str:
    template = Template(
        """
<document>
{{ doc }}
</document>

Here is the chunk we want to situate within the whole document above:

<chunk>
{{ chunk }}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else.
"""
    )

    content = template.render(doc=doc, chunk=chunk)

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=1000,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": content,
            }
        ],
        response_model=SituatedContext,
    )
    return response.context


async def process_chunk(node: TextNode, document_dict: dict) -> TextNode:
    doc_id: str = node.source_node.node_id  # type: ignore
    doc: Document = document_dict[doc_id]

    if doc.metadata["tokens"] > 120_000:
        # Tokenize the document text
        encoding = tiktoken.encoding_for_model("gpt-4o-mini")
        tokens = encoding.encode(doc.get_content())

        # Trim to 120,000 tokens
        trimmed_tokens = tokens[:120_000]

        # Decode back to text
        trimmed_text = encoding.decode(trimmed_tokens)

        # Update the document with trimmed text
        doc = Document(text=trimmed_text, metadata=doc.metadata)
        doc.metadata["tokens"] = 120_000

    context: str = await situate_context(doc.get_content(), node.text)
    node.text = f"{node.text}\n\n{context}"
    return node


async def process(
    documents: List[Document], semaphore_limit: int = 50
) -> List[TextNode]:

    # From the document, we create chunks
    pipeline = IngestionPipeline(
        transformations=[SentenceSplitter(chunk_size=800, chunk_overlap=0)]
    )
    all_nodes: list[TextNode] = pipeline.run(documents=documents, show_progress=True)
    print(f"Number of nodes: {len(all_nodes)}")

    document_dict: dict[str, Document] = {doc.doc_id: doc for doc in documents}

    semaphore = asyncio.Semaphore(semaphore_limit)

    async def process_with_semaphore(node):
        async with semaphore:
            result = await process_chunk(node, document_dict)
            await asyncio.sleep(0.1)
            return result

    tasks = [process_with_semaphore(node) for node in all_nodes]

    results: List[TextNode] = await tqdm.gather(*tasks, desc="Processing chunks")

    # pdb.set_trace()

    return results


async def main():
    documents: List[Document] = create_docs("data/all_sources_data.jsonl")
    enhanced_nodes: List[TextNode] = await process(documents)

    with open("data/all_sources_contextual_nodes.pkl", "wb") as f:
        pickle.dump(enhanced_nodes, f)

    # pipeline = IngestionPipeline(
    #     transformations=[SentenceSplitter(chunk_size=800, chunk_overlap=0)]
    # )
    # all_nodes: list[TextNode] = pipeline.run(documents=documents, show_progress=True)
    # print(all_nodes[7933])
    # pdb.set_trace()

    with open("data/all_sources_contextual_nodes.pkl", "rb") as f:
        enhanced_nodes: list[TextNode] = pickle.load(f)

    for i, node in enumerate(enhanced_nodes):
        print(f"Chunk {i + 1}:")
        print(f"Node: {node}")
        print(f"Text: {node.text}")
        # pdb.set_trace()
        break


if __name__ == "__main__":
    asyncio.run(main())
