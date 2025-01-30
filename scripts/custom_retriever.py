import asyncio
import os
import time
import traceback
from typing import List, Optional

import logfire
import tiktoken
from cohere import AsyncClient
from dotenv import load_dotenv
from llama_index.core import Document, QueryBundle
from llama_index.core.async_utils import run_async_tasks
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.retrievers import (
    BaseRetriever,
    KeywordTableSimpleRetriever,
    VectorIndexRetriever,
)
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle, TextNode
from llama_index.core.vector_stores import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.postprocessor.cohere_rerank.base import CohereRerank

load_dotenv()


class AsyncCohereRerank(CohereRerank):
    def __init__(
        self,
        top_n: int = 5,
        model: str = "rerank-english-v3.0",
        api_key: Optional[str] = None,
    ) -> None:
        super().__init__(top_n=top_n, model=model, api_key=api_key)
        self._api_key = api_key
        self._model = model
        self._top_n = top_n

    async def apostprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if query_bundle is None:
            raise ValueError("Query bundle must be provided.")

        if len(nodes) == 0:
            return []

        async_client = AsyncClient(api_key=self._api_key)

        with self.callback_manager.event(
            CBEventType.RERANKING,
            payload={
                EventPayload.NODES: nodes,
                EventPayload.MODEL_NAME: self._model,
                EventPayload.QUERY_STR: query_bundle.query_str,
                EventPayload.TOP_K: self._top_n,
            },
        ) as event:
            texts = [
                node.node.get_content(metadata_mode=MetadataMode.EMBED)
                for node in nodes
            ]

            results = await async_client.rerank(
                model=self._model,
                top_n=self._top_n,
                query=query_bundle.query_str,
                documents=texts,
            )

            new_nodes = []
            for result in results.results:
                new_node_with_score = NodeWithScore(
                    node=nodes[result.index].node, score=result.relevance_score
                )
                new_nodes.append(new_node_with_score)
            event.on_end(payload={EventPayload.NODES: new_nodes})

        return new_nodes


class CustomRetriever(BaseRetriever):
    """Custom retriever that performs both semantic search and hybrid search."""

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        document_dict: dict,
        keyword_retriever=None,
        mode: str = "AND",
    ) -> None:
        """Init params."""
        self._vector_retriever = vector_retriever
        self._document_dict = document_dict
        self._keyword_retriever = keyword_retriever
        if mode not in ("AND", "OR"):
            raise ValueError("Invalid mode.")
        self._mode = mode
        super().__init__()

    async def _process_retrieval(
        self, query_bundle: QueryBundle, is_async: bool = True
    ) -> List[NodeWithScore]:
        """Common processing logic for both sync and async retrieval."""
        # Clean query string
        query_bundle.query_str = query_bundle.query_str.replace(
            "\ninput is ", ""
        ).rstrip()
        logfire.info(f"Retrieving nodes with string: '{query_bundle}'")

        start = time.time()

        # Get nodes from both retrievers
        if is_async:
            nodes = await self._vector_retriever.aretrieve(query_bundle)
        else:
            nodes = self._vector_retriever.retrieve(query_bundle)

        keyword_nodes = []
        if self._keyword_retriever:
            if is_async:
                keyword_nodes = await self._keyword_retriever.aretrieve(query_bundle)
            else:
                keyword_nodes = self._keyword_retriever.retrieve(query_bundle)

        logfire.info(f"Number of vector nodes: {len(nodes)}")
        logfire.info(f"Number of keyword nodes: {len(keyword_nodes)}")

        # # Filter keyword nodes based on metadata filters from vector retriever
        # if (
        #     hasattr(self._vector_retriever, "_filters")
        #     and self._vector_retriever._filters
        # ):
        #     filtered_keyword_nodes = []
        #     for node in keyword_nodes:
        #         node_source = node.node.metadata.get("source")
        #         # Check if node's source matches any of the filter conditions
        #         for filter in self._vector_retriever._filters.filters:
        #             if (
        #                 isinstance(filter, MetadataFilter)
        #                 and filter.key == "source"
        #                 and filter.operator == FilterOperator.EQ
        #                 and filter.value == node_source
        #             ):
        #                 filtered_keyword_nodes.append(node)
        #                 break
        #     keyword_nodes = filtered_keyword_nodes
        #     logfire.info(
        #         f"Number of keyword nodes after filtering: {len(keyword_nodes)}"
        #     )

        # Combine results based on mode
        vector_ids = {n.node.node_id for n in nodes}
        keyword_ids = {n.node.node_id for n in keyword_nodes}
        combined_dict = {n.node.node_id: n for n in nodes}
        combined_dict.update({n.node.node_id: n for n in keyword_nodes})

        # If no keyword retriever or no keyword nodes, just use vector nodes
        if not self._keyword_retriever or not keyword_nodes:
            retrieve_ids = vector_ids
        else:
            retrieve_ids = (
                vector_ids.intersection(keyword_ids)
                if self._mode == "AND"
                else vector_ids.union(keyword_ids)
            )

        nodes = [combined_dict[rid] for rid in retrieve_ids]
        logfire.info(f"Number of combined nodes: {len(nodes)}")

        # Filter unique doc IDs
        nodes = self._filter_nodes_by_unique_doc_id(nodes)
        logfire.info(f"Number of nodes without duplicate doc IDs: {len(nodes)}")

        # Process node contents
        for node in nodes:
            doc_id = node.node.source_node.node_id
            if node.metadata["retrieve_doc"]:
                doc = self._document_dict[doc_id]
                node.node.text = doc.text
            node.node.node_id = doc_id

        # Rerank results
        try:
            reranker = (
                AsyncCohereRerank(top_n=5, model="rerank-english-v3.0")
                if is_async
                else CohereRerank(top_n=5, model="rerank-english-v3.0")
            )
            nodes = (
                await reranker.apostprocess_nodes(nodes, query_bundle)
                if is_async
                else reranker.postprocess_nodes(nodes, query_bundle)
            )
        except Exception as e:
            error_msg = f"Error during reranking: {type(e).__name__}: {str(e)}\n"
            error_msg += "Traceback:\n"
            error_msg += traceback.format_exc()
            logfire.error(error_msg)

        # Filter by score and token count
        nodes_filtered = self._filter_by_score_and_tokens(nodes)

        duration = time.time() - start
        logfire.info(f"Retrieving nodes took {duration:.2f}s")
        logfire.info(f"Nodes sent to LLM: {nodes_filtered[:5]}")

        return nodes_filtered[:5]

    def _filter_nodes_by_unique_doc_id(
        self, nodes: List[NodeWithScore]
    ) -> List[NodeWithScore]:
        """Filter nodes to keep only unique doc IDs."""
        unique_nodes = {}
        for node in nodes:
            doc_id = node.node.source_node.node_id
            if doc_id is not None and doc_id not in unique_nodes:
                unique_nodes[doc_id] = node
        return list(unique_nodes.values())

    def _filter_by_score_and_tokens(
        self, nodes: List[NodeWithScore]
    ) -> List[NodeWithScore]:
        """Filter nodes by score and token count."""
        nodes_filtered = []
        total_tokens = 0
        enc = tiktoken.encoding_for_model("gpt-4o-mini")

        for node in nodes:
            if node.score < 0.10:
                logfire.info(f"Skipping node with score {node.score}")
                continue

            node_tokens = len(enc.encode(node.node.text))
            if total_tokens + node_tokens > 100_000:
                logfire.info("Skipping node due to token count exceeding 100k")
                break

            total_tokens += node_tokens
            nodes_filtered.append(node)

        return nodes_filtered

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Async retrieve nodes given query."""
        return await self._process_retrieval(query_bundle, is_async=True)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Sync retrieve nodes given query."""
        return asyncio.run(self._process_retrieval(query_bundle, is_async=False))
