import asyncio
import html
import json
import logging
import os
import pdb
import pickle
import random
import time
from typing import Dict, List, Optional, Tuple

import aiofiles
import chromadb
import logfire
import pandas as pd
from custom_retriever import CustomRetriever
from llama_index.agent.openai import OpenAIAgent
from llama_index.core import Document, SimpleKeywordTableIndex, VectorStoreIndex
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.bridge.pydantic import Field, SerializeAsAny
from llama_index.core.chat_engine.types import (
    AGENT_CHAT_RESPONSE_TYPE,
    AgentChatResponse,
    ChatResponseMode,
)
from llama_index.core.evaluation import (
    AnswerRelevancyEvaluator,
    BatchEvalRunner,
    EmbeddingQAFinetuneDataset,
    FaithfulnessEvaluator,
    RelevancyEvaluator,
)
from llama_index.core.evaluation.base import EvaluationResult
from llama_index.core.evaluation.retrieval.base import (
    BaseRetrievalEvaluator,
    RetrievalEvalMode,
    RetrievalEvalResult,
)
from llama_index.core.indices.base_retriever import BaseRetriever
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.retrievers import (
    BaseRetriever,
    KeywordTableSimpleRetriever,
    VectorIndexRetriever,
)
from llama_index.core.schema import ImageNode, NodeWithScore, QueryBundle, TextNode
from llama_index.core.tools import RetrieverTool, ToolMetadata
from llama_index.core.vector_stores import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
from prompts import system_message_openai_agent
from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm_asyncio

# from setup import (
#     AVAILABLE_SOURCES,
#     AVAILABLE_SOURCES_UI,
#     custom_retriever_all_sources,
#     custom_retriever_langchain,
#     custom_retriever_llama_index,
#     custom_retriever_openai_cookbooks,
#     custom_retriever_peft,
#     custom_retriever_transformers,
#     custom_retriever_trl,
# )


class RotatingJSONLWriter:
    def __init__(
        self, base_filename: str, max_size: int = 10**6, backup_count: int = 5
    ):
        """
        Initialize the rotating JSONL writer.

        Args:
            base_filename (str): The base filename for the JSONL files.
            max_size (int): Maximum size in bytes before rotating.
            backup_count (int): Number of backup files to keep.
        """
        self.base_filename = base_filename
        self.max_size = max_size
        self.backup_count = backup_count
        self.current_file = base_filename

    async def write(self, data: dict):
        # Rotate if file exceeds max size
        if (
            os.path.exists(self.current_file)
            and os.path.getsize(self.current_file) > self.max_size
        ):
            await self.rotate_files()

        async with aiofiles.open(self.current_file, "a", encoding="utf-8") as f:
            await f.write(json.dumps(data, ensure_ascii=False) + "\n")

    async def rotate_files(self):
        # Remove the oldest backup if it exists
        oldest_backup = f"{self.base_filename}.{self.backup_count}"
        if os.path.exists(oldest_backup):
            os.remove(oldest_backup)

        # Rotate existing backups
        for i in range(self.backup_count - 1, 0, -1):
            src = f"{self.base_filename}.{i}"
            dst = f"{self.base_filename}.{i + 1}"
            if os.path.exists(src):
                os.rename(src, dst)

        # Rename current file to backup
        os.rename(self.current_file, f"{self.base_filename}.1")


class AsyncKeywordTableSimpleRetriever(KeywordTableSimpleRetriever):
    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._retrieve, query_bundle)


class SampleableEmbeddingQADataset:
    def __init__(self, dataset: EmbeddingQAFinetuneDataset):
        self.dataset = dataset

    def sample(self, n: int) -> EmbeddingQAFinetuneDataset:
        """
        Sample n queries from the dataset.

        Args:
            n (int): Number of queries to sample.

        Returns:
            EmbeddingQAFinetuneDataset: A new dataset with the sampled queries.
        """
        if n > len(self.dataset.queries):
            raise ValueError(
                f"n ({n}) is greater than the number of queries ({len(self.dataset.queries)})"
            )

        sampled_query_ids = random.sample(list(self.dataset.queries.keys()), n)

        sampled_queries = {qid: self.dataset.queries[qid] for qid in sampled_query_ids}
        sampled_relevant_docs = {
            qid: self.dataset.relevant_docs[qid] for qid in sampled_query_ids
        }

        # Collect all unique document IDs from the sampled relevant docs
        sampled_doc_ids = set()
        for doc_ids in sampled_relevant_docs.values():
            sampled_doc_ids.update(doc_ids)

        sampled_corpus = {
            doc_id: self.dataset.corpus[doc_id] for doc_id in sampled_doc_ids
        }

        return EmbeddingQAFinetuneDataset(
            queries=sampled_queries,
            corpus=sampled_corpus,
            relevant_docs=sampled_relevant_docs,
            mode=self.dataset.mode,
        )

    def __getattr__(self, name):
        return getattr(self.dataset, name)


class RetrieverEvaluator(BaseRetrievalEvaluator):
    """Retriever evaluator.

    This module will evaluate a retriever using a set of metrics.

    Args:
        metrics (List[BaseRetrievalMetric]): Sequence of metrics to evaluate
        retriever: Retriever to evaluate.
        node_postprocessors (Optional[List[BaseNodePostprocessor]]): Post-processor to apply after retrieval.
    """

    retriever: BaseRetriever = Field(..., description="Retriever to evaluate")
    node_postprocessors: Optional[List[SerializeAsAny[BaseNodePostprocessor]]] = Field(
        default=None, description="Optional post-processor"
    )

    async def _aget_retrieved_ids_and_texts(
        self,
        query: str,
        mode: RetrievalEvalMode = RetrievalEvalMode.TEXT,
        source: str = "",
    ) -> Tuple[List[str], List[str]]:
        """Get retrieved ids and texts, potentially applying a post-processor."""
        try:
            retrieved_nodes: list[NodeWithScore] = await self.retriever.aretrieve(query)
            logfire.info(f"Retrieved {len(retrieved_nodes)} nodes for: '{query}'")
        except Exception as e:
            return ["00000000-0000-0000-0000-000000000000"], [str(e)]

        if len(retrieved_nodes) == 0 or retrieved_nodes is None:
            print(f"No nodes retrieved for {query}")
            return ["00000000-0000-0000-0000-000000000000"], ["No nodes retrieved"]

        if self.node_postprocessors:
            for node_postprocessor in self.node_postprocessors:
                retrieved_nodes = node_postprocessor.postprocess_nodes(
                    retrieved_nodes, query_str=query
                )

        return (
            [node.node.node_id for node in retrieved_nodes],
            [node.node.text for node in retrieved_nodes],  # type: ignore
        )


class OpenAIAgentRetrieverEvaluator(BaseRetrievalEvaluator):
    agent: OpenAIAgent = Field(description="The OpenAI agent used for retrieval")

    async def _aget_retrieved_ids_and_texts(
        self,
        query: str,
        mode: RetrievalEvalMode = RetrievalEvalMode.TEXT,
        source: str = "",
    ) -> Tuple[List[str], List[str]]:

        self.agent.memory.reset()

        try:
            logfire.info(f"Executing agent with query: {query}")
            response: AgentChatResponse = await self.agent.achat(query)
        except Exception as e:
            # await self._save_response_data_async(
            #     source, query, ["Error retrieving nodes"], "Error retrieving nodes"
            # )
            return ["00000000-0000-0000-0000-000000000000"], [str(e)]

        retrieved_nodes: list[NodeWithScore] = get_nodes_with_score(response)
        logfire.info(f"Retrieved {len(retrieved_nodes)} to answer: '{query}'")
        retrieved_nodes = retrieved_nodes[:6]  # Limit to first 6 retrieved nodes

        if len(retrieved_nodes) == 0 or retrieved_nodes is None:
            # await self._save_response_data_async(
            #     source, query, ["No retrieved nodes"], "No retrieved nodes"
            # )
            return ["00000000-0000-0000-0000-000000000000"], ["No nodes retrieved"]

        retrieved_ids = [node.node.node_id for node in retrieved_nodes]
        retrieved_texts = [node.node.text for node in retrieved_nodes]  # type: ignore

        # Will not save context as its too long (token wise), costly and takes too much time.
        await self._save_response_data_async(
            source=source, query=query, context="", response=response.response
        )

        return retrieved_ids, retrieved_texts

    async def _save_response_data_async(self, source, query, context, response):
        data = {
            "source": source,
            "question": query,
            # "context": context,
            "answer": response,
        }
        await rotating_writer.write(data)


def get_nodes_with_score(completion) -> list[NodeWithScore]:
    retrieved_nodes = []
    for source in completion.sources:  # completion.sources = list[ToolOutput]
        if source.is_error == True:
            continue
        for node in source.raw_output:  # source.raw_output = list[NodeWithScore]
            retrieved_nodes.append(node)
    return retrieved_nodes


def setup_basic_database(db_collection, dict_file_name, keyword_retriever):
    db = chromadb.PersistentClient(path=f"data/{db_collection}")
    chroma_collection = db.get_or_create_collection(db_collection)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # embed_model = OpenAIEmbedding(model="text-embedding-3-large", mode="similarity")
    embed_model = CohereEmbedding(
        api_key=os.environ["COHERE_API_KEY"],
        model_name="embed-english-v3.0",
        input_type="search_query",
    )
    # client = embed_model._get_client()
    # aclient = embed_model._get_aclient()
    # logfire.instrument_openai(client)
    # logfire.instrument_openai(aclient)

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        show_progress=True,
    )
    vector_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=15,
        embed_model=embed_model,
    )
    with open(f"data/{db_collection}/{dict_file_name}", "rb") as f:
        document_dict = pickle.load(f)

    return CustomRetriever(vector_retriever, document_dict, keyword_retriever, "OR")


def update_query_engine_tools(selected_sources, custom_retriever_all_sources):
    tools = []
    source_mapping = {
        # "Transformers Docs": (
        #     custom_retriever_transformers,
        #     "Transformers_information",
        #     """Useful for general questions asking about the artificial intelligence (AI) field. Employ this tool to fetch information on topics such as language models (LLMs) models such as Llama3 and theory (transformer architectures), tips on prompting, quantization, etc.""",
        # ),
        # "PEFT Docs": (
        #     custom_retriever_peft,
        #     "PEFT_information",
        #     """Useful for questions asking about efficient LLM fine-tuning. Employ this tool to fetch information on topics such as LoRA, QLoRA, etc.""",
        # ),
        # "TRL Docs": (
        #     custom_retriever_trl,
        #     "TRL_information",
        #     """Useful for questions asking about fine-tuning LLMs with reinforcement learning (RLHF). Includes information about the Supervised Fine-tuning step (SFT), Reward Modeling step (RM), and the Proximal Policy Optimization (PPO) step.""",
        # ),
        # "LlamaIndex Docs": (
        #     custom_retriever_llama_index,
        #     "LlamaIndex_information",
        #     """Useful for questions asking about retrieval augmented generation (RAG) with LLMs and embedding models. It is the documentation of a framework, includes info about fine-tuning embedding models, building chatbots, and agents with llms, using vector databases, embeddings, information retrieval with cosine similarity or bm25, etc.""",
        # ),
        # "OpenAI Cookbooks": (
        #     custom_retriever_openai_cookbooks,
        #     "openai_cookbooks_info",
        #     """Useful for questions asking about accomplishing common tasks with the OpenAI API. Returns example code and guides stored in Jupyter notebooks, including info about ChatGPT GPT actions, OpenAI Assistants API,  and How to fine-tune OpenAI's GPT-4o and GPT-4o-mini models with the OpenAI API.""",
        # ),
        # "LangChain Docs": (
        #     custom_retriever_langchain,
        #     "langchain_info",
        #     """Useful for questions asking about the LangChain framework. It is the documentation of the LangChain framework, includes info about building chains, agents, and tools, using memory, prompts, callbacks, etc.""",
        # ),
        "All Sources": (
            custom_retriever_all_sources,
            "all_sources_info",
            """Useful for all questions, contains information about the field of AI.""",
        ),
    }

    for source in selected_sources:
        if source in source_mapping:
            retriever, name, description = source_mapping[source]
            tools.append(
                RetrieverTool(
                    retriever=retriever,
                    metadata=ToolMetadata(
                        name=name,
                        description=description,
                    ),
                )
            )

    return tools


def setup_agent(custom_retriever_all_sources) -> OpenAIAgent:

    llm = OpenAI(
        temperature=1,
        # model="gpt-4o",
        model="gpt-4o-mini",
        max_tokens=5000,
        max_retries=3,
    )
    client = llm._get_client()
    logfire.instrument_openai(client)
    aclient = llm._get_aclient()
    logfire.instrument_openai(aclient)

    tools_available = [
        # "Transformers Docs",
        # "PEFT Docs",
        # "TRL Docs",
        # "LlamaIndex Docs",
        # "LangChain Docs",
        # "OpenAI Cookbooks",
        "All Sources",
    ]
    query_engine_tools = update_query_engine_tools(
        tools_available, custom_retriever_all_sources
    )

    agent = OpenAIAgent.from_tools(
        llm=llm,
        tools=query_engine_tools,
        system_prompt=system_message_openai_agent,
    )

    return agent


async def evaluate_answers():
    start_time = time.time()

    # Gemini is not async here, maybe it could work with multithreading?
    # llm = Gemini(model="models/gemini-1.5-flash-002", temperature=1, max_tokens=1000)
    llm = OpenAI(model="gpt-4o-mini", temperature=1, max_tokens=1000)
    relevancy_evaluator = AnswerRelevancyEvaluator(llm=llm)

    # Load queries and response strings from JSONL file
    query_response_pairs = []
    with open("response_data.jsonl", "r") as f:
        for line in f:
            data = json.loads(line)
            query_response_pairs.append(
                (data["source"], data["query"], data["response"])
            )

    logfire.info(f"Number of queries and answers: {len(query_response_pairs)}")

    semaphore = asyncio.Semaphore(90)  # Adjust this value as needed

    async def evaluate_query_response(source, query, response):
        async with semaphore:
            try:
                result: EvaluationResult = await relevancy_evaluator.aevaluate(
                    query=query, response=response
                )
                return source, result
            except Exception as e:
                logfire.error(f"Error evaluating query for {source}: {str(e)}")
                return source, None

    # Use asyncio.gather to run all evaluations concurrently
    results = await tqdm_asyncio.gather(
        *[
            evaluate_query_response(source, query, response)
            for source, query, response in query_response_pairs
        ],
        desc="Evaluating answers",
        total=len(query_response_pairs),
    )

    # Process results
    eval_results = {}
    for item in results:
        if isinstance(item, tuple) and len(item) == 2:
            source, result = item
            if result is not None:
                if source not in eval_results:
                    eval_results[source] = []
                eval_results[source].append(result)
        else:
            logfire.error(f"Unexpected result: {item}")

    # Save results for each source
    for source, results in eval_results.items():
        with open(f"eval_answers_results_{source}.pkl", "wb") as f:
            pickle.dump(results, f)

    end_time = time.time()
    logfire.info(f"Total evaluation time: {round(end_time - start_time, 3)} seconds")

    return eval_results


def create_docs(input_file: str) -> List[Document]:
    with open(input_file, "r") as f:
        documents = []
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


def get_sample_size(source: str, total_queries: int) -> int:
    """Determine the number of queries to sample based on the source."""
    # small_datasets = {"peft": 0, "trl": 0, "openai_cookbooks": 0}
    # large_datasets = {
    #     "transformers": 0,
    #     "llama_index": 0,
    #     "langchain": 1,
    #     "tai_blog": 0,
    # }
    small_datasets = {"peft": 49, "trl": 34, "openai_cookbooks": 170}
    large_datasets = {
        "transformers": 200,
        "llama_index": 200,
        "langchain": 200,
        "tai_blog": 200,
    }
    # small_datasets = {"peft": 49, "trl": 34, "openai_cookbooks": 100}
    # large_datasets = {
    #     "transformers": 100,
    #     "llama_index": 100,
    #     "langchain": 100,
    #     "tai_blog": 100,
    # }
    # small_datasets = {"peft": 18, "trl": 12, "openai_cookbooks": 14}
    # large_datasets = {
    #     "transformers": 24,
    #     "llama_index": 8,
    #     "langchain": 6,
    #     "tai_blog": 18,
    # }
    # small_datasets = {"peft": 4, "trl": 4, "openai_cookbooks": 4}
    # large_datasets = {
    #     "transformers": 4,
    #     "llama_index": 4,
    #     "langchain": 5,
    #     "tai_blog": 5,
    # }

    if source in small_datasets:
        return small_datasets[source]
    elif source in large_datasets:
        return large_datasets[source]
    else:
        return min(100, total_queries)  # Default to 100 or all queries if less than 100


async def evaluate_retriever():
    start_time = time.time()
    with open("data/keyword_retriever_async.pkl", "rb") as f:
        keyword_retriever = pickle.load(f)

    custom_retriever_all_sources: CustomRetriever = setup_basic_database(
        "chroma-db-all_sources", "document_dict_all_sources.pkl", keyword_retriever
    )
    # agent = setup_agent(custom_retriever_all_sources)

    # filters = MetadataFilters(
    #     filters=[
    #         MetadataFilter(key="source", operator=FilterOperator.EQ, value="langchain"),
    #     ]
    # )
    # custom_retriever_all_sources._vector_retriever._filters = filters

    end_time = time.time()
    logfire.info(
        f"Time taken for setup the custom retriever: {round(end_time - start_time, 2)} seconds"
    )

    sources_to_evaluate = [
        "transformers",
        "peft",
        "trl",
        "llama_index",
        "langchain",
        "openai_cookbooks",
        "tai_blog",
    ]

    # for k in [5, 7, 9, 11, 13, 15]:
    #     custom_retriever_all_sources._vector_retriever._similarity_top_k = k

    retriever_evaluator = RetrieverEvaluator.from_metric_names(
        ["mrr", "hit_rate"], retriever=custom_retriever_all_sources
    )
    # retriever_evaluator = OpenAIAgentRetrieverEvaluator.from_metric_names(
    #     metric_names=["mrr", "hit_rate"], agent=agent
    # )

    all_query_pairs = []
    for source in sources_to_evaluate:
        rag_eval_dataset = EmbeddingQAFinetuneDataset.from_json(
            f"scripts/rag_eval_{source}.json"
        )
        sampleable_dataset = SampleableEmbeddingQADataset(rag_eval_dataset)
        sample_size = get_sample_size(source, len(sampleable_dataset.queries))
        sampled_dataset = sampleable_dataset.sample(n=sample_size)
        query_expected_ids_pairs = sampled_dataset.query_docid_pairs
        all_query_pairs.extend(
            [(source, pair[0], pair[1]) for pair in query_expected_ids_pairs]
        )

    semaphore = asyncio.Semaphore(220)  # 250 caused a couple of errors
    # semaphore = asyncio.Semaphore(90)  # 100 caused a couple of errors with agent

    async def evaluate_query(source, query, expected_ids):
        async with semaphore:
            try:
                result: RetrievalEvalResult = await retriever_evaluator.aevaluate(
                    query=query,
                    expected_ids=expected_ids,
                    mode=RetrievalEvalMode.TEXT,
                    source=source,
                )
                return source, result
            except Exception as e:
                logfire.error(f"Error evaluating query for {source}: {str(e)}")
                return source, None

    # Use asyncio.gather to run all evaluations concurrently
    results = await tqdm_asyncio.gather(
        *[
            evaluate_query(source, query, expected_ids)
            for source, query, expected_ids in all_query_pairs
        ],
        desc="Evaluating queries",
        total=len(all_query_pairs),
    )

    # Process results
    eval_results = {source: [] for source in sources_to_evaluate}
    for item in results:
        if isinstance(item, tuple) and len(item) == 2:
            source, result = item
            if result is not None:
                eval_results[source].append(result)
        else:
            logfire.error(f"Unexpected result: {item}")

    # Save results for each source
    for source, results in eval_results.items():
        with open(f"eval_results_{source}.pkl", "wb") as f:
            pickle.dump(results, f)
        # print(display_results_retriever(source, results))

    end_time = time.time()
    logfire.info(f"Total evaluation time: {round(end_time - start_time, 3)} seconds")


def display_results_retriever(name, eval_results):
    """Display results from evaluate."""

    metric_dicts = []
    for eval_result in eval_results:
        metric_dict = eval_result.metric_vals_dict
        metric_dicts.append(metric_dict)

    full_df = pd.DataFrame(metric_dicts)

    hit_rate = full_df["hit_rate"].mean()
    mrr = full_df["mrr"].mean()

    metric_df = pd.DataFrame(
        {"Retriever Name": [name], "Hit Rate": [hit_rate], "MRR": [mrr]}
    )

    return metric_df


def display_results():

    sources = [
        "transformers",
        "peft",
        "trl",
        "llama_index",
        "langchain",
        "openai_cookbooks",
        "tai_blog",
    ]
    # retrievers_to_evaluate = [
    #     # "chroma-db-all_sources_400_0",
    #     # "chroma-db-all_sources_400_200",
    #     # "chroma-db-all_sources_500_0",
    #     # "chroma-db-all_sources_500_250",
    #     # "chroma-db-all_sources",
    #     # "chroma-db-all_sources_800_400",
    #     # "chroma-db-all_sources_1000_0",
    #     # "chroma-db-all_sources_1000_500",
    # ]

    # topk = [5, 7, 9, 11, 13, 15]
    # for k in topk:
    # for db_name in retrievers_to_evaluate:
    if True:
        # print("-" * 20)
        # print(f"Retriever {db_name}")
        for source in sources:
            with open(f"eval_results_{source}.pkl", "rb") as f:
                eval_results = pickle.load(f)
            print(display_results_retriever(f"{source}", eval_results))


def display_results_answers():

    sources = [
        "transformers",
        "peft",
        "trl",
        "llama_index",
        "langchain",
        "openai_cookbooks",
        "tai_blog",
    ]

    for source in sources:
        with open(f"eval_answers_results_{source}.pkl", "rb") as f:
            eval_results = pickle.load(f)
        print(
            f"Score for {source}:",
            sum(result.score for result in eval_results) / len(eval_results),
        )


async def main():
    await evaluate_retriever()
    display_results()
    # await evaluate_answers()
    # display_results_answers()
    return


if __name__ == "__main__":

    logfire.configure()
    rotating_writer = RotatingJSONLWriter(
        "response_data.jsonl", max_size=10**7, backup_count=5
    )

    start_time = time.time()
    asyncio.run(main())
    end_time = time.time()
    logfire.info(
        f"Time taken to run script: {round((end_time - start_time), 3)} seconds"
    )

    # # Creating the keyword index and retriever
    # logfire.info("Creating nodes from documents")
    # documents = create_docs("data/all_sources_data.jsonl")
    # pipeline = IngestionPipeline(
    #     transformations=[SentenceSplitter(chunk_size=800, chunk_overlap=0)]
    # )
    # all_nodes = pipeline.run(documents=documents, show_progress=True)
    # # with open("data/all_nodes.pkl", "wb") as f:
    # #     pickle.dump(all_nodes, f)

    # # all_nodes = pickle.load(open("data/all_nodes.pkl", "rb"))
    # logfire.info(f"Number of nodes: {len(all_nodes)}")

    # with open("processed_chunks.pkl", "rb") as f:
    #     all_nodes: list[TextNode] = pickle.load(f)

    # keyword_index = SimpleKeywordTableIndex(
    #     nodes=all_nodes, max_keywords_per_chunk=10, show_progress=True, use_async=False
    # )
    # # with open("data/keyword_index.pkl", "wb") as f:
    # #     pickle.dump(keyword_index, f)
    # # keyword_index = pickle.load(open("data/keyword_index.pkl", "rb"))

    # logfire.info("Creating keyword retriever")
    # keyword_retriever = AsyncKeywordTableSimpleRetriever(index=keyword_index)

    # with open("data/keyword_retriever_async.pkl", "wb") as f:
    #     pickle.dump(keyword_retriever, f)
