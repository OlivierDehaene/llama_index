from typing import List, Optional

from llama_index.bridge.pydantic import Field
from llama_index.callbacks import CallbackManager, CBEventType, EventPayload
from llama_index.postprocessor.types import BaseNodePostprocessor
from llama_index.schema import NodeWithScore, QueryBundle
from llama_index.embeddings.huggingface_utils import format_query, format_text

DEFAULT_URL = "http://127.0.0.1:8080"


class TextEmbeddingsInferenceRerank(BaseNodePostprocessor):
    model_name: str = Field(
        default="unknown", description="The name of the re-ranking model."
    )
    base_url: str = Field(
        default=DEFAULT_URL,
        description="Base URL for the text embeddings service.",
    )
    query_instruction: Optional[str] = Field(
        description="Instruction to prepend to query text."
    )
    text_instruction: Optional[str] = Field(
        description="Instruction to prepend to text."
    )
    timeout: float = Field(
        default=60.0,
        description="Timeout in seconds for the request.",
    )
    truncate_text: bool = Field(
        default=True,
        description="Whether to truncate text or not when generating embeddings.",
    )

    def __init__(
        self,
        model_name: str,
        base_url: str = DEFAULT_URL,
        text_instruction: Optional[str] = None,
        query_instruction: Optional[str] = None,
        timeout: float = 60.0,
        truncate_text: bool = True,
    ):
        try:
            import httpx  # noqa
        except ImportError:
            raise ImportError(
                "TextEmbeddingsInterface requires httpx to be installed.\n"
                "Please install httpx with `pip install httpx`."
            )

        super().__init__(
            base_url=base_url,
            model_name=model_name,
            text_instruction=text_instruction,
            query_instruction=query_instruction,
            timeout=timeout,
            truncate_text=truncate_text,
        )

    @classmethod
    def class_name(cls) -> str:
        return "TextEmbeddingsInferenceRerank"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        import httpx

        if query_bundle is None:
            raise ValueError("Missing query bundle in extra info.")

        query = format_query(query_bundle.query_str, self.model_name, self.query_instruction)
        texts = [format_text(node.node.get_content(), self.model_name, self.text_instruction) for node in nodes]

        with self.callback_manager.event(
            CBEventType.RERANKING,
            payload={
                EventPayload.NODES: nodes,
                EventPayload.MODEL_NAME: self.model_name,
                EventPayload.QUERY_STR: query,
            },
        ) as event:
            headers = {"Content-Type": "application/json"}
            json_data = {"query": query, "texts": texts, "truncate": self.truncate_text}

            with httpx.Client() as client:
                response = client.post(
                    f"{self.base_url}/rerank",
                    headers=headers,
                    json=json_data,
                    timeout=self.timeout,
                )

            results = response.json()

            new_nodes = []
            for result in results:
                new_node_with_score = NodeWithScore(
                    node=nodes[result["index"]].node, score=result["score"]
                )
                new_nodes.append(new_node_with_score)
            event.on_end(payload={EventPayload.NODES: new_nodes})

        return new_nodes
