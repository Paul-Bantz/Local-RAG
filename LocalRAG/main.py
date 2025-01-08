import os

from RAG.Embeddings import EmbeddingInterface
from RAG.LLMInterface import LLMInterface
from RAG.WorkfowGraph import WorkflowGraph

os.environ["TOKENIZERS_PARALLELISM"] = "true"

embedding_interface = EmbeddingInterface.EmbeddingInterface()
llm_interface = LLMInterface(server="http://localhost:11434", 
                             model="llama3.2:3b-instruct-fp16")

workflow_graph = WorkflowGraph(embeding_interface=embedding_interface,
                               llm_interface=llm_interface)

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

embedding_interface.embed_web_documents(urls)

workflow_graph.query("What are the models released today for llama3.2?", 3)