""" Declaration of a RAG agent
"""

import os
from typing import List, Tuple

from langchain_core.messages import AIMessage
from langchain.schema import Document

from RAG.llm_agent import LLMAgent
from RAG.Embeddings.embedding_interface import EmbeddingInterface
from RAG.workflow_graph import WorkflowGraph

class RagAgent:
    """ Main class of declaration of a RAG Agent.

    Initializes a LLM connexion, a vector datasource and a workflow graph
    """

    llm_agent : LLMAgent
    embedding_interface : EmbeddingInterface
    workflow_graph : WorkflowGraph

    def __init__(self):

        llm_host=os.environ.get('LLM_HOST')
        llm_model=os.environ.get('LLM_MODEL')

        self.embedding_interface = EmbeddingInterface()
        self.llm_agent = LLMAgent(server=llm_host,
                                      model=llm_model)

        self.workflow_graph = WorkflowGraph(embeding_interface=self.embedding_interface,
                                            llm_agent=self.llm_agent)

    def query(self, query:str, iterations:int) -> Tuple[AIMessage, List[Document]] :
        """ Submit a query to the workflow graph. Given a max number of iterations
        will generate an answer in context.

        Args:
            query : the question to run through the RAG
            iterations : the number of iterations to better the answer

        Returns:
            A tuple containing the generated answer and a list of documentary
            sources originating from the underlying Vector DB or the web.
        """

        return self.workflow_graph.execute(query, iterations)

    def embed_documents(self, documents : List[tuple]) -> List[tuple]:
        """ Add the urls in the vector store

        Args:
            documents: a list of documents from which to generate embeddings
        """

        self.embedding_interface.embed_web_documents(documents)

    def list_store_contents(self) ->  List[tuple]:
        """ Returns a list of documents sources and associated topics contained
            in the vector store
        """
        return self.embedding_interface.list_store_contents()
