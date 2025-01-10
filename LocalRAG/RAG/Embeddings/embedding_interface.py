""" I/O Interface with a vector datastore
"""

from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader

from . import vector_store

class EmbeddingInterface:
    """
    Attributes:
        vectorstore: the underlying vector store
        text_splitter: documents splitting strategy implementation
    """

    vectorstore : vector_store
    text_splitter : RecursiveCharacterTextSplitter

    def __init__(self):
        """ Initialize an inmemory vector store and a text splitter
        """
        self.vectorstore = vector_store.InMemoryVectorStore()
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000,
                                                                                  chunk_overlap=200)

    def embed_web_documents(self, documents_source: List[tuple]):
        """ Loads a set of urls and store their embeddings in the vector store

        Args:
            documents_source: a list of tuple (urls and topics) from which to
                generate the embeddings
        """
        print("Loading url set in Vector Store")

        # Load documents
        docs = [WebBaseLoader(document[0]).load() for document in documents_source]

        docs_list = [item for sublist in docs for item in sublist]

        # Split documents
        doc_splits = self.text_splitter.split_documents(docs_list)

        # Add to vectorDB
        self.vectorstore.add_documents(doc_splits, documents_source)

    def get_documents(self, query: str):
        """ Retrieve the relevant documents associated with the query string
        """
        return self.vectorstore.get_documents(query)

    def list_store_contents(self) ->  List[tuple]:
        """ Returns a list of documents sources and associated topics contained
            in the vector store
        """
        return self.vectorstore.list_store_contents()

    def get_store_topics(self) -> str:
        """ Returns a string representation of all the topics of the documents
            contained in the vector store
        """
        store_contents = self.vectorstore.list_store_contents()
        topics = list(set([document[1] for document in store_contents]))

        if len(topics) > 1:

            topics_string = ", ".join(topics[:-1])
            topics_string = topics_string + " and " + topics[-1]

        else:
            topics_string = topics[0]

        return topics_string
