""" Definition of vector storage implementations.
"""

from abc import ABC, abstractmethod
from typing import List

from langchain_core.documents import Document
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_community.vectorstores import Chroma
from langchain_nomic import NomicEmbeddings

class VectorStore(ABC):
    """ Abstract class definition for a vector store datasource

    Underlying implementations define the type and mode of access to the
    datastore
    """

    @property
    @abstractmethod
    def store_type(self):
        """ Datastore type
        """

    @property
    @abstractmethod
    def store(self):
        """ Datastore implementation
        """

    @property
    @abstractmethod
    def retriever(self):
        """ Associated retriever implementation
        """

    @abstractmethod
    def add_documents(self, documents: List[Document], documents_reference: List[tuple]):
        """ Adds the documents to the underlying vector store

        Args:
            documents: A list of documents to add to the store
            documents_reference: a list of tuple containing the documents source
                and associated topic
        """

    @abstractmethod
    def get_documents(self, query: str) -> List[Document]:
        """ Retrieve the documents associated with the query string

        Invokes the retriever to get the documents pertaining to the query
        string

        Args:
            query the query string
        """

    @abstractmethod
    def list_store_contents(self) -> List[Document]:
        """ Retrieve the contents of the vector store

        Returns:
            A list of documents and associated metadata contained in the store
        """
class InMemoryVectorStore(VectorStore):
    """ Definition for a simple in-memory vector store

    Attributes:
        store: the underlying vectorstore implementation
        store_type: in memory
        retriever: the retriever for this store
        documents: a dict of documents contained in this store - this attribute
            is necessary to track the contents of the store for this type
    """

    stored_documents_source:List[tuple]

    @property
    def store(self):
        return self._store

    @property
    def store_type(self):
        return "In-Memory"

    @property
    def retriever(self):
        return self._retriever

    def __init__(self):
        """Initializes the an in-memory vector store based on the scikit-learn
        library NearestNeighbor

        The store is initialized with a local nomic embedding model running on
        the gpu
        """
        nomic_embedding = NomicEmbeddings(model="nomic-embed-text-v1.5",
                                         inference_mode="local",
                                         device='gpu')

        self._store = SKLearnVectorStore(embedding=nomic_embedding)
        self._retriever = self._store.as_retriever(k=3)
        self.stored_documents_source = []

    def add_documents(self,  documents: List[Document], documents_reference: List[tuple]):

        self.store.add_documents(documents)
        self.stored_documents_source = list(set(documents_reference + self.stored_documents_source))

    def get_documents(self, query: str) -> List[Document]:
        return self.retriever.invoke(query)

    def list_store_contents(self) -> List[tuple]:
        return self.stored_documents_source
