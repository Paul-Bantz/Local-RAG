from abc import ABC, abstractmethod
from typing import List

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_nomic import NomicEmbeddings

class VectorStore(ABC):
    """ Abstract class definition for a vector store datasource

    Underlying implementations define the type and mode of access to the
    datastore
    """
    
    @property
    @abstractmethod
    def store(self):
        """ Datastore implementation
        """
        pass

    @property
    @abstractmethod
    def retriever(self):
        """ Associated retriever implementation
        """
        pass

    @abstractmethod
    def add_documents(self, documents: List[Document]):
        """ Adds the documents to the underlying vector store

        Args:
            documents: A list of documents to add to the store
        """
        pass
    
    @abstractmethod
    def get_documents(self, query: str) -> List[Document]:
        """ Retrieve the documents associated with the query string

        Invokes the retriever to get the documents pertaining to the query
        string

        Args:
            query the query string
        """
        pass

class InMemoryVectorStore(VectorStore):
    """ Definition for a simple in-memory vector store

    Attributes:
        store: the underlying vectorstore implementation
    """
    
    def __init__(self):
        """Initializes the an in-memory vector store based on the scikit-learn 
        library NearestNeighbor

        The store is initialized with a local nomic embedding model running on 
        the gpu
        """
        nomicEmbedding = NomicEmbeddings(model="nomic-embed-text-v1.5", 
                                         inference_mode="local",
                                         device='gpu')
        
        self._store = SKLearnVectorStore(embedding=nomicEmbedding)
        self._retriever = self._store.as_retriever(k=3)

    def add_documents(self,  documents: List[Document]):

        self.store.add_documents(documents)

    def get_documents(self, query: str) -> List[Document]:
        return self.retriever.invoke(query)
    
    @property
    def store(self):
        return self._store
    
    @property
    def retriever(self):
        return self._retriever