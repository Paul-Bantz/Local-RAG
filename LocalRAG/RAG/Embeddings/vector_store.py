""" Definition of vector storage implementations.
"""

from abc import ABC, abstractmethod
import os
from typing import List
import uuid

import chromadb
from chromadb import ClientAPI

from langchain_core.embeddings import Embeddings
from chromadb.api.types import EmbeddingFunction, Documents

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
    def store_type(self):
        """ Datastore type
        """

    @abstractmethod
    def add_documents(self,
                      documents: List[str],
                      metatadas: List[dict],
                      ids: List):
        """ Adds the documents to the underlying vector store

        Args:
            documents: A list of documents to add to the store
            metatadas: associated document metadata
            ids: list of associated ids
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
    def list_store_contents(self) -> set[tuple]:
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
    store: SKLearnVectorStore
    retriever: VectorStoreRetriever

    @property
    def store_type(self):
        return "In-Memory"

    def __init__(self):
        """Initializes the an in-memory vector store based on the scikit-learn
        library NearestNeighbor

        The store is initialized with a local nomic embedding model running on
        the gpu
        """

        embedding_model =os.environ.get('NOMIC_EMBEDDING_MODEL')
        embedding_device =os.environ.get('GPU_DEVICE')

        nomic_embedding = NomicEmbeddings(model=embedding_model,
                                         inference_mode="local",
                                         device=embedding_device)

        self.store = SKLearnVectorStore(embedding=nomic_embedding)
        self.retriever = self.store.as_retriever(k=3)
        self.stored_documents_source = []

    def add_documents(self,
                      documents: List[str],
                      metatadas: List[dict],
                      ids: List):

        documents_to_add = []
        light_metadata = set

        for idx in range(len(documents)):
            documents_to_add.append(Document(page_content=documents[idx],
                                             metadata=metatadas[idx]))

            light_metadata.add((metatadas[idx]['source'],
                                metatadas[idx]['topic']))

        self.store.add_documents(documents_to_add)
        self.stored_documents_source = [set(light_metadata + self.stored_documents_source)]

    def get_documents(self, query: str) -> List[Document]:
        return self.retriever.invoke(query)

    def list_store_contents(self) -> set[tuple]:
        return self.stored_documents_source

class ChromaVectorStore(VectorStore):
    """ Definition for a chroma vector store connector

    Attributes:
        store: the underlying vectorstore implementation
        store_type: in memory
        retriever: the retriever for this store
        documents: a dict of documents contained in this store - this attribute
            is necessary to track the contents of the store for this type
    """

    embeddings : NomicEmbeddings

    chroma_client: ClientAPI

    base_collection: str = "local-rag"

    @property
    def store_type(self):
        return "Chroma"

    def __init__(self):
        """Initializes the an in-memory vector store based on the scikit-learn
        library NearestNeighbor

        The store is initialized with a local nomic embedding model running on
        the gpu
        """

        embedding_model =os.environ.get('NOMIC_EMBEDDING_MODEL')
        embedding_device =os.environ.get('GPU_DEVICE')

        chroma_host =os.environ.get('CHROMA_HOST')
        chroma_port =os.environ.get('CHROMA_PORT')

        self.embeddings = NomicEmbeddings(model=embedding_model,
                                         inference_mode="local",
                                         device=embedding_device)

        self.chroma_client = chromadb.HttpClient(host=chroma_host,
                                                 port=chroma_port)

        self.chroma_client.heartbeat()

    def add_documents(self,
                      documents: List[str],
                      metatadas: List[dict],
                      ids: List):

        collection = self.chroma_client.get_or_create_collection(name=self.base_collection,
                                                                 embedding_function=LangChainEmbeddingAdapter(self.embeddings))

        collection.add(documents=documents,
                       metadatas=metatadas,
                       ids=ids)

    def get_documents(self, query: str) -> List[Document]:

        collection = self.chroma_client.get_or_create_collection(name=self.base_collection,
                                                                 embedding_function=LangChainEmbeddingAdapter(self.embeddings))

        query_result = collection.query(query_texts=query)
        return query_result.items

    def list_store_contents(self) -> set[tuple]:
        """ Naive implementation - not intended for a large volumetry
        """

        collection = self.chroma_client.get_or_create_collection(name=self.base_collection,
                                                                 embedding_function=LangChainEmbeddingAdapter(self.embeddings))

        all_metadatas = collection.get(include=["metadatas"]).get('metadatas')
        distinct_keys = set([(x.get('source'), x.get('topic')) for x  in all_metadatas])

        return distinct_keys

class LangChainEmbeddingAdapter(EmbeddingFunction[Documents]):
    """ Adapter for Langchain embedder to Chroma format
    """

    def __init__(self, ef: Embeddings):
        self.ef = ef

    def __call__(self, input: Documents) -> Embeddings:

        return self.ef.embed_documents(input)