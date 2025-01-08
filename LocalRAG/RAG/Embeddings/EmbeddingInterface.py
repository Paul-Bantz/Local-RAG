from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader

from RAG.Embeddings import VectorStore

class EmbeddingInterface:
    """ I/O Interface with a vector datastore

    Attributes:
        vectorstore: the underlying vector store
        text_splitter: documents splitting strategy implementation
    """

    vectorstore : VectorStore
    text_splitter : RecursiveCharacterTextSplitter

    def __init__(self):
        """ Initialize an inmemory vector store and a text splitter
        """
        self.vectorstore = VectorStore.InMemoryVectorStore()
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, 
                                                                                  chunk_overlap=200)

    def embed_web_documents(self, urls):
        """ Loads a set of urls and store their embeddings in the vector store

        Args:
            urls: the list of urls from which to create the embeddings
        """
        # Load documents
        docs = [WebBaseLoader(url).load() for url in urls]
        docs_list = [item for sublist in docs for item in sublist]

        # Split documents
        doc_splits = self.text_splitter.split_documents(docs_list)

        # Add to vectorDB
        self.vectorstore.add_documents(doc_splits)

    def get_documents(self, query: str):
        """ Retrieve the relevant documents associated with the query string
        """
        return self.vectorstore.get_documents(query)