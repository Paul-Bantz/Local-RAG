
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from typing import List

class Rag:
    """ RAG declaration

    Attributes:
        rag_prompt: llm prompt for RAG
    """

    rag_prompt : str = """You are an assistant for question-answering tasks. 

        Here is the context to use to answer the question:

        {context} 

        Think carefully about the above context. 

        Now, review the user question:

        {question}

        Provide an answer to this questions using only the above context. 

        Use three sentences maximum and keep the answer concise.

        Answer:"""

    def format_docs(self, docs:list[Document]) -> str :
        """ Formats a list of documents        
        """
        return "\n\n".join(doc.page_content for doc in docs)

    def query(self, llm: BaseChatModel, 
              documents: List[Document],
              question: str) -> str :
        """ Given a set of documents and a question outputs a formatted answer

        Args:
            documents: the list of documents to process
            question: a question pertaining to the set of documents
        
        """
        docs_txt = self.format_docs(documents)

        rag_prompt_formatted = self.rag_prompt.format(context=docs_txt, 
                                                      question=question)
        
        return llm.invoke([HumanMessage(content=rag_prompt_formatted)])