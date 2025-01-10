""" Interface for LLM interaction
"""

import json

from typing import List

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from langchain_core.documents import Document

class LLMAgent:
    """ Interface for LLM interaction

    Attributes:
        llm: llm in text mode
        llm_json_mode: llm in json mode (will reply with json formatted strings)
        rag_prompt: default base prompt
    """

    llm : ChatOllama
    llm_json_mode : ChatOllama

    rag_prompt : str = """You are an assistant for question-answering tasks.

        Here is the context to use to answer the question:

        {context}

        Think carefully about the above context.

        Now, review the user question:

        {question}

        Provide an answer to this questions using only the above context.

        Use three sentences maximum and keep the answer concise.

        Answer:"""

    def __init__(self, server:str, model:str):
        """ Initialization of an Ollama based LLM given a server and a model

        Args:
            server: hostname and port of the server running the llm
            model: the model currently running on the server

        """
        self.llm = ChatOllama(base_url=server,
                              model=model,
                              temperature=0)

        self.llm_json_mode = ChatOllama(base_url=server,
                                        model=model,
                                        temperature=0,
                                        format="json")

    def concatenate_documents(self, docs:list[Document]) -> str :
        """ Formats a list of documents as a single string

        Args:
            docs: the list of documents to concatenate

        Return:
            The concatenation of all the documents
        """
        return "\n\n".join(doc.page_content for doc in docs)

    def run_query(self,
                  json_mode: bool,
                  instructions: str,
                  query: str) -> str:
        """ Formats a list of documents as a single string

        Args:
            docs: the list of documents to concatenate

        Return:
            The concatenation of all the documents
        """
        if json_mode :
            result = self.llm_json_mode.invoke([SystemMessage(content=instructions)]
                                                + [HumanMessage(content=query)])
        else :
            result = self.llm.invoke([SystemMessage(content=instructions)]
                                    + [HumanMessage(content=query)])

        return json.loads(result.content)

    def run_rag_query_on_documents(self,
              json_mode: bool,
              documents: List[Document],
              question: str) -> str :
        """ Given a set of documents and a question, outputs a formatted answer

        Args:
            json_mode: whether the answer should be returned as a json object
                or plain text generation
            documents: the list of documents to process
            question: a question pertaining to the set of documents

        Returns:
            An answer to the question related to the documents in json or plain
            text
        """
        docs_txt = self.concatenate_documents(documents)

        rag_prompt_formatted = self.rag_prompt.format(context=docs_txt,
                                                      question=question)

        if json_mode :
            return_value = self.llm_json_mode.invoke([HumanMessage(content=rag_prompt_formatted)])
        else:
            return_value = self.llm.invoke([HumanMessage(content=rag_prompt_formatted)])

        return return_value
