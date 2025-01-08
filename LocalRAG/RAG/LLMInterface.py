
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document

class LLMInterface:
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
        """ Initialization of a Ollama based LLM given a server and a model

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