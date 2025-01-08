### Router
import json
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

class Router:
    """ Router agent declaration

    Given a user question decide whether to query a vector store or perform a
    web request

    Attributes:
        router_instructions: base llm instruction for routing a user question
        either to a vector store or a web search
    """

    router_instructions : str = """You are an expert at routing a user question to a vectorstore or web search.

        The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.

        Use the vectorstore for questions on these topics. For all else, and especially for current events, use web-search.

        Return JSON with single key, datasource, that is 'websearch' or 'vectorstore' depending on the question."""


    def route(self, llm: BaseChatModel, question: str) -> str:
        """ Decide whether a given question should be routed to the vector store
        or a websearch 

        Returns:
            A json string with a single key 'websearch' or 'vectorstore' 
            depending on the routing decision for the question        
        """

        routing_result = llm.invoke([SystemMessage(content=self.router_instructions)]
                                     + [HumanMessage(content=question)])
        
        return json.loads(routing_result.content)
