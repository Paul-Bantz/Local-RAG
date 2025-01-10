""" Router agent declaration

Given a user question decide whether to query a vector store or perform a
web request

"""

from .llm_agent import LLMAgent

class Router:
    """ Routing implementation

    Attributes:
        router_instructions: base llm instruction for routing a user question
        either to a vector store or a web search
    """

    # pylint: disable=line-too-long
    router_instructions : str = """You are an expert at routing a user question to a vectorstore or web search.

        The vectorstore contains documents related to {topic_of_interest}.

        Use the vectorstore for questions on these topics. For all else, and especially for current events, use web-search.

        Return JSON with single key, datasource, that is 'websearch' or 'vectorstore' depending on the question."""


    def route(self, llm_agent: LLMAgent, question: str, topics: str) -> str:
        """ Decide whether a given question should be routed to the vector store
        or a websearch

        Args:
            llm_agent: the llm to be queried for routing
            question: the query to be routed
            topics: a string representation of the topics covered by the store
        Returns:
            A json string with a single key 'websearch' or 'vectorstore'
            depending on the routing decision for the question
        """

        updated_instructions = self.router_instructions.format(topic_of_interest=topics)

        return llm_agent.run_query(json_mode=True,
                                   instructions=updated_instructions,
                                   query=question)
