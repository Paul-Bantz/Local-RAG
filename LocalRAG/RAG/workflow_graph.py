""" Rag execution graph declaration.

"""

import operator
import os
import logging

from typing import List, Annotated, Tuple
from typing_extensions import TypedDict

from langchain_core.messages import AIMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document

from langgraph.graph import END
from langgraph.graph import StateGraph

from .Embeddings.embedding_interface import EmbeddingInterface
from .grader import AnswerGrader, HallucinationGrader, RetrievalGrader
from .llm_agent import LLMAgent
from .router import Router

class GraphState(TypedDict):
    """
    Graph state is a dictionary that contains information we want to
    propagate to, and modify in each graph node.
    """

    question: str           # User question
    generation: str         # LLM generation
    web_search: str         # Binary decision to run web search
    max_retries: int        # Max number of retries for answer generation
    answers: int            # Number of answers generated
    documents: List[str]    # List of retrieved documents
    loop_step: Annotated[int, operator.add]

class WorkflowGraph:
    """Rag Workflow declaration

    Given an input question and a number of iterations will establish an answer
    with information store in the underlying vector store and / or a websearch
    """

    logger = logging.getLogger(__name__)

    state_graph : StateGraph

    embedding_interface : EmbeddingInterface
    llm_agent : LLMAgent

    router : Router

    retrieval_grader : RetrievalGrader
    hallucination_grader : HallucinationGrader
    answer_grader : AnswerGrader

    web_search_tool : TavilySearchResults

    def __init__(self,
                 embeding_interface: EmbeddingInterface,
                 llm_agent: LLMAgent ):

        logging.basicConfig(level=logging.INFO)

        self.embedding_interface = embeding_interface
        self.llm_agent = llm_agent

        self.router = Router()

        self.answer_grader = AnswerGrader()
        self.hallucination_grader = HallucinationGrader()
        self.retrieval_grader = RetrievalGrader()

        if "TAVILY_API_KEY" in os.environ:
            self.web_search_tool = TavilySearchResults(k=3)
        else:
            self.logger.error('🔴 Tavily API Key not found - Skipping initialization')

        self.construct_workflow()

    def construct_workflow(self) :
        """ Builds an execution graph to generate an answer given a specific
        question.
        """

        workflow = StateGraph(GraphState)

        # Define the nodes
        workflow.add_node("websearch", self.web_search)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("grade_documents", self.grade_documents)
        workflow.add_node("generate", self.generate)

        # Build graph
        workflow.set_conditional_entry_point(

            self.route_question,
            {
                "websearch": "websearch",
                "vectorstore": "retrieve",
            },
        )

        workflow.add_edge("websearch", "generate")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "websearch": "websearch",
                "generate": "generate",
            },
        )
        workflow.add_conditional_edges(
            "generate",
            self.grade_generation_v_documents_and_question,
            {
                "not supported": "generate",
                "useful": END,
                "not useful": "websearch",
                "max retries": END,
            },
        )

        self.state_graph = workflow.compile()

    def get_workflow_visualisation(self) -> bytes :
        """Get a visual representation of the workflow graph

        Returns:
            A png as bytes representing the graph
        """

        return self.state_graph.get_graph().draw_mermaid_png()

    def execute(self, question:str, max_retries:int) -> Tuple[AIMessage, List[Document]]:
        """ Runs the graph given a question and a maximum number of retries.

        Args:
            question : the question to run through the RAG
            max_retries : the number of iterations to better the answer

        Returns:
            A tuple containing the generated answer and a list of documentary
            sources originating from the underlying Vector DB or the web.
        """

        inputs = {
            "question": question,
            "max_retries": max_retries,
        }

        last_generation_ai_message = AIMessage("No content found")
        last_generation_documents = []

        for event in self.state_graph.stream(inputs, stream_mode="values"):

            if "generation" in event :
                last_generation_ai_message = event.get("generation", last_generation_ai_message)
                last_generation_documents = event.get("documents", last_generation_documents)

        # End of graph, return answer
        return last_generation_ai_message, last_generation_documents

    ### Nodes
    def retrieve(self, state : dict):
        """
        Retrieve documents from vectorstore

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        self.logger.info('- Retrieving documents from vectorstore')
        question = state["question"]

        # Write retrieved documents to documents key in state
        documents = self.embedding_interface.get_documents(query=question)

        return {"documents": documents}


    def generate(self, state : dict):
        """
        Generate answer using RAG on retrieved documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        self.logger.info('- RAG Generation')
        question = state["question"]
        documents = state["documents"]
        loop_step = state.get("loop_step", 0)

        # RAG generation
        generation = self.llm_agent.run_rag_query_on_documents(json_mode=False,
                                              documents=documents,
                                              question=question)

        return {"generation": generation, "loop_step": loop_step + 1}


    def grade_documents(self, state : dict):
        """
        Determines whether the retrieved documents are relevant to the question
        If any document is not relevant, we will set a flag to run web search

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Filtered out irrelevant documents and updated web_search state
        """

        self.logger.info('- Checking document relevance to the question')
        question = state["question"]
        documents = state["documents"]

        # Score each doc
        filtered_docs = []
        web_search = "No"

        for d in documents:

            grade = self.retrieval_grader.execute(llm_agent=self.llm_agent,
                                                  question=question,
                                                  answer=d.page_content)["binary_score"]

            # Document relevant
            if grade.lower() == "yes":
                self.logger.info('- 🔵 Document is relevant')
                filtered_docs.append(d)
            # Document not relevant
            else:
                self.logger.info('- 🔴 Document is not relevant')
                # We do not include the document in filtered_docs
                # We set a flag to indicate that we want to run web search
                web_search = "Yes"
                continue

        return {"documents": filtered_docs, "web_search": web_search}

    def web_search(self, state : dict):
        """
        Web search based based on the question

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Appended web results to documents
        """

        self.logger.info('- Running websearch...')
        question = state["question"]
        documents = state.get("documents", [])

        # Web search
        if self.web_search_tool :

            docs = self.web_search_tool.invoke({"query": question})

            for web_document in docs :

                documents.append(Document(page_content=web_document["content"],
                                          metadata={"source": web_document["url"],
                                                    "title": web_document["url"],
                                                    "origin": "web search"}))

        return {"documents": documents}

    ### Edges Definition

    def route_question(self,state : dict):
        """
        Route question to web search or RAG

        Args:
            state (dict): The current graph state

        Returns:
            str: Next node to call
        """

        self.logger.info('- Routing question')

        source = self.router.route(self.llm_agent,
                                   state["question"],
                                   self.embedding_interface.get_store_topics())["datasource"]

        if source == "websearch":
            self.logger.info('\t→ Routing to websearch')
            return "websearch"
        elif source == "vectorstore":
            self.logger.info('\t→ Routing to RAG')
            return "vectorstore"


    def decide_to_generate(self, state : dict):
        """
        Determines whether to generate an answer, or add web search

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """

        self.logger.info('- Assessing whether performing further websearch or not')
        web_search = state["web_search"]

        if web_search == "Yes":
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            self.logger.info('\t → All documents are not relevant, routing to websearch')
            return "websearch"
        else:
            # We have relevant documents, so generate answer
            self.logger.info('\t → Documents are relevant - generate answer')
            return "generate"


    def grade_generation_v_documents_and_question(self, state):
        """
        Determines whether the generation is grounded in the document and answers question

        Args:
            state (dict): The current graph state

        Returns:
            str: Decision for next node to call
        """

        self.logger.info('- Checking for hallucinations')
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        max_retries = state.get("max_retries", 3)  # Default to 3 if not provided

        documents_as_text = self.llm_agent.concatenate_documents(documents)

        grade = self.hallucination_grader.execute(llm_agent=self.llm_agent,
                                                  question=documents_as_text,
                                                  answer=generation.content)

        # Check hallucination
        if grade == "yes":

            self.logger.info('\t→ Verifying pertinence of answer vs question')

            # Check question-answering
            grade = self.answer_grader.execute(llm_agent=self.llm_agent,
                                               question=question,
                                               answer=generation.content)["binary_score"]

            if grade == "yes":
                self.logger.info('\t→ 🔵 Generation addresses question.')
                return "useful"
            elif state["loop_step"] <= max_retries:
                self.logger.info('\t→ 🔴 Generation does not addresses question.')
                return "not useful"
            else:
                self.logger.info('\t→ ⭕ Max retry count reached')
                return "max retries"

        elif state["loop_step"] <= max_retries:
            self.logger.info('- Generation is not grounded in documents, retrying')
            return "not supported"
        else:
            self.logger.info('Max retry count reached')
            return "max retries"
