import operator

from typing_extensions import TypedDict
from typing import List, Annotated
from langchain.schema import Document
from langgraph.graph import END

from langgraph.graph import StateGraph

from langchain_community.tools.tavily_search import TavilySearchResults

from RAG.Embeddings.EmbeddingInterface import EmbeddingInterface
from RAG.Graders.Grader import AnswerGrader, HallucinationGrader, RetrievalGrader
from RAG.LLMInterface import LLMInterface
from RAG.Router import Router
from RAG.Rag import Rag

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

    state_graph : StateGraph

    embedding_interface : EmbeddingInterface
    llm_interface : LLMInterface

    router : Router
    rag: Rag

    retrieval_grader : RetrievalGrader
    hallucination_grader : HallucinationGrader
    answer_grader : AnswerGrader
    
    web_search_tool : TavilySearchResults

    def __init__(self, 
                 embeding_interface: EmbeddingInterface, 
                 llm_interface: LLMInterface ):

        self.embedding_interface = embeding_interface
        self.llm_interface = llm_interface

        self.rag = Rag()
        self.router = Router()

        self.answer_grader = AnswerGrader()
        self.hallucination_grader = HallucinationGrader()
        self.retrieval_grader = RetrievalGrader()

        self.web_search_tool = TavilySearchResults(k=3)

        self.construct_workflow()
        
    def construct_workflow(self) :

        workflow = StateGraph(GraphState)

        # Define the nodes
        workflow.add_node("websearch", self.web_search)  # web search
        workflow.add_node("retrieve", self.retrieve)  # retrieve
        workflow.add_node("grade_documents", self.grade_documents) # grade documents
        workflow.add_node("generate", self.generate)  # generate

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

    def query(self, question:str, max_retries:int) : 

        inputs = {
            "question": question,
            "max_retries": max_retries,
        }

        for event in self.state_graph.stream(inputs, stream_mode="values"):
            print(event)

    ### Nodes
    def retrieve(self, state : dict):
        """
        Retrieve documents from vectorstore

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        print("---RETRIEVE---")
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
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]
        loop_step = state.get("loop_step", 0)

        # RAG generation
        generation = self.rag.query(llm=self.llm_interface.llm, 
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

        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]

        # Score each doc
        filtered_docs = []
        web_search = "No"

        for d in documents:

            grade = self.retrieval_grader.grade(llm=self.llm_interface.llm_json_mode, 
                                                question=question, 
                                                answer=d.page_content)["binary_score"]

            # Document relevant
            if grade.lower() == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            # Document not relevant
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
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

        print("---WEB SEARCH---")
        question = state["question"]
        documents = state.get("documents", [])

        # Web search
        docs = self.web_search_tool.invoke({"query": question})
        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=web_results)
        documents.append(web_results)
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

        print("---ROUTE QUESTION---")
        
        source = self.router.route(self.llm_interface.llm_json_mode, 
                                   state["question"])["datasource"]
        
        if source == "websearch":
            print("---ROUTE QUESTION TO WEB SEARCH---")
            return "websearch"
        elif source == "vectorstore":
            print("---ROUTE QUESTION TO RAG---")
            return "vectorstore"


    def decide_to_generate(state):
        """
        Determines whether to generate an answer, or add web search

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """

        print("---ASSESS GRADED DOCUMENTS---")
        web_search = state["web_search"]

        if web_search == "Yes":
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            print(
                "---DECISION: NOT ALL DOCUMENTS ARE RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
            )
            return "websearch"
        else:
            # We have relevant documents, so generate answer
            print("---DECISION: GENERATE---")
            return "generate"


    def grade_generation_v_documents_and_question(self, state):
        """
        Determines whether the generation is grounded in the document and answers question

        Args:
            state (dict): The current graph state

        Returns:
            str: Decision for next node to call
        """

        print("---CHECK HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        max_retries = state.get("max_retries", 3)  # Default to 3 if not provided

        grade = self.hallucination_grader.grade(llm=self.llm_interface.llm_json_mode, 
                                                question=self.rag.format_docs(documents), 
                                                answer=generation.content)

        # Check hallucination
        if grade == "yes":

            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            # Check question-answering
            print("---GRADE GENERATION vs QUESTION---")

            grade = self.answer_grader.grade(llm=self.llm_interface.llm_json_mode,
                                             question=question, 
                                             answer=generation.content)["binary_score"]
            
            if grade == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            elif state["loop_step"] <= max_retries:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
            else:
                print("---DECISION: MAX RETRIES REACHED---")
                return "max retries"
            
        elif state["loop_step"] <= max_retries:
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RETRY---")
            return "not supported"
        else:
            print("---DECISION: MAX RETRIES REACHED---")
            return "max retries"