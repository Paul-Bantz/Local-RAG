import abc
import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

class Grader(metaclass=abc.ABCMeta):
    """ Abstract class definition for a grading agent

    Allows the generation of a grade for a given answer and a question

    Attributes:
        instructions: base grader llm instruction
        prompt: complete llm prompt for grader initialisation
    """

    instructions: str = None
    prompt: str = None

    def grade(self, llm: BaseChatModel, question: str, answer:str) -> str:
        """ Grade an answer given its associated question

        The result will vary depending on the base instructions and prompt of
        the grader

        Args:
            question: the question corresponding to the given answer
            answer: the answer to be graded for correctness
        
        """
        prompt_formatted = self.prompt.format(
                question=question, 
                answer=answer
            )
        
        result = llm.invoke(
                [SystemMessage(content=self.instructions)]
                + [HumanMessage(content=prompt_formatted)]
            )
        
        return json.loads(result.content)

class AnswerGrader(Grader):
    
    def __init__(self):
        
        self.prompt = """QUESTION: \n\n {question} \n\n STUDENT ANSWER: {answer}. 

        Return JSON with two two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER meets the criteria. And a key, explanation, that contains an explanation of the score."""
        
        self.instructions = """You are a teacher grading a quiz. 

        You will be given a QUESTION and a STUDENT ANSWER. 

        Here is the grade criteria to follow:

        (1) The STUDENT ANSWER helps to answer the QUESTION

        Score:

        A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score. 

        The student can receive a score of yes if the answer contains extra information that is not explicitly asked for in the question.

        A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.

        Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

        Avoid simply stating the correct answer at the outset."""

class HallucinationGrader(Grader):

    def __init__(self):

        self.prompt = """FACTS: \n\n {question} \n\n STUDENT ANSWER: {answer}. 

        Return JSON with two two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER is grounded in the FACTS. And a key, explanation, that contains an explanation of the score."""

        self.instructions = """

        You are a teacher grading a quiz. 

        You will be given FACTS and a STUDENT ANSWER. 

        Here is the grade criteria to follow:

        (1) Ensure the STUDENT ANSWER is grounded in the FACTS. 

        (2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

        Score:

        A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score. 

        A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.

        Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

        Avoid simply stating the correct answer at the outset."""

class RetrievalGrader(Grader):

    def __init__(self):
        
        self.prompt = """Here is the retrieved document: \n\n {document} \n\n Here is the user question: \n\n {question}. 

        This carefully and objectively assess whether the document contains at least some information that is relevant to the question.

        Return JSON with single key, binary_score, that is 'yes' or 'no' score to indicate whether the document contains at least some information that is relevant to the question."""

        self.instructions = """You are a grader assessing relevance of a retrieved document to a user question.

        If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant."""
