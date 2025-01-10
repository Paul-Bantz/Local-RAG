""" Definition for various grading logics. Given a an input and an output,
grade the relevance of the output corresponding to the input.
"""

import abc
from LocalRAG.RAG.llm_agent import LLMAgent

class Grader(metaclass=abc.ABCMeta):
    """ Abstract class definition for a grading agent

    Allows the generation of a grade for a given answer and a question

    Attributes:
        instructions: base grader llm instruction
        prompt: complete llm prompt for grader initialisation
    """

    instructions: str = None
    prompt: str = None

    def grade(self, llm_agent: LLMAgent, json_mode: bool, question: str, answer:str) -> str:
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

        return llm_agent.run_query(json_mode=json_mode,
                                   instructions=self.instructions,
                                   query=prompt_formatted)

class AnswerGrader(Grader):
    """ Grade an answer given a question
    """

    def __init__(self):

        # pylint: disable=line-too-long
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

    def execute(self, llm_agent: LLMAgent, question: str, answer:str) -> str:
        """ Given a question, grade the answer and return a json object containing
        two keys :
        - binary_score : (yes or no if the answer is relevant to the
        question)
        - explanation : giving the reasoning for the grading

        Args:
            llm_agent : the agent used for the grading
            question : the question
            answer : the answer to grade for relevance in relation to the
                question
        """

        return super().grade(llm_agent=llm_agent,
                             json_mode=True,
                             question=question,
                             answer=answer)


class HallucinationGrader(Grader):
    """ Determine if an answer is in context of given a question
    """

    def __init__(self):

        # pylint: disable=line-too-long
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

    def execute(self, llm_agent: LLMAgent, question: str, answer:str) -> str:
        """ Given a question, grade the answer and return a json object containing
        two keys :
        - binary_score : (yes or no if the answer is out of subject in relation
         to the question)
        - explanation : giving the reasoning for the grading

        Args:
            llm_agent : the agent used for the grading
            question : the question
            answer : the answer to grade for relevance in relation to the
                question
        """
        return super().grade(llm_agent=llm_agent,
                             json_mode=True,
                             question=question,
                             answer=answer)

class RetrievalGrader(Grader):
    """ Determine if an answer is relevant to the question : ie in context,
    determine if the returned output of the rag is relevant to the question.
    """

    def __init__(self):

        # pylint: disable=line-too-long
        self.prompt = """Here is the retrieved document: \n\n {answer} \n\n Here is the user question: \n\n {question}.

        This carefully and objectively assess whether the document contains at least some information that is relevant to the question.

        Return JSON with single key, binary_score, that is 'yes' or 'no' score to indicate whether the document contains at least some information that is relevant to the question."""

        # pylint: disable=line-too-long
        self.instructions = """You are a grader assessing relevance of a retrieved document to a user question.

        If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant."""

    def execute(self, llm_agent: LLMAgent, question: str, answer:str) -> str:
        """ Given a question, grade the answer and return a json object containing
        two keys :
        - binary_score : (yes or no if the answer is out of subject in relation
         to the question)
        - explanation : giving the reasoning for the grading

        Args:
            llm_agent : the agent used for the grading
            question : the question
            answer : the answer to grade for relevance in relation to the
                question
        """

        return super().grade(llm_agent=llm_agent,
                             json_mode=True,
                             question=question,
                             answer=answer)
