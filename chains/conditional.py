from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.2)

parser = StrOutputParser()

class Feedback(BaseModel):
    feedback: Literal["positive", "negative"] = Field(description="Give the sentiment of the Feedback")

parser1 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template = 'Classify the following feedback as positive or negative \n {feedback} \n {format_instructions}',
    input_variables = ['feedback'],
    partial_variables = {'format_instructions': parser1.get_format_instructions()}
)

classify_chain = prompt1 | model | parser1

prompt2 = PromptTemplate(
    template = 'Give an appropriate response to this positive feedback \n {feedback}',
    input_variables = ['feedback']
)

prompt3 = PromptTemplate(
    template = 'Give an appropriate response to this negative feedback \n {feedback}',
    input_variables = ['feedback']
)

branch_chain = RunnableBranch(
    (lambda x: x.feedback == "positive", prompt2 | model | parser),
    (lambda x: x.feedback == "negative", prompt3 | model |parser),
    RunnableLambda(lambda x: "Could not classify the feedback")
)

chain = classify_chain | branch_chain

print(chain.invoke({'feedback': 'The product is amazing!'}))

chain.get_graph().print_ascii()