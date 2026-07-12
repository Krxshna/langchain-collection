from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableSequence, RunnableParallel, RunnablePassthrough,RunnableLambda

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.2)

prompt1 = PromptTemplate(
    template = 'Write a detailed report on {topic}',
    input_variables = ['topic']
)

prompt2 = PromptTemplate(
    template = 'Summarize the following text in 4 lines: {text}',
    input_variables = ['text']
)

parser = StrOutputParser()

report_gen_chain = prompt1 | model | parser

print(report_gen_chain.invoke({'topic': 'Utilizing AI for Observability in Kubernetes: Techniques and Best Practices'}))

branch_chain = RunnableBranch(
    (lambda x: len(x.split())>300, prompt2 | model | parser),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_gen_chain, branch_chain)

print(final_chain.invoke({'topic': 'Utilizing AI for Observability in Kubernetes: Techniques and Best Practices'}))