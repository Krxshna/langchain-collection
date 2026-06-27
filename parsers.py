from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.2)

template1 = PromptTemplate(
    template = 'Write a detailed report on {topic}',
    input_variables = ['topic']
    )

template2 = PromptTemplate(
    template = 'Summarize the following text in 10 lines: {text}',
    input_variables = ['text']
    )

parser = StrOutputParser()

chain = template1 | model |parser | template2 | model | parser

result = chain.invoke({'topic': 'Fine tuning ai model for any use case: technical view'})
print(result)