from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

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

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic': 'IPL 2026'})
print(result)

chain.get_graph().print_ascii()