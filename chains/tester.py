from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.2)

prompt = PromptTemplate(
    template = 'Give me a detailed explanation of {topic}',
    input_variables = ['topic']
    )

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({'topic': 'techniques of Fine tuning ai model'})

print(result)

chain.get_graph().print_ascii()