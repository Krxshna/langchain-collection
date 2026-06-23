from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

parser = JsonOutputParser()

template = PromptTemplate(
    template = 'Give details like name, age, profession, and hobbies about a fictional character \n {format_instructions}',
    input_variables = [],
    partial_variables = {'format_instructions': parser.get_format_instructions()}
    )

chain = template | model | parser

result = chain.invoke({})
print(result)