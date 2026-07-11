from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

load_dotenv()

model1 = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.2)

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(temperature=0.5, max_new_tokens=100)
)

model2 = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template = 'Generate 10 tough interview questions on {topic1}',
    input_variables = ['topic']
    )

prompt2 = PromptTemplate(
    template = 'Generate 5 moderate interview questions on {topic2}',
    input_variables = ['topic']
    )

prompt3 = PromptTemplate(
    template = 'Merge the interview questions and get 10 total inteview question and answers for \n ai/ml -> {ai}, and data structures -> {ds}',
    input_variables = ['ai', 'ds']
    )

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {
    'ai': prompt1 | model1 | parser, 
    'ds': prompt2 | model2 | parser
    }
    )

merged_chain = prompt3 | model1 | parser

chain = parallel_chain | merged_chain

topic1 = 'ai/ml'
topic2 = 'data structures'

result = chain.invoke({'topic1': topic1, 'topic2': topic2})

print(result)

chain.get_graph().print_ascii()
