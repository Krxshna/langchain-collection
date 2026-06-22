from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate,load_prompt
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.2)

chat_history = [
    SystemMessage(content="You are a helpful Personal Coach that provides advice on fitness and nutrition. You provide personalized recommendations based on the user's goals and preferences. Always be supportive and encouraging in your responses."),
]

while True:
    user_input = input("User: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input.lower() in ['exit', 'quit']:
        print("Exiting the chatbot. Goodbye!")
        break
    result = model.invoke(chat_history)
    print("Chatbot:", result.content)
    chat_history.append(AIMessage(content=result.content))
    
print("Chat history: ", chat_history)