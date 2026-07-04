from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.2)

class reviews(TypedDict):
    key_themes: Annotated[list[str], "List of key themes mentioned in the review"]
    summary: Annotated[str, "Brief summary of the review"]
    sentiment: Annotated[str, "Overall sentiment of the review (positive/negative/neutral)"]
    pros: Annotated[Optional[list[str]], "List of pros mentioned in the review, if any"]
    cons: Annotated[Optional[list[str]], "List of cons mentioned in the review, if any]"]

structured_model = model.with_structured_output(reviews)
    
result = structured_model.invoke("""
                                 Samsung s23 is really good, I love the camera and the battery life is amazing and the display is really bright and vibrant. However, the price is a bit high and the phone is a bit heavy.
                                 Pros:
                                   - Great camera with 48MP sensor
                                   - Long battery life with 3900mAh capacity
                                   - Bright and vibrant AmOLED display
                                   - nice design
                                   - light weight
                                 Cons:
                                   - Expensive
                                   - Heating issues
                                   - No headphone jack
                                   - No expandable storage
                                 Overall, I would recommend the Samsung s23 to anyone looking for a high-end smartphone with excellent camera and battery life, but be prepared to pay a premium price.   
                                 
                                 """)

print(result)