from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant"
)

recipe_agent = Agent(
    role="Recipe Creator",
    goal="Create tasty, clear, step-by-step recipes",
    backstory="You are a chef with excellent cooking skills.",
    llm=llm
)

nutrition_agent = Agent(
    role="Nutrition Expert",
    goal="Give simple nutritional breakdown",
    backstory="You are a nutrition coach.",
    llm=llm
)

improvement_agent = Agent(
    role="Health Improver",
    goal="Make the recipe healthier and easier",
    backstory="You suggest improvements to meals.",
    llm=llm
)
