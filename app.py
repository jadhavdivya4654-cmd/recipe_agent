import streamlit as st
from crewai import Agent, Task, Crew, LLM
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()

# Setup Groq LLM for CrewAI
llm = LLM(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

st.title("🍳 Recipe Suggestor AI — CrewAI Version")

# --------------------------
# Create Agents
# --------------------------

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

# --------------------------
# Create Recipe Function
# --------------------------

def generate_recipe_with_crewai(ingredients):
    task1 = Task(
        description=f"Create a recipe using these ingredients: {ingredients}",
        agent=recipe_agent,
    )

    task2 = Task(
        description="Give the nutrition details of this recipe.",
        agent=nutrition_agent,
    )

    task3 = Task(
        description="Improve this recipe for better health.",
        agent=improvement_agent,
    )

    crew = Crew(
        agents=[recipe_agent, nutrition_agent, improvement_agent],
        tasks=[task1, task2, task3]
    )

    return crew.kickoff()

# --------------------------
# Streamlit UI
# --------------------------

ingredients = st.text_input("Enter ingredients (comma separated):")

if st.button("Generate Recipe"):
    if ingredients.strip():
        with st.spinner("Generating..."):
            result = generate_recipe_with_crewai(ingredients)
            st.success("Your Recipe is Ready!")
            st.write(result)
    else:
        st.warning("Please enter ingredients.")
