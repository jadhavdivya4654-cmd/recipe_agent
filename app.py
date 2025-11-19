import streamlit as st
from crewai import Agent, Task, Crew
from groq import Groq
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.title("🍳 Recipe Suggestor AI — CrewAI Version")

# --------------------------
# Create Agents
# --------------------------

recipe_agent = Agent(
    role="Recipe Creator",
    goal="Create tasty, clear, step-by-step recipes",
    backstory="You are a chef with excellent cooking skills."
)

nutrition_agent = Agent(
    role="Nutrition Expert",
    goal="Give simple nutritional breakdown",
    backstory="You are a nutrition coach."
)

improvement_agent = Agent(
    role="Health Improver",
    goal="Make the recipe healthier and easier",
    backstory="You suggest improvements to meals."
)

# --------------------------
# Create Recipe Function
# --------------------------

def generate_recipe_with_crewai(ingredients):
    task1 = Task(
        description=f"Create a recipe using these ingredients: {ingredients}",
        agent=recipe_agent,
        expected_output="A complete step-by-step recipe."
    )

    task2 = Task(
        description="Analyze nutrition of this recipe.",
        agent=nutrition_agent,
        expected_output="A simple, clear nutrition breakdown."
    )

    task3 = Task(
        description="Improve the recipe to make it healthier.",
        agent=improvement_agent,
        expected_output="A healthier version of the recipe."
    )

    crew = Crew(
        agents=[recipe_agent, nutrition_agent, improvement_agent],
        tasks=[task1, task2, task3],
        verbose=True
    )

    return crew.kickoff()

# --------------------------
# Streamlit UI
# --------------------------

ingredients = st.text_input("Enter ingredients (comma separated):")

if st.button("Generate Recipe"):
    if ingredients.strip() == "":
        st.warning("Please enter ingredients!")
    else:
        with st.spinner("Cooking the best recipe for you..."):
            output = generate_recipe_with_crewai(ingredients)
            st.success("Your Recipe is Ready!")
            st.write(output)
