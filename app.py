import streamlit as st
from crewai import Agent, Task, Crew
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

# Create Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Create a simple LLM wrapper for CrewAI
def groq_llm(prompt):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message["content"]

# Streamlit UI
st.title("🍳 Recipe Suggestor AI — CrewAI Version")
ingredients = st.text_input("Enter ingredients (comma separated):")

# Agents
recipe_agent = Agent(
    role="Recipe Creator",
    goal="Create step-by-step tasty recipes",
    backstory="Expert chef with years of culinary knowledge.",
    llm=groq_llm
)

nutrition_agent = Agent(
    role="Nutrition Specialist",
    goal="Explain nutrition",
    backstory="Certified dietician.",
    llm=groq_llm
)

improvement_agent = Agent(
    role="Health Improver",
    goal="Suggest healthier version",
    backstory="Fitness and health expert.",
    llm=groq_llm
)

# CrewAI Process
def generate_recipe_with_crewai(ingredients):
    task1 = Task(
        description=f"Create a full recipe using: {ingredients}",
        agent=recipe_agent,
    )

    task2 = Task(
        description="Provide the nutrition of this recipe.",
        agent=nutrition_agent,
    )

    task3 = Task(
        description="Improve the recipe to be healthier.",
        agent=improvement_agent,
    )

    crew = Crew(
        agents=[recipe_agent, nutrition_agent, improvement_agent],
        tasks=[task1, task2, task3]
    )

    return crew.kickoff()

# Streamlit button
if st.button("Generate Recipe"):
    if ingredients.strip():
        with st.spinner("Preparing your recipe..."):
            result = generate_recipe_with_crewai(ingredients)
            st.success("Done!")
            st.write(result)
    else:
        st.warning("Please enter ingredients.")
