import streamlit as st
from crewai import Agent, Task, Crew, LLM
import os
from dotenv import load_dotenv
import json
import re

# Load API key
load_dotenv()

# Setup Groq LLM for CrewAI
llm = LLM(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

st.set_page_config(page_title="Recipe Suggestor AI", layout="wide")
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
# Clean & Extract Sections
# --------------------------

def extract_section(text, header):
    """Extracts a block that starts with a header like 'Ingredients:'."""
    pattern = rf"{header}[:\n]+(.*?)(?:\n[A-Z][A-Za-z ]+[:\n]|$)"
    match = re.search(pattern, text, re.S)
    return match.group(1).strip() if match else None

# --------------------------
# Create Recipe Function
# --------------------------

def generate_recipe_with_crewai(ingredients):
    task1 = Task(
        description=f"Create a recipe using these ingredients: {ingredients}",
        agent=recipe_agent,
        expected_output="A clear, step-by-step recipe."
    )

    task2 = Task(
        description="Give the nutrition details of this recipe.",
        agent=nutrition_agent,
        expected_output="A simple nutrition breakdown."
    )

    task3 = Task(
        description="Improve this recipe for better health.",
        agent=improvement_agent,
        expected_output="A healthier version of the recipe."
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

        # Convert to string
        raw = str(result)

        # -------- Extract Parts Automatically --------
        recipe = extract_section(raw, "Recipe") or "Not Found"
        ingredients_block = extract_section(raw, "Ingredients")
        steps_block = extract_section(raw, "Instructions") or extract_section(raw, "Steps")
        nutrition_block = extract_section(raw, "Nutrition")
        improvement_block = extract_section(raw, "Improvements") or extract_section(raw, "Healthier Version")

        col1, col2 = st.columns([2, 1])

        # ---------- LEFT SIDE ----------
        with col1:
            st.header("📌 Final Recipe")
            st.write(recipe)

            if ingredients_block:
                st.subheader("🥗 Ingredients")
                st.write(ingredients_block)

            if steps_block:
                st.subheader("👩‍🍳 Instructions")
                st.write(steps_block)

            if improvement_block:
                with st.expander("💡 Healthy Improvements"):
                    st.write(improvement_block)

        # ---------- RIGHT SIDE ----------
        with col2:
            st.header("🍎 Nutrition Breakdown")
            st.write(nutrition_block or "Nutrition not found")

        # Raw Debug
        st.markdown("---")
        with st.expander("🔍 Raw CrewAI Output (Debug)"):
            st.write(raw)

    else:
        st.warning("Please enter ingredients.")
