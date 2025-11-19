def generate_recipe_with_crewai(ingredients):
    task1 = Task(
        description=f"Create a detailed and clear recipe using these ingredients: {ingredients}",
        agent=recipe_agent,
        expected_output="A complete step-by-step recipe."
    )

    task2 = Task(
        description="Analyze nutrition of the recipe.",
        agent=nutrition_agent,
        expected_output="A simple nutrition breakdown."
    )

    task3 = Task(
        description="Improve the recipe for better health.",
        agent=improvement_agent,
        expected_output="A healthier and easier version of the recipe."
    )

    crew = Crew(
        agents=[recipe_agent, nutrition_agent, improvement_agent],
        tasks=[task1, task2, task3],
        verbose=True
    )

    return crew.kickoff()

