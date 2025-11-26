from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from typing import List
from crewai.agents.agent_builder.base_agent import BaseAgent

# Import tools you created
from tools import ExtractFoodIngredients
from tools import (
    extract_ingredients,
    nutrition_lookup,
    diabetic_impact,
    meal_plan_generator,
    glucose_predictor,
    grocery_advisor,
    exercise_recommender,
    habit_analyzer
)


@CrewBase
class DiabetesAppCrew:

    agents: List[BaseAgent]
    tasks: List[task]

    # ---------------------------------------------------------
    # 1. FOOD ANALYSIS AGENT
    # ---------------------------------------------------------
    @agent
    def food_analysis_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["food_analysis_agent"],
            tools=[
                extract_ingredients,   # Vision → ingredient list
                nutrition_lookup,      # API → nutritional facts
                diabetic_impact       # attach your tool here
            ]
        )

    # ---------------------------------------------------------
@agent
def meal_planner_agent(self) -> Agent:
    return Agent(
        config=self.agents_config["meal_planner_agent"],
        tools=[meal_plan_generator]
    )

@agent
def glucose_prediction_agent(self) -> Agent:
    return Agent(
        config=self.agents_config["glucose_prediction_agent"],
        tools=[glucose_predictor]
    )

@agent
def shopping_assistant_agent(self) -> Agent:
    return Agent(
        config=self.agents_config["shopping_assistant_agent"],
        tools=[grocery_advisor]
    )

@agent
def activity_coach_agent(self) -> Agent:
    return Agent(
        config=self.agents_config["activity_coach_agent"],
        tools=[exercise_recommender]
    )

@agent
def habit_tracker_agent(self) -> Agent:
    return Agent(
        config=self.agents_config["habit_tracker_agent"],
        tools=[habit_analyzer]
    )