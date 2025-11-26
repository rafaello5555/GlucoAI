from dotenv import load_dotenv
import os

load_dotenv()  # load variables from .env

WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")
WATSONX_URL = os.getenv("WATSONX_URL")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")

USDA_API_KEY = os.getenv("USDA_API_KEY")

print(WATSONX_API_KEY, WATSONX_URL, WATSONX_PROJECT_ID)
print(USDA_API_KEY)




# app.py
import gradio as gr
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

# -------------------------
# 1. Food Analysis Function
# -------------------------
def analyze_food(image_or_url):
    # Step 1: Extract ingredients
    ingredients = extract_ingredients(image_or_url)
    
    # Step 2: Get nutrition facts for each ingredient
    nutrition_data = []
    for item in ingredients.split(","):
        nutrition_data.append(nutrition_lookup(item.strip()))
    
    # Step 3: Combine nutrition data and calculate diabetic impact
    impact_results = []
    for data in nutrition_data:
        impact_results.append(diabetic_impact(data))
    
    return "\n\n".join(impact_results)

# -------------------------
# 2. Meal Planner Function
# -------------------------
def plan_meals(preferences):
    return meal_plan_generator(preferences)

# -------------------------
# 3. Glucose Predictor Function
# -------------------------
def predict_glucose_level(carbs, previous_glucose=100, hours_since_last_meal=3):
    return glucose_predictor(carbs, previous_glucose, hours_since_last_meal)

# -------------------------
# 4. Grocery Advisor Function
# -------------------------
def recommend_groceries(items):
    return grocery_advisor(items)

# -------------------------
# 5. Exercise Recommender
# -------------------------
def recommend_exercise(current_glucose, fitness_level="moderate"):
    return exercise_recommender(current_glucose, fitness_level)

# -------------------------
# 6. Habit Analyzer
# -------------------------
def analyze_habits(meals, steps, sleep_hours):
    logs = {
        "meals": meals,
        "steps": steps,
        "sleep_hours": sleep_hours
    }
    return habit_analyzer(logs)

# -------------------------
# Gradio Interface
# -------------------------
with gr.Blocks() as demo:
    gr.Markdown("# Diabetes Assistant App")
    
    with gr.Tab("Food Analysis"):
        img_input = gr.Textbox(label="Image URL or local path")
        food_output = gr.Textbox(label="Food Analysis Result")
        btn_food = gr.Button("Analyze Food")
        btn_food.click(analyze_food, inputs=img_input, outputs=food_output)
    
    with gr.Tab("Meal Planner"):
        meal_pref = gr.Dropdown(["keto", "vegan", "vegetarian", "low-carb"], label="Meal Preference")
        meal_output = gr.Textbox(label="Weekly Meal Plan")
        btn_meal = gr.Button("Generate Meal Plan")
        btn_meal.click(plan_meals, inputs=meal_pref, outputs=meal_output)
    
    with gr.Tab("Glucose Prediction"):
        carbs_input = gr.Number(label="Carbs (g)")
        prev_glucose = gr.Number(label="Previous Glucose (mg/dL)", value=100)
        hours_since_meal = gr.Number(label="Hours Since Last Meal", value=3)
        glucose_output = gr.Textbox(label="Predicted Glucose")
        btn_glucose = gr.Button("Predict Glucose")
        btn_glucose.click(predict_glucose_level, inputs=[carbs_input, prev_glucose, hours_since_meal], outputs=glucose_output)
    
    with gr.Tab("Grocery Advisor"):
        items_input = gr.Textbox(label="Grocery Items (comma-separated)")
        grocery_output = gr.Textbox(label="Recommendations")
        btn_grocery = gr.Button("Recommend")
        btn_grocery.click(recommend_groceries, inputs=items_input, outputs=grocery_output)
    
    with gr.Tab("Exercise Recommender"):
        curr_glucose = gr.Number(label="Current Glucose (mg/dL)")
        fitness_lvl = gr.Dropdown(["light", "moderate"], label="Fitness Level")
        exercise_output = gr.Textbox(label="Exercise Recommendation")
        btn_exercise = gr.Button("Recommend Exercise")
        btn_exercise.click(recommend_exercise, inputs=[curr_glucose, fitness_lvl], outputs=exercise_output)
    
    with gr.Tab("Habit Tracker"):
        meals = gr.Number(label="Meals Today")
        steps = gr.Number(label="Steps Today")
        sleep_hours = gr.Number(label="Sleep Hours")
        habit_output = gr.Textbox(label="Weekly Habit Insights")
        btn_habit = gr.Button("Analyze Habits")
        btn_habit.click(analyze_habits, inputs=[meals, steps, sleep_hours], outputs=habit_output)

# Launch app
demo.launch()
