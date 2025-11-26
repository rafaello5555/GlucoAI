from dotenv import load_dotenv
import os
import gradio as gr

load_dotenv()  # Load environment variables from .env

# Optional: verify keys are loaded
print(os.getenv("WATSONX_API_KEY"))
print(os.getenv("WATSONX_URL"))
print(os.getenv("WATSONX_PROJECT_ID"))
print(os.getenv("USDA_API_KEY"))

from  tools import extract_ingredients,  nutrition_lookup, diabetic_impact, meal_plan_generator, glucose_predictor, grocery_advisor, exercise_recommender,  habit_analyzer

# -------------------------
# 1. Food Analysis Function
# -------------------------
# app.py (or wherever analyze_food is defined)
def analyze_food(image_or_url):
    # Step 1: Extract ingredients from image
    ingredients_text = extract_ingredients(image_or_url)  # WatsonX returns a string

    # Step 2: Clean and split ingredients into a list
    # Remove bullets, extra characters, and split by lines
    ingredients_list = []
    for line in ingredients_text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Remove leading bullets or numbering
        if line.startswith("* "):
            line = line[2:].strip()
        elif line[0].isdigit() and line[1] in [".", ")"]:
            line = line[2:].strip()
        ingredients_list.append(line)

    # Step 3: Lookup nutrition info for each ingredient
    nutrition_data = []
    for item in ingredients_list:
        try:
            result = nutrition_lookup(item)
        except Exception as e:
            result = f"USDA API error for '{item}': {str(e)}"
        nutrition_data.append(result)

    # Step 4: Analyze diabetic impact for each ingredient
    impact_results = []
    for data in nutrition_data:
        try:
            impact = diabetic_impact(data)
        except Exception as e:
            impact = f"Error analyzing impact for '{data}': {str(e)}"
        impact_results.append(impact)

    return {
        "ingredients": ingredients_list,
        "nutrition": nutrition_data,
        "diabetic_impact": impact_results
    }


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
     img_file = gr.File(label="Upload Image")          # for local files
     img_url = gr.Textbox(label="Or enter Image URL")  # optional URL
     food_output = gr.Textbox(label="Food Analysis Result")
     btn_food = gr.Button("Analyze Food")

    def handle_food_input(file, url):
        if file:  # If user uploaded a file
            return analyze_food(file.name)
        elif url:  # If user provided a URL
            return analyze_food(url)
        else:
            return "Please upload a file or provide a URL."

    btn_food.click(handle_food_input, inputs=[img_file, img_url], outputs=food_output)
    
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
