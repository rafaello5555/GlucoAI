# tools.py
import os
import base64
import requests
from io import BytesIO
import random
from dotenv import load_dotenv
from langchain.tools import tool
from ibm_watsonx_ai.foundation_models import ModelInference

# ------------------------------
# Load environment variables
# ------------------------------
load_dotenv()

WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
WATSONX_CREDS = {
    "apikey": os.getenv("WATSONX_API_KEY"),
    "url": os.getenv("WATSONX_URL")
}

USDA_API_KEY = os.getenv("USDA_API_KEY")

# ------------------------------
# 1. IMAGE → INGREDIENT EXTRACTOR
# ------------------------------
@tool("extract_ingredients", return_direct=False)
def extract_ingredients(image_input: str) -> str:
    """
    Extracts ingredients from a food image using WatsonX LLaMA Vision model.
    Accepts: Local file path or URL.
    Returns: A text list of ingredients.
    """
    if image_input.startswith("http"):
        response = requests.get(image_input)
        response.raise_for_status()
        image_bytes = BytesIO(response.content)
    else:
        if not os.path.isfile(image_input):
            raise FileNotFoundError(f"No file found at: {image_input}")
        with open(image_input, "rb") as f:
            image_bytes = BytesIO(f.read())

    encoded_image = base64.b64encode(image_bytes.read()).decode("utf-8")

    model = ModelInference(
        model_id="meta-llama/llama-3-2-90b-vision-instruct",
        credentials=WATSONX_CREDS,
        project_id=WATSONX_PROJECT_ID,
        params={"max_tokens": 400},
    )

    response = model.chat(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "List all ingredients you see in this food image."},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + encoded_image}}
                ]
            }
        ]
    )

    return response["choices"][0]["message"]["content"]

# ------------------------------
# 2. NUTRITION API (USDA)
# ------------------------------
@tool("nutrition_lookup", return_direct=False)
def nutrition_lookup(food_name: str) -> str:
    """
    Uses USDA FoodData Central API to retrieve nutrition facts:
    - calories, carbs, sugar, protein, fat, fiber
    """
    search_url = "https://api.nal.usda.gov/fdc/v1/foods/search"
    params = {"api_key": USDA_API_KEY, "query": food_name, "pageSize": 1}

    search_response = requests.get(search_url, params=params)
    search_response.raise_for_status()
    search_data = search_response.json()

    if not search_data.get("foods"):
        return f"No USDA data found for '{food_name}'."

    food = search_data["foods"][0]
    nutrients = {n["nutrientName"]: n.get("value", 0) for n in food.get("foodNutrients", [])}

    result = {
        "food": food.get("description"),
        "calories": nutrients.get("Energy", 0),
        "carbs": nutrients.get("Carbohydrate, by difference", 0),
        "sugar": nutrients.get("Sugars, total including NLEA", 0),
        "fat": nutrients.get("Total lipid (fat)", 0),
        "protein": nutrients.get("Protein", 0),
        "fiber": nutrients.get("Fiber, total dietary", 0)
    }

    return str(result)

# ------------------------------
# 3. DIABETIC IMPACT ANALYZER
# ------------------------------
@tool("diabetic_impact", return_direct=False)
def diabetic_impact(nutrition_json: str) -> str:
    """
    Evaluates how a food affects blood sugar:
    - high/medium/low glycemic risk
    - recommended portion size
    - advice for diabetics
    """
    prompt = f"""
Analyze the following nutrition data and estimate diabetic impact.

Nutrition Data:
{nutrition_json}

Provide:
- Summary (carbs & sugar impact)
- Estimated glycemic load category (low/medium/high)
- Blood sugar impact
- Safe serving size
- Advice for diabetic individuals
"""

    model = ModelInference(
        model_id="ibm/granite-3-3-8b-instruct",
        credentials=WATSONX_CREDS,
        project_id=WATSONX_PROJECT_ID,
        params={"max_tokens": 300},
    )

    response = model.chat(messages=[{"role": "user", "content": prompt}])
    return response["choices"][0]["message"]["content"]

# ------------------------------
# 4. MEAL PLANNER
# ------------------------------
@tool("meal_plan_generator", return_direct=True)
def meal_plan_generator(preferences: str) -> str:
    """
    Generates a weekly diabetic-friendly meal plan.
    preferences: "keto", "vegan", "vegetarian", "low-carb"
    """
    sample_meals = {
        "keto": ["Egg & spinach scramble", "Grilled chicken salad", "Zucchini noodles with pesto", "Almonds & cheese snack"],
        "vegan": ["Oatmeal with berries", "Chickpea salad", "Tofu stir-fry", "Hummus & veggies"],
        "vegetarian": ["Greek yogurt with fruits", "Veggie wrap", "Paneer curry with salad", "Nuts & seeds"],
        "low-carb": ["Avocado egg salad", "Grilled salmon with broccoli", "Cauliflower rice stir-fry", "Cheese & cucumber slices"]
    }

    meals = sample_meals.get(preferences.lower(), sample_meals["low-carb"])
    plan = f"""
Weekly Meal Plan ({preferences}):
- Breakfast: {meals[0]}
- Lunch: {meals[1]}
- Dinner: {meals[2]}
- Snack: {meals[3]}
"""
    return plan

# ------------------------------
# 5. GLUCOSE PREDICTOR
# ------------------------------
@tool("glucose_predictor", return_direct=True)
def glucose_predictor(carbs: float, previous_glucose: float = 100, time_since_last_meal: float = 3) -> str:
    """
    Estimates post-meal glucose level based on carbohydrate intake and previous glucose level.
    """
    rise = carbs * 1.5
    predicted = previous_glucose + rise - (time_since_last_meal * 2)
    return f"Estimated glucose level after meal: {predicted:.1f} mg/dL (rise of {rise:.1f} mg/dL)"

# ------------------------------
# 6. GROCERY ADVISOR
# ------------------------------
@tool("grocery_advisor", return_direct=True)
def grocery_advisor(items: str) -> str:
    """
    Suggests diabetic-friendly alternatives for grocery items.
    Input: comma-separated items string
    """
    item_list = [item.strip().lower() for item in items.split(",")]
    alternatives = {
        "bread": "whole grain or almond flour bread",
        "rice": "cauliflower rice",
        "soda": "sparkling water with lemon",
        "pasta": "zucchini noodles or shirataki noodles",
        "sugar": "stevia or monk fruit sweetener"
    }

    suggestions = []
    for item in item_list:
        if item in alternatives:
            suggestions.append(f"{item} → {alternatives[item]}")
        else:
            suggestions.append(f"{item} → No alternative needed")

    return "Grocery Recommendations:\n" + "\n".join(suggestions)

# ------------------------------
# 7. EXERCISE RECOMMENDER
# ------------------------------
@tool("exercise_recommender", return_direct=True)
def exercise_recommender(current_glucose: float, fitness_level: str = "moderate") -> str:
    """
    Recommends safe exercise based on current glucose level and fitness.
    """
    if current_glucose < 70:
        return "Glucose is low! Recommend light activity: 10-15 min walking and have a small snack first."
    elif current_glucose > 180:
        return "Glucose is high! Avoid intense exercise. Recommend gentle stretching or short walk."
    else:
        intensity = "moderate" if fitness_level == "moderate" else "light"
        duration = random.choice([20, 25, 30])
        return f"Glucose safe. Recommend {intensity} exercise for {duration} minutes."

# ------------------------------
# 8. HABIT ANALYZER
# ------------------------------
@tool("habit_analyzer", return_direct=True)
def habit_analyzer(logs: dict) -> str:
    """
    Analyzes daily habits (meals, sleep, steps) and provides weekly insights for glucose control.
    """
    insights = []

    if logs.get("steps", 0) < 7000:
        insights.append("Try to increase daily steps to at least 7000 for better glucose control.")
    else:
        insights.append("Great job on staying active!")

    if logs.get("sleep_hours", 0) < 7:
        insights.append("Increase sleep to 7-8 hours for optimal health.")
    else:
        insights.append("Sleep duration is good.")

    if logs.get("meals", 0) > 4:
        insights.append("Consider reducing snacking to avoid glucose spikes.")

    return "Weekly Habit Insights:\n" + "\n".join(insights)
