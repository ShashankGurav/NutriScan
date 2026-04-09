# diet.py
import json
import re
import logging
from datetime import datetime
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from config import client, MODEL_NAME, IST
import os
import asyncio
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

log = logging.getLogger("food-vision")
load_dotenv()

# Hugging Face Client for Image Generation
hf_client = InferenceClient(token=os.getenv("HF_TOKEN"))

if not os.getenv("HF_TOKEN"):
    log.warning("⚠️ HF_TOKEN not found in .env file. Image generation will be disabled.")

diet_router = APIRouter(prefix="/diet", tags=["Diet Recommendation"])


# ── Request Model ─────────────────────────────────────────────────────────────
class DietRequest(BaseModel):
    gender: str = Field(..., example="Male")
    age: int = Field(..., gt=12, lt=100, example=28)
    activity_level: str = Field(..., example="Moderately Active")
    meal_preference: str = Field(..., example="Veg")
    cuisine_preference: str = Field(..., example="North Indian")
    primary_goal: str = Field(..., example="Weight Loss")
    chronic_diseases: List[str] = Field(default=[], example=["Diabetes"])
    allergies: List[str] = Field(default=[], example=["Peanuts"])


# ── Response Models ───────────────────────────────────────────────────────────
class Meal(BaseModel):
    time: str
    name: str
    calories: int
    protein: int
    carbs: int
    fat: int
    description: str
    image_prompt: str
    image_url: str = ""

class DailyPlan(BaseModel):
    day: str
    meals: List[Meal]
    total_calories: int
    total_protein: int
    total_carbs: int
    total_fat: int


class SmartSwap(BaseModel):
    instead_of: str
    swap_with: str
    reason: str


class DietResponse(BaseModel):
    success: bool
    daily_calories_target: int
    plan: List[DailyPlan]
    smart_swaps: List[SmartSwap]
    foods_to_avoid: List[str]
    ai_notes: str
    generated_at: str


# ── System Prompt ─────────────────────────────────────────────────────────────
DIET_SYSTEM_PROMPT = """You are an expert Indian dietitian.

Create a realistic, tasty 7-day weekly diet plan based on the user's cuisine preference and health goals.

Return ONLY valid JSON in this exact format. No extra text, no markdown, no code blocks.

CRITICAL: Every meal object MUST contain all 8 fields: time, name, calories, protein, carbs, fat, description, image_prompt.
Missing "carbs" or "fat" from ANY meal is a fatal error.

{
  "daily_calories_target": 1800,
  "plan": [
    {
      "day": "Day 1 - Monday",
      "meals": [
        {
          "time": "08:00 AM BREAKFAST",
          "name": "Masala Oats Upma",
          "calories": 350,
          "protein": 12,
          "carbs": 45,
          "fat": 8,
          "description": "Oats with mustard seeds, curry leaves, mixed veggies",
          "image_prompt": "masala oats upma bowl, Indian breakfast"
        },
        {
          "time": "01:00 PM LUNCH",
          "name": "Dal Tadka with Brown Rice",
          "calories": 480,
          "protein": 20,
          "carbs": 72,
          "fat": 10,
          "description": "Yellow dal with cumin and garlic, served with brown rice",
          "image_prompt": "dal tadka brown rice, Indian lunch"
        },
        {
          "time": "04:00 PM SNACK",
          "name": "Roasted Chana",
          "calories": 150,
          "protein": 8,
          "carbs": 22,
          "fat": 3,
          "description": "Crunchy roasted chickpeas with chaat masala",
          "image_prompt": "roasted chana bowl, healthy snack"
        },
        {
          "time": "08:00 PM DINNER",
          "name": "Palak Paneer with 2 Rotis",
          "calories": 420,
          "protein": 22,
          "carbs": 38,
          "fat": 15,
          "description": "Spinach curry with paneer and whole wheat rotis",
          "image_prompt": "palak paneer rotis, Indian dinner"
        }
      ],
      "total_calories": 1400,
      "total_protein": 62,
      "total_carbs": 177,
      "total_fat": 36
    }
  ],
  "smart_swaps": [
    {"instead_of": "White Rice", "swap_with": "Brown Rice", "reason": "Lower glycemic index, more fiber"},
    {"instead_of": "Maida Roti", "swap_with": "Whole Wheat Roti", "reason": "Higher fiber, stabilizes blood sugar"},
    {"instead_of": "Full Fat Milk", "swap_with": "Skimmed Milk", "reason": "Lower saturated fat, same protein"},
    {"instead_of": "Fruit Juice", "swap_with": "Whole Fruit", "reason": "Fiber slows sugar absorption"},
    {"instead_of": "Fried Snacks", "swap_with": "Roasted Alternatives", "reason": "Lower in calories and fat"}
  ],
  "foods_to_avoid": [
    "Fried snacks like samosa, pakora, and vada",
    "Sugary drinks like cola, packaged juices, and energy drinks",
    "White bread and maida-based products like pav and naan",
    "Processed foods like chips, biscuits, and instant noodles",
    "High-sugar sweets like gulab jamun, jalebi, and rasgulla",
    "Alcohol and carbonated beverages"
  ],
  "ai_notes": "Plan tailored for your goal. Stay hydrated with 2.5-3L water daily."
}

STRICT RULES:
- MANDATORY meal fields (all 8, no exceptions): time, name, calories, protein, carbs, fat, description, image_prompt.
- Each day must have exactly 4 meals: Breakfast, Lunch, Evening Snack, Dinner.
- Every day must include total_calories, total_protein, total_carbs, total_fat as integers.
- Meals MUST match cuisine_preference (South Indian: idli/dosa/sambar; North Indian: roti/dal/sabzi; Continental: salads/wraps/soups).
- Strictly respect meal_preference (Veg / Non-Veg / Eggitarian / Vegan).
- smart_swaps: exactly 5 items. foods_to_avoid: exactly 6 items.
- description: max 12 words. image_prompt: max 8 words.
- All numeric values must be integers.
- No trailing commas. No markdown. Raw JSON only.
"""

# ── Image Prompt Enhancer for Better Indian Food ─────────────────────────────
def enhance_image_prompt(meal_name: str, base_prompt: str) -> str:
    """Convert short LLM prompt into detailed realistic Indian food prompt"""
    meal_lower = meal_name.lower()
    
    enhancements = {
        "palak paneer": "vibrant green spinach gravy with soft paneer cubes, garnished with fresh cream and coriander leaves",
        "dal makhani": "rich dark creamy black lentil curry with rajma, velvety texture, garnished with coriander",
        "dal tadka": "yellow dal with cumin garlic tadka, garnished with coriander",
        "upma": "coarse masala oats upma with mustard seeds, curry leaves, mixed vegetables, garnished with coriander",
        "roti": "fresh soft whole wheat roti",
        "idli": "steaming hot soft idlis served with coconut chutney and sambar",
        "dosa": "crispy golden dosa with potato masala and chutneys",
        "paneer": "soft paneer cubes in rich gravy",
    }
    
    extra = ""
    for key, desc in enhancements.items():
        if key in meal_lower:
            extra = desc
            break
    
    return (
        f"Authentic Indian food photography of {base_prompt}, {extra}, "
        f"steaming hot, appetizing, professional close-up shot, natural studio lighting, "
        f"high detail, sharp focus, vibrant colors, realistic, mouthwatering, restaurant style"
    )


# ── Generate Single Meal Image ───────────────────────────────────────────────
async def generate_meal_image(meal_name: str, base_prompt: str) -> str:
    """Generate image and return local file path"""
    if not base_prompt or not os.getenv("HF_TOKEN"):
        return ""
    
    enhanced_prompt = enhance_image_prompt(meal_name, base_prompt)
    
    try:
        image = hf_client.text_to_image(
            prompt=enhanced_prompt,
            model="black-forest-labs/FLUX.1-schnell",
            width=1024,
            height=768,
            num_inference_steps=25,
        )
        
        # Save image locally
        safe_name = re.sub(r'[^a-zA-Z0-9]', '_', meal_name.lower())[:40]
        filename = f"meal_images/{safe_name}.png"
        os.makedirs("meal_images", exist_ok=True)
        
        image.save(filename)
        
        return f"/static/meal_images/{os.path.basename(filename)}"
        
    except Exception as e:
        log.error(f"Image generation failed for '{meal_name}': {e}")
        return ""
# ── Robust JSON Cleaner ───────────────────────────────────────────────────────
def clean_and_parse_diet_json(raw: str) -> dict:
    text = raw.strip()

    if text.startswith("```"):
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()

    json_match = re.search(r"(\{[\s\S]*\})", text)
    if json_match:
        text = json_match.group(1)

    text = re.sub(r",\s*([}\]])", r"\1", text)
    text = re.sub(r'"\s*,\s*"', '", "', text)
    text = re.sub(r'(\}\s*)\n\s*\{', r'\1,\n  {', text)

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        log.error(f"Diet JSON parse failed: {e}\nLast 300 chars:\n{text[-300:]}")
        raise HTTPException(
            status_code=500,
            detail={"success": False, "message": "Failed to generate diet recommendation", "error": str(e)}
        )


# ── Endpoint ──────────────────────────────────────────────────────────────────
# @diet_router.post("/recommend-diet", response_model=DietResponse)
# async def recommend_diet(request: DietRequest):
#     user_prompt = f"""
# Gender: {request.gender}
# Age: {request.age} years
# Activity Level: {request.activity_level}
# Meal Preference: {request.meal_preference}
# Cuisine Preference: {request.cuisine_preference}
# Primary Goal: {request.primary_goal}
# Chronic Diseases: {', '.join(request.chronic_diseases) if request.chronic_diseases else 'None'}
# Allergies: {', '.join(request.allergies) if request.allergies else 'None'}

# Generate a complete 7-day diet plan matching the cuisine preference above.
# REMINDER: Every meal must include all 8 fields — especially "carbs" and "fat". Never skip them.
# Include 5 smart swaps and 6 foods to avoid based on the goal and chronic diseases.
# Return raw JSON only — no markdown, no code blocks.
# """

#     response = client.chat.completions.create(
#         model=MODEL_NAME,
#         max_tokens=8000,
#         temperature=0.7,
#         messages=[
#             {"role": "system", "content": DIET_SYSTEM_PROMPT},
#             {"role": "user", "content": user_prompt}
#         ]
#     )

#     raw = response.choices[0].message.content.strip()
#     finish_reason = response.choices[0].finish_reason

#     log.info(f"Diet response: {len(raw)} chars | finish_reason={finish_reason}")

#     if finish_reason == "length":
#         log.warning("⚠️ Diet response truncated — JSON likely incomplete")

#     data = clean_and_parse_diet_json(raw)
#     data["generated_at"] = datetime.now(IST).isoformat()

#     return {"success": True, **data}
@diet_router.post("/recommend-diet", response_model=DietResponse)
async def recommend_diet(request: DietRequest):
    user_prompt = f"""
Gender: {request.gender}
Age: {request.age} years
Activity Level: {request.activity_level}
Meal Preference: {request.meal_preference}
Cuisine Preference: {request.cuisine_preference}
Primary Goal: {request.primary_goal}
Chronic Diseases: {', '.join(request.chronic_diseases) if request.chronic_diseases else 'None'}
Allergies: {', '.join(request.allergies) if request.allergies else 'None'}

Generate a complete 7-day diet plan matching the cuisine preference above.
REMINDER: Every meal must include all 8 fields — especially "carbs" and "fat". Never skip them.
Include 5 smart swaps and 6 foods to avoid based on the goal and chronic diseases.
Return raw JSON only — no markdown, no code blocks.
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=8000,
        temperature=0.7,
        messages=[
            {"role": "system", "content": DIET_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
    )

    raw = response.choices[0].message.content.strip()
    finish_reason = response.choices[0].finish_reason

    log.info(f"Diet response: {len(raw)} chars | finish_reason={finish_reason}")

    if finish_reason == "length":
        log.warning("⚠️ Diet response truncated — JSON likely incomplete")

    data = clean_and_parse_diet_json(raw)
    data["generated_at"] = datetime.now(IST).isoformat()

    # ── Generate Images for All Meals in Parallel ───────────────────────────
    async def enrich_plan_with_images(plan: List[dict]):
        tasks = []
        for day in plan:
            for meal in day.get("meals", []):
                tasks.append(
                    generate_meal_image(
                        meal.get("name", ""), 
                        meal.get("image_prompt", "")
                    )
                )
        
        image_urls = await asyncio.gather(*tasks, return_exceptions=True)
        
        idx = 0
        for day in plan:
            for meal in day.get("meals", []):
                url = image_urls[idx] if idx < len(image_urls) and not isinstance(image_urls[idx], Exception) else ""
                meal["image_url"] = url
                idx += 1

    await enrich_plan_with_images(data["plan"])

    return {"success": True, **data}