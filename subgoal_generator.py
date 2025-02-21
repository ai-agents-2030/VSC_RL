import os
import json
import io
import re
from PIL import Image
import google.generativeai as genai
from tqdm import tqdm

GOOGLE_API_KEY = ""
genai.configure(api_key=GOOGLE_API_KEY)

def process_image(image):
  image = image.resize((image.width // 4, image.height // 4))
  buffer = io.BytesIO()
  image.save(buffer, format="PNG")
  buffer.seek(0)
  image_reloaded = Image.open(buffer)
  return image_reloaded

def prompt_to_generate_subgoals(goal, image, example_goal, example_image):
    prompt = [
    f"""
    =====Examples=====
    Q: TASK: {example_goal}
    Based on the provided screenshot:
    """,
    example_image,
    f""", PLEASE descompse the TASK to the sequence of specific SUBTASKS.
    A: To complete your requested task, you would need to accomplish these subtasks on a device:
    1. Open a browser.
    2. Go to the Walmart website.
    3. Use the search bar to search for "macbook pro".
    4. Click on the first search result.
    5. Click on the "Add to Cart" button on the product page.
    =====Your Turn=====
    Q: TASK: {goal}
    Based on the provided screenshot:
    """,
    image,
    f""", PLEASE descompse the TASK to the sequence of specific SUBTASKS.
    Respond in this format:
    A: To complete your requested task, you would need to accomplish these subtasks on a device:
    <SUBTASKS>
    """
    ]
    return prompt

def load_task_file(assets_path, task_set, task_split):
    all_tasks = []
    with open(os.path.join(assets_path, f"{task_set}_{task_split}.txt")) as fb: 
        for line in fb:
            all_tasks.append(line.strip())
    return all_tasks

def generate_subgoals(goal, example_goal, example_image):
    model = genai.GenerativeModel(model_name="gemini-1.5-pro")
    prompt = prompt_to_generate_subgoals(goal=goal, image=example_image, example_goal=example_goal, example_image=example_image)
    response = model.generate_content(prompt).text
    return response

