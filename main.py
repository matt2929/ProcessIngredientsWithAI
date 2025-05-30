# This is a sample Python script.
import sys

import pandas as pd
from tqdm import tqdm
import json
import ollama
from sentence_transformers import SentenceTransformer, util
import emoji
import torch

# register to watch progress
tqdm.pandas()

model = SentenceTransformer('all-MiniLM-L6-v2')

# map of emoji to description
# Load all emoji descriptions from the emoji package
emoji_map = {
    e: data['en']
    for e, data in emoji.EMOJI_DATA.items()
    if 'en' in data
}
# Encode all descriptions
emoji_descriptions = list(emoji_map.values())
emoji_chars = list(emoji_map.keys())
print(f"Encoding {len(emoji_chars)} emojis...")

simple_examples = [
    "salt",
    "sugar",
    "water",
    "butter",
    "flour",
    "olive oil",
    "garlic",
    "milk",
    "egg",
    "chicken broth",
    "mayonnaise",
    "soy sauce",
    "bbq sauce",
    "vanilla extract",
    "lemon juice",
    "pesto sauce",
    "tomato paste",
    "brown sugar",
]

compound_examples = complex_ingredients = [
    "spicy mango chicken curry",
    "rosemary garlic roasted potatoes",
    "cranberry walnut quinoa salad",
    "smoky chipotle beef stew",
    "lemon basil pasta with shrimp",
    "honey glazed roasted carrots",
    "pumpkin spice overnight oats",
    "maple pecan breakfast granola",
    "roasted garlic and herb hummus",
    "thai peanut chicken stir fry",
    "ginger sesame noodle salad",
    "caramelized onion and bacon quiche",
    "cilantro lime black bean salsa",
    "cranberry orange glazed turkey",
    "chocolate chip banana bread",
    "roasted beet and goat cheese salad",
    "sweet potato and kale hash",
    "blueberry lemon ricotta pancakes",
    "herb crusted pork tenderloin",
    "avocado cucumber sushi rolls",
    "spiced apple cider donuts",
    "grilled pineapple teriyaki chicken",
    "buffalo cauliflower wings",
    "sun-dried tomato pesto pasta",
    "mango coconut chia pudding",
    "balsamic glazed Brussels sprouts",
    "cilantro lime grilled corn",
    "smoked paprika roasted chickpeas",
    "lemon garlic butter salmon",
    "pumpkin coconut curry soup",
    "roasted tomato basil bruschetta",
    "cauliflower fried rice",
    "chili lime roasted cashews",
    "honey mustard glazed carrots",
    "baked spinach and artichoke dip",
    "chocolate peanut butter energy bites",
    "maple roasted butternut squash",
    "zucchini parmesan fritters",
    "cherry almond granola bars",
    "turmeric ginger detox tea",
    "garlic parmesan roasted asparagus",
    "spaghetti squash with marinara",
    "avocado egg salad sandwich",
    "spicy black bean veggie burger",
    "pumpkin spice latte smoothie",
    "cinnamon roasted apple chips",
    "sweet chili glazed shrimp",
    "roasted garlic mashed cauliflower",
    "blueberry coconut overnight oats",
    "chipotle lime chicken tacos",
    "kale and quinoa power bowl",
    "ginger garlic beef stir fry",
    "lemon thyme roasted chicken",
    "mango avocado salsa",
    "coconut curry lentil stew",
    "basil garlic roasted mushrooms",
    "smoked salmon dill cream cheese",
    "honey garlic glazed pork chops",
    "thai green curry with tofu",
    "roasted red pepper and feta dip",
    "cinnamon spiced pumpkin muffins",
    "spinach and feta stuffed chicken",
    "maple glazed brussels sprouts",
    "cauliflower buffalo bites",
    "berry chia seed jam",
    "lite greek pasta salad"
    "roasted garlic lemon hummus",
    "lemon blueberry scones",
    "sweet potato black bean chili",
    "herbed goat cheese crostini",
    "grilled peach and burrata salad",
    "spicy garlic edamame",
    "coconut lime quinoa salad",
    "pumpkin spice granola clusters",
    "roasted beet and orange salad",
    "black garlic mushroom risotto",
    "ginger sesame chicken wings",
    "maple glazed roasted carrots",
    "chocolate avocado mousse",
    "roasted garlic and herb potatoes",
    "spiced cranberry orange sauce",
    "turmeric roasted cauliflower",
    "blueberry lemon yogurt parfait",
    "cinnamon apple baked oats",
    "spicy chipotle black bean soup",
    "roasted garlic tomato soup",
    "lemon basil grilled shrimp",
    "caramelized banana and nutella crepes",
    "pumpkin gingerbread smoothie",
    "garlic parmesan kale chips",
    "honey lime grilled chicken",
    "spiced apple baked brie",
    "coconut mango sticky rice",
    "roasted tomato and garlic salsa",
    "zesty lemon herb chicken salad",
    "spicy mango black bean salsa",
    "ginger turmeric detox smoothie",
    "smoked paprika roasted potatoes",
    "maple roasted acorn squash",
    "cinnamon spiced chai latte",
    "roasted garlic mushroom soup",
    "berry lemon quinoa bowl",
    "spicy peanut soba noodles",
    "caramel apple cinnamon crisp",
]

simple_vecs = model.encode(simple_examples, convert_to_tensor=True)
compound_vecs = model.encode(compound_examples, convert_to_tensor=True)
emoji_vectors = model.encode(emoji_descriptions, convert_to_tensor=True)
# Compute class prototype embeddings by averaging
simple_proto = torch.mean(simple_vecs, dim=0)
compound_proto = torch.mean(compound_vecs, dim=0)

# Configure the ollama client to connect to your custom endpoint
# Replace 'http://ollama:11434' with your actual endpoint if different
client = ollama.Client(host='http://ollama:11434')


def produce_emoji_ollama(ingredient_name: str) -> str:
    response = client.chat(model='llama2', messages=[
        {
            'role': 'user',
            'content': f'return one emoji only that best describes the ingredient {ingredient_name}',
        },
    ])
    output_emoji = response['message']['content']
    print(f"{ingredient_name}={output_emoji}")
    return output_emoji


def is_simple_ingredient_ollama(ingredient_name: str) -> str:
    response = client.chat(model='llama3', messages=[
        {
            'role': 'user',
            'content': f'return the most general possible name for {ingredient_name} limit the response to only the ingredient name ;.l,kj;.l,kmjhgfd ',
        },
    ])
    output_emoji = response['message']['content']
    print(f"{ingredient_name}={output_emoji}")
    return output_emoji


def produce_emoji_bart(ingredient_name: str) -> str:
    query_vec = model.encode(ingredient_name, convert_to_tensor=True)
    similarities = util.cos_sim(query_vec, emoji_vectors)[0]
    best_match_idx = similarities.argmax()
    return emoji_chars[best_match_idx]


def is_simple_ingredient_bart(ingredient_name: str) -> bool:
    vec = model.encode(ingredient_name, convert_to_tensor=True)

    sim_simple = util.cos_sim(vec, simple_proto).item()
    sim_compound = util.cos_sim(vec, compound_proto).item()

    # Choose the closest class
    if sim_compound > sim_simple:
        return False
    else:
        return True


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Read the CSV file
    df = pd.read_csv('archive/RAW_recipes.csv')

    # Display the first few rows of the dataframe
    print("Original DataFrame:")
    print(df.head())

    # Drop specific columns
    columns_to_drop = ["id",
                       "minutes",
                       "contributor_id",
                       "submitted",
                       "tags",
                       "nutrition",
                       "n_steps",
                       "steps",
                       "description",
                       "n_ingredients"
                       ]  # Replace with your actual column names
    df = df.drop(columns=columns_to_drop)

    # Display the first few rows of the dataframe after dropping columns
    print("\nDataFrame after dropping columns:")
    print(df.head(30))

    df.to_json(orient="records")

    all_ingredient_strings = df['name'].tolist() + [item for sublist in df['ingredients'] for item in sublist]
    # Create a mapping from each unique string to a unique integer
    unique_ingredient_strings = list(set(all_ingredient_strings))

    ingredient_data = {
        'ingredient': unique_ingredient_strings
    }
    ingredient_df = pd.DataFrame(ingredient_data)

    print(f"Running Emoji Definition....")
    ingredient_df['emoji'] = ingredient_df['ingredient'].head(1000).progress_apply(produce_emoji_ollama)

    print(ingredient_df.head(1000).to_json)

    filtered_food_ingredient_mapping = 'key_map.json'
    emoji_ingredient_mapping = 'ingredients.json'

    # Write the Python object to a JSON file
    with open(filtered_food_ingredient_mapping, 'w') as dataFile, open(emoji_ingredient_mapping, 'w') as emojiMapping:
        json.dump(df.to_json(), dataFile)
        json.dump(ingredient_df['emoji'].to_json(), emojiMapping)
                    