from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

K_INSTRUCT = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

model = AutoModelForCausalLM.from_pretrained(
    K_INSTRUCT,
    device_map={"": "mps"},
    torch_dtype="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(K_INSTRUCT)

ingredients = [
    "tomatoes",
    "olive oil",
    "garlic",
    "basil",
    "parmesan cheese",
    "chicken breast",
    "spinach",
    "lemon juice",
    "black pepper",
    "mushrooms"
]

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    max_new_tokens=4,
    do_sample=False
)

for ingredient in ingredients:
    prompt = f"Reply with only one emoji that best represents {ingredient}. Do not write anything else."

    output = generator([{"role": "user", "content": f"{prompt}"}])
    print(f"{ingredient} : {output[0]['generated_text'].strip()}")