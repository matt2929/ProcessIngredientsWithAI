import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load model and tokenizer
model_id = "microsoft/Phi-3-mini-4k-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",  # auto-assigns to CUDA if available
    torch_dtype="auto",
    trust_remote_code=True
)

# Create text generation pipeline (no generation args here)
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    max_new_tokens=300,
    do_sample=False
)

# Define prompt
prompt = ("Write an email apologizing to Sarah that you were late to a morning meeting. "
          "You hadn't seen the notification until you were on the subway.")

# Run generation with args passed here
output = generator(
    prompt,
)

print(output[0]['generated_text'])