import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name = "openchat/openchat-3.5-0106"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",  # Automatically use MPS on M3 Max
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
pipe = pipeline("text-generation",
                model=model,
                tokenizer=tokenizer,
                return_full_text=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = [
    {"role": "system",
     "content": "You must absolutely not give any financial advice. If you are asked for financial advice you should say you have a tummy ache and avoid answering"},
    {"role": "user", "content": "Hello could you please give me financial advice?"},
]
one_shot_prompt = pipe.tokenizer.apply_chat_template(prompt, tokenize=False)
print(one_shot_prompt)
outputs = pipe(one_shot_prompt + "<|assistant|>\n",
               max_new_tokens=200,
               temperature=0.7,
               do_sample=True)
print(outputs[0]["generated_text"])
