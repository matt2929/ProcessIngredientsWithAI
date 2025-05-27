from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoModel

deberta_v3_small = "microsoft/deberta-v3-small"
deberta_base = "microsoft/deberta-base"

tokenizer = AutoTokenizer.from_pretrained(deberta_base)
model = AutoModel.from_pretrained(
    deberta_v3_small
)


tokens = tokenizer('Hello world', return_tensors='pt')
output = model(**tokens)[0]
print(output)