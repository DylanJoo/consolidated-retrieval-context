from modeling_llama import LlamaForCausalLM
from transformers import AutoTokenizer

MODEL_PATH='meta-llama/Llama-3.2-1B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.add_special_tokens({"pad_token": "<pad>" })
model = LlamaForCausalLM.from_pretrained(MODEL_PATH, do_cross_attention=True, num_cross_attn_layers=2).to('cuda')

embedding_size = model.get_input_embeddings().weight.shape[0]
if len(tokenizer) > embedding_size:
    model.resize_token_embeddings(len(tokenizer))

doc_template = "Document [{}]: {}\n"
summary_template = "[{}]: {}\n"

B = 2
N = 3
inputs = tokenizer([
    'Document [1]: xxe512 is a dog dog dog dog dog dog dog dog dog dog dog dog dog dog dog.', 
    'Document [2]: xxe512 is a dog dog dog dog dog dog dog dog dog dog dog dog dog dog dog.', 
    'Document [3]: xxe512 is a dog dog dog dog dog dog dog dog dog dog dog dog dog dog dog.', 
    'Document [1]: xxe512 is a dog dog dog dog dog dog dog dog dog dog dog dog dog dog dog.', 
    'Document [2]: xxe512 is a dog dog dog dog dog dog dog dog dog dog dog dog dog dog dog.', 
    'Document [3]: xxe512 is a dog dog dog dog dog dog dog dog dog dog dog dog dog dog dog.', 
], padding=True, truncation=True, return_tensors='pt').to('cuda')

input_ids = inputs['input_ids'].view(B, N, -1)
attention_mask = inputs['attention_mask'].view(B*N, -1)

# initial = tokenizer(["Summarize each documents based on the topic.", "Summarize each documents based on the topic."], return_tensors='pt').to('cuda')
initial = tokenizer(['<|begin_of_text|>', '<|begin_of_text|>'], return_tensors='pt').to('cuda')

output = model.generate(
    input_ids=initial.input_ids,
    attention_mask=initial.attention_mask,
    encoder_input_ids=input_ids, 
    encoder_attention_mask=attention_mask,
    min_new_tokens=10,
    max_new_tokens=20,
    temperature=1e-10
)
print(tokenizer.decode(output[0]))
print(tokenizer.decode(output[1]))
