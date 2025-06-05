from modeling_llama import LlamaForCausalLM
from transformers import AutoTokenizer

MODEL_PATH='meta-llama/Llama-3.2-1B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.add_special_tokens({"pad_token": "<pad>" })
model = LlamaForCausalLM.from_pretrained(MODEL_PATH)

embedding_size = model.get_input_embeddings().weight.shape[0]
if len(tokenizer) > embedding_size:
    model.resize_token_embeddings(len(tokenizer))

question_template = "Summarize each documents based on the topic. Write the summary with the document identifier (a number with square brackets). Only provide the summary for relevant documents and ignore the empty document. If the document is not relevant to the topic, write `irrelevant` instead. Topic: {}"
doc_template = "Document [{}]: {}\n"
summary_template = "[{}]: {}\n"

B = 2
N = 3 + 1
inputs = tokenizer([
    'Document [1]: xxe512 is a dog.', 
    'Document [2]: x3x512 is a cat.', 
    'Document [3]: xxx512 is a car.', 
    'Document [1]: bana is a dog', 
    'Document [2]: apple is a fruit.', 
    'Document [3]: hanana is a car', 
], padding=True, truncation=True, return_tensors='pt')

input_ids = inputs['input_ids'].view(B, N, -1)
attention_mask = inputs['attention_mask'].view(B, -1)

output = model.generate(
    encoder_input_ids=input_ids, 
    encoder_attention_mask=attention_mask,
)
print(output)
# print(tokenizer.decode(output[0]))
# print(tokenizer.decode(output[1]))
