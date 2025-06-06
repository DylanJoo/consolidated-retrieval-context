from modeling_llama import LlamaForCausalLM
from transformers import AutoTokenizer

MODEL_PATH='meta-llama/Llama-3.2-1B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.add_special_tokens({"pad_token": "<pad>" })
model = LlamaForCausalLM.from_pretrained(MODEL_PATH, do_cross_attention=True, num_cross_attn_layers=0).to('cuda')
print(model)

embedding_size = model.get_input_embeddings().weight.shape[0]
if len(tokenizer) > embedding_size:
    model.resize_token_embeddings(len(tokenizer))

B = 2
N = 3
passages = [
]

inputs = tokenizer([
        "Document [1]: The sun dipped below the horizon, casting an amber glow across the quiet valley. Birds chirped their final songs of the day as the wind whispered through the tall grass.",
        "Document [2]: She flipped through the worn pages of the old journal, tracing her fingers over the faded ink. Each entry told a story lost to time, each sentence a fragment of a forgotten world.",
        "Document [3]: Rain tapped gently against the windowpane, a soft rhythm in the silence of the room. Inside, the aroma of fresh coffee mingled with the scent of aged paper and warm wood.",
        "Document [1]: The elevator groaned as it ascended the ancient shaft, its lights flickering with each floor passed. Somewhere above, thunder rumbled like a warning.",
        "Document [2]: He stared at the chessboard, pieces frozen mid-battle, as if the outcome hinged not on logic, but on something far older and stranger than reason.",
        "Document [3]: A narrow path wound through the misty forest, flanked by moss-covered stones and silent, towering pines. The air smelled of rain and secrets.",
], padding=True, truncation=True, return_tensors='pt').to('cuda')

input_ids = inputs['input_ids'].view(B, N, -1)
attention_mask = inputs['attention_mask'].view(B*N, -1)

initial = tokenizer(["Summarize each documents based on the topic.", "Summarize each documents based on the topic."], return_tensors='pt').to('cuda')

output = model.generate(
    input_ids=initial.input_ids,
    attention_mask=initial.attention_mask,
    encoder_input_ids=input_ids, 
    encoder_attention_mask=attention_mask,
    min_new_tokens=10,
    max_new_tokens=30,
    temperature=1e-10
)
print(tokenizer.decode(output[0]))
print(tokenizer.decode(output[1]))
