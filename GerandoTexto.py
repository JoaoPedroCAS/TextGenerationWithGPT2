from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Carregar o modelo e o tokenizador treinados
model = GPT2LMHeadModel.from_pretrained("./gpt2-finetuned")
tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-finetuned")

# Função para gerar texto
def generate_text(input_text, max_length=100, num_return_sequences=1):
    input_ids = tokenizer.encode(input_text, return_tensors="pt", padding=True)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    output = model.generate(input_ids, attention_mask=attention_mask, max_length=max_length, num_return_sequences=num_return_sequences)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Texto de entrada
input_text = "I thinkt that we should"
generated_text = generate_text(input_text)

print(generated_text)
