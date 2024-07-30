# Gerando texto utilizando o GPT-2

Neste repositório encontra-se minha primeira experiência na utilização de uma IA Generativa já pré-treinada na tarefa de gerar textos.



Os processos necessários para preparar o nosso ambiente de trabalho serão descritos a seguir.



##  Preparando o ambiente

### Linguagem e bibliotecas

Nesse caso, utilizei a linguagem de programação Python, um tutorial sobre como preparar sua máquina para executar códigos nessa linguagem pode ser encontrado [aqui](https://www.python.org/).

Além da linguagem, utilizaremos a biblioteca de Transformers do PyTorch, a documentação dessa biblioteca pode ser encontrada [aqui](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html). 



Por fim utilizaremos  os datasets da biblioteca Datasets, a documentação dessa biblioteca pode ser encontrada [aqui](https://huggingface.co/docs/datasets/index).



Para instalar estas duas bibliotecas utilizamos o comando:

```bash
pip install torch transformers datasets
```



## Carregando e pré-processando os dados

O dataset que utilizaremos é o WikiText, que contem mais de 100 milhões de tokens extraídos de artigos da Wikipédia.

Os script estão detalhadamente explicados no arquivo de código.

```python
from datasets import load_dataset

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
```

Para "tokenizar" os dados, utilizamos a biblioteca transformers, com o módulo GPT2Tokenizer.

```Python
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

```

## Treinar o modelo

Primeiro precisamos definir quais são os argumentos de treino

```Python
from transformers import GPT2LMHeadModel, Trainer, TrainingArguments

model = GPT2LMHeadModel.from_pretrained("gpt2")

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
)

```

Após isso podemos criar o treinador e iniciar o treino.

```Python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
)

trainer.train()

```

## Gerando texto com o modelo treinado

Primeiro precisamos carregar o modelo que acabamos de treinar.

```python
model.save_pretrained("./gpt2-finetuned")
tokenizer.save_pretrained("./gpt2-finetuned")

model = GPT2LMHeadModel.from_pretrained("./gpt2-finetuned")
tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-finetuned")

```

Após isso podemos passar um prompt para o modelo e esperar um texto generativo como resposta.

```python
input_text = "Once upon a time"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

output = model.generate(input_ids, max_length=100, num_return_sequences=1)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)

```

## Engenharia de prompting

Podemos experimentar diferentes prompts e parâmetros com a finalidade de entender melhor como cada parâmetro influência a resposta.

```
output = model.generate(input_ids, max_length=100, num_return_sequences=1, temperature=0.7, top_k=50)
```

