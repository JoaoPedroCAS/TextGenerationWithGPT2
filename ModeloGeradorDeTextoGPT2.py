from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments

# Função para tokenizar o texto
def tokenize_function(examples):
    outputs = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    outputs["labels"] = outputs["input_ids"].copy()
    return outputs 

'''
load_dataset():
Carrega um conjunto de dados do Hugging Face Hub ou um conjunto de dados local. 
Um conjunto de dados é um diretório que contém:
    - Alguns arquivos de dados em formatos genéricos (JSON, CSV, Parquet, texto, etc.).
    - Opcionalmente um script de conjunto de dados, se for necessário algum código para ler os arquivos de dados. 
Esta função faz o seguinte por baixo dos panos:
    1. Baixa e importa na biblioteca o script do conjunto de dados do caminho passado se ele ainda não estiver armazenado em cache.
    2. Executa o script do conjunto de dados que irá:
        * Baixar o arquivo do conjunto de dados da URL original se ele ainda não estiver disponível localmente ou armazenado em cache.
        * Processar e armazenar em cache o conjunto de dados em Arrow Tables para armazenamento em cache.
    3. Retorna um conjunto de dados construído a partir das divisões solicitadas em `split` (padrão: all).
'''
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
print("Conjunto de dados carregado!")

'''
GPT2Tokenizer:
Constrói um tokenizador GPT-2. Baseado em Byte-Pair-Encoding de nível de byte.
Este tokenizador foi treinado para tratar espaços como partes dos tokens (um pouco como sentencepiece).
from_pretrained():
Instancia um PreTrainedTokenizerBase (ou uma classe derivada) de um tokenizador predefinido.
'''
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
print("Tokenizador criado!")

# Tokenizando o dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
print("Dataset tokenizado!")

'''
GPT2LMHeadModel:
O transformador do modelo GPT2 com uma cabeça de modelagem de linguagem na parte superior (camada linear com pesos vinculados aos embeddings de entrada).
from_pretrained():
Instancia um modelo pytorch pré-treinado a partir de uma configuração de modelo pré-treinado.
'''
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.config.pad_token_id = tokenizer.eos_token_id
print("Modelo criado!")

'''
TrainingArguments:
TrainingArguments é o subconjunto dos argumentos que usamos em nossos scripts de exemplo que se relacionam ao loop de treinamento em si.
'''
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
)

'''
Trainer:
O Trainer é um loop de treinamento e avaliação simples, mas completo, para PyTorch, otimizado para Transformers.
'''
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
)

# Treinando o modelo
trainer.train()
print("Treino Realizado!")

'''
save_pretrained():
Salva um modelo e seu arquivo de configuração em um diretório para que ele possa ser recarregado usando o método de classe PreTrainedModel.from_pretrained.
'''
model.save_pretrained("./gpt2-finetuned")
tokenizer.save_pretrained("./gpt2-finetuned")
print("Modelo Salvo!")