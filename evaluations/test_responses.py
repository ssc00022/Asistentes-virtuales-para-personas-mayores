import sys
sys.path.append("../prototype/llm_agent (server)")
from llm_api import LLMApi
from agent import Agent
from tools import retrieval_augmented_generation
import pandas as pd
from datasets import Dataset
import os
import json
import ast

# Genera respuestas a partir de una lista de preguntas (batch)
def generate(batch):
    questions = batch["question"]
    responses = []
    for question in questions:
        response = agent.invoke(user_message=question)
        responses.append(response)
    return {"response": responses}

# Aplica el generador de respuestas a todo el dataset
def generate_response(dataset):
    dataset = dataset.map(generate, batched=True, batch_size=5)

    data_dict = {
        f"test_{i + 1}": {
            "question": row["question"],
            "reference": row["reference"],
            "response": row["response"],
            "reference_contexts": row["reference_contexts"],
            "retrieved_contexts": row["retrieved_contexts"]
        }
        for i, row in enumerate(dataset)
    }

    return data_dict, dataset.to_pandas()

# Cargar datos desde CSV
df_orig = pd.read_csv('reranking/cross-encoder_5.csv', converters={'retrieved_contexts': ast.literal_eval})

# Convertir a Dataset de HuggingFace
dt = Dataset.from_pandas(df_orig)

# Crear carpeta de salida si no existe
output_dir = 'llm_responses'
os.makedirs(output_dir, exist_ok=True)

# Modelos LLM disponibles
llm_list = {
    'eurollm': "<YOUR_MODEL_PATH>/EuroLLM-9B-Instruct",
    'salamandra': "<YOUR_MODEL_PATH>/salamandra-7b-instruct",
    'qwen': "<YOUR_MODEL_PATH>/Qwen2.5-7B",
    'llama': "<YOUR_MODEL_PATH>/Llama-3.1-8B-Instruct"
}

# Clave de API
api_key = "<ADA_API_KEY>"

# Prompts del sistema para distintas variantes de prueba
system_prompts = {
    "es-adapted": (
        "Eres un asistente conversacional llamado María, diseñado especialmente para personas de nacionalidad "
        "española mayores de 65 años. Debes usar un lenguaje sencillo, breve y educado para comunicarte con "
        "ellas. "
        "- Sencillo: Usa palabras básicas y evita términos complejos o técnicos. "
        "- Breve: Tus respuestas no deben ser demasiado largas, preferiblemente de unas pocas líneas. "
        "- Educado: Debes dirigirte a la persona mayor con cortesía, utilizando 'usted'. "
        "Tu rol es activo e interactivo, lo que significa que debes proponer y mantener conversaciones sobre "
        "temas cotidianos, como el clima, recuerdos del pasado, aficiones o eventos del día. "
        "Tu objetivo es hacer que la persona mayor se sienta acompañada y entretenida."
    ),
    "en-adapted": (
        "You are a conversational assistant called Maria, designed especially for people over 65 years old "
        "from Spain. You must use simple, brief and polite language to communicate with them. "
        "Your role is active and interactive, encouraging conversations about everyday topics like the weather, "
        "memories, hobbies or daily events. Your goal is to make them feel accompanied and entertained."
    ),
    "es-not-adapted": (
        "Te llamas María, un asistente conversacional diseñado para interactuar de manera clara y efectiva. "
        "Tu rol es responder preguntas sobre una amplia variedad de temas según los intereses del usuario. "
        "Tu objetivo es ofrecer una experiencia fluida y agradable."
    ),
    "en-not-adapted": (
        "Your name is Maria, a conversational assistant designed to interact in a clear and effective way. "
        "Your role is to answer questions on a wide variety of topics based on the user's interests. "
        "Your goal is to provide a smooth and enjoyable experience."
    )
}

# Ejecutar la generación para cada combinación de modelo y prompt
for model_name, model_path in llm_list.items():
    for prompt_type, prompt in system_prompts.items():
        agent = Agent(
            llm=LLMApi(model=model_path, api_key=api_key),
            tools=[retrieval_augmented_generation],
            system_prompt=prompt
        )

        # Generar respuestas
        prepared_data_json, prepared_data_csv = generate_response(dt)

        # Guardar en JSON
        output_json = os.path.join(output_dir, f"{model_name}_{prompt_type}.json")
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(prepared_data_json, f, ensure_ascii=False, indent=4)

        # Guardar en CSV
        output_csv = os.path.join(output_dir, f"{model_name}_{prompt_type}.csv")
        prepared_data_csv.to_csv(output_csv, index=False)

        print(f'Datos guardados en {output_json} y {output_csv}')
