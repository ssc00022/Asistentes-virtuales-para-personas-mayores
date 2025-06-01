import pandas as pd
from datasets import Dataset
from ragas.metrics import NonLLMContextPrecisionWithReference, NonLLMContextRecall
from ragas import evaluate
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ast import literal_eval
from pathlib import Path

def eval_results(df, model_path):
    """
    Aplica evaluación de precisión y recall sobre los contextos recuperados.

    Args:
        df (DataFrame): Dataset con preguntas, contextos recuperados y referencia.
        model_path (str): Nombre del modelo de embeddings.

    Returns:
        DataFrame: Métricas evaluadas (precision, recall).
    """
    # Preprocesamiento
    df['retrieved_contexts'] = df['retrieved_contexts'].apply(literal_eval)
    df['reference'] = df['reference'].fillna("")
    df['reference_contexts'] = df['reference_contexts'].fillna("")
    df['reference_contexts'] = df['reference_contexts'].apply(lambda text: [text])

    # Conversión a HuggingFace Dataset
    dt = Dataset.from_pandas(df)

    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': True}
    embed_model = HuggingFaceEmbeddings(
        model_name=f'sentence-transformers/{model_path}',  # Ruta sustituida
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

    # Evaluación con RAGAS
    results = evaluate(
        dataset=dt,
        metrics=[NonLLMContextPrecisionWithReference(), NonLLMContextRecall()],
        embeddings=LangchainEmbeddingsWrapper(embed_model)
    )

    score_df = pd.DataFrame(results.scores).round(5)
    return score_df

if __name__ == '__main__':
    # Modelos usados para los embeddings
    model_list = {
        'minilm': 'paraphrase-multilingual-MiniLM-L12-v2',
        'mpnet': 'paraphrase-multilingual-mpnet-base-v2'
    }

    input_dir = Path("faiss_contexts")
    output_dir = Path("faiss_evaluated")
    output_dir.mkdir(exist_ok=True)

    # Evaluar todos los archivos CSV de recuperación
    for archivo in input_dir.glob("*.csv"):
        df = pd.read_csv(archivo)
        df = df.iloc[:24]  # Evaluar solo preguntas sobre la base de conocimiento (Provincia de Jaén)

        partes = archivo.stem.split("_")
        clave = next((p for p in partes if p in model_list), None)

        if clave:
            score_df = eval_results(df, model_list[clave])
            final_df = pd.concat([df['question'], score_df], axis=1)
            output_file = output_dir / archivo.name
            final_df.to_csv(output_file, index=False, sep=',', decimal="'")
            print(f'Datos recuperados y guardados en {output_file}')
            print(score_df.mean())
