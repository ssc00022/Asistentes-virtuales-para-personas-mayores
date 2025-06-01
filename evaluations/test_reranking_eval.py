import pandas as pd
from datasets import Dataset
from ragas.metrics import NonLLMContextPrecisionWithReference, NonLLMContextRecall
from ragas import evaluate
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ast import literal_eval
from pathlib import Path

def eval_results(df):
    """
    Evalúa precisión y recall sobre contextos reranqueados.

    Args:
        df (DataFrame): Dataset con contextos recuperados y referencias.

    Returns:
        DataFrame: Métricas calculadas.
    """
    df['retrieved_contexts'] = df['retrieved_contexts'].apply(literal_eval)
    df['reference_contexts'] = df['reference_contexts'].apply(lambda text: [text])

    dt = Dataset.from_pandas(df)

    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': True}
    embed_model = HuggingFaceEmbeddings(
        model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2',  # Ruta oculta
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

    results = evaluate(
        dataset=dt,
        metrics=[NonLLMContextPrecisionWithReference(), NonLLMContextRecall()],
        embeddings=LangchainEmbeddingsWrapper(embed_model)
    )

    score_df = pd.DataFrame(results.scores).round(5)
    return score_df

if __name__ == '__main__':
    input_dir = Path("reranking")
    output_dir = Path("reranking_evaluated")
    output_dir.mkdir(exist_ok=True)

    # Evaluar todos los archivos generados tras reranking
    for archivo in input_dir.glob("*.csv"):
        df = pd.read_csv(archivo)
        df = df.iloc[:24]  # Evaluación sobre preguntas de la base de conocimiento (Provincia de Jaén)

        score_df = eval_results(df)
        final_df = pd.concat([df['question'], score_df], axis=1)

        output_file = output_dir / archivo.name
        final_df.to_csv(output_file, index=False, sep=',', decimal="'")

        print(f'Datos recuperados y guardados en {output_file}')
        print(score_df.mean())
