import os
from langchain_community.retrievers import TFIDFRetriever, BM25Retriever
from tqdm import tqdm
import pandas as pd
import json
from sentence_transformers import CrossEncoder
from ast import literal_eval

def rerank_documents(df: pd.DataFrame, strategy, k, retrievers):
    """
    Reranquea documentos utilizando BM25 o TF-IDF con el número de documentos deseado.

    Args:
        df (DataFrame): Datos originales con preguntas y contextos.
        strategy (str): Estrategia de recuperación ('bm25' o 'tfidf').
        k (int): Número de documentos a recuperar.
        retrievers (dict): Diccionario con clases de retrievers.

    Returns:
        list: Lista de listas de documentos reranqueados.
    """
    retriever_class = retrievers[strategy]
    return df.apply(lambda row: [
        doc.page_content for doc in retriever_class.from_texts(row["retrieved_contexts"], k=k).invoke(row["question"], k=k)
    ], axis=1).tolist()

def rerank_with_cross_encoder(df: pd.DataFrame, reranker: CrossEncoder, k: int):
    """
    Reranquea usando un modelo CrossEncoder para reordenar documentos según la similitud con la pregunta.
    """
    for index, row in df.iterrows():
        pairs = [(row['question'], doc) for doc in row['retrieved_contexts']]
        scores = reranker.predict(pairs)
        reranked_docs = [doc for _, doc in sorted(zip(scores, row['retrieved_contexts']), reverse=True)][:k]
        df.at[index, 'retrieved_contexts'] = reranked_docs

def prepare_data(df_orig: pd.DataFrame, strategy, k, retrievers):
    """
    Aplica reranking con la estrategia indicada y devuelve los datos preparados.

    Args:
        df_orig (DataFrame): Datos originales.
        strategy (str): Estrategia de reranking ('bm25', 'tfidf' o 'cross-encoder').
        k (int): Número de documentos a conservar.
        retrievers (dict): Diccionario de retrievers.

    Returns:
        Tuple[dict, DataFrame]: Datos en formato JSON y DataFrame.
    """
    new_df = df_orig.copy()
    reranker = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

    if strategy in ['bm25', 'tfidf']:
        new_df['retrieved_contexts'] = rerank_documents(new_df, strategy, k, retrievers)
    elif strategy == 'cross-encoder':
        rerank_with_cross_encoder(new_df, reranker, k)

    data_list = new_df.to_dict(orient='records')
    data_dict = {f"test_{i+1}": entry for i, entry in enumerate(data_list)}

    return data_dict, pd.DataFrame(data_list)

if __name__ == '__main__':
    # Cargar el dataset generado por recuperación semántica
    df = pd.read_csv("faiss_contexts/mpnet_full_similarity20.csv", index_col=0)

    # Preprocesamiento
    df['reference'] = df['reference'].fillna("")
    df['reference_contexts'] = df['reference_contexts'].fillna("")
    df['retrieved_contexts'] = df['retrieved_contexts'].apply(literal_eval)
    df['reference_contexts'] = df['reference_contexts'].apply(lambda text: [text])

    strategies = ['bm25', 'tfidf', 'cross-encoder']
    n_docs = [15, 10, 5]
    retriever_configs = [(strategy, k) for strategy in strategies for k in n_docs]
    retrievers = {'bm25': BM25Retriever, 'tfidf': TFIDFRetriever}

    output_dir = 'reranking'
    os.makedirs(output_dir, exist_ok=True)

    for strategy, k in tqdm(retriever_configs, desc="Procesando estrategias de recuperación"):
        prepared_data_json, prepared_data_csv = prepare_data(df, strategy, k, retrievers)

        # Guardar resultados
        output_json = f'{output_dir}/{strategy}_{k}.json'
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(prepared_data_json, f, ensure_ascii=False, indent=4)

        output_csv = f'{output_dir}/{strategy}_{k}.csv'
        prepared_data_csv.to_csv(output_csv, index=False)

        print(f'Datos guardados en {output_json} y {output_csv}')
