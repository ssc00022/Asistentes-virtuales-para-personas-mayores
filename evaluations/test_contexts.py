from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import pandas as pd
import spacy
import json
import math

def prepare_data(df_orig, retriever, ner):
    """
    Genera los contextos recuperados a partir de las preguntas del CSV original.

    Args:
        df_orig (DataFrame): Dataset con preguntas y referencias.
        retriever: Objeto que permite recuperar documentos relevantes.
        ner (bool): Si True, aplica extracción de sustantivos con spaCy.

    Returns:
        Tuple[dict, DataFrame]: Datos en formato JSON y DataFrame.
    """
    questions = df_orig['question'].tolist()
    references = df_orig['reference'].tolist()
    ids = df_orig['id_context'].tolist()
    reference_contexts = df_orig['reference_contexts'].tolist()

    data_list = []
    data_dict = {}

    if ner:
        nlp = spacy.load('es_core_news_lg')

    for idx, question in enumerate(questions):
        query = extraer_sustantivos(nlp(question)) if ner else question

        retrieved_docs = retriever.invoke(query)
        retrieved_contexts = [{'content': doc.page_content, 'metadata': doc.metadata} for doc in retrieved_docs]

        entry = {
            "question": question,
            "query": query,
            "id_context": int(ids[idx]) if not math.isnan(ids[idx]) else -1,
            "retrieved_contexts": retrieved_contexts,
            "reference": references[idx],
            "reference_contexts": reference_contexts[idx]
        }

        data_dict[f"test_{idx + 1}"] = entry
        data_list.append({
            "question": question,
            "retrieved_contexts": [doc.page_content for doc in retrieved_docs],
            "reference": references[idx],
            "reference_contexts": reference_contexts[idx]
        })

    return data_dict, pd.DataFrame(data_list)

def extraer_sustantivos(doc):
    """
    Extrae los sustantivos comunes y propios de un documento spaCy.
    """
    palabras_filtradas = [token.text for token in doc if token.pos_ in ("NOUN", "PROPN")]
    return " ".join(palabras_filtradas)

if __name__ == '__main__':
    # Carga de datasets de preguntas según tamaño de chunk
    csv_files = {
        "full": pd.read_csv("data/preguntas_wikipedia_jaen_full.csv"),
        "512": pd.read_csv("data/preguntas_wikipedia_jaen_512.csv"),
        "1024": pd.read_csv("data/preguntas_wikipedia_jaen_1024.csv"),
    }

    model_list = [
        ['minilm', 'paraphrase-multilingual-MiniLM-L12-v2'],
        ['mpnet', 'paraphrase-multilingual-mpnet-base-v2']
    ]
    size_list = ["full", "512", "1024"]

    retriever_configs = [
        ['similarity5', {"search_type": "similarity", "search_kwargs": {"k": 5}}],
        ['similarity10', {"search_type": "similarity", "search_kwargs": {"k": 10}}],
        ['similarity20', {"search_type": "similarity", "search_kwargs": {"k": 20}}],
        ['mmr5', {"search_type": "mmr", "search_kwargs": {"k": 5, "fetch_k": 25}}],
        ['mmr10', {"search_type": "mmr", "search_kwargs": {"k": 10, "fetch_k": 50}}],
        ['mmr20', {"search_type": "mmr", "search_kwargs": {"k": 20, "fetch_k": 100}}],
    ]

    ner_list = [
        ['', False],
        ['ner', True]
    ]

    for model_name, model_path in model_list:
        model_kwargs = {'device': 'cuda'}
        encode_kwargs = {'normalize_embeddings': True}
        embed_model = HuggingFaceEmbeddings(
            model_name=f'sentence-transformers/{model_path}',  # Ruta sustituida
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

        for size in size_list:
            persist_dir = f"storage/faiss_index_{model_name}_{size}"
            vectorstore = FAISS.load_local(persist_dir, embeddings=embed_model, allow_dangerous_deserialization=True)

            for strategy, config in retriever_configs:
                retriever = vectorstore.as_retriever(**config)

                for suffix, use_ner in ner_list:
                    df_orig = csv_files[size]

                    prepared_data_json, prepared_data_csv = prepare_data(df_orig, retriever, use_ner)

                    output_json = f'faiss_contexts/{model_name}_{size}_{strategy}{suffix}.json'
                    with open(output_json, 'w', encoding='utf-8') as f:
                        json.dump(prepared_data_json, f, ensure_ascii=False, indent=4)

                    output_csv = f'faiss_contexts/{model_name}_{size}_{strategy}{suffix}.csv'
                    prepared_data_csv.to_csv(output_csv, index=True)

                    print(f'Datos guardados en {output_json} y {output_csv}')
