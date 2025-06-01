from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import NLTKTextSplitter
from langchain.schema import Document
import pandas as pd
import faiss
import json
from typing import List

def import_data(file_path):
    """
    Carga los artículos desde un CSV y los convierte en objetos Document de LangChain.

    Args:
        file_path (str): Ruta del archivo CSV.

    Returns:
        list: Lista de documentos procesados.
    """
    df = pd.read_csv(file_path)

    # Validar que el archivo contiene las columnas requeridas
    required_columns = {'content', 'title', 'categories'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"El archivo debe contener las columnas: {required_columns}")

    # Crear lista de documentos
    documents = [
        Document(
            page_content=row['content'],
            metadata={
                'id': i + 1,
                'title': row['title'],
                'categories': row['categories']
            },
        )
        for i, row in df.iterrows()
    ]

    return documents

def split_documents(docs: List[Document], size: int) -> List[Document]:
    """
    Divide los documentos en chunks utilizando NLTK.

    Args:
        docs (List[Document]): Lista de documentos a dividir.
        size (int): Tamaño de chunk.

    Returns:
        List[Document]: Lista de chunks generados.
    """
    splitter = NLTKTextSplitter(
        chunk_size=size,
        chunk_overlap=size // 4,
        language='spanish',
        separator=' '
    )
    return splitter.split_documents(docs)

def chunks_to_json(chunks, filename):
    """
    Guarda los chunks como archivo JSON estructurado.

    Args:
        chunks: Lista de documentos divididos.
        filename: Nombre del archivo de salida.
    """
    docs = [{'content': doc.page_content, 'metadata': doc.metadata} for doc in chunks]

    data = {f"chunk_{i}": doc for i, doc in enumerate(docs)}

    output_file = f'chunks/{filename}'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    model_list = [
        ['minilm', 'paraphrase-multilingual-MiniLM-L12-v2'],
        ['mpnet', 'paraphrase-multilingual-mpnet-base-v2']
    ]
    size_list = ['full', '512', '1024']
    file_path = 'data/wikipedia_jaen.csv'

    docs = import_data(file_path)

    for model_name, model_path in model_list:
        # Cargar el modelo de embeddings
        model_kwargs = {'device': 'cuda'}
        encode_kwargs = {'normalize_embeddings': True}
        embed_model = HuggingFaceEmbeddings(
            model_name=f'sentence-transformers/{model_path}',  # Ruta sustituida
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

        # Comprobar dimensión de embeddings
        sample_text = "texto de ejemplo"
        embedding = embed_model.embed_query(sample_text)
        print(f"Dimensión del embedding: {len(embedding)}")

        for size in size_list:
            if size == 'full':
                chunks = docs
            else:
                chunks = split_documents(docs, int(size))
                filename = f'chunks_{model_name}_{size}.json'
                chunks_to_json(chunks, filename)

            # Crear índice FAISS
            d = len(embedding)
            faiss_index = faiss.IndexFlatL2(d)
            imdocstore = InMemoryDocstore({})
            vectorstore = FAISS(
                index=faiss_index,
                docstore=imdocstore,
                index_to_docstore_id={},
                embedding_function=embed_model,
                distance_strategy='COSINE'
            )

            vectorstore.add_documents(chunks)

            # Guardar el índice en disco
            persist_dir = f"storage/faiss_index_{model_name}_{size}"
            vectorstore.save_local(persist_dir)
            print(f'COMPLETADO: {model_name} {size}')
