from langchain.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
import json

# Ruta donde se guardan los vectores de FAISS
persist_directory = "storage"

# Modelo de embeddings multilingüe
embedding_model = HuggingFaceEmbeddings(
    model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True},
)

# Cargar el vector store
vector_store = FAISS.load_local(persist_directory, embeddings=embedding_model, allow_dangerous_deserialization=True)

# Configuración del recuperador
retriever_config = {"search_type": "similarity", "search_kwargs": {"k": 20}}
retriever = vector_store.as_retriever(**retriever_config)

# Modelo de reranking para reordenar resultados según relevancia
reranker = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")


@tool
def retrieval_augmented_generation(query: str) -> str:
    """
    Recupera documentos relevantes usando FAISS + reranking con un modelo CrossEncoder.
    
    :param query: Consulta del usuario.
    :return: Documentos rerankeados como JSON.
    """
    # Recuperación inicial
    retrieved_docs = retriever.invoke(query)
    retrieved_texts = [doc.page_content[:1024] for doc in retrieved_docs]

    # Emparejar consulta con cada contexto recuperado
    pairs = [(query, context) for context in retrieved_texts]

    # Obtener puntuaciones de relevancia
    scores = reranker.predict(pairs)

    # Selección de los 5 mejores documentos tras reranking
    top_docs = [doc for _, doc in sorted(zip(scores, retrieved_texts), reverse=True)][:5]

    return json.dumps(
        {i + 1: doc for i, doc in enumerate(top_docs)},
        ensure_ascii=False,
        indent=4
    )
