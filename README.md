# Desarrollo de asistentes virtuales personalizados para personas mayores mediante técnicas avanzadas de PLN

El presente repositorio contiene el código fuente del TFM titulado "Desarrollo de asistentes virtuales personalizados para personas mayores mediante técnicas avanzadas de PLN", realizado por el alumno Samuel Sánchez Carrasco, estudiante del Máster de Ingeniería informática en la Universidad de Jaén.

## Estructura

- `prototype/` Contiene el prototipo completo del asistente conversacional.
  - `app/` Aplicación cliente (React Native + Expo Go) que permite grabar audio, enviar consultas al backend y recibir respuestas en texto y audio.
    - `App.js`, `FormScreen.js`, `MainScreen.js`, `index.js`: lógica de navegación y pantallas.
    - `config.js`: configuración de la URL base del backend.
    - `babel.config.js`, `package.json`, `yarn.lock`, `assets/`: configuración y recursos de la app.
  - `llm_agent/` Backend del sistema (Python + FastAPI + Ngrok).
    - `agent.py`: implementa un agente conversacional con LangGraph y clasificación de consultas.
    - `backend.py`: API con transcripción de audio mediante Whisper, generación de respuestas y text-to-speech mediante Edge-TTS.
    - `llm_api.py`: cliente para acceder al LLM alojado en un servidor externo.
    - `tools.py`: definición de herramientas de recuperación con FAISS y reranking.
    - `storage/`: contiene el índice FAISS utilizado para RAG.
- `evaluations/` Conjunto de scripts y datos para construir el corpus, indexar, recuperar, generar y evaluar respuestas automáticas.
  - `chunks/` Chunks generados a partir del corpus para distintos modelos y tamaños.
  - `data/` Datos de entrada del sistema: corpus de Wikipedia sobre la provincia Jaén según división en chunks.
  - `faiss_contexts/` Contextos recuperados para cada pregunta usando distintas configuraciones (modelo, chunking, estrategia y NER).
  - `faiss_evaluated/` Métricas de evaluación aplicadas a los contextos recuperados (precisión y recall).
  - `llm_responses/` Respuestas generadas por diferentes LLMs y prompts adaptados/no adaptados.
  - `reranking/` Contextos reranqueados usando BM25, TF-IDF y CrossEncoder. 
  - `reranking_evaluated/` Resultados de evaluación sobre contextos reranqueados.
  - `stats/` Estadísticas del corpus y visualizaciones: histogramas, boxplots y estadísticas de tokens y preguntas evaluadas.
  - `storage/` Almacena los índices FAISS generados para cada modelo y configuración de chunking.
  - `download_wikipedia.py`: descarga y limpia artículos de Wikipedia sobre la provincia de Jaén.
  - `data_stats.py`: analiza la distribución de tokens en el corpus.
  - `test_store.py`: genera chunks, calcula embeddings y construye índices FAISS.
  - `test_contexts.py`: recupera contextos relevantes para cada pregunta.
  - `test_reranking.py`: aplica reranking con BM25, TF-IDF y CrossEncoder.
  - `test_responses.py`: genera respuestas con LLMs usando los contextos recuperados.
  - `test_contexts_eval.py`: evalúa la calidad de la recuperación según métricas de Ragas.
  - `test_reranking_eval.py`: evalúa la calidad de los contextos reranqueados según métricas de Ragas.
  - `test_contexts_resume.py`: resume los resultados de evaluación.

## Instalación

### Requisitos generales

* Python 3.10+
* Node.js y npm (o yarn)
* ffmpeg (necesario para Whisper)
* git (para instalar whisper desde el repositorio oficial)

### Backend (Python)

```bash
cd prototype/llm_agent
python -m venv venv
source venv/bin/activate  # o venv\Scripts\activate en Windows
```

Instalar dependencias:

```bash
pip install -r requirements.txt  # si está disponible
```

O manualmente:

```bash
pip install fastapi uvicorn aiofiles whisper edge-tts langchain langgraph sentence-transformers faiss-cpu json-repair pyngrok requests langfuse
```

Instalar Whisper desde el repositorio oficial si no está en PyPI:

```bash
pip install git+https://github.com/openai/whisper.git
```

Instalar modelo de spaCy usado en NER:

```bash
python -m spacy download es_core_news_lg
```

### Cliente (React Native)

```bash
cd prototype/app
npm install  # o yarn
npx expo start
```

> Requiere tener instalada la app Expo Go en el móvil para escanear el código QR.

## Uso

### Ejecutar backend

```bash
cd prototype/llm_agent
python backend.py  # abrirá automáticamente un túnel ngrok
```

### Ejecutar cliente

```bash
cd prototype/app
npx expo start
```

Escanea el QR con la app Expo Go en tu teléfono.

### Pipeline de evaluación

```bash
cd evaluations
python download_wikipedia.py
python data_stats.py
python test_store.py
python test_contexts.py
python test_reranking.py
python test_responses.py
python test_contexts_eval.py
python test_reranking_eval.py
python test_contexts_resume.py
```

Esto genera y evalúa las respuestas automáticas y la calidad de la recuperación.


## Contacto
Para dudas o sugerencias, contactar a [ssc00022@red.ujaen.es].