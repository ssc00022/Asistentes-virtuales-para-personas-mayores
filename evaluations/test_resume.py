import os
import pandas as pd

# Directorios que contienen los archivos evaluados (CSV con métricas)
folders = ["faiss_evaluated", "reranking_evaluated"]

for folder_path in folders:
    # Listas para acumular resultados
    results = []
    all_data = []
    
    # Recorrer todos los archivos .csv del directorio
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)

            # Cargar archivo
            df = pd.read_csv(file_path, sep=",")

            # Reemplazar comillas simples por punto decimal en caso de errores regionales
            df = df.replace({"'": "."}, regex=True)

            # Convertir columnas a tipo numérico de forma segura
            df["non_llm_context_precision_with_reference"] = pd.to_numeric(
                df["non_llm_context_precision_with_reference"], errors='coerce'
            )
            df["non_llm_context_recall"] = pd.to_numeric(
                df["non_llm_context_recall"], errors='coerce'
            )

            # Calcular medias sobre los primeros 24 ejemplos
            mean_values = df[[
                "non_llm_context_precision_with_reference",
                "non_llm_context_recall"
            ]].mean()

            # Guardar resultado resumido por archivo
            results.append({
                "filename": filename[:-4],
                "precision_mean": mean_values["non_llm_context_precision_with_reference"],
                "recall_mean": mean_values["non_llm_context_recall"]
            })

            # Acumular todas las preguntas para resumen posterior
            all_data.append(df)

    # Crear resumen global por archivo
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"stats/{folder_path}_contexts.csv", index=False, sep=',', decimal="'")

    # Mostrar los 10 mejores en cada métrica
    print("Top 10 archivos por precisión:")
    print(results_df.nlargest(10, 'precision_mean'))

    print("Top 10 archivos por recall:")
    print(results_df.nlargest(10, 'recall_mean'))

    # Calcular media por pregunta (acumulada entre archivos)
    all_data_df = pd.concat(all_data).groupby("question", as_index=False).mean()

    # Guardar resultados por pregunta
    all_data_df.to_csv(f"stats/{folder_path}_questions.csv", index=False, sep=',', decimal="'")
    print("Dataset con la media de todas las preguntas:")
    print(all_data_df)
