import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer
from nltk.tokenize import word_tokenize

# Descargar recursos de NLTK si no están disponibles
nltk.download("punkt")

# Cargar el dataset generado desde Wikipedia
df = pd.read_csv("data/wikipedia_jaen.csv")  # Alternativamente: pd.read_parquet("dataset.parquet")

# Rellenar valores nulos en el contenido
df["content"] = df["content"].fillna("")

# Tokenización con NLTK
df["num_tokens_nltk"] = df["content"].apply(lambda x: len(word_tokenize(str(x))))

# Tokenización con modelos de Hugging Face (MPNet y MiniLM)
mpnet_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
df["num_tokens_mpnet"] = df["content"].apply(lambda x: len(mpnet_tokenizer.tokenize(x)))

minilm_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
df["num_tokens_minilm"] = df["content"].apply(lambda x: len(minilm_tokenizer.tokenize(x)))

# Función para calcular estadísticas descriptivas de una columna
def calcular_estadisticas(columna, name):
    return pd.DataFrame([{
        "Nombre": name,
        "Moda": columna.mode()[0] if not columna.mode().empty else None,
        "Media": columna.mean(),
        "Mediana": columna.median(),
        "Mínimo": columna.min(),
        "Máximo": columna.max(),
        "Desviación estándar": columna.std(),
        "Varianza": columna.var(),
        "Cuartil 1 (Q1)": columna.quantile(0.25),
        "Cuartil 2 (Q2 - Mediana)": columna.quantile(0.50),
        "Cuartil 3 (Q3)": columna.quantile(0.75),
        "Percentil 5%": columna.quantile(0.05),
        "Percentil 95%": columna.quantile(0.95),
    }])

# Eliminar valores atípicos usando IQR
def filtrar_outliers(columna):
    q1 = columna.quantile(0.25)
    q3 = columna.quantile(0.75)
    iqr = q3 - q1
    return columna[(columna >= q1 - 1.5 * iqr) & (columna <= q3 + 1.5 * iqr)]

# Calcular estadísticas originales y sin outliers
stats_df = pd.concat([
    calcular_estadisticas(df["num_tokens_nltk"], "NLTK - Original"),
    calcular_estadisticas(df["num_tokens_mpnet"], "MPNet - Original"),
    calcular_estadisticas(df["num_tokens_minilm"], "MiniLM - Original"),
    calcular_estadisticas(filtrar_outliers(df["num_tokens_nltk"]).dropna(), "NLTK - Sin outliers"),
    calcular_estadisticas(filtrar_outliers(df["num_tokens_mpnet"]).dropna(), "MPNet - Sin outliers"),
    calcular_estadisticas(filtrar_outliers(df["num_tokens_minilm"]).dropna(), "MiniLM - Sin outliers"),
], ignore_index=True)

# Guardar estadísticas en CSV
stats_df.to_csv("stats/token_stats.csv", index=False)

# Histograma con valores originales
plt.figure(figsize=(7, 5))
sns.histplot(df["num_tokens_nltk"], bins=30, kde=True, color="blue", label="NLTK", alpha=0.5)
sns.histplot(df["num_tokens_mpnet"], bins=30, kde=True, color="green", label="MPNet", alpha=0.5)
sns.histplot(df["num_tokens_minilm"], bins=30, kde=True, color="red", label="MiniLM", alpha=0.5)
plt.xlabel("Número de tokens")
plt.ylabel("Frecuencia")
plt.title("Distribución del número de tokens (Original)")
plt.legend()
plt.savefig("stats/histograma_tokens.png")
plt.close()

# Boxplot con valores originales
plt.figure(figsize=(7, 5))
sns.boxplot(data=df[["num_tokens_nltk", "num_tokens_mpnet", "num_tokens_minilm"]],
            palette=["blue", "green", "red"])
plt.xticks([0, 1, 2], ["NLTK", "MPNet", "MiniLM"])
plt.title("Boxplot de longitudes de tokens (Original)")
plt.ylabel("Número de tokens")
plt.savefig("stats/boxplot_tokens.png")
plt.close()

# Filtrar outliers para visualización limpia
df_filtrado = df.copy()
df_filtrado["num_tokens_nltk"] = filtrar_outliers(df["num_tokens_nltk"])
df_filtrado["num_tokens_mpnet"] = filtrar_outliers(df["num_tokens_mpnet"])
df_filtrado["num_tokens_minilm"] = filtrar_outliers(df["num_tokens_minilm"])

# Histograma sin outliers
plt.figure(figsize=(7, 5))
sns.histplot(df_filtrado["num_tokens_nltk"].dropna(), bins=30, kde=True, color="blue", label="NLTK", alpha=0.5)
sns.histplot(df_filtrado["num_tokens_mpnet"].dropna(), bins=30, kde=True, color="green", label="MPNet", alpha=0.5)
sns.histplot(df_filtrado["num_tokens_minilm"].dropna(), bins=30, kde=True, color="red", label="MiniLM", alpha=0.5)
plt.title("Distribución del número de tokens (Sin outliers)")
plt.xlabel("Número de tokens")
plt.ylabel("Frecuencia")
plt.legend()
plt.savefig("stats/histograma_sin_outliers.png")
plt.close()

# Boxplot sin outliers
plt.figure(figsize=(7, 5))
sns.boxplot(data=df_filtrado[["num_tokens_nltk", "num_tokens_mpnet", "num_tokens_minilm"]],
            palette=["blue", "green", "red"])
plt.xticks([0, 1, 2], ["NLTK", "MPNet", "MiniLM"])
plt.title("Boxplot de longitudes de tokens (Sin outliers)")
plt.ylabel("Número de tokens")
plt.savefig("stats/boxplot_sin_outliers.png")
plt.close()

# Mostrar características de tokenizadores
print("Características de los tokenizadores:")
print("\nMPNet:\n", mpnet_tokenizer)
print("\nMiniLM:\n", minilm_tokenizer)

# Ejemplo práctico de tokenización
ejemplo_texto = "Jaén es una ciudad histórica con una gran tradición cultural."
print("\nEjemplo de tokenización:")
print("Texto original:", ejemplo_texto)
print("Tokens NLTK:", word_tokenize(ejemplo_texto))
print("Tokens MPNet:", mpnet_tokenizer.tokenize(ejemplo_texto))
print("Tokens MiniLM:", minilm_tokenizer.tokenize(ejemplo_texto))
