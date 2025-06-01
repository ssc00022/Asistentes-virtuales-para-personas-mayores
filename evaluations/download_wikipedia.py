import requests
import pandas as pd
import re
import mwparserfromhell

# Crear un DataFrame vacío para almacenar los artículos
articles_df = pd.DataFrame(columns=["title", "content", "categories"])

# Lista de categorías que se excluirán de la exploración
exclude_categories = [
    "Casa de Alburquerque",
    "Río Guadalquivir",
    "Río Segura",
    "Ríos de Sierra Morena",
    "Reino de Murcia"
]

def fetch_article(title):
    """Descarga el contenido plano de un artículo desde Wikipedia."""
    url = "https://es.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": True,
        "titles": title
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        pages = data.get('query', {}).get('pages', {})
        for page_id, page_content in pages.items():
            if "extract" in page_content and page_content["extract"].strip():
                return {
                    "title": page_content["title"],
                    "content": page_content["extract"]
                }
            else:
                print(f"El artículo '{title}' no tiene contenido.")
    except requests.RequestException as e:
        print(f"Error al descargar el artículo '{title}': {e}")
    return None

def add_article(title, content, category, articles_df):
    """Agrega un artículo nuevo al DataFrame."""
    print(f"Importando artículo: {title}\nCategoría: {category}")
    new_row = pd.DataFrame([{
        "title": title,
        "content": content,
        "categories": [category]
    }])
    return pd.concat([articles_df, new_row], ignore_index=True)

def update_article(title, category, articles_df):
    """Actualiza las categorías asociadas a un artículo ya incluido."""
    idx = articles_df[articles_df["title"] == title].index[0]
    current_categories = set(articles_df.at[idx, "categories"])
    print(f"Actualizando categorías para '{title}'. Categorías existentes: {current_categories}")
    current_categories.add(category)
    articles_df.at[idx, "categories"] = list(current_categories)
    return articles_df

def explore_category(category, visited_categories, articles_df):
    """Explora una categoría y sus subcategorías recursivamente."""
    if category in visited_categories or category in exclude_categories:
        print(f"Ignorando categoría ya visitada: {category}")
        return articles_df
    visited_categories.add(category)

    url = "https://es.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "list": "categorymembers",
        "cmtitle": f"Categoría:{category}",
        "cmlimit": "max"
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        for member in response.json().get("query", {}).get("categorymembers", []):
            title = member["title"]
            ns = member["ns"]

            if ns == 0:  # Artículo
                if title in articles_df["title"].values:
                    articles_df = update_article(title, category, articles_df)
                else:
                    article = fetch_article(title)
                    if article:
                        articles_df = add_article(article["title"], article["content"], category, articles_df)
            elif ns == 14:  # Subcategoría
                subcat = title.replace("Categoría:", "")
                if subcat not in exclude_categories:
                    print(f"Explorando subcategoría: {subcat}")
                    articles_df = explore_category(subcat, visited_categories, articles_df)
                else:
                    print(f"Ignorando categoría excluida: {subcat}")
            elif ns == 100:  # Portal
                print(f"Ignorando portal: {title}")
    except requests.RequestException as e:
        print(f"Error al explorar categoría '{category}': {e}")
    return articles_df

def clean_text(text):
    """Limpia el texto eliminando etiquetas y caracteres no deseados."""
    wikicode = mwparserfromhell.parse(text)
    plain_text = wikicode.strip_code()
    return re.sub(r'\s+', ' ', plain_text).strip()

def remove_irrelevant_sections(text):
    """Elimina secciones no útiles como referencias o enlaces externos."""
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'== Véase también ==.*', '', text, flags=re.DOTALL)
    text = re.sub(r'== Referencias ==.*', '', text, flags=re.DOTALL)
    text = re.sub(r'== Enlaces externos ==.*', '', text, flags=re.DOTALL)
    return text.strip()

# Punto de entrada del script
if __name__ == "__main__":
    initial_category = "Provincia de Jaén"
    visited_categories = set()
    articles_df = explore_category(initial_category, visited_categories, articles_df)

    # Limpiar el contenido de los artículos
    articles_df['content'] = articles_df['content'].apply(remove_irrelevant_sections)
    articles_df['content'] = articles_df['content'].apply(clean_text)

    # Exportar el resultado a CSV
    articles_df.to_csv("data/wikipedia_jaen.csv", index=False, encoding="utf-8")
    print("Exploración completada. Artículos guardados en 'wikipedia_jaen.csv'.")
