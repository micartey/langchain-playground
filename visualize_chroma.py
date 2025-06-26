import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="mxbai-embed-large")
db = Chroma(
    collection_name="test_collection",
    embedding_function=embeddings,
    persist_directory=".chromadb"
)

# --- 2. Daten und Embeddings aus ChromaDB abrufen ---
# Dokumentation: Die .get() Methode ist der Schlüssel zum Extrahieren von Inhalten.
# Wir übergeben `include=['embeddings', 'documents']`, um genau die Daten zu erhalten, die wir benötigen.
retrieved_data = db.get(include=['embeddings', 'documents'])

# Dokumentation: Wir extrahieren die Dokumente und die Embeddings in separate Variablen.
# Die Embeddings werden für die Dimensionalitätsreduktion in ein NumPy-Array umgewandelt.
retrieved_documents = retrieved_data['documents']
embeddings_array = np.array(retrieved_data['embeddings'])

print(f"Found embeddings of shape: {embeddings_array.shape}")

# Reduces dimensions
tsne = TSNE(
    n_components=2, # amount of dimensions
    perplexity=3,   # number of neighbors
    random_state=42 # seed
)

# Dokumentation: Wir wenden t-SNE auf unsere hochdimensionalen Embeddings an.
# Das Ergebnis ist ein neues Array mit denselben Zeilen, aber nur 2 Spalten (x, y).
tsne_results = tsne.fit_transform(embeddings_array)

print("Dimensionalitätsreduktion mit t-SNE ist abgeschlossen.")


# --- 4. Visualisierung mit Matplotlib ---
# Dokumentation: Wir erstellen eine Figur und eine Achse für unseren Plot.
fig, ax = plt.subplots(figsize=(12, 12))

# Dokumentation: Wir erstellen ein Streudiagramm (Scatter Plot) mit den 2D-Koordinaten.
# tsne_results[:, 0] sind die x-Koordinaten, tsne_results[:, 1] die y-Koordinaten.
ax.scatter(tsne_results[:, 0], tsne_results[:, 1])

print("Erstelle den Plot. Füge Annotationen für jeden Punkt hinzu...")

# Dokumentation: Wir fügen jedem Punkt eine Beschriftung (das Originaldokument) hinzu.
# Dies macht die Visualisierung aussagekräftig.
for i, doc in enumerate(retrieved_documents):
    # Wir kürzen lange Texte für eine bessere Lesbarkeit
    print(doc)
    short_doc = doc[:30] + '...' if len(doc) > 30 else doc
    ax.annotate(short_doc, (tsne_results[i, 0], tsne_results[i, 1] + 3), fontsize=9)

# Dokumentation: Titel und Achsenbeschriftungen hinzufügen.
ax.set_title("Embeddings", fontsize=16)
# ax.set_xlabel("t-SNE Komponente 1", fontsize=12)
# ax.set_ylabel("t-SNE Komponente 2", fontsize=12)
ax.grid(False)

plt.show()
