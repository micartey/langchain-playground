from sklearn.decomposition import PCA
import plotly.express as px
import numpy as np

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Get embeddings
embeddings_model = OllamaEmbeddings(model="mxbai-embed-large")
chroma = Chroma(
    collection_name="test_collection",
    embedding_function=embeddings_model,
    persist_directory=".chromadb"
)
retrieved_data = chroma.get(include=['embeddings'])
embeddings = retrieved_data['embeddings']
docs = retrieved_data['ids']


# Reduce the embedding dimensionality
pca = PCA(n_components=3)
vis_dims = pca.fit_transform(embeddings)

# Calculate the color scale factor based in the data
scale_factor = np.max(np.abs(vis_dims)) * 20

# Create a list of 'rgb(r,g,b)' color strings
colors = [
    f"rgb({int(127 + r / scale_factor * 255)},{int(127 + g / scale_factor * 255)},{int(127 + b / scale_factor * 255)})"
    for r, g, b in vis_dims
]

fig = px.scatter_3d(
    x=vis_dims[:, 0],
    y=vis_dims[:, 1],
    z=vis_dims[:, 2],
    text=docs,
    title='Embeddings by Orientation',
    color=colors, # Apply the custom HSL colors
)
# Set marker style for better appearance
fig.update_traces(marker=dict(size=6, line=dict(width=0)))

# Style the plot: remove background, grids, and the color bar
fig.update_layout(
    scene=dict(
        xaxis=dict(showbackground=False, showgrid=False, zeroline=False, title=''),
        yaxis=dict(showbackground=False, showgrid=False, zeroline=False, title=''),
        zaxis=dict(showbackground=False, showgrid=False, zeroline=False, title=''),
        aspectmode='cube'
    ),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    margin=dict(l=0, r=0, b=0, t=40),
    coloraxis_showscale=False
)

fig.show()
