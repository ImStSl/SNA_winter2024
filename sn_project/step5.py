import pickle
from networkx.algorithms import community
import matplotlib.pyplot as plt
import networkx as nx
def load_graph():
    try:
        # Try to load the graph from a saved file
        with open('undirected_graph.pkl', 'rb') as file:
            G = pickle.load(file)
        print("Graph loaded from file.")
    except FileNotFoundError:
        print("Graph loading failed.")
    return G

G = load_graph()
# Perform community detection using the Louvain method
communities = list(community.greedy_modularity_communities(G))

# Print the communities with video titles
for i, community in enumerate(communities):
    titles = [G.nodes[video_id].get('title', f'Title not available for {video_id}') for video_id in community]
    print(f"Community {i + 1}: {titles}")

color_map = {}
for i, community in enumerate(communities):
    for video_id in community:
        color_map[video_id] = i

# Draw the graph with nodes colored by community
pos = nx.spring_layout(G, seed=42)  # Using seed for reproducibility
plt.figure(figsize=(16, 12))
nx.draw(G, pos, node_color=[color_map[node] for node in G.nodes], with_labels=False, node_size=20, cmap='viridis', alpha=0.7)
plt.title('Community Detection Visualization')
plt.show()