import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pickle

# Function to build or load the graph
def build_or_load_graph():
    try:
        # Try to load the graph from a saved file
        with open('undirected_weighted_graph.pkl', 'rb') as file:
            G = pickle.load(file)
        print("Graph loaded from file.")
    except FileNotFoundError:
        # If the file doesn't exist, build the graph and save it
        G = build_graph()
        with open('undirected_weighted_graph.pkl', 'wb') as file:
            pickle.dump(G, file)
        print("Graph built and saved to file.")
    return G

# Function to build the undirected graph
def build_graph():
    # Load the dataset with proper handling of mixed types
    df = pd.read_csv('../dataset/usdataset.csv', low_memory=False)
    # Create an undirected graph
    G = nx.Graph()

    # Add channel nodes with 'node_type' attribute set to 'channel'
    for index, row1 in df.iterrows():
        G.add_node(row1['channelId'], title=row1['channelTitle'], node_type='channel')

    # Add video nodes with 'node_type' attribute set to 'video'
    videos = set(df['video_id'])
    G.add_nodes_from(videos, node_type='video')

    # Add edges with weights based on views, comments, likes, and dislikes
    for index, row in df.iterrows():
        channel_id = row['channelId']
        video_id = row['video_id']
        views = row['view_count']
        comments = row['comment_count']
        likes = row['likes']
        dislikes = row['dislikes']

        # You can customize how you want to calculate the edge weight
        weight = views + 4 * comments + 2 * likes - 2 * dislikes

        G.add_edge(channel_id, video_id, weight=weight)

    return G

# Build or load the undirected graph
G = build_or_load_graph()

# Calculate centrality measures on the entire graph G
degree_centrality = nx.degree_centrality(G)
weighted_degree_centrality = dict(G.degree(weight='weight'))
weighted_eigenvector_centrality = nx.eigenvector_centrality_numpy(G, weight='weight')
weighted_pagerank = nx.pagerank(G, weight='weight')

# Print top 15 channels by each centrality measure
print("\nTop 15 channels by Degree Centrality:")
top_channels_degree = sorted(degree_centrality, key=degree_centrality.get, reverse=True)[:15]
for channel in top_channels_degree:
    title = G.nodes[channel].get('title', 'Title not available')
    degree = G.degree(channel)
    w_degree = weighted_degree_centrality[channel]
    w_eigenvector = weighted_eigenvector_centrality[channel]
    w_pagerank = weighted_pagerank[channel]
    print(f"Channel ID: {channel}, Title: {title}, Degree: {degree}, Weighted Degree: {w_degree}, Weighted Eigenvector: {w_eigenvector}, Weighted PageRank: {w_pagerank}")
