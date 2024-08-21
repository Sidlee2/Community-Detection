import random
import networkx as nx
import numpy as np
from community import community_louvain
import matplotlib.pyplot as plt

def generate_network(n):
    '''
    This function will generate a random weighted network associated with the user-specified
    number of nodes.
    
    params:
        n (Integer): The number of nodes you want in your network
    
    returns:
        A networkX multi-graph
        
    example:
        G = generate_network(n)
    '''
    # initialize dictionary with nodes
    graph_dct = {node: [] for node in range(n)}
    nodes = list(range(n))
    
    # generate edges
    for n, edge_list in graph_dct.items():
        edge_c = random.randint(min(nodes), int(max(nodes) / 2))
        el = random.sample(nodes, edge_c)
        graph_dct[n] = el
    
    # create networkx multi-edge graph
    G = nx.MultiGraph(graph_dct)
    return G

def generate_fuzzy_membership_matrix(num_nodes, num_communities):
    fuzzy_membership_matrix = np.zeros((num_nodes, num_communities))

    # Assign nodes to communities sequentially
    for i in range(num_nodes):
        # Generate random fuzzy measure values between 0 and 1
        fuzzy_membership_matrix[i] = np.random.uniform(0.3, 0.7, num_communities)

        # Normalize to sum to 1
        fuzzy_membership_matrix[i] = fuzzy_membership_matrix[i] / np.sum(fuzzy_membership_matrix[i])

    return fuzzy_membership_matrix

def compute_modularity(adj_matrix, mu, C):
    m = np.sum(adj_matrix) / 2
    degrees = np.sum(adj_matrix, axis=1)
    Qe = 0

    num_nodes = len(adj_matrix)

    for i in range(num_nodes):
        for j in range(num_nodes):
            s_ij = max([min(mu[i, c], mu[j, c]) for c in range(C)])
            A_ij = adj_matrix[i, j]
            ki = degrees[i]
            kj = degrees[j]
            Qe += (A_ij - (ki * kj / (2 * m))) * s_ij

    Qe = Qe / (2 * m)
    return Qe

def create_network(G, fuzzy_membership_matrix):
    n = len(G.nodes)
    
    # Print the fuzzy membership matrix before Louvain
    print("\nFuzzy Membership Matrix before Louvain:")
    print(fuzzy_membership_matrix)

    # Compute modularity before Louvain
    modularity_before = compute_modularity(nx.adjacency_matrix(G).toarray(), fuzzy_membership_matrix, fuzzy_membership_matrix.shape[1])
    print(f"\nModularity before Louvain: {modularity_before:.3f}")

    # Visualize the graph before Louvain
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=75, font_size=8, font_color='black', font_weight='bold', cmap=plt.cm.tab10, edge_color='gray', width=1, alpha=0.7)
    plt.title('Graph before Louvain')
    plt.show()

    # Perform community detection using the Louvain method
    partition = community_louvain.best_partition(G)

    # Determine the number of communities found by Louvain
    num_communities_after = max(partition.values()) + 1

    # Update fuzzy membership matrix based on Louvain results
    fuzzy_membership_matrix_after = np.zeros((n, num_communities_after))
    for i, node in enumerate(G.nodes):
        fuzzy_membership_matrix_after[i] = np.zeros(num_communities_after)
        fuzzy_membership_matrix_after[i][partition[node]] = 1.0

    # Print statements before the plot
    print("\nCommunity Assignment before Louvain:")
    for node, comm_id in partition.items():
        print(f"Node {node} is in community {comm_id}")

    # Visualize the graph with community highlights
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=75, font_size=8, font_color='black', font_weight='bold', node_color=list(partition.values()), cmap=plt.cm.tab10, edge_color='gray', width=1, alpha=0.7)
    plt.title('Graph with Community Highlights before Louvain')
    plt.show()

    # Print the fuzzy membership matrix after Louvain
    print("\nFuzzy Membership Matrix after Louvain:")
    print(fuzzy_membership_matrix_after)

    # Compute modularity after Louvain
    modularity_after = compute_modularity(nx.adjacency_matrix(G).toarray(), fuzzy_membership_matrix_after, num_communities_after)
    print(f"\nModularity after Louvain: {modularity_after:.3f}")

    return fuzzy_membership_matrix_after

# Generate a random graph
n = 13
G = generate_network(n)

# Generate fuzzy membership matrix
num_communities = 3
fuzzy_membership_matrix = generate_fuzzy_membership_matrix(n, num_communities)

# Create a network from the adjacency matrix and fuzzy membership matrix
updated_fuzzy_membership_matrix = create_network(G, fuzzy_membership_matrix)
