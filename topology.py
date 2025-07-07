import torch

#### Topology ####
#### Take number of nodes, return a List[List[Number]] -> Adjacent list of a connected graph
#### Note that node/client is indexed from 0.
def adj_list_gen(num_nodes = 20, topo = "ring"):
    """
    Params:
        num_nodes: number of node in the graph
        topo: Topology type, must be in ["ring", "fully", "star", "random"]
    Return:
        List[List[Number]] -> Adjacent list of a graph that is always connected
    """
    if num_nodes == 1:
        return [[]]
    if topo == "fully":
        return fully_connected_graph(num_nodes)
    elif topo == "ring":
        return ring_graph(num_nodes)
    elif topo == "star":
        return star_graph(num_nodes)
    elif topo == "random":
        return random_graph(num_nodes)
    elif topo == "dis":
        return disconnected_graph(num_nodes)
    else:
        raise ValueError("In adj_list_gen(): topo is not approriate")


def fully_connected_graph(num_nodes = 20):
    return [[j for j in range(num_nodes) if j != i] for i in range(num_nodes)]


def ring_graph(num_nodes = 20):
    return [[(i+1)%num_nodes,(i+num_nodes-1)%num_nodes] for i in range(num_nodes)]


def star_graph(num_nodes = 20):
    ### Center node is 0
    ls = [[i for i in range(1,num_nodes)]]
    for i in range(1, num_nodes):
        ls.append([0])
    return ls


def random_graph(num_nodes = 20, prob = 0.5):
    # Simple Random scheme:
    # Let i from 1 to num_nodes-1, and j from 0 to i-1, the there is {prod} probability to
    # make i, j connected. To gurantee the graph is connected, make sure at least one edge is
    # formed during iteration on i.
    # Drawback: not really random.

    ls = [[] for i in range(num_nodes)]
    for i in range(1, num_nodes):
        connect = False
        for j in range(0, i):
            if torch.rand(1).item() <= prob:
                ls[i].append(j)
                ls[j].append(i)
                connect = True

        if not connect:
            j = torch.randint(i, (1,)).item()
            ls[i].append(j)
            ls[j].append(i)

    return ls


def disconnected_graph(num_nodes = 20):
    return [[] for i in range(num_nodes)]


def adj_list_to_matrix(adj_list):
    # Convert Adj list to 2D torch tensor
    N = len(adj_list)
    matrix = torch.zeros(N, N)
    for i in range(N):
        for j in adj_list[i]:
            matrix[i][j] = 1
            matrix[j][i] = 1
    return matrix
