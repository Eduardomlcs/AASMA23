import heapq

# Define the possible movements (up, down, left, right)

MOVEMENTS = [(0, -1), (0, 1), (-1, 0), (1, 0)]

def convert_path(path):
    new_path = []
    for step in path:
        if step[1] % 2 != 0:
            new_path = new_path + [(step[0]-1,step[1]//2)]
    return new_path

def heuristic(p1, p2):
    # Manhattan distance heuristic
    return abs(p2[0] - p1[0]) + abs(p2[1] - p1[1])

def get_neighbors(node, grid):
    # Returns the valid neighbors of a given node
    neighbors = []
    for movement in MOVEMENTS:
        new_row = node[0] + movement[0]
        new_col = node[1] + movement[1]
        if 1 <= new_row < len(grid) - 1 and 0 <= new_col < len(grid[0]) and grid[new_row][new_col] != b'|' and grid[new_row][new_col] != b'C':
            neighbors.append((new_row, new_col))
    return neighbors

def a_star_search(start, goal, grid):
    # Initialize the open and closed lists
    open_list = []
    closed_set = set()

    # Create a dictionary to store the parent node of each visited node
    parents = {}

    # Create a dictionary to store the cost of reaching each node
    g_scores = {start: 0}

    # Add the start node to the open list with its estimated cost
    heapq.heappush(open_list, (heuristic(start, goal), start))

    while open_list:
        # Pop the node with the lowest cost from the open list
        current_node = heapq.heappop(open_list)[1]

        if current_node == goal:
            # Reached the goal, reconstruct the path
            path = []
            while current_node in parents:
                path.append(current_node)
                current_node = parents[current_node]
            path.append(start)
            path.reverse()
            return convert_path(path)
        closed_set.add(current_node)

        # Explore the neighbors of the current node
        neighbors = get_neighbors(current_node, grid)
        for neighbor in neighbors:
            if neighbor in closed_set:
                continue

            # Calculate the tentative g-score for the neighbor
            neighbor_g_score = g_scores[current_node] + 1

            if neighbor not in [item[1] for item in open_list] or neighbor_g_score < g_scores[neighbor]:
                # Update the parent and g-score of the neighbor
                parents[neighbor] = current_node
                g_scores[neighbor] = neighbor_g_score
                f_score = neighbor_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_list, (f_score, neighbor))

    # No path found
    return None

