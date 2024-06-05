
# import nltk
# nltk.download('words')
from nltk.corpus import words
import Levenshtein

# Step 1

word_list = words.words()

# Prompt the user to enter the start and end words
start_word = input('Please enter a start word: ').strip().lower()
end_word = input('Please enter an end word, make sure it has the same length as the start word: ').strip().lower()
# Ensure the start and end words are the same length
while len(start_word) != len(end_word):
    end_word = input('Wrong length, please enter an end word again, make sure it has the same length as the start word: ').strip().lower()

# Filter the words to only include those of the same length as the start word
filtered_words = []

for w in word_list:
    if len(w) == len(start_word):
        filtered_words.append(w.lower())

# Create an adjacency list where each word is a key, and the value is a list of adjacent words
adjacency_list = {}

for w in filtered_words:
    adjacency_list[w] = []

for w in adjacency_list:
    for match in filtered_words:
        if w != match and Levenshtein.distance(w, match) == 1:
            adjacency_list[w].append(match)

class WordPathNode:
    """
    Represents a node in the word path search.
    
    Attributes:
        word (str): The word at this node.
        parent (WordPathNode): The parent node in the path.
    """
    def __init__(self, word, parent=None):
        self.word = word
        self.parent = parent
    
    def __eq__(self, other):
        return isinstance(other, WordPathNode) and self.word == other.word
    
    def __hash__(self):
        return hash(self.word)
    
    def get_path_to_root(self):
        """
        Constructs the path from the start word to this node.
        
        Returns:
            list: The path from the start word to this node.
        """
        path = []
        current_node = self
        while current_node:
            path.append(current_node.word)
            current_node = current_node.parent
        return path[::-1]
    
# Step 2: Implement Breadth-First Search (BFS)
def bfs(start_word, end_word, adjacency_list):
    """
    Performs BFS to find the shortest path from start_word to end_word.
    
    Args:
        start_word (str): The starting word.
        end_word (str): The target word.
        adjacency_list (dict): The adjacency list of words.
    
    Returns:
        list: The path from start_word to end_word, or None if no path is found.
    """
    start_node = WordPathNode(start_word)
    que = [start_node]
    seen = set([start_word])

    while que:
        cur = que.pop(0)

        if cur.word == end_word:
            return cur.get_path_to_root()
        
        for neighbor in adjacency_list[cur.word]:
            if neighbor not in seen:
                seen.add(neighbor)
                que.append(WordPathNode(neighbor, cur))
        
    return None

# Testing BFS
bfs_test = bfs(start_word, end_word, adjacency_list)
if bfs_test:
    print(f"Path found: {' -> '.join(bfs_test)}")
else:
    print("No path found.")

# Step 3: Implement Depth-First Search (DFS)
def dfs(start_word, end_word, adjacency_list):
    """
    Performs DFS to find a path from start_word to end_word.
    
    Args:
        start_word (str): The starting word.
        end_word (str): The target word.
        adjacency_list (dict): The adjacency list of words.
    
    Returns:
        list: The path from start_word to end_word, or None if no path is found.
    """
    start_node = WordPathNode(start_word)
    stack = [start_node]
    seen = set([start_word])

    while stack:
        cur = stack.pop()

        if cur.word == end_word:
            return cur.get_path_to_root()
        
        for neighbor in adjacency_list[cur.word]:
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(WordPathNode(neighbor, cur))

    return None

# Testing DFS
dfs_test = dfs(start_word, end_word, adjacency_list)
if dfs_test:
    print(f"DFS Path found: {' -> '.join(dfs_test)}")
else:
    print("No path found.")

# Step 4: Implement Iterative Deepening Depth-First Search (IDDFS)
def depth_limited_search(start_word, end_word, adjacency_list, depth_limit):
    """
    Performs depth-limited DFS to find a path from start_word to end_word.
    
    Args:
        start_word (str): The starting word.
        end_word (str): The target word.
        adjacency_list (dict): The adjacency list of words.
        depth_limit (int): The current depth limit.
    
    Returns:
        list: The path from start_word to end_word, or None if no path is found.
    """
    stack = [(WordPathNode(start_word), 0)]   # node, current depth
    seen = set()

    while stack:
        cur, depth = stack.pop()

        if cur.word == end_word:
            return cur.get_path_to_root()
        
        if depth < depth_limit:
            seen.add(cur.word)
            for neighbor in adjacency_list[cur.word]:
                if neighbor not in seen:
                    stack.append((WordPathNode(neighbor, cur), depth + 1))

    return None

def iterative_deepening(start_word, end_word, adjacency_list):
    """
    Performs IDDFS to find the shortest path from start_word to end_word.
    
    Args:
        start_word (str): The starting word.
        end_word (str): The target word.
        adjacency_list (dict): The adjacency list of words.
    
    Returns:
        tuple: The path from start_word to end_word and the depth at which it was found, or None if no path is found.
    """
    depth = 0
    while True:
        res = depth_limited_search(start_word, end_word, adjacency_list, depth)
        if res:
            return (res, depth)
        depth += 1

# Testing IDDFS
deepening = iterative_deepening(start_word, end_word, adjacency_list)
if deepening:
    print(f"Iterative Deepening Path found: {' -> '.join(deepening[0])}")
    print(f"At the depth of {deepening[1]}")
else:
    print("No path found.")

# Step 5: Implement A* Search
def a_star_search(start_word, end_word, adjacency_list):
    """
    Performs A* search to find the shortest path from start_word to end_word.
    
    Args:
        start_word (str): The starting word.
        end_word (str): The target word.
        adjacency_list (dict): The adjacency list of words.
    
    Returns:
        list: The path from start_word to end_word, or None if no path is found.
    """
    def hamming_distance(word1, word2):
        """
        Calculates the Hamming distance between two words.
        
        Args:
            word1 (str): The first word.
            word2 (str): The second word.
        
        Returns:
            int: The Hamming distance.
        """
        return sum(c1 != c2 for c1, c2 in zip(word1, word2))
    
    def reconstruct_path(came_from, current_node):
        """
        Reconstructs the path from the start word to the goal word.
        
        Args:
            came_from (dict): A dictionary mapping nodes to their parent nodes.
            current_node (WordPathNode): The current node.
        
        Returns:
            list: The path from the start word to the goal word.
        """
        path = [current_node.word]
        while current_node in came_from:
            current_node = came_from[current_node]
            path.append(current_node.word)
        return path[::-1]
    
    start_node = WordPathNode(start_word)
    
    open_set = [(0, start_node)]
    
    came_from = {}
    g_costs = {start_word: 0}
    f_costs = {start_word: hamming_distance(start_word, end_word)}
    
    while open_set:
        open_set.sort(key=lambda x: x[0])
        _, current_node = open_set.pop(0)
        
        if current_node.word == end_word:
            return reconstruct_path(came_from, current_node)
        
        for neighbor in adjacency_list[current_node.word]:
            tentative_g = g_costs[current_node.word] + 1  # Each edge has a cost of 1
            
            if neighbor not in g_costs or tentative_g < g_costs[neighbor]:
                neighbor_node = WordPathNode(neighbor)
                came_from[neighbor_node] = current_node
                g_costs[neighbor] = tentative_g
                f_costs[neighbor] = tentative_g + hamming_distance(neighbor, end_word)
                
                open_set.append((f_costs[neighbor], neighbor_node))
    
    return None

# Testing A* Search
path_astar = a_star_search(start_word, end_word, adjacency_list)
if path_astar:
    print(f"A* Search Path found: {' -> '.join(path_astar)}")
else:
    print("No path found using A* Search.")