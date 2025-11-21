import numpy as np
import random
import heapq
import matplotlib.pyplot as plt

DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)]

def in_bounds(p, H, W): return 0 <= p[0] < H and 0 <= p[1] < W

def neighbors(p, H, W, grid):
    for dr, dc in DIRS:
        nr, nc = p[0] + dr, p[1] + dc
        if in_bounds((nr, nc), H, W) and grid[nr, nc] == 0: yield (nr, nc)

def manhattan(a, b): return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(start, goal, H, W, grid):
    open_set = []
    heapq.heappush(open_set, (manhattan(start, goal), 0, start, None))
    came = {}
    g = {start: 0}
    while open_set:
        f, dist, cur, parent = heapq.heappop(open_set)
        if cur in came: continue
        came[cur] = parent
        if cur == goal:
            path = []
            p = cur
            while p:
                path.append(p)
                p = came[p] if p in came else None
            return path[::-1]
        for nb in neighbors(cur, H, W, grid):
            nd = dist + 1
            if nb not in g or nd < g[nb]:
                g[nb] = nd
                heapq.heappush(open_set, (nd + manhattan(nb, goal), nd, nb, cur))
    return None

class GridWorld:
    def __init__(self, H, W):
        self.H, self.W = H, W
        self.grid = np.zeros((H, W), dtype=int)
    def add_walls(self, prob=0.12, seed=0):
        random.seed(seed)
        for r in range(self.H):
            for c in range(self.W):
                if random.random() < prob: self.grid[r, c] = 1

class Agent:
    def __init__(self, id, start):
        self.id = id
        self.pos = start
        self.history = [start]
    def set_history(self, h):
        self.history = h
        self.pos = h[-1]

H, W = 12, 12
world = GridWorld(H, W)
world.add_walls(prob=0.12, seed=5)

start1, start2 = (1, 1), (1, W-2)
goal1, goal2 = (H-2, 1), (H-2, W-2)

# Plan independent paths
p1 = astar(start1, goal1, H, W, world.grid) or [start1]
p2 = astar(start2, goal2, H, W, world.grid) or [start2]
hist = {0: list(p1), 1: list(p2)}

# Collision Resolution
t = 0
while t < 200:
    pos0 = hist[0][t] if t < len(hist[0]) else hist[0][-1]
    pos1 = hist[1][t] if t < len(hist[1]) else hist[1][-1]
    
    if pos0 == pos1:
        # Agent 1 waits one step
        hist[1].insert(t, hist[1][t-1] if t > 0 else hist[1][0])
    t += 1

agents = [Agent(0, start1), Agent(1, start2)]
for i, a in enumerate(agents): a.set_history(hist[i])

# Visualize
canvas = np.ones((H, W, 3)) * 0.95
canvas[world.grid == 1] = np.array([0.1, 0.1, 0.1])
for a in agents:
    for p in a.history: canvas[p] = np.array([0.8, 0.8, 0.6])
    r, c = a.pos
    canvas[r, c] = np.array([0.2 + 0.3 * (a.id % 3), 0.2 + 0.4 * ((a.id + 1) % 3), 0.3 + 0.3 * ((a.id + 2) % 3)])

plt.figure(figsize=(6, 6))
plt.imshow(canvas)
plt.title('Cooperative Path Planners Final')
plt.axis('off')
plt.show()