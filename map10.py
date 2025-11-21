import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)]

def in_bounds(p, H, W): return 0 <= p[0] < H and 0 <= p[1] < W

def neighbors(p, H, W, grid):
    for dr, dc in DIRS:
        nr, nc = p[0] + dr, p[1] + dc
        if in_bounds((nr, nc), H, W) and grid[nr, nc] == 0: yield (nr, nc)

def bfs(start, goals, H, W, grid):
    q = deque([start])
    parent = {start: None}
    while q:
        cur = q.popleft()
        if cur in goals:
            path = []
            p = cur
            while p:
                path.append(p)
                p = parent[p]
            return path[::-1]
        for nb in neighbors(cur, H, W, grid):
            if nb not in parent:
                parent[nb] = cur
                q.append(nb)
    return None

class GridWorld:
    def __init__(self, H, W):
        self.H, self.W = H, W
        self.grid = np.zeros((H, W), dtype=int)
    def add_walls(self, prob=0.06, seed=0):
        random.seed(seed)
        for r in range(self.H):
            for c in range(self.W):
                if random.random() < prob: self.grid[r, c] = 1

class Agent:
    def __init__(self, id, start):
        self.id = id
        self.pos = start
        self.path = [start]
        self.history = [start]
    def step(self):
        if len(self.path) > 1:
            self.path = self.path[1:]
            self.pos = self.path[0]
            self.history.append(self.pos)

# --- Main Execution ---
H, W = 18, 18
world = GridWorld(H, W)
world.add_walls(prob=0.06, seed=31)

agents = [Agent(0, (1, 1)), Agent(1, (1, W-2)), Agent(2, (H-2, 1))]
unexplored = {(r, c) for r in range(H) for c in range(W) if world.grid[r, c] == 0}

# Partition map into columns for each agent
k = len(agents)
cols_per = W // k
regions = []
for i in range(k):
    cols = range(i * cols_per, (i + 1) * cols_per if i < k - 1 else W)
    regions.append({(r, c) for r in range(H) for c in cols if (r, c) in unexplored})

t = 0
while any(regions) and t < 800:
    t += 1
    for i, a in enumerate(agents):
        reg = regions[i]
        if reg:
            target = min(reg, key=lambda x: abs(x[0] - a.pos[0]) + abs(x[1] - a.pos[1]))
            p = bfs(a.pos, {target}, H, W, world.grid)
            if p: a.path = p
            
        a.step()
        
        # Mark explored
        if a.pos in reg:
            reg.discard(a.pos)

print("Exploration finished approx. Remaining region sizes:", [len(r) for r in regions])

# Visualization
canvas = np.ones((H, W, 3)) * 0.95
canvas[world.grid == 1] = np.array([0.1, 0.1, 0.1])
for a in agents:
    for h in a.history: canvas[h] = np.array([0.8, 0.9, 0.9])
    r, c = a.pos
    canvas[r, c] = np.array([0.2 + 0.3 * (a.id % 3), 0.2 + 0.4 * ((a.id + 1) % 3), 0.3 + 0.3 * ((a.id + 2) % 3)])

plt.figure(figsize=(6, 6))
plt.imshow(canvas)
plt.title('Map Exploration Final')
plt.axis('off')
plt.show()