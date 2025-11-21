import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

# --- Helper Functions ---
DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)]

def in_bounds(pos, H, W):
    return 0 <= pos[0] < H and 0 <= pos[1] < W

def neighbors(pos, H, W, grid):
    for dr, dc in DIRS:
        nr, nc = pos[0] + dr, pos[1] + dc
        if in_bounds((nr, nc), H, W) and grid[nr, nc] == 0:
            yield (nr, nc)

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

# --- Classes ---
class GridWorld:
    def __init__(self, H, W):
        self.H = H
        self.W = W
        self.grid = np.zeros((H, W), dtype=int)

    def add_random_walls(self, prob=0.18, seed=0):
        random.seed(seed)
        for r in range(self.H):
            for c in range(self.W):
                if random.random() < prob:
                    self.grid[r, c] = 1

class Agent:
    def __init__(self, id, start):
        self.id = id
        self.pos = start
        self.path = [start]
        self.history = [start]
        self.task = None

    def set_path(self, p):
        if p: self.path = p

    def step(self):
        if len(self.path) > 1:
            self.path = self.path[1:]
            self.pos = self.path[0]
            self.history.append(self.pos)

# --- Main Execution ---
H, W = 15, 15
world = GridWorld(H, W)
world.add_random_walls(0.18, seed=42)
starts = [(1, 1), (H-2, W-2)]
for s in starts: world.grid[s] = 0

# Place keys
keys = set()
while len(keys) < 8:
    p = (random.randrange(H), random.randrange(W))
    if world.grid[p] == 0 and p not in starts:
        keys.add(p)

agents = [Agent(0, starts[0]), Agent(1, starts[1])]
shared = set(keys)
t = 0
max_steps = 500

while shared and t < max_steps:
    # Claim nearest key
    claims = {}
    for a in agents:
        if not shared: break
        nearest = min(shared, key=lambda x: abs(x[0]-a.pos[0]) + abs(x[1]-a.pos[1]))
        claims[a.id] = nearest
    
    # Resolve duplicates
    assigned = set()
    for a in agents:
        if a.id in claims:
            tgt = claims[a.id]
            if tgt in assigned:
                choices = [r for r in shared if r not in assigned]
                if choices:
                    tgt = min(choices, key=lambda x: abs(x[0]-a.pos[0]) + abs(x[1]-a.pos[1]))
            a.task = tgt
            assigned.add(tgt)
    
    # Move
    for a in agents:
        if a.task:
            p = bfs(a.pos, {a.task}, H, W, world.grid)
            if p: a.set_path(p)
        a.step()
        if a.pos in shared:
            shared.remove(a.pos)
    t += 1

# Visualization
canvas = np.ones((H, W, 3)) * 0.95
canvas[world.grid == 1] = np.array([0.1, 0.1, 0.1])
for k in keys: canvas[k] = np.array([0.8, 0.4, 0.0])
for a in agents:
    for (r, c) in a.history:
        canvas[r, c] = np.array([0.6, 0.6, 0.95]) 
for a in agents:
    r, c = a.pos
    canvas[r, c] = np.array([0.2 + 0.3 * (a.id % 3), 0.2 + 0.4 * ((a.id + 1) % 3), 0.3 + 0.3 * ((a.id + 2) % 3)])

print("Keys collected:", 8 - len(shared))
plt.figure(figsize=(6, 6))
plt.imshow(canvas)
plt.title('Dual Maze Navigators Final')
plt.axis('off')
plt.show()