import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

# --- Constants & Helpers ---
DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)]

def in_bounds(p, H, W):
    return 0 <= p[0] < H and 0 <= p[1] < W

def neighbors(p, H, W, grid):
    for dr, dc in DIRS:
        nr, nc = p[0] + dr, p[1] + dc
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
        self.H, self.W = H, W
        self.grid = np.zeros((H, W), dtype=int)

    def add_walls(self, prob=0.10, seed=0):
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
    
    def step(self):
        if len(self.path) > 1:
            self.path = self.path[1:]
            self.pos = self.path[0]
            self.history.append(self.pos)

# --- Main Execution ---
H, W = 15, 15
world = GridWorld(H, W)
world.add_walls(prob=0.10, seed=3)

starts = [(0, 0), (0, W-1), (H-1, 0)]
agents = [Agent(i, s) for i, s in enumerate(starts)]

victims = set()
while len(victims) < 6:
    p = (random.randrange(H), random.randrange(W))
    if world.grid[p] == 0 and p not in starts:
        victims.add(p)

remaining = set(victims)
t = 0

while remaining and t < 400:
    t += 1
    for a in agents:
        # Re-plan if idle or path finished
        if not a.path or len(a.path) <= 1:
            path = bfs(a.pos, remaining, H, W, world.grid)
            if path: a.path = path
        
        a.step()
        
        if a.pos in remaining:
            remaining.remove(a.pos)

print(f"Rescued {len(victims)-len(remaining)}/{len(victims)} victims.")

# Visualization
canvas = np.ones((H, W, 3)) * 0.95
canvas[world.grid == 1] = np.array([0.1, 0.1, 0.1])
for v in victims: canvas[v] = np.array([0.9, 0.4, 0.4])
for a in agents:
    for p in a.history: canvas[p] = np.array([0.7, 0.9, 0.9])
    r, c = a.pos
    canvas[r, c] = np.array([0.2 + 0.3 * (a.id % 3), 0.2 + 0.4 * ((a.id + 1) % 3), 0.3 + 0.3 * ((a.id + 2) % 3)])

plt.figure(figsize=(6, 6))
plt.imshow(canvas)
plt.title('Rescue Bot Squad Final')
plt.axis('off')
plt.show()