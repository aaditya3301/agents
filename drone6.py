import numpy as np
import random
import heapq
import matplotlib.pyplot as plt

# --- Constants & Helpers ---
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

# --- Classes ---
class GridWorld:
    def __init__(self, H, W):
        self.H, self.W = H, W
        self.grid = np.zeros((H, W), dtype=int)
    def add_walls(self, prob=0.04, seed=0):
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
        self.task = None
    def step(self):
        if len(self.path) > 1:
            self.path = self.path[1:]
            self.pos = self.path[0]
            self.history.append(self.pos)

# --- Main Execution ---
H, W = 16, 16
world = GridWorld(H, W)
world.add_walls(prob=0.04, seed=99)

drones = [Agent(0, (1, 1)), Agent(1, (H-2, W-2))]
packages = [(random.randint(1, H-2), random.randint(1, W-2)) for _ in range(6)]
remaining = set(packages)

t = 0
while remaining and t < 400:
    t += 1
    # Greedy assignment
    for d in drones:
        if d.task is None and remaining:
            d.task = min(remaining, key=lambda x: manhattan(d.pos, x))
            
    for d in drones:
        if d.task:
            p = astar(d.pos, d.task, H, W, world.grid)
            if p: d.path = p
        d.step()
        if d.task and d.pos == d.task:
            remaining.discard(d.task)
            d.task = None

print(f"Delivered {len(packages)-len(remaining)}/{len(packages)} packages.")

# Visualization
canvas = np.ones((H, W, 3)) * 0.95
canvas[world.grid == 1] = np.array([0.1, 0.1, 0.1])
for pk in packages: canvas[pk] = np.array([0.8, 0.6, 0.2])
for d in drones:
    for p in d.history: canvas[p] = np.array([0.7, 0.9, 0.9])
    r, c = d.pos
    canvas[r, c] = np.array([0.2 + 0.3 * (d.id % 3), 0.2 + 0.4 * ((d.id + 1) % 3), 0.3 + 0.3 * ((d.id + 2) % 3)])

plt.figure(figsize=(6, 6))
plt.imshow(canvas)
plt.title('Dual Drone Delivery Final')
plt.axis('off')
plt.show()