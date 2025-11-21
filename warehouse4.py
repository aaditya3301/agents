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
    def add_walls(self, prob=0.05, seed=0):
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

H, W = 14, 14
world = GridWorld(H, W)
world.add_walls(prob=0.05, seed=11)
starts = [(0, 0), (H-1, 0), (0, W-1)]
agents = [Agent(i, s) for i, s in enumerate(starts)]
items = [(random.randint(1, H-2), random.randint(1, W-2)) for _ in range(8)]
remaining = set(items)

t = 0
while remaining and t < 400:
    remaining_list = list(remaining)
    for a in agents:
        if a.task is None and remaining_list:
            a.task = min(remaining_list, key=lambda x: manhattan(a.pos, x))
    
    for a in agents:
        if a.task:
            p = astar(a.pos, a.task, H, W, world.grid)
            if p: a.path = p
        a.step()
        if a.pos == a.task:
            remaining.discard(a.task)
            a.task = None
    t += 1

canvas = np.ones((H, W, 3)) * 0.95
canvas[world.grid == 1] = np.array([0.1, 0.1, 0.1])
for it in items: canvas[it] = np.array([0.8, 0.4, 0.0])
for a in agents:
    for p in a.history: canvas[p] = np.array([0.7, 0.9, 0.9])
    r, c = a.pos
    canvas[r, c] = np.array([0.2 + 0.3 * (a.id % 3), 0.2 + 0.4 * ((a.id + 1) % 3), 0.3 + 0.3 * ((a.id + 2) % 3)])

plt.figure(figsize=(6, 6))
plt.imshow(canvas)
plt.title('Warehouse Pickup Final')
plt.axis('off')
plt.show()