import numpy as np
import heapq
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- A* Logic ---
def astar(start, goal, H, W, grid):
    def h(a, b): return abs(a[0]-b[0]) + abs(a[1]-b[1])
    open_set = [(h(start, goal), 0, start, None)]
    came_from = {}
    g_score = {start: 0}
    
    while open_set:
        _, dist, cur, parent = heapq.heappop(open_set)
        if cur in came_from and g_score.get(cur, float('inf')) < dist: continue
        came_from[cur] = parent
        
        if cur == goal:
            path = []
            while cur:
                path.append(cur)
                cur = came_from[cur]
            return path[::-1]
            
        for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]:
            nr, nc = cur[0]+dr, cur[1]+dc
            if 0<=nr<H and 0<=nc<W and grid[nr, nc] == 0:
                new_g = dist + 1
                if new_g < g_score.get((nr, nc), float('inf')):
                    g_score[(nr, nc)] = new_g
                    heapq.heappush(open_set, (new_g + h((nr, nc), goal), new_g, (nr, nc), cur))
    return None

# --- Setup ---
H, W = 12, 12
grid = np.zeros((H, W), dtype=int)
random.seed(7)
# Add random walls
for r in range(H):
    for c in range(W):
        if random.random() < 0.1: grid[r, c] = 1

# Create Dirt
dirty_cells = set()
for r in range(H):
    for c in range(W):
        if grid[r, c] == 0 and random.random() < 0.3:
            dirty_cells.add((r, c))

agents = [
    {'id': 0, 'pos': (0, 0), 'path': [], 'color': '#FF5722'}, # Deep Orange
    {'id': 1, 'pos': (H-1, W-1), 'path': [], 'color': '#00BCD4'} # Cyan
]

# --- Simulation ---
history_frames = []
max_steps = 400

for step in range(max_steps):
    if not dirty_cells: break

    for i, ag in enumerate(agents):
        # Logic: Agent 0 cleans left side, Agent 1 cleans right side (optimization)
        my_region = [(r,c) for r,c in dirty_cells if (c < W//2 if i==0 else c >= W//2)]
        # If my side is clean, help the other side
        targets = my_region if my_region else list(dirty_cells)
        
        if not ag['path'] and targets:
            target = min(targets, key=lambda x: abs(x[0]-ag['pos'][0]) + abs(x[1]-ag['pos'][1]))
            path = astar(ag['pos'], target, H, W, grid)
            if path: ag['path'] = path
        
        if ag['path']:
            if len(ag['path']) > 1:
                ag['path'].pop(0)
                ag['pos'] = ag['path'][0]
            elif len(ag['path']) == 1:
                 ag['pos'] = ag['path'][0] # Reached end

        if ag['pos'] in dirty_cells:
            dirty_cells.remove(ag['pos'])

    history_frames.append({'agents': [dict(a) for a in agents], 'dirt': list(dirty_cells)})

# --- Render ---
print(f"Cleaning finished. Frames: {len(history_frames)}")
fig, ax = plt.subplots(figsize=(6, 6))
fig.patch.set_facecolor('#212121') 

def update(frame_idx):
    ax.clear(); ax.axis('off')
    data = history_frames[frame_idx]
    
    floor = np.zeros((H, W, 3)) + 0.2
    floor[grid == 1] = [0.1, 0.1, 0.1] 
    ax.imshow(floor, extent=[0, W, H, 0])
    
    if data['dirt']:
        dy, dx = zip(*data['dirt'])
        ax.scatter(dx, dy, c='#795548', s=120, marker='o', alpha=0.8, edgecolors='none')

    for ag in data['agents']:
        ax.scatter(ag['pos'][1], ag['pos'][0], c=ag['color'], s=220, edgecolors='white', linewidth=2)

    ax.set_title(f"CLEANING CREW | Dirt Left: {len(data['dirt'])}", fontsize=10, color='white')
    ax.set_xlim(-0.5, W-0.5); ax.set_ylim(H-0.5, -0.5)

anim = FuncAnimation(fig, update, frames=len(history_frames), interval=50, repeat=False)
plt.show()