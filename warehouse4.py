import numpy as np
import random
import heapq
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- 1. Logic Core: A* Pathfinding ---
def astar(start, goal, H, W, grid):
    def h(a, b): return abs(a[0]-b[0]) + abs(a[1]-b[1])
    # Priority Queue: (f_score, g_score, current_node, parent_node)
    open_set = [(h(start, goal), 0, start, None)]
    came_from = {}
    g_score = {start: 0}
    
    while open_set:
        _, dist, cur, parent = heapq.heappop(open_set)
        
        # Skip if we found a shorter way to this node already
        if cur in came_from and g_score.get(cur, float('inf')) < dist:
            continue
            
        came_from[cur] = parent
        
        if cur == goal:
            path = []
            while cur:
                path.append(cur)
                cur = came_from[cur]
            return path[::-1] # Return reversed path
            
        for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
            nr, nc = cur[0]+dr, cur[1]+dc
            if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] == 0:
                new_g = dist + 1
                if new_g < g_score.get((nr, nc), float('inf')):
                    g_score[(nr, nc)] = new_g
                    heapq.heappush(open_set, (new_g + h((nr, nc), goal), new_g, (nr, nc), cur))
    return None

# --- 2. Setup World ---
H, W = 14, 14
grid = np.zeros((H, W), dtype=int)
random.seed(11)
for r in range(H):
    for c in range(W):
        if random.random() < 0.05: grid[r, c] = 1

# Agents: [id, start_pos, color]
agents = [
    {'id': 0, 'pos': (0, 0), 'path': [], 'task': None, 'color': '#FFC107'}, # Amber
    {'id': 1, 'pos': (H-1, 0), 'path': [], 'task': None, 'color': '#03A9F4'}, # Light Blue
    {'id': 2, 'pos': (0, W-1), 'path': [], 'task': None, 'color': '#8BC34A'}  # Light Green
]

items = set((random.randint(1, H-2), random.randint(1, W-2)) for _ in range(8))

# --- 3. Simulation Loop ---
frames = []
max_steps = 300

for _ in range(max_steps):
    # Stop condition: No items left AND all agents have stopped moving
    if not items and all(not a['path'] for a in agents):
        break
    
    # 1. Assign Tasks
    for a in agents:
        if not a['task'] and items:
            # Find closest item
            target = min(items, key=lambda x: abs(x[0]-a['pos'][0]) + abs(x[1]-a['pos'][1]))
            a['task'] = target
            path = astar(a['pos'], target, H, W, grid)
            if path:
                a['path'] = path
            else:
                # If path fails (blocked), reset task
                a['task'] = None

    # 2. Move Agents
    for a in agents:
        if a['path']:
            if len(a['path']) > 1:
                a['path'].pop(0)       # Remove current position
                a['pos'] = a['path'][0] # Move to next
        
        # 3. Check Pickup
        if a['task'] and a['pos'] == a['task']:
            if a['pos'] in items:
                items.remove(a['pos'])
            a['task'] = None
            a['path'] = [] 

    frames.append({'agents': [a['pos'] for a in agents], 'items': list(items)})

# --- 4. Visualization ---
print(f"Simulation finished. Total frames: {len(frames)}")
fig, ax = plt.subplots(figsize=(6, 6))
fig.patch.set_facecolor('#263238') # Industrial Dark Grey

def render(frame):
    ax.clear()
    ax.axis('off')
    data = frames[frame]
    
    # Draw Floor
    canvas = np.zeros((H, W, 3)) + 0.15
    canvas[grid == 1] = [0.05, 0.05, 0.05]
    ax.imshow(canvas, extent=[0, W, H, 0])

    # Draw Items (Crates)
    if data['items']:
        iy, ix = zip(*data['items'])
        ax.scatter(ix, iy, c='#FF5722', s=150, marker='s', edgecolors='black', linewidth=2, label='Crate')

    # Draw Agents
    for i, pos in enumerate(data['agents']):
        ax.scatter(pos[1], pos[0], c=agents[i]['color'], s=200, edgecolors='white', linewidth=2, zorder=10)
        ax.text(pos[1], pos[0], str(i), ha='center', va='center', color='black', fontweight='bold', fontsize=8)

    status = "WORKING" if data['items'] else "COMPLETE"
    ax.set_title(f"WAREHOUSE | Items Left: {len(data['items'])} | {status}", color='white')
    ax.set_xlim(-0.5, W-0.5)
    ax.set_ylim(H-0.5, -0.5)

anim = FuncAnimation(fig, render, frames=len(frames), interval=100, repeat=False)
plt.show()