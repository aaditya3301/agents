import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- 1. Logic Core ---
DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)]

def bfs_path(start, goal, H, W, grid):
    q = deque([start])
    parent = {start: None}
    visited = {start}
    while q:
        cur = q.popleft()
        if cur == goal:
            path = []
            while cur:
                path.append(cur)
                cur = parent[cur]
            return path[::-1]
        
        for dr, dc in DIRS:
            nr, nc = cur[0] + dr, cur[1] + dc
            if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] == 0 and (nr, nc) not in visited:
                visited.add((nr, nc))
                parent[(nr, nc)] = cur
                q.append((nr, nc))
    return None

class Agent:
    def __init__(self, id, start, color):
        self.id = id
        self.pos = start
        self.path = []
        self.color = color 
        self.history = [start]
        self.task = None

    def step(self):
        if len(self.path) > 1:
            self.path.pop(0)
            self.pos = self.path[0]
            self.history.append(self.pos)
            return True # Moved
        return False # Didn't move

# --- 2. Setup World ---
H, W = 15, 15
grid = np.zeros((H, W), dtype=int)
random.seed(42)

# Reduced wall density slightly to prevent unreachable keys
for r in range(H):
    for c in range(W):
        if random.random() < 0.15: grid[r, c] = 1

starts = [(1, 1), (H-2, W-2)]
for s in starts: grid[s] = 0

keys = set()
while len(keys) < 8:
    p = (random.randrange(H), random.randrange(W))
    if grid[p] == 0 and p not in starts: keys.add(p)

agents = [
    Agent(0, starts[0], (0.0, 1.0, 1.0)), # Cyan
    Agent(1, starts[1], (1.0, 0.0, 1.0))  # Magenta
]

# --- 3. Run Simulation (With "Stuck" Detection) ---
frames = []
max_steps = 300
shared_keys = set(keys)
status_msg = "RUNNING"

for step in range(max_steps):
    # 1. Check Success
    if not shared_keys:
        status_msg = "MISSION COMPLETE"
        # Add one final frame to show the win
        frames.append({'agents': [a.pos for a in agents], 'trails': [list(a.history) for a in agents], 'keys': [], 'status': status_msg})
        break

    # 2. Assign tasks
    for a in agents:
        if not a.path and shared_keys:
            targets = list(shared_keys)
            target = min(targets, key=lambda k: abs(k[0]-a.pos[0]) + abs(k[1]-a.pos[1]))
            a.task = target
            path = bfs_path(a.pos, target, H, W, grid)
            if path: a.path = path

    # 3. Step agents
    moved_any = False
    for a in agents:
        did_move = a.step()
        if did_move: moved_any = True
        
        if a.pos in shared_keys:
            shared_keys.remove(a.pos)
            a.task = None
            a.path = [] 

    # 4. Save frame
    frames.append({
        'agents': [a.pos for a in agents],
        'trails': [list(a.history) for a in agents],
        'keys': list(shared_keys),
        'status': "RUNNING"
    })

    # 5. Stop if stuck (Keys exist but nobody moved and nobody has a path)
    if not moved_any and all(not a.path for a in agents):
        status_msg = "STUCK (Unreachable Key)"
        frames[-1]['status'] = status_msg
        break

# --- 4. Render Animation ---
print(f"Simulation finished. Total Steps: {len(frames)}. Opening window...")

fig, ax = plt.subplots(figsize=(6, 6))
fig.patch.set_facecolor('#121212') 

def render_frame(frame_idx):
    ax.clear()
    ax.axis('off')
    data = frames[frame_idx]
    
    # Draw Grid
    canvas = np.zeros((H, W, 3)) + 0.1
    canvas[grid == 1] = [0.05, 0.05, 0.1]
    ax.imshow(canvas, extent=[0, W, H, 0]) 
    
    # Draw Trails
    for i, trail in enumerate(data['trails']):
        if not trail: continue
        y, x = zip(*trail[-20:]) # Trail length
        color = agents[i].color
        ax.scatter(x, y, c=[color], s=30, alpha=0.3, marker='s')

    # Draw Keys
    if data['keys']:
        ky, kx = zip(*data['keys'])
        ax.scatter(kx, ky, c='#FFD700', s=120, marker='*', edgecolors='white', linewidth=0.5)

    # Draw Agents
    for i, pos in enumerate(data['agents']):
        ax.scatter(pos[1], pos[0], c=[agents[i].color], s=150, edgecolors='white', linewidth=1.5, zorder=10)

    # Dynamic Title
    status = data['status']
    color = 'white'
    if status == "MISSION COMPLETE": color = '#00FF00' # Green
    elif "STUCK" in status: color = '#FF0000' # Red
        
    ax.set_title(f"STEP: {frame_idx} | {status}", color=color, fontsize=12, fontweight='bold')
    ax.set_ylim(H-0.5, -0.5)
    ax.set_xlim(-0.5, W-0.5)

# repeat=False prevents it from restarting loop
anim = FuncAnimation(fig, render_frame, frames=len(frames), interval=100, repeat=False)

plt.show()