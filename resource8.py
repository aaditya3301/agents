import numpy as np
import random
import heapq
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def astar(start, goal, H, W, grid):
    def h(a, b): return abs(a[0]-b[0]) + abs(a[1]-b[1])
    open_set = [(h(start, goal), 0, start, None)]; came = {}; g = {start: 0}
    while open_set:
        _, dist, cur, parent = heapq.heappop(open_set)
        if cur == goal:
            path = []; 
            while cur: path.append(cur); cur = came.get(cur)
            return path[::-1]
        if cur in came and g.get(cur,float('inf'))<dist: continue
        came[cur] = parent
        for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]:
            nr, nc = cur[0]+dr, cur[1]+dc
            if 0<=nr<H and 0<=nc<W and grid[nr,nc]==0:
                if dist+1 < g.get((nr,nc),float('inf')):
                    g[(nr,nc)]=dist+1; heapq.heappush(open_set, (dist+1+h((nr,nc),goal), dist+1, (nr,nc), cur))
    return None

H, W = 14, 14
grid = np.zeros((H, W), dtype=int)
random.seed(13)
for r in range(H):
    for c in range(W):
        if random.random() < 0.07: grid[r, c] = 1

agents = [{'pos':p, 'path':[], 'task':None} for p in [(0,0), (H-1,W-1), (H-1,0)]]
resources = [(random.randint(1,H-2), random.randint(1,W-2)) for _ in range(12)]
queue = list(resources)

frames = []
max_steps = 500

for _ in range(max_steps):
    if not queue and all(not a['task'] for a in agents): break
    
    for a in agents:
        if not a['task'] and queue:
            a['task'] = queue.pop(0)
            path = astar(a['pos'], a['task'], H, W, grid)
            if path: a['path'] = path
        
        if a['path']:
            if len(a['path']) > 1: a['path'].pop(0); a['pos'] = a['path'][0]
        
        if a['pos'] == a['task']:
            a['task'] = None

    active_res = list(queue) + [a['task'] for a in agents if a['task']]
    frames.append({'agents': [a['pos'] for a in agents], 'res': active_res})

print(f"Simulation complete. Frames: {len(frames)}")
fig, ax = plt.subplots(figsize=(6, 6))
fig.patch.set_facecolor('#3E2723')

def render(frame):
    ax.clear(); ax.axis('off')
    data = frames[frame]
    
    canvas = np.zeros((H, W, 3)) + 0.2
    canvas[grid==1] = [0.1, 0.05, 0.05]
    ax.imshow(canvas, extent=[0, W, H, 0])

    for r in data['res']:
        ax.scatter(r[1], r[0], c='#FFD700', marker='h', s=150, edgecolors='orange')
    
    for pos in data['agents']:
        ax.scatter(pos[1], pos[0], c='#CDDC39', s=180, edgecolors='black')
        
    ax.set_title(f"MINING BOTS | Remaining: {len(data['res'])}", color='white')
    ax.set_xlim(-0.5, W-0.5); ax.set_ylim(H-0.5, -0.5)

anim = FuncAnimation(fig, render, frames=len(frames), interval=100, repeat=False)
plt.show()