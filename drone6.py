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
            path = []
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

H, W = 16, 16
grid = np.zeros((H, W), dtype=int)
random.seed(99)
for r in range(H): 
    for c in range(W): 
        if random.random() < 0.04: grid[r, c] = 1

drones = [
    {'pos':(1,1), 'path':[], 'task':None, 'color':'#2979FF'}, 
    {'pos':(H-2,W-2), 'path':[], 'task':None, 'color':'#FF4081'}
]
packages = set((random.randint(1, H-2), random.randint(1, W-2)) for _ in range(6))

frames = []
for _ in range(400):
    if not packages and all(not d['task'] for d in drones): break
    
    for d in drones:
        if not d['task'] and packages:
            target = min(packages, key=lambda x: abs(x[0]-d['pos'][0])+abs(x[1]-d['pos'][1]))
            d['task'] = target
            packages.remove(target)
            path = astar(d['pos'], target, H, W, grid)
            if path: d['path'] = path
        
        if d['path']:
            if len(d['path']) > 1: d['path'].pop(0); d['pos'] = d['path'][0]
        
        if d['pos'] == d['task']: d['task'] = None

    # Visualization data
    active_tasks = [d['task'] for d in drones if d['task']]
    frames.append({'drones': [d['pos'] for d in drones], 'packs': list(packages) + active_tasks})

print(f"Delivery complete. Frames: {len(frames)}")
fig, ax = plt.subplots(figsize=(6, 6))
fig.patch.set_facecolor('#0D47A1')

def render(frame):
    ax.clear(); ax.axis('off')
    data = frames[frame]
    
    canvas = np.zeros((H, W, 3)) + 0.1
    canvas[grid==1] = [0, 0, 0.2]
    ax.imshow(canvas, extent=[0, W, H, 0], alpha=0.6)
    
    for x in range(W): ax.axvline(x-0.5, color='white', alpha=0.1)
    for y in range(H): ax.axhline(y-0.5, color='white', alpha=0.1)

    if data['packs']:
        py, px = zip(*[p for p in data['packs'] if p])
        ax.scatter(px, py, c='#FFD740', marker='D', s=100, edgecolors='black')

    for i, pos in enumerate(data['drones']):
        ax.scatter(pos[1], pos[0], c=drones[i]['color'], s=200, marker='o', edgecolors='white', linewidth=2)
        ax.scatter(pos[1], pos[0], c='white', s=50, marker='+')

    ax.set_title(f"DRONE DELIVERY | Step: {frame}", color='white')
    ax.set_xlim(-0.5, W-0.5); ax.set_ylim(H-0.5, -0.5)

anim = FuncAnimation(fig, render, frames=len(frames), interval=100, repeat=False)
plt.show()