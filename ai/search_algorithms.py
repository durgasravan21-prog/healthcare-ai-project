"""Classical searches on a fictional workflow graph."""
from collections import deque
import heapq
from time import perf_counter

GRAPH = {"Start":["Registration","Emergency Desk"],"Registration":["Basic Review"],
"Emergency Desk":["Basic Review","Priority Review"],"Basic Review":["Specialist Review","Information Desk"],
"Priority Review":["Specialist Review"],"Information Desk":["Specialist Review"],
"Specialist Review":["Follow-up Planning"],"Follow-up Planning":["End"],"End":[]}
COST = {(a,b):1 for a,ns in GRAPH.items() for b in ns}
H = {"Start":5,"Registration":4,"Emergency Desk":4,"Basic Review":3,
"Priority Review":3,"Information Desk":3,"Specialist Review":2,"Follow-up Planning":1,"End":0}


def result(path, seen, start, cost=None):
    return {"path":path,"explored":seen,"steps":len(path)-1 if path else None,
            "cost":cost,"elapsed_ms":round((perf_counter()-start)*1000,3)}


def bfs(graph,start,goal):
    t,q,seen,visited=perf_counter(),deque([(start,[start])]),{start},[]
    while q:
        node,path=q.popleft(); visited.append(node)
        if node==goal: return result(path,visited,t,len(path)-1)
        for nxt in graph.get(node,[]):
            if nxt not in seen: seen.add(nxt); q.append((nxt,path+[nxt]))
    return result(None,visited,t)


def dfs(graph,start,goal):
    t,stack,seen,visited=perf_counter(),[(start,[start])],set(),[]
    while stack:
        node,path=stack.pop()
        if node in seen: continue
        seen.add(node); visited.append(node)
        if node==goal: return result(path,visited,t,len(path)-1)
        stack.extend((n,path+[n]) for n in reversed(graph.get(node,[])) if n not in seen)
    return result(None,visited,t)


def best_first(graph,start,goal):
    t,counter,seen,visited=perf_counter(),0,set(),[]
    q=[(H[start],0,start,[start])]
    while q:
        _,_,node,path=heapq.heappop(q)
        if node in seen: continue
        seen.add(node); visited.append(node)
        if node==goal:
            return result(path,visited,t,sum(COST[a,b] for a,b in zip(path,path[1:])))
        for nxt in graph.get(node,[]):
            if nxt not in seen:
                counter+=1; heapq.heappush(q,(H[nxt],counter,nxt,path+[nxt]))
    return result(None,visited,t)


def astar(graph,start,goal):
    t,counter,best,visited=perf_counter(),0,{start:0},[]
    q=[(H[start],0,0,start,[start])]
    while q:
        _,cost,_,node,path=heapq.heappop(q)
        if cost!=best.get(node): continue
        visited.append(node)
        if node==goal: return result(path,visited,t,cost)
        for nxt in graph.get(node,[]):
            new=cost+COST.get((node,nxt),1)
            if new<best.get(nxt,float("inf")):
                best[nxt]=new; counter+=1
                heapq.heappush(q,(new+H[nxt],new,counter,nxt,path+[nxt]))
    return result(None,visited,t)


def hill_climbing(graph,start,goal):
    t,node,path,visited=perf_counter(),start,[start],[start]
    while node!=goal:
        options=graph.get(node,[])
        if not options: return result(None,visited,t)
        nxt=min(options,key=lambda x:H[x])
        if H[nxt]>=H[node]: return result(None,visited,t)
        node=nxt; path.append(node); visited.append(node)
    return result(path,visited,t,len(path)-1)


def run_all(start="Start",goal="End"):
    if start not in GRAPH or goal not in GRAPH: raise ValueError("Invalid start or goal.")
    methods=[("BFS",bfs),("DFS",dfs),("Greedy Best-First",best_first),
             ("Hill Climbing",hill_climbing),("A*",astar)]
    return {name:fn(GRAPH,start,goal) for name,fn in methods}
