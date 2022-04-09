import math
import matplotlib.pyplot as plt
from matplotlib import collections  as mc
import random
import numpy as np

class Graph:
    """Class defining my graph object to use thru out my code

    vertices: list of vertices in the graph
    edges: list of tuples containing indexes for connect vertices
    addVert: function to add vertex to vertices and returns the index, takes the vertex as input
    addEdge: function to add edge between two vertices, takes two indexes as input
    draw: function to plot the vertices and edges, takes ax as input
    """
    def __init__(self):
        self.vertices = []
        self.edges = []

    def addVert(self, node):
        self.vertices.append(node)
        return len(self.vertices)

    def addEdge(self, node1, node2):
        self.edges.append((node1, node2))

    def draw(self, ax):
        px = [x for x, y in self.vertices]
        py = [y for x, y in self.vertices]
        ax.scatter(px, py, c = 'black', s = 4)
        lines = [(self.vertices[edge[0]], self.vertices[edge[1]]) for edge in self.edges]
        lc = mc.LineCollection(lines, colors='black', linewidths=1)
        ax.add_collection(lc)

def newVertex(randvex, nearvex, stepSize):
    """Creates new vertex after one is randomly generated

    @type randvex: the randomly generated vertex
    @type nearvex: the nearest vertex found to which to make edge
    @type stepSize: the inputted stepSize
    """
    dirn = np.array(randvex) - np.array(nearvex)
    length = np.linalg.norm(dirn)
    dirn = (dirn / length) * min (stepSize, length)

    newvex = (nearvex[0]+dirn[0], nearvex[1]+dirn[1])
    return newvex

def getRandom(cspace):
    """Creates random x,y coordinates in C Space

    @type cspace: the bounds of the c space
    """
    x = random.randint(cspace[0][0], cspace[0][1])
    y = random.randint(cspace[1][0], cspace[1][1])
    return x, y

def distance(a, b):
    """Gets euclidean distance between two coords

    @type a: vertex 1
    @type b: vertex 2
    """
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def getNearest(G, vert, obst, r):
    """Gets nearest vertex in graph to draw edge

    @type G: the graph so far
    @type vert: the randomly generated vertex
    @type obst: the obstacles in the cspace
    @type r: radius of the obstacles
    """
    newVert = None
    indx = None
    minDist = float("inf")

    for i, v in enumerate(G.vertices):
        (p, dirn, dist2) = getLine(v, vert)
        if thruObst(p, dirn, dist2, obst, r):
            continue
        dist = distance(v, vert)
        if dist < minDist:
            minDist = dist
            indx = i
            newVert = v
    return newVert, indx

def thruObst(p, dirn, dist, obst, r):
    """Checks if the edge goes thru an obstacle

    @type p: inputted probability
    @type dirn: calculated value
    @type dist: the calculates distance
    @type obst: the obstacles in the cspace
    @type r: radius of the obstacles
    """
    for obs in obst:
        if checkCollison(p, dirn, dist, obs, r):
            return True
    return False

def getLine(node1, node2):
    """Gets the edge between two vertices

    @type node1: vertex 1
    @type node2: vertex 2
    """
    p = np.array(node1)
    dirn = np.array(node2) - np.array(node1)
    dist = np.linalg.norm(dirn)
    dirn = dirn/dist
    return (p, dirn, dist)

def checkCollison(p, dirn, dist, o, r):
    """Checks if anything collides with obstacles

    @type p: inputted probability
    @type dirn: calculated value
    @type dist: the calculates distance
    @type o: the obstacles in the cspace
    @type r: radius of the obstacles
    """
    x = np.dot(dirn, dirn)
    y = 2 * np.dot(dirn, p - o)
    z = np.dot(p - o, p - o) - r * r

    d = y * y - 4 * x * z
    if d < 0:
        return False
        
    t1 = (-y + np.sqrt(d)) / (2 * x)
    t2 = (-y - np.sqrt(d)) / (2 * x)

    if (t1 < 0 and t2 < 0) or (t1 > dist and t2 > dist):
        return False

    return True

def inObst(vex, obstacles, radius):
    """Checks if vertex is in any of the obstacles

    @type vex: vertex to check
    @type obstacles: the list of obstacles
    @type radius: radius of the obstacles
    """
    for obs in obstacles:
        if distance(obs, vex) < radius:
            return True
    return False

def rrt(G, qG, qI, cspace, stepSize, obst, r, n, p):
    """RRT algo, to generate tree

    @type G: the graph so far
    @type qG: the goal coordinates
    @type qI: the starting coordinates 
    @type cspace:the bounds of the c space
    @type stepSize: how often to generate random vertex
    @type obst: list of obstacles
    @type r: radius of said obstacles
    @type n: how many iterations to run
    @type p: the probability that newPos = qG
    """
    if p != 0:
        newPos = qI
    else:
        newPos = getRandom(cspace)
        while inObst(newPos, obst, r):
            newPos = getRandom(cspace)
    G.addVert(newPos)
    isGoal = False
    for i in range(n):
        newPos = getRandom(cspace)
        isGoal = False
        if p != 0:
            if i % (p*100) == 0:
                newPos = qG
                isGoal = True
        while inObst(newPos, obst, r):
            newPos = getRandom(cspace)
        nearest, indx = getNearest(G, newPos, obst, r)
        if nearest is None:
            continue
        newPos = newVertex(newPos, nearest, stepSize)
        if isGoal:
            newPos = qG
        newIndx = G.addVert(newPos)-1
        G.addEdge(newIndx, indx)
        if isGoal:
            return G
    return G

def getObstacles(dt, center):
    """Gets all the coordinates for a given obstacle

    @type dt: 1-dt makes the radius
    @type center: the coords of the center
    """
    allCoords = []
    for i in range(0, 360):
        rad = i * 0.0174533
        x = center[0] + (1-dt) * math.cos(rad)
        y = center[1] + (1-dt) * math.sin(rad)
        coord = (x, y)
        allCoords.append(coord)
    return allCoords

def getPath(G):
    """Backtracks from the goal position to get the path

    @type G: the graph so far
    """
    path = []
    vertices = G.vertices
    edges = G.edges
    i = len(vertices)-1
    while i > 0:
        path.append(vertices[i])
        i = edges[i-1][1]
    return path

def draw(ax, cspace, obstacles, qI, qG, G, path, title=""):
    """Plot the C-space, obstacles, qI, qG, and graph on the axis ax

    @type ax: axes.Axes, created, e.g., fig, ax = plt.subplots()
    @type cspace: a list [(xmin, xmax), (ymin, ymax)] indicating that the C-space
        is given by [xmin, xmax] \times [ymin, ymax].
    @type obstacles: a list [obs_1, ..., obs_m] of obstacles, where obs_i is a list of coordinates
        on the boundary of the i^{th} obstacle.
    @type qI: a tuple (x, y), indicating the initial configuration.
    @type qG: a tuple (x, y), indicating the goal configuration
    @type path: a list of tuples specifying the sequence of configurations visited along the path
    @type title: a string, indicating the title of the plot
    """

    draw_cspace(ax, cspace, obstacles)
    G.draw(ax)
    if qI is not None:
        if len(qI) == 2:
            ax.plot(qI[0], qI[1], "bx", markersize=7)
        elif len(qI) == 3:
            ax.plot(
                qI[0],
                qI[1],
                marker=(3, 0, qI[2] * 180 / math.pi - 90),
                markersize=15,
                linestyle="None",
                markerfacecolor="blue",
                markeredgecolor="blue",
            )
    if qG is not None:
        if len(qI) == 2:
            ax.plot(qG[0], qG[1], "bo", markersize=7)
        elif len(qG) == 3:
            ax.plot(
                qG[0],
                qG[1],
                marker=(3, 0, qG[2] * 180 / math.pi - 90),
                markersize=15,
                linestyle="None",
                markerfacecolor="red",
                markeredgecolor="red",
            )
    if len(path) > 0:
        ax.plot(
            [state[0] for state in path],
            [state[1] for state in path],
            "b-",
            linewidth=3,
        )
    if len(title) > 0:
        ax.set_title(title, fontsize=15)


def draw_cspace(ax, cspace, obstacles, tick_step=[1, 1]):
    """Draw the C-space and C-space obstacles on the axis ax

    @type cspace: a list [(xmin, xmax), (ymin, ymax)] indicating that the C-space
        is given by [xmin, xmax] \times [ymin, ymax].
    @type obstacles: a list [obs_1, ..., obs_m] of obstacles, where obs_i is a list of coordinates
        on the boundary of the i^{th} obstacle.
    """
    for obs in obstacles:
        ax.plot([v[0] for v in obs], [v[1] for v in obs], "r-", linewidth=2)

    ax.set_xticks(
        range(math.ceil(cspace[0][0]), math.floor(cspace[0][1]) + 1, tick_step[0])
    )
    ax.set_yticks(
        range(math.ceil(cspace[1][0]), math.floor(cspace[1][1]) + 1, tick_step[1])
    )
    ax.set(xlim=cspace[0], ylim=cspace[1])
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)     

if __name__ == '__main__':
    fig, ax = plt.subplots()
    stepSize = 0.1
    dt = 0.02
    p = 0.1
    cspace = [(-3, 3), (-1, 1)]
    obj1 = getObstacles(dt, (0, -1))
    obj2 = getObstacles(dt, (0, 1))
    obstacles = [obj1, obj2]
    qi = (-2, -0.5)
    qg = (2, -0.5)
    G = Graph()
    G = rrt(G, qg, qi, cspace, stepSize, [], 0, 500, 0)
    path = []
    title_1a = "RRT exploration, neglecting obstacles"
    title_1b = "RRT exploration, considering obstacles"
    title_2 = "RRT planning"
    draw(ax, cspace, obstacles, qi, qg, G, path, title_1a)

    fig, ay = plt.subplots()
    G2 = Graph()
    G2 = rrt(G2, qg, qi, cspace, stepSize, [(0, -1), (0, 1)], 1-dt, 500, 0)
    draw(ay, cspace, obstacles, qi, qg, G2, path, title_1b)

    fig, az = plt.subplots()
    G3 = Graph()
    G3 = rrt(G3, qg, qi, cspace, stepSize, [(0, -1), (0, 1)], 1-dt, 500, p)
    path = getPath(G3)
    draw(az, cspace, obstacles, qi, qg, G3, path, title_2)
    plt.show()