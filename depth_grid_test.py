import numpy as np
import math
import matplotlib.pyplot as plt
grid = np.zeros((12,28))
corners = [(0,0),(11,0),(0,27),(11,27)]

def distance(s:tuple,x:tuple):
    return math.sqrt(pow(s[0]-x[0],2)+pow(s[1]-x[1],2))

def getDistances(p,targets):
    S = []
    for t in targets:
        S.append(distance(p,t))
    return S


import math
def NormalDepth(x):
    rho = 1/4
    x = rho*x
    mu = 0; phi = 1
    return  rho*((1/phi*math.sqrt(2*math.pi))*pow(math.e,(-1/2)*pow((x-mu)/phi,2)))
    
def totalDepth(p, targets):
    distances = [math.sqrt(pow(p[0]-x[0],2)+pow(p[1]-x[1],2)) for x in targets]
    # how close is it to the closest corner
    # get the closest corner
    test = dict(zip(distances,targets))
    if distances:
        closest = min(distances)
        closest_corner = test[closest]
        closest_side = min([pow(p[0]-closest_corner[0],2),pow(p[1]-closest_corner[1],2)])
        print(closest_side)
        # add buff for distance to same axis
        return closest+closest_side

for i in range(len(grid)):
    for j in range(len(grid[0])):
        grid[i][j] = totalDepth((i,j),corners)

import matplotlib.pyplot as plt

x = np.arange(grid.shape[1])  # X coordinates (columns)
y = np.arange(grid.shape[0])  # Y coordinates (rows)
x, y = np.meshgrid(x, y)  # Create a meshgrid for X and Y

# Initialize a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(x, y, grid, cmap='viridis')  # You can change the colormap if desired

# Labels and title
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Depth')
ax.set_title('3D Surface Plot of Depth Array')

# Show the plot
plt.show()
