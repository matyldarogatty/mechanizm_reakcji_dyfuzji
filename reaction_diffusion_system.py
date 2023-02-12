import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
import seaborn as sns
sns.set()

step_space = 0.1
step_time = 0.001
D = 2
xmax = 10
tmax = 10
gamma = 100
b = 0.0001

n = int(xmax/step_space)
k = int(tmax/step_time)

f = lambda a, c: gamma * a ** 2 / c - b * gamma * a
g = lambda a, c: gamma * a ** 2 - gamma * c

u = np.zeros((n, k))
v = np.zeros((n, k))
x = np.arange(0, xmax, step_space)
t = np.arange(0, tmax, step_time)


def jump_fun(s):
    if s == 2 or s == 50 or s == 70 :
        s = 10
    else:
        s = np.random.random(n)
    return s

def ssin(x):
    r = (math.sin(x))**2
    if r == 0:
        r += (math.sin(x+1))**2 + 0.001
    return r

def ccos(x):
    r = (math.cos(x))**2
    if r == 0:
        r += (math.sin(x+1))**2 + 0.01
    elif r==1:
        r=10
    return r


m = 1
if m:
    v[:,0] = [ccos(i) for i in range(n)]
    u[:,0] = [ssin(i) for i in range(n)]
else:
    u[:, 0] = np.random.random(n)
    v[:,0] = np.random.random(n)

# v[:, 0] = [i/10+0.01 for i in range(n)]
# u[:, 0] = [2*(i)/100 for i in range(n)]
# for x in range(100):
#     if x//9 == 0:
#         u[x, 0] = 5
# for s in range(100):
#    if u[s,0] == 0.12 or u[s,0]  == 0.96 or u[s,0]  == 1 or u[s,0]  == 1.6:
#        u[s,0] = 10

# u[:, 0] = [100/i+1000 for i in range(1,n+1)]
# v[:, 0] = [1/2*i for i in range(1,n+1)]
# u[:, 0] = [jump_fun(i) for i in range(n)]
# v[:, 0] = [jump_fun(i) for i in range(n)]


l = step_time / step_space ** 2

for j in range(1, k):
    for i in range(n):
        if i == 0:
            u[0, j] = u[0, j - 1] + l * (u[1, j - 1] - u[0, j - 1]) + step_time * f(u[0, j - 1], v[0, j - 1])
            v[i, j] = v[i, j - 1] + D * l * (v[i + 1, j - 1] - v[i, j - 1]) \
                      +step_time * g(u[i, j - 1], v[i, j - 1])
        elif i == n-1:
            u[i, j] = u[i, j - 1] + l * (u[i - 1, j - 1] - u[i, j - 1]) + step_time * f(u[i, j - 1], v[i, j - 1])
            v[i, j] = v[i, j - 1] + D * l * (v[i - 1, j - 1] - v[i, j - 1]) \
                      + step_time * g(u[i, j - 1], v[i, j - 1])
        else:
            try:
                u[i, j] = u[i, j - 1] + l * (u[i + 1, j - 1] + u[i - 1, j - 1] - 2 * u[i, j-1]) + \
                          step_time * f(u[i, j - 1], v[i, j - 1])
                v[i, j] = v[i, j - 1] + D * l * (v[i + 1, j - 1] + v[i - 1, j - 1] - 2 * v[i, j - 1]) \
                          + step_time * g(u[i, j - 1], v[i, j - 1])
            except Exception as error:
                print('here')
                print(u.shape,v.shape,i,j)
                print(u[i + 1, j - 1])
                print(error)
                break

fig, ax = plt.subplots()
ax.set_xlim((0, 5))
ax.set_ylim((-1, 2))
#x = np.arange(0, xmax, step_space)
line, = ax.plot([], [], lw=2)


# def init():
#     line.set_data([], [])
#     return line,
#
#
# def animation(j):
#   line.set_data(x, u[:,j])
#   return line,
#
#
# anim = FuncAnimation(fig, init_func=init, func=animation, frames=k, interval=100, blit=True)
# plt.show()
#
X, Y = np.meshgrid(t, x)
Z = u
# sns.heatmap(u, cmap='vlag')
# plt.show()
# sns.heatmap(v, cmap='vlag')
ax = plt.axes(projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()


