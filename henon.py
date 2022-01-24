import numpy as np
import matplotlib.pyplot as plt
import os

import plotly.graph_objects as go

dt, chyby = np.genfromtxt('energies.txt', unpack=True, delimiter=",")

init_energy = 0.18045

plt.figure()
plt.title("Chyby energie")
plt.loglog(dt, np.abs(chyby - init_energy), 'rx', )
plt.savefig("chyby_energie.png")

dt, chyby = np.genfromtxt('absolutes.txt', unpack=True, delimiter=",")

plt.figure()
plt.title("Chyby L1 distance")
plt.loglog(dt, chyby, 'rx')
plt.savefig("chyby_absoutes.png")

poincarove = os.listdir("poincare")

fig, ax = plt.subplots(2,2, figsize=(10,10))
for poincare in poincarove:
    t, xb, pxb, pyb = np.genfromtxt("poincare/" + poincare, unpack=True, delimiter=",")
    ax[0,0].set_title("[x, px]")
    ax[0,0].plot(xb, pxb, marker='x', linestyle='None', markersize=1)
    ax[1,0].set_title("[x, py]", )
    ax[1,0].plot(xb, pyb, marker='x', linestyle='None', markersize=1)
    ax[0,1].set_title("[px, py]")
    ax[0,1].plot(pxb, pyb, marker='x', linestyle='None', markersize=1)

fig.savefig("poincare.png", dpi=300)

t, xb, yb, pxb, pyb = np.genfromtxt("orbit.txt", unpack=True, delimiter=",")
fig = go.Figure(
    data=go.Scatter3d(
    x = xb,
    y = yb,
    z = np.sqrt(pxb * pxb + pyb * pyb),
    
    marker=dict(
        size=1,
        color=t,
        colorscale='Viridis',
        showscale=True,
        ),
    line=dict(
        color=t,
        width=1,
        colorscale='Viridis',
        ),
    )
)

fig.update_layout(scene = dict(
                    xaxis_title='x',
                    yaxis_title='y',
                    zaxis_title='|p|',),

                    )

fig.write_html("orbit.html")