import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from src.pyspde import anisotropy_from_svg, anisotropy_stationary, Grid, Spde
from scipy.stats import qmc 
import pickle
import inspect
import uuid 

uuid_ = str(uuid.uuid4())[:8]
script = inspect.getsource(inspect.getmodule(inspect.currentframe()))


tag = 'tag'
nx = 600 
ny = 300
a = 3

gamma = 20
beta = 1000

space_samp = 10
n_samps = 100
fps = 6
n_simu = int(fps*5)

#nSeconds = 0.2
drill_index = np.round(np.arange(nx/space_samp/2, nx, nx/space_samp)).astype(int)

Zs = []


atpy = anisotropy_from_svg(r'assets/anticline.svg', beta=beta, gamma=gamma)
#atrp = anisotropy_stationary(theta=theta, gamma=1, beta=8)


pad_factor = ((gamma + beta)**0.5) * a * 2
padx = np.round(pad_factor *  np.abs(atpy._u.mean())).astype(int)
pady = np.round(pad_factor *  np.abs(atpy._v.mean())).astype(int)
print(padx, pady)


grid = Grid(nx=nx, ny=ny, anisotropy=atpy, pady=pady, padx=padx)
sp = Spde(grid, sigma=1, a=a)

reality = sp.simulate(seed=None).squeeze()

breakpoint()

plt.imshow(reality, vmax=2, vmin=-2)
print(reality.mean(), reality.std())
plt.show()


sampler = qmc.Halton(d=2, scramble=False)
samples = sampler.random(n=n_samps)
samples_coords = np.round(qmc.scale(samples, [0, 0], [nx, ny])).astype(int)
samples_z = reality[samples_coords[:,1], samples_coords[:, 0]]

samples_krig = sp.kriging(np.column_stack((samples_coords, samples_z)))

Zs = []

non_cond = sp.simulate(n=n_simu, seed=None)
samples_z_non_cond = non_cond[samples_coords[:,1], samples_coords[:, 0], :]
non_cond_krig = sp.kriging(np.column_stack((samples_coords, samples_z_non_cond)))

#n_ = 1
#plt.imshow(non_cond_krig[:,:,n_], vmax=2, vmin=-2)
#plt.scatter(samples_coords[:,0], samples_coords[:,1], c=samples_z_non_cond[:,n_], s=6, edgecolors='black', linewidths=0.2, vmax=2, vmin=-2)
#plt.show()

cond_sim = samples_krig[:,:,np.newaxis] + (non_cond - non_cond_krig)


fig, ax = plt.subplots()
fig.set_size_inches(10,5)

im = ax.imshow(cond_sim[:, :, 0], vmax=2, vmin=-2)
ax.scatter(samples_coords[:,0], samples_coords[:,1], c=samples_z, s=8, edgecolors='black', linewidths=0.2, vmax=2, vmin=-2)

def animate_func(i):
    im.set_array(cond_sim[:, :, i])
    return [im]

anim = animation.FuncAnimation(
                               fig, 
                               animate_func, 
                               frames = n_simu,
                               repeat = False,
                               #interval = 1000 / fps, # in ms
                               )

fig.subplots_adjust(bottom=0.08, left=0.04,right=0.975, top=0.975,
                    #, wspace=None, hspace=None
                    )
anim.save('test_anim.gif', fps=fps, dpi=600)


pickle.dump({'anim': anim, 'samples_coords': samples_coords, 'samples_z': samples_z, 'cond_sim': cond_sim, 'script': script, 'uuid': uuid_, 'tag': tag}, open(f'simu_{tag}_{uuid_}.pickle','wb'))
