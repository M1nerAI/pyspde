# PySPDE

PySPDE is a Python library for performing simulations and kriging of non-stationary spatial gaussian random fields with Matérn covariance, by solving the following Stochastic Partial Differential Equations (SPDEs):

$$ ({\kappa}^2 - {\nabla} {\cdot} H(x) {\nabla} )Z(x) = {\tau}(x)W(x) \quad x \in \mathbb{R}^2
$$

The theory is proposed in Fuglstad (2014).

## Instalation

On Debian/Ubuntu systems:
```
sudo apt-get install libsuitesparse-dev
pip install pyspde
```

On Windows systems:
```
conda install -c conda-forge suitesparse
pip install pyspde
```

## Basic Usage

Imports:
```
from pyspde import anisotropy, Grid, Spde, anisotropy_from_svg
```

Define the anisotropy and the grid:
```
atpy = anisotropy_from_svg(r'assets/anticline.svg', beta=50, gamma=0.5)
grid = Grid(nx=400, ny=200, anisotropy=atpy)
```

Define SPDE and Simulate:
```
sp = Spde(grid, sigma=1, a=50)
Z = sp.simulate(seed=0)
```

![alt text](https://onedrive.live.com/embed?resid=a94fce3415a299a1%2117697&authkey=%21AMG50D19mGNfdyY&width=660)
