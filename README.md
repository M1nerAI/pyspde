# PySPDE

PySPDE is a Python library for performing simulations and kriging of non-stationary spatial gaussian random fields with Matérn covariance, by solving the following Stochastic Partial Differential Equations (SPDEs):

$$ ({\kappa}^2 - {\nabla} {\cdot} H(x) {\nabla} )Z(x) = {\tau}W(x) \quad x \in \mathbb{R}^2
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
from pyspde import anisotropy, Grid, Spde
```

Define the anisotropy and the grid:
```
atpy = anisotropy_from_svg(r'assets/anicline.svg', beta=10, gamma=0.1)
grid = Grid(nx=200, ny=100, anisotropy=atpy)
```

Define SPDE and Simulate:
```
sp = Spde(grid, sigma=1, a=25)
Z = sp.simulate()
```
