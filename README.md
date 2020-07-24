# GasMix
This repository holds the gas functions, gas cross sections, gas objects, and gas mixing functions used by both PyBoltz and PyGrad**.

** PyGradd is not built yet.

## Setting up and running instructions.
There is two methods to do so. In both of them you will need python3+, cython, and numpy installed. Our setup has python 3.6.7, Cython 0.29.3, and numpy 1.16.1.

### First method.
Simply install the module by using pip through the following command. **Note: this method needs a new version of PIP.**
```
sudo pip3 install --upgrade  git+https://github.com/UTA-REST/PyGasMix --user
```

### Second method.
Clone this repository. Then run the following command. 
```
$ source setup.sh
```
