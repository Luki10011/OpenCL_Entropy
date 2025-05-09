# Objects counting with entropy

The goal of the project is to implement local entropy algorithm and accelerate computation with OpenCL and GPU

# Builidng and running OpenCL application

## Unix
In OpenCl project directory ie. `LocalEntropy`, perform operations listed below.

Creating a Makefile
```
cmake -G "Unix Makefiles"
```
Building application
```
make
```
In case of problems
```
make clean
make
```
Run binary file
```
./bin/x86_64/Release/LocalEntropy
```


## Windows
Łukasz, dopisałbyś byś tu te komendy których ty używasz?

# Requirements
Do poprawy i uzupełnienia - dodałem tak orientacyjnie, żeby nie zapomnieć potem - Roman
### Program model in python
 - python 3.11
 - opencv-python
 - skimage
 - numpy

### OpenCL application
 - OpenCL
 - AMD App SDK 3.0