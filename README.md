# Objects counting with entropy

The goal of the project is to implement local entropy algorithm and accelerate computation with OpenCL and GPU

# Builidng and running OpenCL application

## Unix
In OpenCl project directory ie. `LocalEntropy`, perform operations listed below.

Creating a Makefile
```
mkdir -p build
cd build
cmake .. -G "Unix Makefiles"
```
Building application
```
make
```
In case of problems
```
make clean
make
cd ..
```
Run binary file from `LocalEntropy` directory
```
./bin/x86_64/Release/LocalEntropy --input <input_image_name_without_extension> -f <d/i> -s <struct_shape> -m <struct_height> -n <struct_width>

note: -f (folder) arguement is responsible for handling reading of every image from input folder (if d is selected)
```


## Windows
You need to run this commands in roder to build project on windows
```
mkdir build
cd build
cmake -G "MinGW Makefiles" ..
mingw32-make

```

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
