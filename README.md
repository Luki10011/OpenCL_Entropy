# Local entropy application

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
cmake .. -G "MinGW Makefiles"
mingw32-make

```

# Requirements

### Program model in python
- Python 3.10.12
- imageio==2.37.0
- lazy_loader==0.4
- networkx==3.4.2
- numpy==2.2.6
- opencv-python==4.11.0.86
- packaging==25.0
- pillow==11.2.1
- scikit-image==0.25.2
- scipy==1.15.3
- tifffile==2025.5.10

To install required python packages, in `python_modek` directory run command:
```
pip install -r requirements.txt
```

### OpenCL application
 - C++ 17
 - OpenCL
 - AMD App SDK 3.0
