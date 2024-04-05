# BladeRF Starting Guide
#### *This assumes a Linux system.

## 1: Setup
### Install Dependencies:
- libusb >= 1.0.16
- cmake >= 2.8.5

### Download FPGA Bitstream and Firmware
- https://www.nuand.com/fpga/hostedxA4-latest.rbf
- https://www.nuand.com/fx3/bladeRF_fw_latest.img

## 2: Library Installation
~~~
git clone git@github.com:Nuand/bladeRF.git

cd bladeRF

mkdir -p build && cd build

cmake ../

make

sudo make install

sudo ldconfig
~~~

## 3: Python 3 Bindings Installation
~~~
cd bladeRF/host/libraries/libbladeRF_bindings/python

python3 setup.py install
~~~

## 4: Load Bitstream and Firmware
~~~
bladeRF-cli -f bladeRF_fw_latest.img

bladeRF-cli -L hostedxA4-latest.rbf
~~~

