. bladerfvenv/bin/activate
cd bladeRF/
# sudo rm -r build/*
cd build/
cmake ../
make -j8
sudo make -j8 install
sudo ldconfig
cd ../host/libraries/libbladeRF_bindings/python
python setup.py install