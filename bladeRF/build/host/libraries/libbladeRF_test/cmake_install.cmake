# Install script for directory: /home/jta/bladerf_testing/bladeRF/host/libraries/libbladeRF_test

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/jta/bladerf_testing/bladeRF/build/host/libraries/libbladeRF_test/test_async/cmake_install.cmake")
  include("/home/jta/bladerf_testing/bladeRF/build/host/libraries/libbladeRF_test/test_bootloader_recovery/cmake_install.cmake")
  include("/home/jta/bladerf_testing/bladeRF/build/host/libraries/libbladeRF_test/test_c/cmake_install.cmake")
  include("/home/jta/bladerf_testing/bladeRF/build/host/libraries/libbladeRF_test/test_cpp/cmake_install.cmake")
  include("/home/jta/bladerf_testing/bladeRF/build/host/libraries/libbladeRF_test/test_ctrl/cmake_install.cmake")
  include("/home/jta/bladerf_testing/bladeRF/build/host/libraries/libbladeRF_test/test_freq_hop/cmake_install.cmake")
  include("/home/jta/bladerf_testing/bladeRF/build/host/libraries/libbladeRF_test/test_fw_check/cmake_install.cmake")
  include("/home/jta/bladerf_testing/bladeRF/build/host/libraries/libbladeRF_test/test_open/cmake_install.cmake")
  include("/home/jta/bladerf_testing/bladeRF/build/host/libraries/libbladeRF_test/test_parse/cmake_install.cmake")
  include("/home/jta/bladerf_testing/bladeRF/build/host/libraries/libbladeRF_test/test_peripheral_timing/cmake_install.cmake")
  include("/home/jta/bladerf_testing/bladeRF/build/host/libraries/libbladeRF_test/test_repeater/cmake_install.cmake")
  include("/home/jta/bladerf_testing/bladeRF/build/host/libraries/libbladeRF_test/test_quick_retune/cmake_install.cmake")
  include("/home/jta/bladerf_testing/bladeRF/build/host/libraries/libbladeRF_test/test_repeated_stream/cmake_install.cmake")
  include("/home/jta/bladerf_testing/bladeRF/build/host/libraries/libbladeRF_test/test_rx_discont/cmake_install.cmake")
  include("/home/jta/bladerf_testing/bladeRF/build/host/libraries/libbladeRF_test/test_scheduled_retune/cmake_install.cmake")
  include("/home/jta/bladerf_testing/bladeRF/build/host/libraries/libbladeRF_test/test_sync/cmake_install.cmake")
  include("/home/jta/bladerf_testing/bladeRF/build/host/libraries/libbladeRF_test/test_timestamps/cmake_install.cmake")
  include("/home/jta/bladerf_testing/bladeRF/build/host/libraries/libbladeRF_test/test_tune_timing/cmake_install.cmake")
  include("/home/jta/bladerf_testing/bladeRF/build/host/libraries/libbladeRF_test/test_unused_sync/cmake_install.cmake")
  include("/home/jta/bladerf_testing/bladeRF/build/host/libraries/libbladeRF_test/test_version/cmake_install.cmake")
  include("/home/jta/bladerf_testing/bladeRF/build/host/libraries/libbladeRF_test/test_digital_loopback/cmake_install.cmake")
  include("/home/jta/bladerf_testing/bladeRF/build/host/libraries/libbladeRF_test/test_interleaver/cmake_install.cmake")
  include("/home/jta/bladerf_testing/bladeRF/build/host/libraries/libbladeRF_test/test_rx_meta/cmake_install.cmake")
  include("/home/jta/bladerf_testing/bladeRF/build/host/libraries/libbladeRF_test/test_fpga_load/cmake_install.cmake")

endif()

