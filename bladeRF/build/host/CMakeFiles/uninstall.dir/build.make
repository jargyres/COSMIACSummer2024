# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jta/bladerf_testing/bladeRF

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jta/bladerf_testing/bladeRF/build

# Utility rule file for uninstall.

# Include any custom commands dependencies for this target.
include host/CMakeFiles/uninstall.dir/compiler_depend.make

# Include the progress variables for this target.
include host/CMakeFiles/uninstall.dir/progress.make

host/CMakeFiles/uninstall:
	cd /home/jta/bladerf_testing/bladeRF/build/host && /usr/bin/cmake -P /home/jta/bladerf_testing/bladeRF/build/host/cmake_uninstall.cmake

uninstall: host/CMakeFiles/uninstall
uninstall: host/CMakeFiles/uninstall.dir/build.make
.PHONY : uninstall

# Rule to build all files generated by this target.
host/CMakeFiles/uninstall.dir/build: uninstall
.PHONY : host/CMakeFiles/uninstall.dir/build

host/CMakeFiles/uninstall.dir/clean:
	cd /home/jta/bladerf_testing/bladeRF/build/host && $(CMAKE_COMMAND) -P CMakeFiles/uninstall.dir/cmake_clean.cmake
.PHONY : host/CMakeFiles/uninstall.dir/clean

host/CMakeFiles/uninstall.dir/depend:
	cd /home/jta/bladerf_testing/bladeRF/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jta/bladerf_testing/bladeRF /home/jta/bladerf_testing/bladeRF/host /home/jta/bladerf_testing/bladeRF/build /home/jta/bladerf_testing/bladeRF/build/host /home/jta/bladerf_testing/bladeRF/build/host/CMakeFiles/uninstall.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : host/CMakeFiles/uninstall.dir/depend
