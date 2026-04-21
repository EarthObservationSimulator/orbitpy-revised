cmake_minimum_required(VERSION 3.20)
project(orbitpy LANGUAGES CXX)

add_subdirectory(extern/CoverageKinematics)

install(TARGETS GeometricTools LIBRARY DESTINATION .)
install(TARGETS kcl LIBRARY DESTINATION .)