cmake_minimum_required(VERSION 3.16)
project(run_geant4)

find_package(Geant4 REQUIRED)
include(${Geant4_USE_FILE})

add_executable(run_geant4 run_geant4.cc)
target_link_libraries(run_geant4 ${Geant4_LIBRARIES})
