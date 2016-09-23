cmake_minimum_required(VERSION 2.8)

project(Network)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion
                OUTPUT_VARIABLE GCC_VERSION)
if (GCC_VERSION VERSION_GREATER 5 OR GCC_VERSION VERSION_EQUAL 5)
else()
    message(FATAL "gcc is version" ${GCC_VERSION} ", You need at least 5")
endif()

set(GCC_WARNING_FLAGS "-Wall -Werror -Wextra -pedantic")

# ftrapv checks for overflow and crashes the program if it happens.
set(GCC_DEBUG_FLAGS "-g -ftrapv")
set(GCC_SANITIZE_FLAGS "-fuse-ld=gold -fsanitize=undefined,address")

set(GCC_DIALECT_FLAGS "-std=c++14")
set(GCC_CODE_COVERAGE_FLAGS "-O0 --coverage -pg -g -fprofile-arcs -ftest-coverage")

if(CMAKE_BUILD_TYPE MATCHES "[Rr][Ee][Ll][Ee][Aa][Ss][Ee]")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 -mavx -DNDEBUG ${GCC_WARNING_FLAGS} ${GCC_DIALECT_FLAGS}")
elseif(CMAKE_BUILD_TYPE MATCHES "[Dd][Ee][Bb][Uu][Gg]" OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_WARNING_FLAGS} ${GCC_DIALECT_FLAGS} ${GCC_SANITIZE_FLAGS} ${GCC_DEBUG_FLAGS}")
elseif(CMAKE_BUILD_TYPE MATCHES "[Pp][Rr][Oo][Ff][Ii][Ll][Ee]")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mavx -DNDEBUG -pg ${GCC_DIALECT_FLAGS} ${GCC_WARNING_FLAGS} -fno-omit-frame-pointer")
elseif(CMAKE_BUILD_TYPE MATCHES "[Cc][Oo][Vv][Ee][Rr][Aa][Gg][Ee]")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG ${GCC_DIALECT_FLAGS} ${GCC_CODE_COVERAGE_FLAGS} ${GCC_WARNING_FLAGS} -fno-omit-frame-pointer")
else()
    message(FATAL "Build type not recognised")
endif()

# Set directories variable to look use in looking up the headers
set(CORTEXICA_SHARED_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../SharedLibraries)
set(THIRD_PARTY_LIBS ${CMAKE_CURRENT_SOURCE_DIR}/../3rdPartyLibs)
set(APPLICATIONS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/applications)
set(CORTEXICA_SHARED_SCRIPTS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../SharedScripts)

# Put binaries under bin
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
message(STATUS "Executables will be placed in ${EXECUTABLE_OUTPUT_PATH}")

# Put libraries under lib
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
message(STATUS "Libraries go to ${LIBRARY_OUTPUT_PATH}")

# Define macro used copy directories when installing
macro(install_subdir folder_name)
    install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/${folder_name} DESTINATION
            "$ENV{HOME}/.VP/${PROJECT_NAME}"
            DIRECTORY_PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE
                                  GROUP_WRITE GROUP_READ GROUP_EXECUTE
                                  WORLD_WRITE WORLD_READ WORLD_EXECUTE
            FILE_PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE
                             GROUP_WRITE GROUP_READ GROUP_EXECUTE
                             WORLD_WRITE WORLD_READ WORLD_EXECUTE
            PATTERN ".gitignore" EXCLUDE)
endmacro()

macro(install_project_directory)
    install(DIRECTORY DESTINATION
            "$ENV{HOME}/.VP/${PROJECT_NAME}" DIRECTORY_PERMISSIONS
                        OWNER_WRITE OWNER_READ OWNER_EXECUTE
                        GROUP_WRITE GROUP_READ GROUP_EXECUTE
                        WORLD_WRITE WORLD_READ WORLD_EXECUTE)
endmacro()

# Define macro used to include subfolders
macro(subdirlist result curdir)
  file(GLOB children RELATIVE ${curdir} ${curdir}/*)
  set(dirlist "")
  foreach(child ${children})
    if(IS_DIRECTORY ${curdir}/${child})
        list(APPEND dirlist ${child})
    endif()
  endforeach()
  set(${result} ${dirlist})
endmacro()

add_subdirectory(Network)
add_subdirectory(XorTrainingDataGenerator)
