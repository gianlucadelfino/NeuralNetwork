project(Network)

FILE(GLOB INCLUDES include/*.h)
FILE(GLOB SOURCES src/*.cpp)
FILE(GLOB LIB_DIR_INCLUDE ${LIB_DIR}/*/include)

source_group("Header Files" FILES ${INCLUDES})
source_group("Source Files" FILES ${SOURCES})
add_executable(${PROJECT_NAME} ${INCLUDES} ${SOURCES})

include_directories(${PROJECT_SOURCE_DIR}/include)

target_link_libraries(${PROJECT_NAME})

install(TARGETS ${PROJECT_NAME} DESTINATION ${PROJECT_NAME}/bin)

# make a local folder in config for logs and resources
install_project_directory()
