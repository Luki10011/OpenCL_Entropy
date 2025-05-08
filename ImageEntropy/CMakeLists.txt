cmake_minimum_required(VERSION 3.15)
project(ImageEntropyCalculator LANGUAGES CXX)

# Konfiguracja dla Windows
if(WIN32)
    add_definitions(-DWIN32_LEAN_AND_MEAN -DNOMINMAX)
    if(MSVC)
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    endif()
endif()

# Generowanie struktury katalogów
set(PROJECT_DIRECTORIES
    src/
    src/opencl
    src/opencl/kernels
    include/
)

foreach(dir ${PROJECT_DIRECTORIES})
    if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${dir})
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${dir})
        message(STATUS "Created directory: ${dir}")
    endif()
endforeach()

# Wyszukiwanie OpenCL z odpowiednią obsługą błędów
find_package(OpenCL REQUIRED)

if(NOT OpenCL_FOUND)
    message(FATAL_ERROR "OpenCL not found - please install appropriate SDK")
else()
    message(STATUS "Found OpenCL:")
    message(STATUS "  Includes: ${OpenCL_INCLUDE_DIRS}")
    message(STATUS "  Libraries: ${OpenCL_LIBRARIES}")
endif()

# Sprawdzenie czy istnieją pliki źródłowe
set(SOURCE_FILES)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)
    list(APPEND SOURCE_FILES src/main.cpp)
else()
    # Tworzenie podstawowego pliku main.cpp jeśli nie istnieje
    file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
        "#include <iostream>\n#include \"opencl_utils.h\"\n\nint main() {\n    std::cout << \"OpenCL Entropy Calculator\\n\";\n    return 0;\n}\n")
    list(APPEND SOURCE_FILES src/main.cpp)
    message(STATUS "Created basic main.cpp")
endif()

# Tworzenie podstawowego pliku nagłówkowego OpenCL jeśli nie istnieje
if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/include/opencl_utils.h)
    file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/include/opencl_utils.h
        "#pragma once\n\n#include <CL/cl.h>\n\nvoid init_opencl();\nvoid cleanup_opencl();\n")
endif()

# Konfiguracja projektu
add_executable(${PROJECT_NAME} ${SOURCE_FILES})

# Ustawienia kompilacji
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
target_include_directories(${PROJECT_NAME} PRIVATE 
    include
    ${OpenCL_INCLUDE_DIRS}
)

# Linkowanie bibliotek
target_link_libraries(${PROJECT_NAME} PRIVATE 
    ${OpenCL_LIBRARIES}
)

# Kopiowanie plików kernelów do folderu output
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/opencl/kernels/entropy.cl)
    configure_file(
        src/opencl/kernels/entropy.cl
        ${CMAKE_CURRENT_BINARY_DIR}/entropy.cl
        COPYONLY
    )
    
    # Dla Windows - dodatkowa kopia do folderu wykonywalnego
    if(WIN32)
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/src/opencl/kernels/entropy.cl
            $<TARGET_FILE_DIR:${PROJECT_NAME}>/entropy.cl
            COMMENT "Copying OpenCL kernel to output directory"
        )
    endif()
endif()

message(STATUS "\nProject configured successfully!")
message(STATUS "Source directory: ${CMAKE_CURRENT_SOURCE_DIR}/src")
message(STATUS "Include directory: ${CMAKE_CURRENT_SOURCE_DIR}/include")
message(STATUS "OpenCL kernels: ${CMAKE_CURRENT_SOURCE_DIR}/src/opencl/kernels")