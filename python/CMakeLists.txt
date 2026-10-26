# Python bindings (module lokib), built only with -DLOKIB_PYTHON=ON. Needs pybind11 (find_package).
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(lokib lokib.cpp)
target_link_libraries(lokib PRIVATE loki_cpp::loki-b)
# find libloki-b next to the module or in the build tree
set_target_properties(lokib PROPERTIES BUILD_RPATH "$<TARGET_FILE_DIR:loki-b>" INSTALL_RPATH "$ORIGIN;@loader_path")
