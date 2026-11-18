# Python bindings (module lokib), built only with -DLOKIB_PYTHON=ON. Needs pybind11 (find_package).
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(lokib lokib.cpp)
target_link_libraries(lokib PRIVATE loki_cpp::loki-b)
# find libloki-b next to the module or in the build tree
set_target_properties(lokib PROPERTIES BUILD_RPATH "$<TARGET_FILE_DIR:loki-b>" INSTALL_RPATH "$ORIGIN;@loader_path")
# A Python extension module leaves the interpreter's symbols (Py_*, _Py_*) undefined: Python
# provides them when it loads the module, and the module must not link libpython itself. The
# top-level CMakeLists.txt forbids undefined symbols for every link (-no-undefined on Linux,
# -undefined,error on macOS), which breaks this link on Linux (on macOS pybind11's later
# -undefined dynamic_lookup happens to override it). Drop those two options for this target only.
get_target_property(_lokib_link_options lokib LINK_OPTIONS)
if(_lokib_link_options)
  list(REMOVE_ITEM _lokib_link_options "LINKER:-no-undefined" "LINKER:-undefined,error")
  set_target_properties(lokib PROPERTIES LINK_OPTIONS "${_lokib_link_options}")
endif()
