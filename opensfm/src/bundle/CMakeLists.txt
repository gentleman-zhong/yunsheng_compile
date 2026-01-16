# --- OpenSfM Bundle Module CMake Configuration ---

# 1. 定义源文件列表
set(BUNDLE_FILES
    bundle_adjuster.h
    reconstruction_alignment.h
    data/camera.h
    data/data.h
    data/pose.h
    data/shot.h
    error/error_utils.h
    error/motion_prior_errors.h
    error/position_functors.h
    error/projection_errors.h
    error/absolute_motion_errors.h
    error/parameters_errors.h
    error/relative_motion_errors.h
    src/bundle_adjuster.cc
)

# 2. 探测 CUDA 工具包 (必需，用于解决 undefined symbol)
find_package(CUDAToolkit REQUIRED)

# 3. 编译 bundle 静态库
add_library(bundle ${BUNDLE_FILES})

# 强制要求 C++17，确保与 Ceres 2.1.0 的标准一致
target_compile_features(bundle PRIVATE cxx_std_17)

target_link_libraries(bundle
  PUBLIC  # <--- 将 PRIVATE 改为 PUBLIC，这样依赖会传递给 pybundle
    ${CERES_LIBRARIES}
    foundation
    CUDA::cudart
    CUDA::cusolver
    CUDA::cublas
    CUDA::cusparse
  PRIVATE # 仅内部使用的库可以保留为 PRIVATE
    ${LAPACK_LIBRARIES}
    ${SUITESPARSE_LIBRARIES}
)

# 5. 指定包含路径
if (LAPACK_FOUND)
    target_include_directories(bundle PRIVATE ${LAPACK_INCLUDE_DIRS})
endif()
if (SUITESPARSE_FOUND)
    target_include_directories(bundle PRIVATE ${SUITESPARSE_INCLUDE_DIRS})
endif()

# 包含 Ceres 及其内置的 miniglog
target_include_directories(bundle PRIVATE 
    ${CMAKE_SOURCE_DIR} 
    ${CERES_INCLUDE_DIRS}
    ${ADDITIONAL_INCLUDE_DIRS}/ceres/internal/miniglog
)

# --- 6. 单元测试部分 (如果开启) ---
if (OPENSFM_BUILD_TESTS)
    set(BUNDLE_TEST_FILES
        test/reprojection_errors_test.cc
        test/bundle_data_test.cc
    )
    add_executable(bundle_test ${BUNDLE_TEST_FILES})
    target_compile_features(bundle_test PRIVATE cxx_std_17)
    target_include_directories(bundle_test
                        PRIVATE
                        ${CMAKE_SOURCE_DIR}
                        ${CERES_INCLUDE_DIRS}
                        ${GMOCK_INCLUDE_DIRS})
    target_link_libraries(bundle_test
                        PUBLIC
                        bundle
                        geometry
                        Eigen3::Eigen
                        ${TEST_MAIN}
                        CUDA::cudart) # 测试程序也需要链接 CUDA
    add_test(bundle_test bundle_test)
endif()

# --- 7. 编译 Python 绑定模块 (pybundle) ---
pybind11_add_module(pybundle python/pybind.cc)

# 确保 pybundle 使用 C++17
target_compile_features(pybundle PRIVATE cxx_std_17)

# 关键：使用 --whole-archive 标志链接静态库 libceres.a
target_link_libraries(pybundle PRIVATE
  bundle
  geometry
  foundation
  pybind11
  
  # --- 核心修改如下 ---
  -Wl,--whole-archive "${CERES_LIBRARIES}" -Wl,--no-whole-archive
  
  CUDA::cudart
  CUDA::cusolver
  CUDA::cublas
  CUDA::cusparse
)

# 设置输出目录到 opensfm 源码根目录
set_target_properties(pybundle PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${opensfm_SOURCE_DIR}/.."
)