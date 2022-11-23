# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT IREE_TARGET_BACKEND_LLVM_CPU OR
   NOT IREE_HAL_DRIVER_LOCAL_SYNC OR
   NOT IREE_HAL_EXECUTABLE_LOADER_EMBEDDED_ELF)
  return()
endif()

# NOTE: this example uses clang to target x86_64 as that's what the example.mlir
# file is hardcoded to accept. It's possible to use arbitrary compilers when
# using either the system linker or static library output options of the
# CPU backend but embedded dynamic libraries have strict requirements and it's
# easiest to always use clang (plus it is good at cross-compiling).
#
# This example just shows how users can link in custom objects and is not
# intended to demonstrate the infrastructure to produce the object files: when
# using this custom kernel approach it is up to the user to handle that work.
find_program(CLANG clang)
if(NOT CLANG)
  message(STATUS "IREE custom_dispatch/cpu/embedded ignored -- clang not found")
  return()
endif()

set(_OBJ_TARGET iree_samples_custom_dispatch_cpu_embedded_obj)
add_custom_command(
  OUTPUT functions.o
  DEPENDS functions.c
  COMMAND ${CLANG}
      -target x86_64-unknown-unknown-eabi-elf
      -std=c17
      -fvisibility=hidden
      -fno-plt
      -fno-rtti
      -fno-exceptions
      -fdata-sections
      -ffunction-sections
      -funique-section-names
      -c ${CMAKE_CURRENT_SOURCE_DIR}/functions.c
      -o ${CMAKE_CURRENT_BINARY_DIR}/functions.o
  VERBATIM
)
add_custom_target(${_OBJ_TARGET} DEPENDS
  ${CMAKE_CURRENT_BINARY_DIR}/functions.o
)
add_dependencies(iree-test-deps "${_OBJ_TARGET}")
add_dependencies(iree-sample-deps "${_OBJ_TARGET}")

iree_lit_test_suite(
  NAME
    example
  SRCS
    "example.mlir"
  DATA
    ${_OBJ_TARGET}
  TOOLS
    FileCheck
    iree-compile
    iree-run-module
  LABELS
    "driver=local-sync"
    "hostonly"
)
