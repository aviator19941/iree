// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/cuda/registration/driver_module.h"

#include <inttypes.h>
#include <stddef.h>

#include "iree/base/api.h"
#include "iree/base/internal/flags.h"
#include "iree/base/internal/path.h"
#include "iree/base/tracing.h"
#include "iree/hal/drivers/cuda/api.h"

// Force using CUDA streams until we support command buffer caching to avoid the
// overhead of graph creation.
IREE_FLAG(
    bool, cuda_use_streams, true,
    "Use CUDA streams for executing command buffers (instead of graphs).");

IREE_FLAG(bool, cuda_allow_inline_execution, false,
          "Allow command buffers to execute inline against CUDA streams when "
          "possible.");

IREE_FLAG(int32_t, cuda_default_index, 0, "Index of the default CUDA device.");

static iree_status_t iree_hal_cuda_driver_factory_enumerate(
    void* self, iree_host_size_t* out_driver_info_count,
    const iree_hal_driver_info_t** out_driver_infos) {
  // NOTE: we could query supported cuda versions or featuresets here.
  static const iree_hal_driver_info_t driver_infos[1] = {{
      .driver_name = iree_string_view_literal("cuda"),
      .full_name = iree_string_view_literal("CUDA (dynamic)"),
  }};
  *out_driver_info_count = IREE_ARRAYSIZE(driver_infos);
  *out_driver_infos = driver_infos;
  return iree_ok_status();
}

static iree_string_view_t* get_driver_names(iree_string_view_t driver_names, iree_host_size_t num_drivers) {
  iree_string_view_t* drivers = malloc(num_drivers * sizeof(iree_string_view_t));
  printf("PARSE_CUDA_FROM_DRIVER_NAME: %s\n", driver_names.data);
  // insert driver_name into driver
  char* name_str;
  for (int i = 0; i < num_drivers; i++) {
    iree_string_view_t key_value;
    iree_string_view_split(driver_names, ',', &key_value, &driver_names);
    name_str = (char*) malloc((key_value.size) * sizeof(char));
    memset(name_str, '\0', (key_value.size) * sizeof(char));
    strncpy(name_str, key_value.data, key_value.size);
    printf("name_str: %s, %zu\n", name_str, strlen(name_str));
    drivers[i] = iree_string_view_trim(iree_make_string_view(
      name_str, strlen(name_str)));
    printf("DRIVERS[%d]: %s, %zu\n", i, drivers[i].data, drivers[i].size);
  }

  return drivers;
}

static iree_status_t iree_hal_cuda_driver_factory_try_create(
    void* self, iree_string_view_t driver_name, iree_allocator_t host_allocator,
    iree_hal_driver_t** out_driver) {
  IREE_ASSERT_ARGUMENT(out_driver);
  *out_driver = NULL;
  if (!iree_string_view_equal(driver_name, IREE_SV("cuda"))) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "no driver '%.*s' is provided by this factory",
                            (int)driver_name.size, driver_name.data);
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  printf("DRIVER NAME: %s\n", driver_name.data);
  //iree_string_view_t uri = iree_string_view_trim(iree_make_string_view(
  //    driver_name.data, strlen(driver_name.data)));  

  iree_hal_cuda_device_params_t default_params;
  iree_hal_cuda_device_params_initialize(&default_params);
  if (FLAG_cuda_use_streams) {
    default_params.command_buffer_mode =
        IREE_HAL_CUDA_COMMAND_BUFFER_MODE_STREAM;
  }
  default_params.allow_inline_execution = FLAG_cuda_allow_inline_execution;

  iree_hal_cuda_driver_options_t driver_options;
  iree_hal_cuda_driver_options_initialize(&driver_options);
  driver_options.default_device_index = FLAG_cuda_default_index;
  // device=cuda://all -> call list_devices=cuda
  // store all devices in array in driver_options

  iree_status_t status =
      iree_hal_cuda_driver_create(driver_name, &default_params, &driver_options,
                                  host_allocator, out_driver);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t
iree_hal_cuda_driver_module_register(iree_hal_driver_registry_t* registry) {
  static const iree_hal_driver_factory_t factory = {
      .self = NULL,
      .enumerate = iree_hal_cuda_driver_factory_enumerate,
      .try_create = iree_hal_cuda_driver_factory_try_create,
  };
  return iree_hal_driver_registry_register_factory(registry, &factory);
}
