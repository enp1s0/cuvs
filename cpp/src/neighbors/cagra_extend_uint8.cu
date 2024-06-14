/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cagra.cuh"
#include <cuvs/neighbors/cagra.hpp>

namespace cuvs::neighbors::cagra {

#define RAFT_INST_CAGRA_EXTEND(T, IdxT)                                                       \
  void extend(raft::resources const& handle,                                                  \
              raft::device_matrix_view<const T, int64_t, raft::row_major> additional_dataset, \
              cuvs::neighbors::cagra::index<T, IdxT>& idx,                                    \
              const cagra::extend_params& params,                                             \
              const extend_memory_buffers<T, IdxT>& mb)                                       \
  {                                                                                           \
    cuvs::neighbors::cagra::extend<T, IdxT>(handle, additional_dataset, idx, params, mb);     \
  }                                                                                           \
                                                                                              \
  void extend(raft::resources const& handle,                                                  \
              raft::host_matrix_view<const T, int64_t, raft::row_major> additional_dataset,   \
              cuvs::neighbors::cagra::index<T, IdxT>& idx,                                    \
              const cagra::extend_params& params,                                             \
              const extend_memory_buffers<T, IdxT>& mb)                                       \
  {                                                                                           \
    cuvs::neighbors::cagra::extend<T, IdxT>(handle, additional_dataset, idx, params, mb);     \
  }

RAFT_INST_CAGRA_EXTEND(uint8_t, uint32_t);

#undef RAFT_INST_CAGRA_EXTEND

}  // namespace cuvs::neighbors::cagra