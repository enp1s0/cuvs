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
#include "detail/cagra/cagra_build.cuh"
#include <cuvs/neighbors/cagra.hpp>

namespace cuvs::neighbors::cagra {

#define RAFT_INST_CAGRA_BUILD(T, IdxT)                                                      \
  auto build(raft::resources const& handle,                                                 \
             const cuvs::neighbors::cagra::index_params& params,                            \
             raft::device_matrix_view<const T, int64_t, raft::row_major> dataset)           \
    ->cuvs::neighbors::cagra::index<T, IdxT>                                                \
  {                                                                                         \
    return cuvs::neighbors::cagra::build<T, IdxT>(handle, params, dataset);                 \
  }                                                                                         \
                                                                                            \
  auto build(raft::resources const& handle,                                                 \
             const cuvs::neighbors::cagra::index_params& params,                            \
             raft::host_matrix_view<const T, int64_t, raft::row_major> dataset)             \
    ->cuvs::neighbors::cagra::index<T, IdxT>                                                \
  {                                                                                         \
    return cuvs::neighbors::cagra::build<T, IdxT>(handle, params, dataset);                 \
  }                                                                                         \
                                                                                            \
  void build_knn_graph(raft::resources const& res,                                          \
                       raft::host_matrix_view<const T, int64_t, raft::row_major> dataset,   \
                       raft::host_matrix_view<IdxT, int64_t, raft::row_major> knn_graph,    \
                       cuvs::neighbors::nn_descent::index_params build_params)              \
  {                                                                                         \
    cuvs::neighbors::cagra::detail::build_knn_graph(res, dataset, knn_graph, build_params); \
  }                                                                                         \
  void build_knn_graph(raft::resources const& res,                                          \
                       raft::device_matrix_view<const T, int64_t, raft::row_major> dataset, \
                       raft::host_matrix_view<IdxT, int64_t, raft::row_major> knn_graph,    \
                       cuvs::neighbors::nn_descent::index_params build_params)              \
  {                                                                                         \
    cuvs::neighbors::cagra::detail::build_knn_graph(res, dataset, knn_graph, build_params); \
  }

RAFT_INST_CAGRA_BUILD(float, uint32_t);

#undef RAFT_INST_CAGRA_BUILD

void optimize(raft::resources const& res,
              raft::host_matrix_view<const uint32_t, int64_t, raft::row_major> knn_graph,
              raft::host_matrix_view<uint32_t, int64_t, raft::row_major> new_graph,
              const bool guarantee_connectivity)
{
  auto input_knn_graph_unconst = raft::make_device_matrix_view(
    const_cast<uint32_t*>(knn_graph.data_handle()), knn_graph.extent(0), knn_graph.extent(1));
  cuvs::neighbors::cagra::detail::optimize(
    res, input_knn_graph_unconst, new_graph, guarantee_connectivity);
}

void optimize(raft::resources const& res,
              raft::device_matrix_view<const uint32_t, int64_t, raft::row_major> knn_graph,
              raft::host_matrix_view<uint32_t, int64_t, raft::row_major> new_graph,
              const bool guarantee_connectivity)
{
  auto input_knn_graph_unconst = raft::make_device_matrix_view(
    const_cast<uint32_t*>(knn_graph.data_handle()), knn_graph.extent(0), knn_graph.extent(1));
  cuvs::neighbors::cagra::detail::optimize(
    res, input_knn_graph_unconst, new_graph, guarantee_connectivity);
}
}  // namespace cuvs::neighbors::cagra
