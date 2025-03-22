#pragma once

#include "cagra.hpp"

namespace cuvs::neighbors::cagra {
void build_knn_graph(raft::resources const& res,
                     raft::device_matrix_view<const float, int64_t, raft::row_major> dataset,
                     raft::host_matrix_view<uint32_t, int64_t, raft::row_major> knn_graph,
                     cuvs::neighbors::nn_descent::index_params build_params);

void build_knn_graph(raft::resources const& res,
                     raft::device_matrix_view<const float, int64_t, raft::row_major> dataset,
                     raft::device_matrix_view<uint32_t, int64_t, raft::row_major> knn_graph,
                     cuvs::neighbors::nn_descent::index_params build_params);

void optimize(raft::resources const& res,
              raft::host_matrix_view<uint32_t, int64_t, raft::row_major> knn_graph,
              raft::host_matrix_view<uint32_t, int64_t, raft::row_major> new_graph,
              const bool guarantee_connectivity = false);

void optimize(raft::resources const& res,
              raft::device_matrix_view<uint32_t, int64_t, raft::row_major> knn_graph,
              raft::host_matrix_view<uint32_t, int64_t, raft::row_major> new_graph,
              const bool guarantee_connectivity = false);

}  // namespace cuvs::neighbors::cagra
