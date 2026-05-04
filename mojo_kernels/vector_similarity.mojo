# Mojo vector similarity kernels.
#
# This file is intentionally standalone so the Python API can run even when Mojo is absent.
# The Python control plane can call these kernels through a thin bridge once a stable FFI/CLI
# contract is selected.

from math import sqrt


fn dot_product(a: Pointer[Float64], b: Pointer[Float64], n: Int) -> Float64:
    var acc = 0.0
    for i in range(n):
        acc += a[i] * b[i]
    return acc


fn vector_norm(a: Pointer[Float64], n: Int) -> Float64:
    return sqrt(dot_product(a, a, n))


fn cosine_similarity(a: Pointer[Float64], b: Pointer[Float64], n: Int) -> Float64:
    var denom = vector_norm(a, n) * vector_norm(b, n)
    if denom == 0.0:
        return 0.0
    return dot_product(a, b, n) / denom


fn write_cosine_scores(
    query: Pointer[Float64],
    vectors: Pointer[Float64],
    vector_count: Int,
    dims: Int,
    out_scores: Pointer[Float64],
):
    for row in range(vector_count):
        var dot = 0.0
        var query_norm_sq = 0.0
        var vector_norm_sq = 0.0
        for col in range(dims):
            var q = query[col]
            var v = vectors[row * dims + col]
            dot += q * v
            query_norm_sq += q * q
            vector_norm_sq += v * v

        var denom = sqrt(query_norm_sq) * sqrt(vector_norm_sq)
        if denom == 0.0:
            out_scores[row] = 0.0
        else:
            out_scores[row] = dot / denom


fn argmax_score(scores: Pointer[Float64], count: Int) -> Int:
    if count <= 0:
        return -1

    var best_index = 0
    var best_score = scores[0]
    for i in range(1, count):
        if scores[i] > best_score:
            best_index = i
            best_score = scores[i]
    return best_index


fn main():
    print("Mojo vector similarity kernels ready")
