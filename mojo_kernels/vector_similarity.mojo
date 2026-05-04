# Mojo vector similarity scaffold.
# This file is intentionally standalone so the Python API can run even when Mojo is absent.

fn dot_product(a: Pointer[Float64], b: Pointer[Float64], n: Int) -> Float64:
    var acc = 0.0
    for i in range(n):
        acc += a[i] * b[i]
    return acc

fn main():
    print("Mojo vector kernel scaffold ready")
