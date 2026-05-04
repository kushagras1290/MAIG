import argparse
import random
import time

from mojo_ai_gateway.vector_math import top_k_cosine


def random_vector(dims: int) -> list[float]:
    return [random.uniform(-1.0, 1.0) for _ in range(dims)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors", type=int, default=5000)
    parser.add_argument("--dims", type=int, default=768)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    random.seed(7)
    query = random_vector(args.dims)
    vectors = [random_vector(args.dims) for _ in range(args.vectors)]

    start = time.perf_counter()
    result = top_k_cosine(query, vectors, args.top_k)
    elapsed = time.perf_counter() - start

    print({"vectors": args.vectors, "dims": args.dims, "top_k": args.top_k, "seconds": elapsed, "best": result[0]})


if __name__ == "__main__":
    main()
