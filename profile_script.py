import cProfile
from benchmark_local import run_benchmark

cProfile.run('run_benchmark(80, max_iter=3)', 'stats.prof')
