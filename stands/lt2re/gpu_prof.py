import cProfile, pstats, sys, runpy
sys.argv = ["gpu_dump_test.py", "0"]
cProfile.run('runpy.run_path("gpu_dump_test.py", run_name="__main__")', "/home/trim/wblogs/gpu_prof.out")
p = pstats.Stats("/home/trim/wblogs/gpu_prof.out"); p.sort_stats("tottime").print_stats(30)
