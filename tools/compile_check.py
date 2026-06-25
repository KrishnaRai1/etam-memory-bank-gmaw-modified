import py_compile
files=['src/run_benchmark.py','src/benchmark/data_validation.py','src/benchmark/mask_metrics.py','src/benchmark/track_metrics.py','tools/check_frame_alignment.py']
for f in files:
    try:
        py_compile.compile(f, doraise=True)
        print('OK', f)
    except Exception as e:
        print('ERR', f, e)
