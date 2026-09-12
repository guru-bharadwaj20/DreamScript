# 12.2.8 - stage `export` failed

```
Traceback (most recent call last):
  File "C:\Users\Temp\Desktop\DreamScript\src\llm\pipeline.py", line 2635, in run_pipeline
    done = stage.run(ctx)
           ^^^^^^^^^^^^^^
  File "C:\Users\Temp\Desktop\DreamScript\src\llm\pipeline.py", line 2081, in stage_export
    result = run_job(
             ^^^^^^^^
  File "C:\Users\Temp\Desktop\DreamScript\src\llm\pipeline.py", line 171, in run_job
    raise RuntimeError(f"job {name} failed rc={code}:\n{tail}")
RuntimeError: job export failed rc=1:
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "C:\Users\Temp\Desktop\DreamScript\src\llm\pipeline.py", line 2708, in <module>
    sys.exit(main())
             ^^^^^^
  File "C:\Users\Temp\Desktop\DreamScript\src\llm\pipeline.py", line 2692, in main
    return job_main(argv[1], argv[2])
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Temp\Desktop\DreamScript\src\llm\pipeline.py", line 405, in job_main
    result = JOBS[kind](spec)
             ^^^^^^^^^^^^^^^^
  File "C:\Users\Temp\Desktop\DreamScript\src\llm\pipeline.py", line 381, in job_export
    Path(spec["adapter"]),
    ^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Temp\AppData\Roaming\uv\python\cpython-3.11-windows-x86_64-none\Lib\pathlib.py", line 871, in __new__
    self = cls._from_parts(args)
           ^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Temp\AppData\Roaming\uv\python\cpython-3.11-windows-x86_64-none\Lib\pathlib.py", line 509, in _from_parts
    drv, root, parts = self._parse_args(args)
                       ^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Temp\AppData\Roaming\uv\python\cpython-3.11-windows-x86_64-none\Lib\pathlib.py", line 493, in _parse_args
    a = os.fspath(a)
        ^^^^^^^^^^^^
TypeError: expected str, bytes or os.PathLike object, not NoneType


```
