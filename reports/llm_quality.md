# 12.3.1, 12.3.2, 12.3.3, 12.3.8 - stage `quality` failed

```
Traceback (most recent call last):
  File "C:\Users\Temp\Desktop\DreamScript\src\llm\pipeline.py", line 2635, in run_pipeline
    done = stage.run(ctx)
           ^^^^^^^^^^^^^^
  File "C:\Users\Temp\Desktop\DreamScript\src\llm\pipeline.py", line 1658, in stage_quality
    ref_scored, ref_summary = score(
                              ^^^^^^
  File "C:\Users\Temp\Desktop\DreamScript\src\llm\pipeline.py", line 487, in score
    scored = scoring.score_rows(rows, split, workers=workers)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Temp\Desktop\DreamScript\src\llm\score.py", line 306, in score_rows
    return list(pool.map(job, rows))
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Temp\AppData\Roaming\uv\python\cpython-3.11-windows-x86_64-none\Lib\concurrent\futures\_base.py", line 619, in result_iterator
    yield _result_or_cancel(fs.pop())
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Temp\AppData\Roaming\uv\python\cpython-3.11-windows-x86_64-none\Lib\concurrent\futures\_base.py", line 317, in _result_or_cancel
    return fut.result(timeout)
           ^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Temp\AppData\Roaming\uv\python\cpython-3.11-windows-x86_64-none\Lib\concurrent\futures\_base.py", line 456, in result
    return self.__get_result()
           ^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Temp\AppData\Roaming\uv\python\cpython-3.11-windows-x86_64-none\Lib\concurrent\futures\_base.py", line 401, in __get_result
    raise self._exception
  File "C:\Users\Temp\AppData\Roaming\uv\python\cpython-3.11-windows-x86_64-none\Lib\concurrent\futures\thread.py", line 58, in run
    result = self.fn(*self.args, **self.kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Temp\Desktop\DreamScript\src\llm\score.py", line 303, in job
    return score_one(row, pair, pairs_mod.diagram_for(pair))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Temp\Desktop\DreamScript\src\llm\score.py", line 278, in score_one
    sig = functional.signature(code, diagram)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Temp\Desktop\DreamScript\src\llm\functional.py", line 544, in signature
    return json.loads(stdout.rsplit(marker, 1)[1].strip().splitlines()[0])
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Temp\AppData\Roaming\uv\python\cpython-3.11-windows-x86_64-none\Lib\json\__init__.py", line 346, in loads
    return _default_decoder.decode(s)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Temp\AppData\Roaming\uv\python\cpython-3.11-windows-x86_64-none\Lib\json\decoder.py", line 337, in decode
    obj, end = self.raw_decode(s, idx=_w(s, 0).end())
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Temp\AppData\Roaming\uv\python\cpython-3.11-windows-x86_64-none\Lib\json\decoder.py", line 353, in raw_decode
    obj, end = self.scan_once(s, idx)
               ^^^^^^^^^^^^^^^^^^^^^^
json.decoder.JSONDecodeError: Unterminated string starting at: line 1 column 1048565 (char 1048564)

```
