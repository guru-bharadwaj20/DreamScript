"""Phase 16.1 - the app backend.

    uvicorn app.backend.main:app --host 0.0.0.0 --port 3000

The three things 15.11's docstring says are *not* in the model server live here: streaming
stage-by-stage progress (16.1.2), the hardened sandbox runner (16.1.3), and rate and size limits
(16.1.4). What is also here, and is the reason this is a service rather than a client library:
the request identity a phone needs in order to upload a photograph once and then read the IR, the
code and a regeneration off it, and the correction log that makes 16.2.8's taps training data.
"""
