"""FastAPI service exposing data profiling and the reasoning engine over HTTP.

This is the thin layer the future frontend (and anyone scripting against
reclab) talks to. It should stay a thin wrapper — no logic here that belongs
in `data_profiler` or `reasoning_engine`.
"""
