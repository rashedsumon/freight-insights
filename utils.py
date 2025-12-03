# utils.py
# Small helpers used by app.py and others.

import time
from functools import wraps
import streamlit as st


def cached_run(ttl_seconds=300):
    """
    Simple decorator for caching in-memory across Streamlit sessions.
    If deploying multiple workers, replace with Redis or other external cache.
    """
    def decorator(func):
        cache_key = f"_cache_{func.__name__}"

        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            cache = st.session_state.get(cache_key, {})
            if cache and cache.get("ts") and (now - cache["ts"] < ttl_seconds):
                return cache["val"]
            val = func(*args, **kwargs)
            st.session_state[cache_key] = {"ts": now, "val": val}
            return val
        return wrapper
    return decorator
