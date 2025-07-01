# disk_cache.py  ───────────────────────────────────────────────────────────
"""
Very small decorator that stores a function’s return value as JSON on disk
keyed by a hash of its arguments. Stamp-and-go: no database, no expiry.
Creates a local .cache/ directory next to this file.
"""
import json, hashlib, pathlib, functools

CACHE_DIR = pathlib.Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)           # make ./ .cache  if missing

def _key(hashable) -> str:
    raw = json.dumps(hashable, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha1(raw).hexdigest()[:16] + ".json"

def disk_memoize(fn):
    """Decorator: cache fn(args, kwargs) -> result as JSON on disk."""
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        fname = CACHE_DIR / f"{fn.__name__}_{_key([args, kwargs])}"
        if fname.exists():
            return json.loads(fname.read_text(encoding="utf-8"))
        out = fn(*args, **kwargs)
        fname.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        return out
    return wrapped
