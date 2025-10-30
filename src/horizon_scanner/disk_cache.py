# disk_cache.py  ───────────────────────────────────────────────────────────
"""
Very small decorator that stores a function's return value as JSON on disk
keyed by a hash of its arguments. Stamp-and-go: no database, no expiry.
Creates a cache directory in a writable location.
"""
import json, hashlib, pathlib, functools, os, tempfile

# Try to find a writable cache directory
def _get_cache_dir():
    """Get a writable cache directory, trying multiple locations."""
    # 1. Check for explicit environment variable
    if "HORIZON_SCANNER_CACHE_DIR" in os.environ:
        cache_dir = pathlib.Path(os.environ["HORIZON_SCANNER_CACHE_DIR"])
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            return cache_dir
        except (PermissionError, OSError):
            pass
    
    # 2. Try temp directory
    try:
        cache_dir = pathlib.Path(tempfile.gettempdir()) / "horizon_scanner_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
    except (PermissionError, OSError):
        pass
    
    # 3. Try home directory
    try:
        cache_dir = pathlib.Path.home() / ".cache" / "horizon_scanner"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
    except (PermissionError, OSError):
        pass
    
    # 4. Fall back to current directory (original behavior)
    try:
        cache_dir = pathlib.Path(".cache")
        cache_dir.mkdir(exist_ok=True)
        return cache_dir
    except (PermissionError, OSError):
        # If all else fails, use a temp directory without persistence
        return pathlib.Path(tempfile.mkdtemp(prefix="horizon_scanner_cache_"))

CACHE_DIR = _get_cache_dir()

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
