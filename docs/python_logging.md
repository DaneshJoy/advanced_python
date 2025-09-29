# Logging — short & practical

Below are four tiny examples you can drop into your EEG generator project. Replace the toy `generate_eeg()` with your real generator.

------

## 1) Simple logging

Minimal setup — prints INFO+ to console.

```python
import logging

logging.basicConfig(level=logging.INFO)        # call once at program start
logger = logging.getLogger(__name__)

def generate_eeg(n=5):
    for i in range(n):
        sample = f"eeg_sample_{i}"
        logger.info("generated sample %d", i)
        yield sample

if __name__ == "__main__":
    for s in generate_eeg(3):
        print(s)
```

------

## 2) Using a decorator

Decorator that logs entry/exit, exceptions and *works with generator functions* (it yields values through).

```python
import functools
import time
import logging

logger = logging.getLogger(__name__)

def log_calls(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.info("ENTER %s args=%s kwargs=%s", func.__name__, args, kwargs)
        start = time.time()
        try:
            for item in func(*args, **kwargs):     # assumes func is a generator
                logger.debug("YIELD %r", item)
                yield item
            logger.info("EXIT %s (%.3fs)", func.__name__, time.time() - start)
        except Exception:
            logger.exception("EXCEPTION in %s", func.__name__)
            raise
    return wrapper

@log_calls
def generate_eeg(n=3):
    for i in range(n):
        yield f"sample_{i}"

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    for s in generate_eeg(2):
        print(s)
```

------

## 3) Save logs to a file

Simple file-only logging (no console).

```python
import logging

logging.basicConfig(
    filename="eeg.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)

def generate_eeg(n=3):
    logger.info("start generator")
    for i in range(n):
        logger.info("sample %d", i)
        yield f"sample_{i}"

# run — logs go to eeg.log
for _ in generate_eeg(2):
    pass
```

------

## 4) Show in terminal **and** save to file

A small `setup_logging()` that creates both handlers. Call it once at startup.

```python
import logging
from logging import StreamHandler, FileHandler

def setup_logging(logfile="eeg.log", level=logging.DEBUG):
    root = logging.getLogger()
    root.setLevel(level)
    if root.handlers:   # avoid duplicate handlers on re-import
        return root
    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s",
                            "%Y-%m-%d %H:%M:%S")

    fh = FileHandler(logfile)
    fh.setLevel(level)
    fh.setFormatter(fmt)

    ch = StreamHandler()           # console
    ch.setLevel(logging.INFO)      # console less noisy than file
    ch.setFormatter(fmt)

    root.addHandler(fh)
    root.addHandler(ch)
    return root

# use decorator from step 2 (assumes it's available)
if __name__ == "__main__":
    setup_logging()   # must call early
    logger = logging.getLogger(__name__)

    @log_calls
    def generate_eeg(n=4):
        for i in range(n):
            logger.debug("creating sample %d", i)
            yield f"sample_{i}"

    for s in generate_eeg(3):
        print("got:", s)
```

------

That’s it — drop these into your project. If you want a version with `RotatingFileHandler`, json logs, or different formats/levels per module, tell me which and I’ll show the tiny change.

---

Quick breakdown:

------

### `logging.basicConfig(level=logging.INFO)`

- Initializes logging once (usually at program start).
- `level=logging.INFO` → only **INFO and above** (`INFO`, `WARNING`, `ERROR`, `CRITICAL`) get shown.
- Lower-level logs (`DEBUG`) are ignored.

------

### `format="..."` and `datefmt="..."`

This controls how each log line looks.

- `%(asctime)s` → timestamp (formatted by `datefmt`)
- `%(levelname)s` → log level (INFO, DEBUG, ERROR, …)
- `%(name)s` → logger’s name (usually module name via `logging.getLogger(__name__)`)
- `%(message)s` → the actual log text you wrote

`datefmt="%Y-%m-%d %H:%M:%S"` just says: show date/time like `2025-09-27 14:23:10`.

So a log line will look like:

```
2025-09-27 14:23:10 INFO [my_module] generated sample 3
```

------

### Suppressing log levels at runtime?

You can change the level anytime:

```python
logger.setLevel(logging.WARNING)   # now only WARNING+ shown
logger.setLevel(logging.DEBUG)     # show everything
```

Or globally:

```python
logging.getLogger().setLevel(logging.ERROR)
```

So you can dynamically suppress (or enable) logs depending on runtime state (debug mode, prod mode, etc.).

------

## Logging for Packages

The **recommended way** for Python packages.

------

### In your **package** (EEG generator)

- Don’t call `basicConfig()` inside your library.
- Just get a logger and use it:

```python
import logging

logger = logging.getLogger(__name__)

def generate_eeg(n=3):
    for i in range(n):
        logger.info("generating sample %d", i)
        yield f"sample_{i}"
```

That’s it — no config inside the package.

------

### In the **app that uses your package**

The application owns logging configuration. For example:

```python
import logging
from eeg_generator import generate_eeg

logging.basicConfig(
    level=logging.DEBUG,   # app decides level
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

for s in generate_eeg(3):
    print("got:", s)
```

------

### Why?

- If your package calls `basicConfig()` itself, you’d override whatever the app wanted.
- Best practice for libraries: **never configure logging, only log.**
- Best practice for apps: **configure logging** once at startup.

------

✅ So you can set the log level in another app, import your package, and it will respect that.

---

## NullHandler

A tiny pattern (`NullHandler`) usually is added to packages so they stay silent until the app configures logging:

Here’s the usual **Python package logging pattern** with `NullHandler`:

------

### In your **package** (`eeg_generator/__init__.py` or wherever)

```python
import logging

# Create package-level logger
logger = logging.getLogger(__name__)

# Add a NullHandler so "No handler found" warnings don’t appear
logger.addHandler(logging.NullHandler())

def generate_eeg(n=3):
    for i in range(n):
        logger.info("generating sample %d", i)
        yield f"sample_{i}"
```

- `NullHandler` → swallows logs silently until the **app** sets up logging.
- This way, your package won’t pollute stdout or surprise users.

------

### In the **app that uses your package**

```python
import logging
from eeg_generator import generate_eeg

# The app decides how logging works
logging.basicConfig(
    level=logging.DEBUG,  # could be INFO, WARNING, etc.
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

for s in generate_eeg(3):
    print("got:", s)
```

------

👉 Result:

- Your package logs nothing by default.
- If the app configures logging, your package’s logs flow into that config.

------

## Package-wide logger

The **"package-wide logger" style** (`logging.getLogger("eeg_generator")`) so users can tweak just your package’s log level without affecting the rest of the app:

Here’s how you make your **whole package share a single logger namespace** so users can control just your package’s logs:

------

### Inside your **package** (`eeg_generator/__init__.py`)

```python
import logging

# Package-wide logger
logger = logging.getLogger("eeg_generator")
logger.addHandler(logging.NullHandler())   # stays quiet unless app configures

def generate_eeg(n=3):
    for i in range(n):
        logger.info("generating sample %d", i)
        yield f"sample_{i}"
```

If you have submodules, they can all use the same namespace:

```python
# eeg_generator/signals.py
import logging
logger = logging.getLogger("eeg_generator.signals")

def filter_signal(x):
    logger.debug("filtering signal")
    return x
```

------

### In the **app**

```python
import logging
from eeg_generator import generate_eeg
from eeg_generator.signals import filter_signal

# Global logging config
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# BUT: fine-tune just your package’s logger
logging.getLogger("eeg_generator").setLevel(logging.DEBUG)

for s in generate_eeg(2):
    filter_signal(s)
```

------

### ✅ Effect

- By default → only WARNING+ logs (global rule).
- But since we bumped `eeg_generator` to DEBUG, your package logs everything while the rest of the app stays quiet.

------

## Full Example

A **ready-to-drop `logging.py` utility module** inside your package that users can call (`from eeg_generator.logging import setup_logging`) if they want an easy default.

That way they don’t have to write `basicConfig()` themselves unless they want full control.

Here’s an **enhanced `logging.py` utility** with options for logging to **terminal, file, or both**:

------

### 1️⃣ `eeg_generator/logging.py` (enhanced)

```python
import logging
import sys
from pathlib import Path

PACKAGE_NAME = "eeg_generator"

def setup_logging(
    level: int = logging.INFO,
    log_to_console: bool = True,
    log_to_file: str | Path | None = None,
    fmt: str = "%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
):
    """
    Setup package-wide logger.

    Parameters
    ----------
    level : int
        Logging level (default: INFO)
    log_to_console : bool
        Whether to log to terminal (default: True)
    log_to_file : str or Path, optional
        File path to log to (default: None)
    fmt : str
        Log message format
    datefmt : str
        Datetime format
    """
    logger = logging.getLogger(PACKAGE_NAME)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        logger.addHandler(console_handler)

    # File handler
    if log_to_file:
        log_path = Path(log_to_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str | None = None):
    """
    Return a logger in this package namespace. If name is None, returns the root package logger.
    """
    if name:
        return logging.getLogger(f"{PACKAGE_NAME}.{name}")
    return logging.getLogger(PACKAGE_NAME)
```

------

### 2️⃣ `eeg_generator/__init__.py`

```python
from .logging import get_logger, setup_logging

# Package-wide logger; lazy default (quiet unless configured)
logger = get_logger()
logger.addHandler(logging.NullHandler())

# Optional: auto-setup default logging (console only, INFO level)
# Uncomment if you want the package to always log by default
# setup_logging(level=logging.INFO, log_to_console=True)
```

------

### 3️⃣ Submodules (`signals.py`, `generator.py`, etc.)

```python
from eeg_generator.logging import get_logger

logger = get_logger(__name__)  # "eeg_generator.signals"

def filter_signal(x):
    logger.debug("Filtering signal")
    return x
```

------

### 4️⃣ Usage in an app

```python
from eeg_generator import generate_eeg
from eeg_generator.signals import filter_signal
from eeg_generator.logging import setup_logging

# Configure logging: both console and file
setup_logging(level=logging.DEBUG, log_to_console=True, log_to_file="logs/eeg.log")

for s in generate_eeg(2):
    filter_signal(s)
```

### ✅ Usage examples

**Log to terminal only:**

```python
from eeg_generator.logging import setup_logging

setup_logging(level=logging.DEBUG, log_to_console=True, log_to_file=None)
```

**Log to a file only:**

```python
setup_logging(level=logging.INFO, log_to_console=False, log_to_file="logs/eeg.log")
```

**Log to both terminal and file:**

```python
setup_logging(level=logging.DEBUG, log_to_console=True, log_to_file="logs/eeg.log")
```

✅ **Benefits**

- Single package-wide logger namespace.
- Users can choose console, file, or both.
- Submodules automatically inherit correct logger names (`eeg_generator.submodule`).
- Clears old handlers to avoid duplicate logs.
- Drop-in default logging works without forcing `basicConfig()` in the app.

------

If you want, I can also make a **one-liner “auto default logger”** in `__init__.py` that logs to console **INFO** by default **only if no logger exists**, so apps get sensible logs automatically without calling `setup_logging()`. Do you want me to do that?