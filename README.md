# Shumka Analyzer

This repository contains `analyzer.py` for extracting noisy segments from video sources. Helper scripts are provided to run the analyzer on different platforms.

## Usage

### Unix

```bash
./run_analyzer.sh <path_to_source>
```

### Windows

```bat
run_analyzer.bat <path_to_source>
```

You may also set the environment variable `ANALYZER_SOURCE` with the path to the folder or archive and run the scripts without arguments. If neither an argument nor the environment variable is provided, the scripts display this usage hint.

Additional arguments are forwarded to `analyzer.py`, which falls back to its own default settings when not specified.
