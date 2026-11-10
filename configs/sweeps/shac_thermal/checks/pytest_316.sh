#!/bin/bash
# The full test suite on the mjlab316 stack (CPU node).
cd "$SLURM_SUBMIT_DIR"
source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate mjlab316
python -m pytest -q -rfE -p no:cacheprovider ${PYTEST_ARGS:-tests/} 2>&1 | grep -v "Warning:\|warn(\|warnings.warn" 
