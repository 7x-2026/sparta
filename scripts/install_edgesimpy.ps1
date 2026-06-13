$ErrorActionPreference = "Stop"

python -m pip install -r requirements-edgesimpy.txt
python src\simulation\check_edgesimpy_install.py
