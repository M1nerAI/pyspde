env=pypi

rm -rf dist/

source /c/Users/jimen/miniconda3/etc/profile.d/conda.sh
conda activate general

# Build
python -m build

# Install twine
python -m pip install --upgrade twine

# Upload
python -m twine upload --config-file ~/.pyirc --repository $env dist/* 
