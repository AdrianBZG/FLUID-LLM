CONFIG_PATH="configs/training1.json"
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python src/main.py --config_path=$CONFIG_PATH