from pathlib import Path

PROJECT_DIR = Path(__file__).parent.absolute()
DATA_DIR = PROJECT_DIR / 'data'
EXAMPLES_DIR = PROJECT_DIR / 'examples'
EXPERIMENTS_DIR = PROJECT_DIR / 'experiments'
STORAGE_DIR = PROJECT_DIR / 'storage'
CONFIGS_DIR = PROJECT_DIR / 'configs'

DATASETS_DIR = STORAGE_DIR / 'datasets'
EXAMPLES_DATA_DIR = EXAMPLES_DIR / 'data'
EXPERIMENTS_OUTPUT_DIR = STORAGE_DIR / 'output' / 'experiments'
MODELS_DIR = STORAGE_DIR / 'models'

AZURE_CONFIG_FILE_PATH = PROJECT_DIR / 'config.json'
