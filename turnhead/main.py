import os
from pathlib import Path

# BPJDet imports matplotlib. Keep its cache within this project as well.
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).parent / '.cache' / 'matplotlib'))

from src.pipeline import cli

if __name__ == '__main__':
    cli()
