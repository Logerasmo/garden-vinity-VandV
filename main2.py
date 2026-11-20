import sys
from pathlib import Path

# предполагается, что normalize.py лежит в той же папке или в sys.path
from normalize import main

exit_code = main([
    "data/devices.csv",        # input (обязательный)
    "-c", "config.json",       # конфиг
    "--inplace",               # писать обратно в active_area / area_unit
    "-o", "data/normalized/",  # выходная папка
    "-v",                      # подробный лог
])
print("exit code:", exit_code)
