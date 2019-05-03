import json
import os
from typing import Any, AnyStr, IO


def parse() -> Any:
    file = open_file("Алексей63688521416324.json")
    return json.load(file)


def open_file(file_name: AnyStr) -> IO:
    os.chdir("..")
    files = os.listdir("resources")
    if file_name in files:
        file = open(os.path.join(os.getcwd(), "resources", file_name), "r")
        os.chdir("src")
        return file
    raise FileNotFoundError(f"File {file_name} is not found!")