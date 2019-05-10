import json
import os
from typing import Any, AnyStr, IO, List, Dict

from json_parser.mapper import gaze_data_from_dict, GazeData


def parse_all() -> Dict[str, List[List[GazeData]]]:
    result = {}
    for file_name in get_file_names():
        file_name_parts = file_name.split("_")
        name = file_name_parts[0]
        index = int(file_name_parts[1])
        if not name in result:
            result[name] = [[]] * 6
        data = parse_file(file_name)
        result[name][index - 1] = data
    return result


def parse_file(filename: str) -> List[GazeData]:
    file = open_file(filename)
    json_content = json.load(file)
    return gaze_data_from_dict(json_content)


def open_file(file_name: AnyStr) -> IO:
    if file_name in get_file_names():
        return open(os.path.join(os.getcwd(), "..", "resources", file_name), "r")
    raise FileNotFoundError(f"File {file_name} is not found!")


def get_file_names() -> List[str]:
    os.chdir("..")
    file_names = os.listdir("resources")
    os.chdir("src")
    return file_names
