import json_parser.parser as parser
from json_parser.mapper import gaze_data_from_dict

def main():
    data = parser.parse()
    gaze_data = gaze_data_from_dict(data)

if __name__ == '__main__':
    main()