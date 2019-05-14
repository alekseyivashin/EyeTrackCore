# To use this code, make sure you
#
#     import json
#
# and then, to convert JSON from a string, do
#
#     result = gaze_data_from_dict(json.loads(json_string))

from dataclasses import dataclass
from typing import Any, List, TypeVar, Callable

T = TypeVar("T")


def from_float(x: Any) -> float:
    assert isinstance(x, (float, int)) and not isinstance(x, bool)
    return float(x)


def from_int(x: Any) -> int:
    assert isinstance(x, int) and not isinstance(x, bool)
    return x


def from_list(f: Callable[[Any], T], x: Any) -> List[T]:
    assert isinstance(x, list)
    result_list = []
    for y in x:
        try:
            result_list.append(f(y))
        except AssertionError:
            continue
    return result_list


@dataclass
class PositionInCoordinates:
    x: float
    y: float
    z: float

    @staticmethod
    def from_dict(obj: Any) -> 'PositionInCoordinates':
        assert isinstance(obj, dict)
        assert isinstance(obj.get("X"), float)
        assert isinstance(obj.get("Y"), float)
        assert isinstance(obj.get("Z"), float)
        x = from_float(obj.get("X"))
        y = from_float(obj.get("Y"))
        z = from_float(obj.get("Z"))
        return PositionInCoordinates(x, y, z)


@dataclass
class GazeOrigin:
    position_in_user_coordinates: PositionInCoordinates
    position_in_track_box_coordinates: PositionInCoordinates
    validity: int

    @staticmethod
    def from_dict(obj: Any) -> 'GazeOrigin':
        assert isinstance(obj, dict)
        position_in_user_coordinates = PositionInCoordinates.from_dict(obj.get("PositionInUserCoordinates"))
        position_in_track_box_coordinates = PositionInCoordinates.from_dict(obj.get("PositionInTrackBoxCoordinates"))
        validity = from_int(obj.get("Validity"))
        return GazeOrigin(position_in_user_coordinates, position_in_track_box_coordinates, validity)


@dataclass
class PositionOnDisplayArea:
    x: float
    y: float

    @staticmethod
    def from_dict(obj: Any) -> 'PositionOnDisplayArea':
        assert isinstance(obj, dict)
        assert isinstance(obj.get("X"), float)
        assert isinstance(obj.get("Y"), float)
        x = from_float(obj.get("X"))
        y = from_float(obj.get("Y"))
        return PositionOnDisplayArea(x, y)


@dataclass
class GazePoint:
    position_on_display_area: PositionOnDisplayArea
    position_in_user_coordinates: PositionInCoordinates
    validity: int

    @staticmethod
    def from_dict(obj: Any) -> 'GazePoint':
        assert isinstance(obj, dict)
        position_on_display_area = PositionOnDisplayArea.from_dict(obj.get("PositionOnDisplayArea"))
        position_in_user_coordinates = PositionInCoordinates.from_dict(obj.get("PositionInUserCoordinates"))
        validity = from_int(obj.get("Validity"))
        return GazePoint(position_on_display_area, position_in_user_coordinates, validity)


@dataclass
class TEye:
    gaze_point: GazePoint
    gaze_origin: GazeOrigin

    @staticmethod
    def from_dict(obj: Any) -> 'TEye':
        assert isinstance(obj, dict)
        gaze_point = GazePoint.from_dict(obj.get("GazePoint"))
        gaze_origin = GazeOrigin.from_dict(obj.get("GazeOrigin"))
        return TEye(gaze_point, gaze_origin)


@dataclass
class GazeData:
    left_eye: TEye
    right_eye: TEye
    average_display_coordinate: PositionOnDisplayArea
    average_user_coordinate: PositionInCoordinates
    device_time_stamp: int
    system_time_stamp: int

    @staticmethod
    def from_dict(obj: Any) -> 'GazeData':
        assert isinstance(obj, dict)
        left_eye = TEye.from_dict(obj.get("LeftEye"))
        right_eye = TEye.from_dict(obj.get("RightEye"))
        average_display_coordinate = PositionOnDisplayArea(
            (left_eye.gaze_point.position_on_display_area.x + right_eye.gaze_point.position_on_display_area.x) / 2,
            (left_eye.gaze_point.position_on_display_area.y + right_eye.gaze_point.position_on_display_area.y) / 2,
        )
        average_user_coordinate = PositionInCoordinates(
            (
                        left_eye.gaze_origin.position_in_user_coordinates.x + right_eye.gaze_origin.position_in_user_coordinates.x) / 2,
            (
                        left_eye.gaze_origin.position_in_user_coordinates.y + right_eye.gaze_origin.position_in_user_coordinates.y) / 2,
            (
                        left_eye.gaze_origin.position_in_user_coordinates.z + right_eye.gaze_origin.position_in_user_coordinates.z) / 2
        )
        device_time_stamp = from_int(obj.get("DeviceTimeStamp"))
        system_time_stamp = from_int(obj.get("SystemTimeStamp"))
        return GazeData(left_eye, right_eye, average_display_coordinate, average_user_coordinate, device_time_stamp,
                        system_time_stamp)


def gaze_data_from_dict(s: Any) -> List[GazeData]:
    return from_list(GazeData.from_dict, s)
