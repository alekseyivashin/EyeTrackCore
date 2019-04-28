# To use this code, make sure you
#
#     import json
#
# and then, to convert JSON from a string, do
#
#     result = gaze_data_from_dict(json.loads(json_string))

from dataclasses import dataclass
from typing import Any, List, TypeVar, Type, cast, Callable

T = TypeVar("T")


def from_float(x: Any) -> float:
    assert isinstance(x, (float, int)) and not isinstance(x, bool)
    return float(x)


def from_union(fs, x):
    for f in fs:
        try:
            return f(x)
        except:
            pass
    assert False


def to_float(x: Any) -> float:
    assert isinstance(x, float)
    return x


def from_int(x: Any) -> int:
    assert isinstance(x, int) and not isinstance(x, bool)
    return x


def to_class(c: Type[T], x: Any) -> dict:
    assert isinstance(x, c)
    return cast(Any, x).to_dict()


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

    def to_dict(self) -> dict:
        result: dict = {}
        result["X"] = to_float(self.x)
        result["Y"] = to_float(self.y)
        result["Z"] = to_float(self.z)
        return result


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

    def to_dict(self) -> dict:
        result: dict = {}
        result["PositionInUserCoordinates"] = to_class(PositionInCoordinates, self.position_in_user_coordinates)
        result["PositionInTrackBoxCoordinates"] = to_class(PositionInCoordinates,
                                                           self.position_in_track_box_coordinates)
        result["Validity"] = from_int(self.validity)
        return result


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

    def to_dict(self) -> dict:
        result: dict = {}
        result["X"] = to_float(self.x)
        result["Y"] = to_float(self.y)
        return result


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

    def to_dict(self) -> dict:
        result: dict = {}
        result["PositionOnDisplayArea"] = to_class(PositionOnDisplayArea, self.position_on_display_area)
        result["PositionInUserCoordinates"] = to_class(PositionInCoordinates, self.position_in_user_coordinates)
        result["Validity"] = from_int(self.validity)
        return result


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

    def to_dict(self) -> dict:
        result: dict = {}
        result["GazePoint"] = to_class(GazePoint, self.gaze_point)
        result["GazeOrigin"] = to_class(GazeOrigin, self.gaze_origin)
        return result


@dataclass
class GazeData:
    left_eye: TEye
    right_eye: TEye
    device_time_stamp: int
    system_time_stamp: int

    @staticmethod
    def from_dict(obj: Any) -> 'GazeData':
        assert isinstance(obj, dict)
        left_eye = TEye.from_dict(obj.get("LeftEye"))
        right_eye = TEye.from_dict(obj.get("RightEye"))
        device_time_stamp = from_int(obj.get("DeviceTimeStamp"))
        system_time_stamp = from_int(obj.get("SystemTimeStamp"))
        return GazeData(left_eye, right_eye, device_time_stamp, system_time_stamp)

    def to_dict(self) -> dict:
        result: dict = {}
        result["LeftEye"] = to_class(TEye, self.left_eye)
        result["RightEye"] = to_class(TEye, self.right_eye)
        result["DeviceTimeStamp"] = from_int(self.device_time_stamp)
        result["SystemTimeStamp"] = from_int(self.system_time_stamp)
        return result


def gaze_data_from_dict(s: Any) -> List[GazeData]:
    return from_list(GazeData.from_dict, s)


def gaze_data_to_dict(x: List[GazeData]) -> Any:
    return from_list(lambda x: to_class(GazeData, x), x)
