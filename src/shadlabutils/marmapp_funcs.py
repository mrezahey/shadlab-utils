import json
import logging
import struct
from abc import ABC
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)


@dataclass(repr=False)
class BaseDataClass(ABC):
    def __str__(self) -> str:
        def convert(obj):
            if isinstance(obj, Enum):
                return obj.value
            return obj

        d = {"type": self.__class__.__name__}
        raw = asdict(self)
        d.update({k: convert(v) for k, v in raw.items()})
        return json.dumps(d, ensure_ascii=False)


@dataclass()
class EyeTrackerData(BaseDataClass):
    time_received: float
    time_device: float

    left_raw_x: float
    left_raw_y: float
    left_cal_x: float
    left_cal_y: float

    left_cal_vx: float
    left_cal_vy: float

    right_raw_x: float
    right_raw_y: float
    right_cal_x: float
    right_cal_y: float

    right_cal_vx: float
    right_cal_vy: float

    left_pupil: float
    right_pupil: float

    left_blink: bool
    right_blink: bool

    din: int


def read_compressed_values(file_path: str | Path) -> list[Union[str, float, bool, int]]:
    results: list[Union[str, float, bool, int]] = []
    file_path = Path(file_path)

    with file_path.open("rb") as f:
        version = int.from_bytes(f.read(1), byteorder="big", signed=False)

        if version == 1:
            while True:
                length_bytes = f.read(4)
                if not length_bytes:
                    break  # end of file

                if len(length_bytes) != 4:
                    return results

                block_len = struct.unpack(">I", length_bytes)[0]
                payload = f.read(block_len)

                if len(payload) != block_len:
                    logger.warning("Corrupted file: incomplete compressed block, skipping the rest of the file")
                    return results

                d_ = _decode_value(payload, version=version)
                if d_ is not None:
                    results.append(d_)

    return results


def _decode_value(payload: bytes, version: int) -> Union[str, float, bool, int, None]:
    if version == 1:
        if not payload:
            logger.warning("Empty payload")
            return None

        type_tag = payload[0:1]
        body = payload[1:]

        if type_tag == b"b":
            if len(body) != 1:
                logger.warning("Invalid bool payload")
                return None
            return struct.unpack(">?", body)[0]

        if type_tag == b"f":
            if len(body) != 8:
                logger.warning("Invalid float payload")
                return None
            return struct.unpack(">d", body)[0]

        if type_tag == b"i":
            length = struct.unpack(">I", body[:4])[0]
            int_bytes = body[4:4 + length]
            return int.from_bytes(int_bytes, byteorder="big", signed=True)

        if type_tag == b"s":
            if len(body) < 4:
                logger.warning("Invalid string payload")
                return None
            str_len = struct.unpack(">I", body[:4])[0]
            str_bytes = body[4:4 + str_len]
            if len(str_bytes) != str_len:
                logger.warning("Invalid string payload length")
            return str_bytes.decode("utf-8")

        logger.warning(f"Unknown type tag: {type_tag!r}")
    return None
