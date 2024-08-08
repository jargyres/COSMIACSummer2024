import sys
from typing import cast

import numpy as np

PREAMBLE = [0, 1, 0]
START_FRAME_DELIMITER = [1] * 8


PACKET_INDEX_BYTES = 4


def create_packet(payload: bytes, packet_idx: int) -> list[int]:
    packet_idx_bytes = packet_idx.to_bytes(PACKET_INDEX_BYTES, sys.byteorder)

    # print("Preamble:{}\nStartFrameDelimiter:{}\nLenPayload:{}\nPacketIndex:{}\nPayload:{}".format(PREAMBLE, START_FRAME_DELIMITER, _bytes_to_bits([len(payload)]), _bytes_to_bits(list(packet_idx_bytes)), _bytes_to_bits(list(payload))))

    return (
        PREAMBLE
        + START_FRAME_DELIMITER
        + _bytes_to_bits([len(payload)] + list(packet_idx_bytes) + list(payload))
    )


def extract_payload(bits: list[int]) -> tuple[int, bytes]:
    bits_str = "".join(str(b) for b in bits)
    sfd_str = "".join(str(b) for b in START_FRAME_DELIMITER)

    try:
        frame_start_idx = bits_str.index(sfd_str) + len(START_FRAME_DELIMITER)
    except ValueError:
        return (-1, b"")

    payload_length = _bits_to_bytes(bits[frame_start_idx : frame_start_idx + 8])[0]

    payload_start_idx = frame_start_idx + 8 + 8 * PACKET_INDEX_BYTES
    payload_end_idx = payload_start_idx + payload_length * 8

    packet_idx = int.from_bytes(
        bytes(_bits_to_bytes(bits[frame_start_idx + 8 : payload_start_idx])),
        sys.byteorder,
    )

    return (
        packet_idx,
        bytes(_bits_to_bytes(bits[payload_start_idx:payload_end_idx])),
    )


def _bytes_to_bits(bytes: list[int]) -> list[int]:
    return cast(list[int], np.unpackbits(np.array(bytes, dtype=np.uint8)).tolist())


def _bits_to_bytes(bits: list[int]) -> list[int]:
    return cast(list[int], np.packbits(np.array(bits)).tolist())
