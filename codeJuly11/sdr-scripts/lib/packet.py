from typing import cast

import numpy as np

PREAMBLE = [0, 1, 0]
START_FRAME_DELIMITER = [1] * 8


def create_packet(msg: str, packet_idx: int) -> list[int]:
    return (
        PREAMBLE
        + START_FRAME_DELIMITER
        + _bytes_to_bits([len(msg), packet_idx] + list(msg.encode()))
    )


def extract_message(bits: list[int]) -> tuple[int, str]:
    bits_str = "".join(str(b) for b in bits)
    sfd_str = "".join(str(b) for b in START_FRAME_DELIMITER)

    try:
        frame_start_idx = bits_str.index(sfd_str) + len(START_FRAME_DELIMITER)
    except ValueError:
        return (-1, "")

    msg_len = _bits_to_bytes(bits[frame_start_idx : frame_start_idx + 8])[0]
    packet_idx = _bits_to_bytes(bits[frame_start_idx + 8 : frame_start_idx + 16])[0]

    msg_start_idx = frame_start_idx + 16
    msg_end_idx = msg_start_idx + msg_len * 8

    try:
        return (
            packet_idx,
            bytes(_bits_to_bytes(bits[msg_start_idx:msg_end_idx])).decode(),
        )
    except Exception:
        print("Failed to decode received message.")
        return (-1, "")


def _bytes_to_bits(bytes: list[int]) -> list[int]:
    return cast(list[int], np.unpackbits(np.array(bytes, dtype=np.uint8)).tolist())


def _bits_to_bytes(bits: list[int]) -> list[int]:
    return cast(list[int], np.packbits(np.array(bits)).tolist())
