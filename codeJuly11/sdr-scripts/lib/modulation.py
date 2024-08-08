import numpy as np
import numpy.typing as npt


def modulate_ask(
    data_bits: list[int],
    carrier_freq: float,
    sample_rate: float,
    samples_per_symbol: int,
) -> npt.NDArray[np.complex128]:
    sample_amplitudes = np.array(data_bits).repeat(samples_per_symbol)
    num_samples = len(sample_amplitudes)

    t = np.linspace(0, num_samples / sample_rate, num_samples)
    carrier_samples: npt.NDArray[np.complex128] = np.exp(
        1j * 2 * np.pi * carrier_freq * t
    )

    return carrier_samples * sample_amplitudes


def demodulate_ask(
    complex_samples: npt.NDArray[np.complex128], samples_per_symbol: int
) -> list[int]:
    sample_magnitudes = np.abs(complex_samples)
    first_data_sample_idx: int | None = None

    for i, sample_magnitude in enumerate(sample_magnitudes[1:]):
        if abs(sample_magnitude - sample_magnitudes[0]) > 0.5:
            first_data_sample_idx = i + 1
            break

    if first_data_sample_idx is None:
        return []

    num_symbols = int(
        (len(sample_magnitudes) - first_data_sample_idx) / samples_per_symbol
    )

    data_bits: list[int] = []

    for symbol_idx in range(num_symbols):
        symbol_start_idx = first_data_sample_idx + samples_per_symbol * symbol_idx
        symbol_end_idx = symbol_start_idx + samples_per_symbol

        avg_symbol_magnitude = np.mean(
            sample_magnitudes[symbol_start_idx:symbol_end_idx]
        )

        if abs(avg_symbol_magnitude - 1) < 0.5:
            data_bits.append(1)
        else:
            data_bits.append(0)

    return data_bits
