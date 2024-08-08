import threading
import time

from bladerf import _bladerf

BYTES_PER_SAMPLE = 4


class BladeRFWrapper:
    def __init__(self, bladerf_serial: str) -> None:
        if bladerf_serial:
            self._bladerf = _bladerf.BladeRF(f"*:serial={bladerf_serial}")
        else:
            self._bladerf = _bladerf.BladeRF()

    @property
    def bladerf(self) -> _bladerf.BladeRF:
        return self._bladerf

    def close(self) -> None:
        self._bladerf.close()

    def enable_rx(
        self, *, frequency: float, sample_rate: float, gain: float, buffer_size: int
    ) -> None:
        self._rx_ch = self._bladerf.Channel(_bladerf.CHANNEL_RX(0))
        self._rx_ch.frequency = frequency
        self._rx_ch.sample_rate = sample_rate
        self._rx_ch.bandwidth = sample_rate / 2
        self._rx_ch.gain_mode = _bladerf.GainMode.Manual
        self._rx_ch.gain = gain

        self._bladerf.sync_config(
            layout=_bladerf.ChannelLayout.RX_X1,
            fmt=_bladerf.Format.SC16_Q11,
            num_buffers=16,
            buffer_size=buffer_size,
            num_transfers=8,
            stream_timeout=3500,
        )

        self._rx_ch.enable = True

    def disable_rx(self) -> None:
        self._rx_ch.enable = False

    def receive(self, buf: bytearray, num_samples: int) -> None:
        self._bladerf.sync_rx(buf, num_samples)

    def enable_tx(
        self, *, frequency: float, sample_rate: float, gain: float, buffer_size: int
    ) -> None:
        self._tx_ch = self._bladerf.Channel(_bladerf.CHANNEL_TX(0))
        self._tx_ch.frequency = frequency
        self._tx_ch.sample_rate = sample_rate
        self._tx_ch.bandwidth = sample_rate / 2
        self._tx_ch.gain = gain

        self._bladerf.sync_config(
            layout=_bladerf.ChannelLayout.TX_X1,
            fmt=_bladerf.Format.SC16_Q11,
            num_buffers=16,
            buffer_size=buffer_size,
            num_transfers=8,
            stream_timeout=3500,
        )

        self._tx_ch.enable = True

    def disable_tx(self) -> None:
        self._tx_ch.enable = False

    def transmit(self, buf: bytes, num_samples: int) -> None:
        self._bladerf.sync_tx(buf, num_samples)


class SimBladeRFWrapper:
    def __init__(self) -> None:
        print("Using simulated SDR.")

        self._samples_buf: bytearray | None = None
        self._samples_buf_lock = threading.Lock()

    def close(self) -> None:
        pass

    def enable_rx(
        self, *, frequency: float, sample_rate: float, gain: float, buffer_size: int
    ) -> None:
        self._rx_sample_rate = sample_rate

        with self._samples_buf_lock:
            self._samples_buf = bytearray(buffer_size * BYTES_PER_SAMPLE)

    def disable_rx(self) -> None:
        pass

    def receive(self, buf: bytearray, num_samples: int) -> None:
        time.sleep(num_samples / self._rx_sample_rate)

        with self._samples_buf_lock:
            if self._samples_buf is None:
                return

            num_bytes = num_samples * BYTES_PER_SAMPLE
            buf[:num_bytes] = self._samples_buf[:num_bytes]

    def enable_tx(
        self, *, frequency: float, sample_rate: float, gain: float, buffer_size: int
    ) -> None:
        self._tx_sample_rate = sample_rate

    def disable_tx(self) -> None:
        pass

    def transmit(self, buf: bytes, num_samples: int) -> None:
        time.sleep(num_samples / self._tx_sample_rate)

        with self._samples_buf_lock:
            if self._samples_buf is None:
                return

            merged_buf = self._samples_buf + buf[: num_samples * BYTES_PER_SAMPLE]
            self._samples_buf = merged_buf[-len(self._samples_buf) :]
