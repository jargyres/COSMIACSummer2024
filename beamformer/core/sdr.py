import threading
import time
from multiprocessing.pool import ThreadPool


from bladerf import _bladerf
import bladerf

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

    def enable_rx_mimo(
        self, *, frequency: float, sample_rate: float, gain: float, num_samples_per_channel: int
    ) -> None:
        print("Setting up 2 RX MIMO Receiving")
        bandwidth = int(sample_rate / 2)

        ch_rx_0 = _bladerf.CHANNEL_RX(0)
        ch_rx_1 = _bladerf.CHANNEL_RX(1)


        #set both rx to manual gain mode to simplify things

        self._bladerf.set_gain_mode(ch_rx_0, 1)
        self._bladerf.set_gain_mode(ch_rx_1, 1)


        #set the sample rate for the RX chain
        self._bladerf.set_sample_rate(ch_rx_0, sample_rate)
        self._bladerf.set_sample_rate(ch_rx_1, sample_rate)


        #set bandidth for RX chain
        self._bladerf.set_bandwidth(ch_rx_0, bandwidth)
        self._bladerf.set_bandwidth(ch_rx_1, bandwidth)


        #set the gain for the RX chain
        self._bladerf.set_gain(ch_rx_0, gain)
        self._bladerf.set_gain(ch_rx_1, gain)


        #set center frequency for RX chain
        self._bladerf.set_frequency(ch_rx_0, frequency)
        self._bladerf.set_frequency(ch_rx_1, frequency)

        #Trigger setup
        # sig = bladerf.TRIGGER_SIGNAL.J51_1.value
        


        # masterChannel = _bladerf.CHANNEL_TX(0)

        # masterRole = bladerf.TRIGGER_ROLE.Master.value

        # self.master_Trigger = self._bladerf.trigger_init(masterChannel, masterRole, sig)

        # bladerfslaveChannel_0 = _bladerf.CHANNEL_RX(0)

        # slaveRole = bladerf.TRIGGER_ROLE.Slave.value

        # self.slave_Trigger = self._bladerf.trigger_init(bladerfslaveChannel_0, slaveRole, sig)

        # print("Arming All Triggers")
        # self._bladerf.trigger_arm(self.master_Trigger, True)
        # self._bladerf.trigger_arm(self.slave_Trigger, True)

        NUM_CHANNELS = 2
        bytesinint16 = 2
        buf_size = 2 * num_samples_per_channel * NUM_CHANNELS * bytesinint16

        self._bladerf.sync_config(layout = _bladerf.ChannelLayout.RX_X2,
                            fmt            = _bladerf.Format.SC16_Q11,
                            num_buffers    = 32,
                            buffer_size    = buf_size,
                            num_transfers  = 8,
                            stream_timeout = 100)


        rx0 = self._bladerf.Channel(_bladerf.CHANNEL_RX(0))
        rx0.enable = True
        rx1 = self._bladerf.Channel(_bladerf.CHANNEL_RX(1))
        rx1.enable = True

        # self.rxthreadpool = ThreadPool(processes=1)

        print("Done Setting up RX Chain")



    def receive_mimo(self, buf: bytearray, num_samples: int):
        # self._bladerf.trigger_arm(self.master_Trigger, True)
        # self._bladerf.trigger_arm(self.slave_Trigger, True)

        # result = self.rxthreadpool.apply_async(self._bladerf.sync_rx, (buf, num_samples * 2))
        self._bladerf.sync_rx(buf=buf, num_samples=num_samples * 2, timeout_ms=100)
        # print("Fired Trigger")
        # for i in range(100):
        #     self._bladerf.trigger_fire(self.master_Trigger)

        # print("Awaiting result")
        # result.wait()
        # print("Result done")







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

    def print_all_info(self):
        print(">>>>>> DEV INFO <<<<<<")
        print(self._bladerf.get_devinfo())
        print(">>> DEV SPEED")
        print(self._bladerf.get_device_speed())
        print(">>> DEV Serial")
        print(self._bladerf.get_serial())
        print(">>> RX CHANNEL COUNT")
        print(self._bladerf.rx_channel_count)
        print(">>> TX CHANNEL COUNT")
        print(self._bladerf.tx_channel_count)
        print(">>> GAIN SETTINGS")

        print(">>>> GAIN RX0 = {} dB".format(self._bladerf.get_gain(_bladerf.CHANNEL_RX(0))))
        print(">>>> GAIN RX0 = {} dB".format(self._bladerf.get_gain(_bladerf.CHANNEL_RX(1))))
        print(">>>> GAIN TX0 = {} dB".format(self._bladerf.get_gain(_bladerf.CHANNEL_TX(0))))
        print(">>>> GAIN TX0 = {} dB".format(self._bladerf.get_gain(_bladerf.CHANNEL_TX(1))))




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
