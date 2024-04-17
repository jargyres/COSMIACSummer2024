import bladerf
from bladerf import _bladerf
import time

d = bladerf.BladeRF()

sig = bladerf.TRIGGER_SIGNAL.J51_1.value



masterChannel = bladerf.CHANNEL_TX(0)

masterRole = bladerf.TRIGGER_ROLE.Master.value

masterTrigger = d.master_trigger_init(masterChannel, sig)



slaveChannel = bladerf.CHANNEL_RX(0)

slaveRole = bladerf.TRIGGER_ROLE.Slave.value

slaveTrigger = d.trigger_init(slaveChannel, slaveRole, sig)

print("Arming Both Triggers")
d.trigger_arm(masterTrigger, True)
d.trigger_arm(slaveTrigger, True)


d.trigger_state(masterTrigger)
d.trigger_state(slaveTrigger)

d.sync_config(layout = _bladerf.ChannelLayout.RX_X1,
                       fmt            = _bladerf.Format.SC16_Q11,
                       num_buffers    = 16,
                       buffer_size    = 8192,
                       num_transfers  = 8,
                       stream_timeout = 10000)

s = d.Channel(bladerf.CHANNEL_RX(0))
s.enable = True


# d.sync_rx()
bytes_per_sample = 4
buf = bytearray(1024*bytes_per_sample)

num_samples = 1024



d.trigger_fire(masterTrigger)

d.sync_rx(buf, num_samples=num_samples)

# Disable module
print( "RX: Stop" )
s.enable = False

print( "RX: Done" )




