h_freq ([batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, fft_size], tf.complex)
h_freq is non-normalized  – If normalized, the channel is normalized over the resource grid

a ([batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps])
tau [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]


In cir file, we have cir "a" and non-normalized delay "tau" (arrive time of first path isn't 0)

1 TX: 2*2 4 antennas
1 RX: 8*8 64 antennas

In sionna:

orientation=[yaw, pitch, roll], rads

Yaw — rotation around the vertical (Z) axis

Pitch — rotation around the lateral (Y) axis

Roll — rotation around the longitudinal (X) axis

ue_rows = 1
ue_cols = 8
orientation=[0,0,0] vertically

