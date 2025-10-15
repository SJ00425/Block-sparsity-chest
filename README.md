h_freq ([batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, fft_size], tf.complex)

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

