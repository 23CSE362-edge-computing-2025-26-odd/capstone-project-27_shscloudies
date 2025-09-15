import spidev
import time

spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 500000

print("SPI loopback test — waiting for 0x55 bytes")

try:
    while True:
        byte = spi.xfer2([0x00])[0]  # Send dummy byte to read
        if byte == 0x55:
            print("Received 0x55!")
        time.sleep(0.1)
except KeyboardInterrupt:
    spi.close()
    print("Test stopped")
