from SX127x.LoRa import LoRa
from SX127x.board_config import BOARD

BOARD.setup()

lora = LoRa(verbose=True)
lora.set_mode(lora.MODE_STDBY)

version = lora.get_version()
print("LoRa chip version: 0x{:02X}".format(version))

BOARD.teardown()
