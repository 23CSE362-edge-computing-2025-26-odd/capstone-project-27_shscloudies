from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import os

# AES Encrypt
def aes_encrypt(key: bytes, data: bytes) -> bytes:
    """
    Encrypts data using AES-CBC with PKCS7 padding.
    Prepends a random IV to the ciphertext.
    """
    iv = os.urandom(16)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    encrypted = cipher.encrypt(pad(data, AES.block_size))
    return iv + encrypted  # IV + ciphertext

# AES Decrypt
def aes_decrypt(key: bytes, payload: bytes) -> bytes:
    """
    Decrypts AES-CBC encrypted payload prepended with 16-byte IV.
    """
    if len(payload) < 16:
        raise ValueError("Payload too short for IV + data")

    iv = payload[:16]
    ciphertext = payload[16:]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    decrypted = unpad(cipher.decrypt(ciphertext), AES.block_size)
    return decrypted
