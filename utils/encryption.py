import base64
from cryptography.fernet import Fernet
from shared.constants import ENCRYPTION_KEY


class Encryptor:
    def __init__(self):
        self.cipher = Fernet(ENCRYPTION_KEY)

    def encrypt(self, data: str):
        data_bytes = data.encode("utf-8")
        encrypted_data = self.cipher.encrypt(data_bytes)
        return encrypted_data

    def decrypt(self, data: str):
        data_bytes = data[2:-1].encode() if data.startswith("b'") else data.encode()
        decrypted_data = self.cipher.decrypt(data_bytes)
        return decrypted_data.decode("utf-8")

    def encrypt_json(self, data: str):
        encrypted_data = self.encrypt(data)
        encrypted_data_str = base64.b64encode(encrypted_data).decode("utf-8")
        return encrypted_data_str

    def decrypt_json(self, data: str):
        encrypted_data_bytes = base64.b64decode(data)
        decrypted_data = self.decrypt(encrypted_data_bytes.decode("utf-8"))
        return decrypted_data
