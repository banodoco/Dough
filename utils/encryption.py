import base64
import hashlib
import os
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


def validate_file_hash(file_path, expected_hash_list):
    if not os.path.exists(file_path):
        return False

    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    calculated_hash = hash_md5.hexdigest()

    return len(expected_hash_list) and str(calculated_hash) in expected_hash_list


def generate_file_hash(file_path):
    if not os.path.exists(file_path):
        return None

    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)

    return hash_md5.hexdigest()
