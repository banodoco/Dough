from cryptography.fernet import Fernet

from banodoco_settings import ENCRYPTION_KEY

class Encryptor:
    def __init__(self):
        self.cipher = Fernet(ENCRYPTION_KEY)
    
    def encrypt(self, data: str):
        data_bytes = data.encode('utf-8')
        encrypted_data = self.cipher.encrypt(data_bytes)
        return encrypted_data
    
    def decrypt(self, data: str):
        data_bytes = data[2:-1].encode()
        decrypted_data = self.cipher.decrypt(data_bytes)
        return decrypted_data.decode('utf-8')