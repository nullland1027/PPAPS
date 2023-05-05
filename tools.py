import hashlib


class Tools:

    @staticmethod
    def password_encode(pwd):
        return hashlib.sha256(pwd.encode('utf-8')).hexdigest()
