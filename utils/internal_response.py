class InternalResponse:
    def __init__(self, data, message, status):
        self.status = status
        self.message = message
        self.data = data