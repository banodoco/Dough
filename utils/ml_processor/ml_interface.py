

from utils.logging.logging import AppLogger


def get_ml_client():
    from utils.ml_processor.replicate.replicate import ReplicateProcessor
    
    return ReplicateProcessor()

class MachineLearningProcessor:
    def __init__(self):
        self.logger = AppLogger()
