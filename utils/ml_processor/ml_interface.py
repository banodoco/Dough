from abc import ABC


def get_ml_client():
    from utils.ml_processor.replicate.replicate import ReplicateProcessor
    
    return ReplicateProcessor()

class MachineLearningProcessor(ABC):
    pass