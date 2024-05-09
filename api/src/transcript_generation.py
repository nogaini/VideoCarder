import stable_whisper
from whisper.model import Whisper


def load_stable_whisper_model(model_name: str) -> Whisper:
    """
    Loads a stable Whisper model based on its name.
    
    Parameters:
        model_name (str): The name of the Whisper model to be loaded. It should match one of 
                           the available models in the library, such as 'small', 'medium', etc.
                           
    Returns:
        Whisper: An instance of the stable-whisper's `Whisper` class representing the loaded model.
        
    """
    model = stable_whisper.load_model(model_name)
    return model