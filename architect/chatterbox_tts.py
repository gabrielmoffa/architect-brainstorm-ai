import nltk
import torch
import warnings
import numpy as np
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

warnings.filterwarnings(
    "ignore",
    message="torch.nn.utils.weight_norm isdeprecated in favor of torch.nn.utils.parametrizations.weight_norm.",
)


class TextToSpeechService:
    def __init__(self, device: str | None = None):
        """
        Initializes the TextToSpeechService class with ChatterBox TTS.

        Args:
            device (str, optional): The device to be used for the model. If None, will auto-detect.
                Can be "cuda", "mps", or "cpu".
        """
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print(f"Using device: {self.device}")

        if self.device == "cuda" and not torch.cuda.is_available():
            print("CUDA requested but not available, falling back to CPU")
            self.device = "cpu"

        self._patch_torch_load()
        self.model = ChatterboxTTS.from_pretrained(device=self.device)
        self.sample_rate = self.model.sr

    def _patch_torch_load(self):
        """
        Patches torch.load to automatically map tensors to the correct device.
        This is needed because ChatterBox models may have been saved on CUDA.
        """
        map_location = torch.device(self.device)

        if not hasattr(torch, '_original_load'):
            torch._original_load = torch.load

        def patched_torch_load(*args, **kwargs):
            if 'map_location' not in kwargs:
                kwargs['map_location'] = map_location
            return torch._original_load(*args, **kwargs)

        torch.load = patched_torch_load

    def synthesize(self, text: str):
        """
        Synthesizes audio from the given text using ChatterBox TTS.

        Args:
            text (str): The input text to be synthesized.

        Returns:
            tuple: A tuple containing the sample rate and the generated audio array.
        """
        wav = self.model.generate(
            text
        )

        # Convert tensor to numpy array format compatible with sounddevice
        audio_array = wav.squeeze().cpu().numpy()
        return self.sample_rate, audio_array

    def long_form_synthesize(self, text: str):
        """
        Synthesizes audio from the given long-form text using ChatterBox TTS.

        Args:
            text (str): The input text to be synthesized.

        Returns:
            tuple: A tuple containing the sample rate and the generated audio array.
        """
        pieces = []
        sentences = nltk.sent_tokenize(text)
        silence = np.zeros(int(0.1 * self.sample_rate)) # Reduced silence for faster generation

        for sent in sentences:
            sample_rate, audio_array = self.synthesize(
                sent
            )
            pieces += [audio_array] # Removed silence for continuous audio

        return self.sample_rate, np.concatenate(pieces)