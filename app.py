import time
import threading
import numpy as np
import whisper
import sounddevice as sd
import argparse
import os
from queue import Queue
from rich.console import Console
# Updated imports for modern LangChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_ollama import OllamaLLM
from tts import TextToSpeechService
import pyttsx3

class PyTTSX3Service:
    def __init__(self):
        self.engine = pyttsx3.init()

    def synthesize(self, text: str, **kwargs):
        self.engine.say(text)
        self.engine.runAndWait()
        return 0, np.array([]) # Return dummy values for compatibility


console = Console()
stt = whisper.load_model("base.en")

# Parse command line arguments
parser = argparse.ArgumentParser(description="Local Voice Assistant with ChatterBox TTS")
parser.add_argument("--tts-provider", type=str, default="chatterbox", choices=["chatterbox", "pyttsx3"], help="TTS provider to use (chatterbox or pyttsx3)")
parser.add_argument("--voice", type=str, help="Path to voice sample for cloning (ChatterBox only)")
parser.add_argument("--exaggeration", type=float, default=0.5, help="Emotion exaggeration (0.0-1.0) (ChatterBox only)")
parser.add_argument("--cfg-weight", type=float, default=0.5, help="CFG weight for pacing (0.0-1.0) (ChatterBox only)")
parser.add_argument("--provider", type=str, default="ollama", choices=["ollama", "openai"], help="LLM provider to use (ollama or openai)")
parser.add_argument("--ollama-model", type=str, default="gemma3", help="Ollama model to use")
parser.add_argument("--openai-model", type=str, default="gpt-4o-mini", help="OpenAI model to use")
parser.add_argument("--save-voice", action="store_true", help="Save generated voice samples (ChatterBox only)")
args = parser.parse_args()

# Initialize TTS based on provider
if args.tts_provider == "chatterbox":
    tts = TextToSpeechService()
elif args.tts_provider == "pyttsx3":
    tts = PyTTSX3Service()
else:
    raise ValueError("Invalid TTS provider specified. Choose 'chatterbox' or 'pyttsx3'.")

# Modern prompt template using ChatPromptTemplate
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful and friendly AI assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Initialize LLM
# Initialize LLM based on provider
if args.provider == "ollama":
    llm = OllamaLLM(model=args.ollama_model, base_url="http://localhost:11434")
elif args.provider == "openai":
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model=args.openai_model)
else:
    raise ValueError("Invalid LLM provider specified. Choose 'ollama' or 'openai'.")

# Create the chain with modern LCEL syntax
chain = prompt_template | llm

# Chat history storage
chat_sessions = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Get or create chat history for a session."""
    if session_id not in chat_sessions:
        chat_sessions[session_id] = InMemoryChatMessageHistory()
    return chat_sessions[session_id]

# Create the runnable with message history
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

def record_audio(stop_event, data_queue):
    """
    Captures audio data from the user's microphone and adds it to a queue for further processing.

    Args:
        stop_event (threading.Event): An event that, when set, signals the function to stop recording.
        data_queue (queue.Queue): A queue to which the recorded audio data will be added.

    Returns:
        None
    """
    def callback(indata, frames, time, status):
        if status:
            console.print(status)
        data_queue.put(bytes(indata))

    with sd.RawInputStream(
        samplerate=16000, dtype="int16", channels=1, callback=callback
    ):
        while not stop_event.is_set():
            time.sleep(0.1)


def transcribe(audio_np: np.ndarray) -> str:
    """
    Transcribes the given audio data using the Whisper speech recognition model.

    Args:
        audio_np (numpy.ndarray): The audio data to be transcribed.

    Returns:
        str: The transcribed text.
    """
    result = stt.transcribe(audio_np, fp16=False)  # Set fp16=True if using a GPU
    text = result["text"].strip()
    return text


def get_llm_response(text: str) -> str:
    """
    Generates a response to the given text using the language model.

    Args:
        text (str): The input text to be processed.

    Returns:
        str: The generated response.
    """
    # Use a default session ID for this simple voice assistant
    session_id = "voice_assistant_session"

    # Invoke the chain with history
    response = chain_with_history.invoke(
        {"input": text},
        config={"session_id": session_id}
    )

    # The response is now a string from the LLM, no need to remove "Assistant:" prefix
    # since we're using a proper chat model setup
    return response.content.strip()


def play_audio(sample_rate, audio_array):
    """
    Plays the given audio data using the sounddevice library.

    Args:
        sample_rate (int): The sample rate of the audio data.
        audio_array (numpy.ndarray): The audio data to be played.

    Returns:
        None
    """
    sd.play(audio_array, sample_rate)
    sd.wait()


def analyze_emotion(text: str) -> float:
    """
    Simple emotion analysis to dynamically adjust exaggeration.
    Returns a value between 0.3 and 0.9 based on text content.
    """
    # Keywords that suggest more emotion
    emotional_keywords = ['amazing', 'terrible', 'love', 'hate', 'excited', 'sad', 'happy', 'angry', 'wonderful', 'awful', '!', '?!', '...']

    emotion_score = 0.5  # Default neutral

    text_lower = text.lower()
    for keyword in emotional_keywords:
        if keyword in text_lower:
            emotion_score += 0.1

    # Cap between 0.3 and 0.9
    return min(0.9, max(0.3, emotion_score))


if __name__ == "__main__":
    console.print("[cyan]🤖 Local Voice Assistant")
    console.print("[cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    if args.tts_provider == "chatterbox":
        if args.voice:
            console.print(f"[green]Using voice cloning from: {args.voice}")
        else:
            console.print("[yellow]Using default voice (no cloning)")

        console.print(f"[blue]Emotion exaggeration: {args.exaggeration}")
        console.print(f"[blue]CFG weight: {args.cfg_weight}")
    elif args.tts_provider == "pyttsx3":
        console.print("[yellow]Using simple pyttsx3 for TTS (robotic voice)")
    if args.provider == "ollama":
        console.print(f"[blue]LLM provider: Ollama (model: {args.ollama_model})")
    elif args.provider == "openai":
        console.print(f"[blue]LLM provider: OpenAI (model: {args.openai_model})")
    console.print("[cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    console.print("[cyan]Press Ctrl+C to exit.\n")

    # Create voices directory if saving voices
    if args.save_voice:
        os.makedirs("voices", exist_ok=True)

    response_count = 0

    try:
        while True:
            console.input(
                "🎤 Press Enter to start recording, then press Enter again to stop."
            )

            data_queue = Queue()  # type: ignore[var-annotated]
            stop_event = threading.Event()
            recording_thread = threading.Thread(
                target=record_audio,
                args=(stop_event, data_queue),
            )
            recording_thread.start()

            input()
            stop_event.set()
            recording_thread.join()

            audio_data = b"".join(list(data_queue.queue))
            audio_np = (
                np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            )

            if audio_np.size > 0:
                with console.status("Transcribing...", spinner="dots"):
                    text = transcribe(audio_np)
                console.print(f"[yellow]You: {text}")

                with console.status("Generating response...", spinner="dots"):
                    response = get_llm_response(text)

                    # Analyze emotion and adjust exaggeration dynamically
                    dynamic_exaggeration = analyze_emotion(response)

                    # Use lower cfg_weight for more expressive responses
                    dynamic_cfg = args.cfg_weight * 0.8 if dynamic_exaggeration > 0.6 else args.cfg_weight

                    if args.tts_provider == "chatterbox":
                        sample_rate, audio_array = tts.long_form_synthesize(
                            response,
                            audio_prompt_path=args.voice,
                            exaggeration=dynamic_exaggeration,
                            cfg_weight=dynamic_cfg
                        )
                    elif args.tts_provider == "pyttsx3":
                        # pyttsx3 handles audio playback internally
                        tts.synthesize(response)
                        sample_rate, audio_array = 0, np.array([]) # Dummy values

                console.print(f"[cyan]Assistant: {response}")
                if args.tts_provider == "chatterbox":
                    console.print(f"[dim](Emotion: {dynamic_exaggeration:.2f}, CFG: {dynamic_cfg:.2f})[/dim]")

                # Save voice sample if requested (ChatterBox only)
                if args.save_voice and args.tts_provider == "chatterbox":
                    response_count += 1
                    filename = f"voices/response_{response_count:03d}.wav"
                    tts.save_voice_sample(response, filename, args.voice)
                    console.print(f"[dim]Voice saved to: {filename}[/dim]")

                if args.tts_provider == "chatterbox":
                    play_audio(sample_rate, audio_array)
            else:
                console.print(
                    "[red]No audio recorded. Please ensure your microphone is working."
                )

    except KeyboardInterrupt:
        console.print("\n[red]Exiting...")

    console.print("[blue]Session ended. Thank you for using ChatterBox Voice Assistant!")
