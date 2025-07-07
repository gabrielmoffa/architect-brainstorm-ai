import os
import time
import threading
import numpy as np
import whisper
import sounddevice as sd
import argparse
from queue import Queue
from rich.console import Console
from rich.panel import Panel
from dotenv import load_dotenv

# LangChain imports
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain.tools import tool

# Local imports
from . import ai_tools
from .tts_services import PyTTSX3Service
from .chatterbox_tts import TextToSpeechService

# Load environment variables
load_dotenv()

# --- Tool Setup ---
@tool
def create_object(type: str, title: str, description: str, parent_id: str = None):
    """Create a new project object and assign unique ID"""
    return ai_tools.create_object(type, title, description, parent_id)

@tool
def update_object(obj_id: str, new_title: str = None, new_description: str = None):
    """Update existing object with additional information. Provide either new_title or new_description, or both."""
    return ai_tools.update_object(obj_id, new_title, new_description)

@tool
def delete_object(obj_id: str):
    """Delete an existing project object by its ID."""
    return ai_tools.delete_object(obj_id)

@tool
def link_objects(obj1_id: str, obj2_id: str, relationship: str):
    """Create relationship between objects (solves, depends_on, targets, requires, etc.)"""
    return ai_tools.link_objects(obj1_id, obj2_id, relationship)

@tool
def delete_link(obj1_id: str, obj2_id: str, relationship: str):
    """Delete a specific relationship between two objects."""
    return ai_tools.delete_link(obj1_id, obj2_id, relationship)

@tool
def update_mermaid_diagram():
    """Generate/update Mermaid diagram based on current objects and relationships"""
    return ai_tools.update_mermaid_diagram()

@tool
def show_project_summary():
    """Display current project objects and their relationships"""
    return ai_tools.show_project_summary()

@tool
def customize_object_types(new_types: list):
    """Allow user to define custom object types beyond defaults"""
    return ai_tools.customize_object_types(new_types)

@tool
def open_diagram_viewer():
    """Opens the project diagram in a web browser with zoom and pan functionality."""
    import subprocess
    import sys
    
    # Check if the viewer is already running (simple check for now)
    # In a real app, you'd use a more robust PID management
    try:
        # Try to connect to the server to see if it's alive
        import requests
        requests.get('http://127.0.0.1:5000/', timeout=1)
        return "Diagram viewer is already running. Opening in browser..."
    except requests.exceptions.ConnectionError:
        pass # Server is not running, proceed to start it

    # Start the Flask server in a separate process
    # Use sys.executable to ensure the correct python interpreter is used
    subprocess.Popen([sys.executable, "viewer.py"])
    return "Opening diagram viewer in your web browser."


# --- AI Setup ---
SYSTEM_PROMPT = r"""
You are Architect, an expert project planning AI that helps users structure their projects through intelligent conversation and visual planning.

CORE MISSION:
Help users think "right to left" - start with outcomes/goals, work backwards to execution details.

OBJECT CREATION STRATEGY:
- Listen for key project elements in conversation
- ALWAYS begin by using `show_project_summary` to understand the current state and available object IDs.
- Create objects immediately when you have enough information.
- Before creating a new object, check if a similar object already exists in the `show_project_summary` output. If it does, use `update_object` with its existing ID and specify `new_title` or `new_description` as appropriate.
- When a relationship between objects is implied or discussed, IMMEDIATELY use the `link_objects` tool with the exact IDs from `show_project_summary` and a precise relationship type.
  - Specifically, when creating a `Task`, always link it to the `Solution` it is part of (using `broken into`) and any `Resource` it requires (using `requires`). Ask follow-up questions if this information is not provided.
  - For Risks, consider linking Solutions that `mitigate` them, or Tasks that are `mitigation task`s.
- If an object becomes irrelevant or redundant, use the `delete_object` tool to remove it, using its exact ID from `show_project_summary`.
- **To remove a specific relationship between two objects, use the `delete_link` tool.**

CONVERSATION FLOW:
1. Start broad - understand the overall vision/goal
2. Identify key problems being solved
3. Understand target users/stakeholders  
4. Explore proposed solutions
5. Break down into actionable tasks
6. Identify required resources
7. Assess risks and success metrics

QUESTIONING PRINCIPLES:
- Ask ONE focused question at a time
- Build on previous answers naturally
- Dig deeper when answers are vague
- Connect related concepts explicitly
- Know when you have enough information

MERMAID DIAGRAM RULES:
- Use appropriate node shapes for different object types:
  - Goals: Rounded rectangles `(Goal Text)`
  - Problems: Rectangles `[Problem Text]`
  - Users: Circles `((User Text))`
  - Solutions: Hexagons `{{Solution Text}}`
  - Tasks: Subroutines `[[Task Text]]`
  - Resources: Diamonds `{{Resource Text}}`
  - Risks: Trapezoids `[/Risk Text\]`
  - Metrics: Stadium shape `([[Metric Text]])`

- Use logical relationship arrows:
  - Goals --> Problems (addresses)
  - Problems --> Users (affects)
  - Goals --> Solutions (achieved through)
  - Solutions --> Tasks (broken into)
  - Tasks --> Resources (requires)
  - Solutions --> Risks (mitigates)
  - Risks --> Solutions (mitigated by)
  - Goals --> Metrics (measured by)

- Keep diagrams readable - group related items
- Use colors/styles for different object types

STOPPING CRITERIA:
You have enough information when you can clearly articulate:
- What they want to achieve (Goals)
- What problem they're solving (Problems)
- Who they're solving it for (Users)
- How they'll solve it (Solutions)
- What work needs to be done (Tasks)
- What they need to succeed (Resources)
- What could go wrong (Risks)
- How they'll measure success (Metrics)

CONVERSATION STYLE:
- Conversational and encouraging, not interrogative
- Show understanding by briefly summarizing their input
- Make connections between their answers explicit
- Celebrate progress and insights
- Offer to show the diagram regularly

DIAGRAM GENERATION:
- **Remember: Diagram generation tools (`update_mermaid_diagram`) only VISUALIZE the current state of objects and their relationships. They do NOT create or modify relationships.**
- For all diagram needs, use `update_mermaid_diagram`.
"""

def get_agent_executor(llm):
    """Creates and returns the LangChain agent executor."""
    tools = [
        create_object,
        update_object,
        delete_object,
        link_objects,
        delete_link, # Added new tool
        update_mermaid_diagram,
        show_project_summary,
        customize_object_types,
        open_diagram_viewer,
    ]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    chat_sessions = {}

    def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
        """Get or create chat history for a session."""
        if session_id not in chat_sessions:
            chat_sessions[session_id] = InMemoryChatMessageHistory()
        return chat_sessions[session_id]

    return RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

# --- Voice Interaction ---
stt = whisper.load_model("base.en")
console = Console()

# Global event for stopping AI speech playback
stop_ai_speaking_event = threading.Event()

def record_audio(stop_event, data_queue):
    """Captures audio data from the user's microphone."""
    def callback(indata, frames, time, status):
        if status:
            console.print(status)
        data_queue.put(bytes(indata))

    with sd.RawInputStream(samplerate=16000, dtype="int16", channels=1, callback=callback):
        while not stop_event.is_set():
            time.sleep(0.1)

def transcribe(audio_np: np.ndarray) -> str:
    """Transcribes audio using Whisper."""
    result = stt.transcribe(audio_np, fp16=False)
    return result["text"].strip()

def play_audio_interruptible(sample_rate, audio_array, stop_event):
    sd.play(audio_array, sample_rate)
    # Wait for playback to finish or stop_event to be set
    while sd.get_status().output_active and not stop_event.is_set():
        time.sleep(0.1) # Small sleep to avoid busy-waiting
    sd.stop() # Ensure playback is stopped

def listen_for_interrupt_thread(stop_event):
    # This input() will block this thread until Enter is pressed
    # It will consume the Enter key press
    try:
        input()
    except EOFError: # Handle cases where stdin might be closed
        pass
    stop_event.set() # Signal to stop playback

# --- CLI Interface ---
def main():
    parser = argparse.ArgumentParser(description="Architect AI Voice Assistant")
    parser.add_argument("--project-name", type=str, default="default_project", help="The name of the project to work on.")
    parser.add_argument("--tts-provider", type=str, default="pyttsx3", choices=["chatterbox", "pyttsx3"], help="TTS provider to use.")
    parser.add_argument("--voice", type=str, help="Path to voice sample for cloning (ChatterBox only).")
    parser.add_argument("--exaggeration", type=float, default=0.5, help="Emotion exaggeration (0.0-1.0) (ChatterBox only).")
    parser.add_argument("--cfg-weight", type=float, default=0.5, help="CFG weight for pacing (0.0-1.0) (ChatterBox only).")
    parser.add_argument("--provider", type=str, default="openai", choices=["openai"], help="LLM provider to use.")
    parser.add_argument("--openai-model", type=str, default="gpt-4o", help="OpenAI model to use.")
    parser.add_argument("--save-voice", action="store_true", help="Save generated voice samples (ChatterBox only).")
    args = parser.parse_args()

    console.print(Panel("️ Welcome to Architect! Let's plan your project.", title="[bold green]Architect AI[/bold green]"))

    # Set project name for ai_tools
    ai_tools.PROJECT_NAME = args.project_name

    # Initialize TTS
    if args.tts_provider == "chatterbox":
        tts = TextToSpeechService()
    else:
        tts = PyTTSX3Service()

    # Initialize LLM
    llm = ChatOpenAI(model=args.openai_model)
    console.print(f"[blue]LLM provider: OpenAI (model: {args.openai_model})")

    agent_with_history = get_agent_executor(llm)

    console.print(f"[blue]TTS provider: {args.tts_provider}")
    console.print("[cyan]Press Ctrl+C to exit.\n")

    if args.save_voice:
        os.makedirs("voices", exist_ok=True)
    response_count = 0

    try:
        while True:
            console.input("🎤 Press Enter to start recording, then press Enter again to stop.")
            
            data_queue = Queue()
            stop_event = threading.Event()
            recording_thread = threading.Thread(target=record_audio, args=(stop_event, data_queue))
            recording_thread.start()

            input()
            stop_event.set()
            recording_thread.join()

            audio_data = b"".join(list(data_queue.queue))
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            if audio_np.size > 0:
                with console.status("Transcribing...", spinner="dots"):
                    text = transcribe(audio_np)
                console.print(f"[yellow]You: {text}")

                if text:
                    with console.status("Generating response...", spinner="dots"):
                        response = agent_with_history.invoke(
                            {"input": text},
                            config={"configurable": {"session_id": args.project_name}}
                        )
                        response_text = response['output']

                    console.print(f"[cyan]Architect:[/cyan] {response_text}")

                    if args.tts_provider == "chatterbox":
                        dynamic_exaggeration = analyze_emotion(response_text)
                        dynamic_cfg = args.cfg_weight * 0.8 if dynamic_exaggeration > 0.6 else args.cfg_weight
                        sample_rate, audio_array = tts.long_form_synthesize(
                            response_text,
                            audio_prompt_path=args.voice,
                            exaggeration=dynamic_exaggeration,
                            cfg_weight=dynamic_cfg
                        )
                        if args.save_voice:
                            response_count += 1
                            filename = f"voices/response_{response_count:03d}.wav"
                            tts.save_voice_sample(response_text, filename, args.voice)
                            console.print(f"[dim]Voice saved to: {filename}[/dim]")
                        
                        # Play audio interruptibly
                        stop_ai_speaking_event.clear()
                        interrupt_thread = threading.Thread(target=listen_for_interrupt_thread, args=(stop_ai_speaking_event,))
                        interrupt_thread.start()
                        play_audio_interruptible(sample_rate, audio_array, stop_ai_speaking_event)
                        interrupt_thread.join(timeout=0.1) # Give a moment for thread to exit if it did

                        if stop_ai_speaking_event.is_set():
                            console.print("[yellow]AI speech interrupted. Press Enter to start recording your response.[/yellow]")
                        else:
                            console.print("[green]AI finished speaking. Press Enter to start recording your response.[/green]")

                    else: # pyttsx3
                        tts.long_form_synthesize(response_text)
                        console.print("[green]AI finished speaking (pyttsx3 is not interruptible). Press Enter to start recording your response.[/green]")
            else:
                console.print("[red]No audio recorded. Please ensure your microphone is working.")

    except (KeyboardInterrupt, EOFError):
        console.print("\n[red]Exiting...")

    console.print("[blue]Session ended. Thank you for using Architect AI!")

if __name__ == "__main__":
    main()
