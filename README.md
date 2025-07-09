# Architect AI

Architect AI is an intelligent project planning assistant that helps users structure their projects through conversational interaction and visual diagramming.

## Installation

To get started with Architect AI, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/gabrielmoffa/architect-brainstorm-ai.git
    cd architect-brainstorm-ai
    ```

2.  **Install `uv` (if you haven't already):**
    `uv` is a fast Python package installer and resolver. You can install it via pip:
    ```bash
    pip install uv
    ```

3.  **Create and Activate a Virtual Environment:**
    It's highly recommended to use a virtual environment to manage project dependencies.
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

4.  **Install Python Dependencies:**
    Architect AI uses `uv` for dependency management. The `uv.lock` file ensures reproducible builds.
    ```bash
    uv sync
    ```

5.  **Download NLTK data:**
    For sentence tokenization in TTS, you need to download NLTK's `punkt` tokenizer:
    ```bash
    python -c "import nltk; nltk.download('punkt')"
    ```

6.  **Install `mermaid-cli` (for SVG diagram generation):**
    `mermaid-cli` is a Node.js tool. If you don't have Node.js, it's recommended to use `nvm` (Node Version Manager) or download from [nodejs.org](https://nodejs.org/).

    a.  **Install Node.js (if needed):**
        *   **Using `nvm` (recommended):**
            ```bash
            curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.1/install.sh | bash
            nvm install node # Installs the latest LTS version
            nvm use node
            ```
        *   **Direct download:** Visit [https://nodejs.org/](https://nodejs.org/) and download the installer for your operating system.

    b.  **Install `mermaid-cli`:** Once Node.js is installed, open your terminal and run:
        ```bash
        npm install -g @mermaid-js/mermaid-cli
        ```

## How to Use the Architect AI

Architect AI guides you through project planning by asking clarifying questions and visualizing your ideas. It operates conversationally, primarily through voice.

1.  **Start the application:**
    ```bash
    python -m architect.main [OPTIONS]
    ```

    **Available Options:**
    *   `--project-name <NAME>`: Specify a name for your project (default: `default_project`).
    *   `--tts-provider <PROVIDER>`: Choose your Text-to-Speech provider. Options: `chatterbox`, `pyttsx3` (default: `pyttsx3`).

    **Example Usage:**
    ```bash
    # Start with default settings
    python -m architect.main

    # Use ChatterBox TTS
    python -m architect.main --tts-provider chatterbox
    ```

2.  **Interact with the AI:**
    *   The AI will prompt you to press Enter to start recording your voice. Press Enter once to begin recording, and press Enter again to stop.
    *   Speak clearly into your microphone.
    *   The AI will transcribe your speech, process your request, and respond verbally.
    *   **To interrupt the AI while it's speaking (if using ChatterBox TTS), press Enter.**

3.  **Key Conversational Interactions:**
    *   **Project Structuring:** The AI will ask questions to help you define your project's Goals, Problems, Users, Solutions, Tasks, Resources, Risks, and Metrics.
    *   **Creating Objects:** Describe your project elements (e.g., "Create a Goal: Launch a new product").
    *   **Updating Objects:** Refer to objects by their name or a clear description (e.g., "Update the 'Launch new product' goal to 'Achieve market leadership'"). The AI will ask for clarification if unsure.
    *   **Linking Objects:** Discuss relationships between elements (e.g., "Link the 'Create website' task to the 'Online Presence' solution as 'broken into'"). The AI will establish these connections.
    *   **Deleting Objects:** Tell the AI the name or description of the object to remove (e.g., "Delete the 'High Cost' risk").
    *   **Deleting Links:** Specify the two objects and the relationship type to remove (e.g., "Delete the link between 'Task A' and 'Solution B' as 'requires'").
    *   **Project Summary:** Ask the AI to "Show project summary" to see a list of all current objects and their relationships.

4.  **View Diagrams:**
    *   As you discuss your project, the AI automatically updates `project_diagram.md` and `project_diagram.svg`.
    *   To view the interactive diagram in your web browser (with zoom and pan), tell the AI: **"Open the diagram viewer."**
    *   **Troubleshooting the Diagram Viewer:** If the browser doesn't open automatically, or the page is blank/shows an error, you can manually start the viewer server in a separate terminal session:
        1.  Open a new terminal window.
        2.  Navigate to your project directory: `cd /path/to/your/project`
        3.  Activate your virtual environment: `source .venv/bin/activate`
        4.  Run the viewer: `python viewer.py`
        5.  Then, manually open your web browser and go to: `http://127.0.0.1:5000/`

5.  **Exiting the application:**
    *   Press `Ctrl+C` in the terminal where the AI is running.

## Mermaid Diagram Visualization

Architect AI uses Mermaid to generate visual diagrams of your project structure. These are saved as `project_diagram.md` (Mermaid code) and `project_diagram.svg` (SVG image) in the project root.

To view the `.md` file, you can use any markdown editor with Mermaid support (e.g., VS Code with a Mermaid extension, Typora, Obsidian).

## Dependency Management (`uv.lock`)

This project uses `uv` for dependency management. The `uv.lock` file, located in the project's root directory, ensures reproducible builds by precisely recording the exact versions and cryptographic hashes of all direct and transitive dependencies. This guarantees that your development environment is consistent and helps avoid dependency-related issues.

To set up your environment using `uv.lock`:

```bash
# Ensure uv is installed (e.g., pip install uv)
uv sync
```
