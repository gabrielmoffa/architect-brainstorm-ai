### Discontinued project, may not work as expected, a bit of Claude Code magic and will work as expected :) 

### How to Use the Architect AI

To start a conversation with the Architect AI and begin planning your project:

1.  **Run the application:**
    ```bash
    python -m architect.main
    ```

2.  **Interact via voice:**
    *   The AI will prompt you to press Enter to start recording. Press Enter once to start, and again to stop recording.
    *   Speak clearly into your microphone.
    *   The AI will transcribe your speech, process your request, and respond verbally.
    *   **To interrupt the AI while it's speaking (if using ChatterBox TTS), press Enter.**

3.  **Generate and View Diagrams:**
    *   As you discuss your project, the AI will automatically update the `project_diagram.md` and `project_diagram.svg` files.
    *   To view the interactive diagram in your web browser (with zoom and pan functionality), simply tell the AI: **"Open the diagram viewer."**

4.  **Key Interactions:**
    *   **Creating Objects:** Describe your project elements (Goals, Problems, Solutions, Tasks, Users, Resources, Risks, Metrics), and the AI will create them.
    *   **Updating Objects:** If you want to change an object's title or description, refer to it by its ID (e.g., "Update goal-xyz123 with new title 'My New Goal'").
    *   **Linking Objects:** When discussing relationships, the AI should automatically link objects. If not, you can explicitly ask (e.g., "Link task-abc to solution-def as 'broken into'").
    *   **Deleting Objects:** To remove an object, tell the AI its ID (e.g., "Delete task-abc").
    *   **Deleting Links:** To remove a specific relationship, tell the AI the IDs of the two objects and the relationship type (e.g., "Delete the link between task-abc and solution-def as 'broken into'").
    *   **Summarize Project:** Ask the AI to "Show project summary" to see a list of all current objects and their IDs.

5.  **Exiting the application:**
    *   Press `Ctrl+C` in the terminal where the AI is running.
