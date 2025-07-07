from flask import Flask, render_template, Response
import os
import threading
import webbrowser
import time

app = Flask(__name__)

@app.route('/')
def index():
    print("Viewer: / route accessed, rendering viewer.html")
    return render_template('viewer.html')

@app.route('/project_diagram.svg')
def get_svg():
    svg_path = os.path.join(os.getcwd(), 'project_diagram.svg')
    print(f"Viewer: /project_diagram.svg route accessed. Looking for: {svg_path}")
    try:
        with open(svg_path, 'r') as f:
            svg_content = f.read()
        print("Viewer: project_diagram.svg found and read successfully.")
        return Response(svg_content, mimetype='image/svg+xml')
    except FileNotFoundError:
        print("Viewer: ERROR - project_diagram.svg not found!")
        return "SVG file not found", 404
    except Exception as e:
        print(f"Viewer: ERROR - An unexpected error occurred while reading SVG: {e}")
        return f"Server Error: {e}", 500

def run_viewer():
    # Use a separate thread to open the browser after the server starts
    def open_browser_after_delay():
        # Give the server more time to start up
        time.sleep(3) 
        try:
            print("Viewer: Attempting to open browser to http://127.0.0.1:5000/")
            webbrowser.open('http://127.0.0.1:5000/')
        except Exception as e:
            print(f"Viewer: ERROR - Failed to open web browser: {e}")

    threading.Thread(target=open_browser_after_delay).start()
    print("Viewer: Starting Flask server...")
    app.run(debug=False, use_reloader=False) # Disable reloader for cleaner threading

if __name__ == '__main__':
    run_viewer()
