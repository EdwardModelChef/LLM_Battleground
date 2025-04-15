# LLM Battleground - MVP

A minimal web application demonstrating two multimodal LLMs playing chess against each other based on visual board input.

## Setup

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <your-repo-url>
    cd LLM_Battleground
    ```

2.  **Create a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: You might need to install system libraries for `cairosvg` or `Pillow` depending on your OS (e.g., `brew install cairo libffi pkg-config` on macOS, `sudo apt-get install libcairo2-dev libffi-dev pkg-config` on Debian/Ubuntu).*

4.  **Set up environment variables:**
    *   Create a file named `.env` in the project root (`LLM_Battleground/`).
    *   Add your API keys to the `.env` file:
        ```env
        # GEMINI_API_KEY="your_actual_google_gemini_api_key"
        OPENAI_API_KEY="your_actual_openai_api_key"
        ANTHROPIC_API_KEY="your_actual_anthropic_api_key"
        ```
    *   **Important:** Ensure `.env` is listed in your `.gitignore` file to avoid committing your API keys.
    *   *Note:* The application will still run if some keys are missing, but the corresponding models will be unavailable for selection.

5.  **(Optional) Font Setup:** The backend uses Pillow to generate board images with piece characters. It tries to load a default system font. If you encounter errors related to fonts or see squares instead of pieces, you might need to:
    *   Install a common font like Arial or DejaVu Sans.
    *   Modify the `font_path` logic within the `generate_board_image_pil` function in `backend/app.py` to point to a valid `.ttf` font file on your system.

## Running the Application

1.  **Activate the virtual environment (if not already active):**
    ```bash
    source venv/bin/activate
    ```

2.  **Start the Flask backend server:**
    ```bash
    cd backend
    python app.py
    ```
    The server will typically start on `http://127.0.0.1:5001` or `http://0.0.0.0:5001`.

3.  **Open the web interface:**
    *   Open your web browser and navigate to the address provided by the Flask server (e.g., `http://127.0.0.1:5001`).

4.  **Select Models and Start the game:**
    *   Choose the desired LLM for the White and Black players using the dropdown menus.
    *   Click the "Start New Game" button.
    *   The selected LLMs will automatically start playing against each other. The board and status will update with each move.

## Project Structure

```
LLM_Battleground/
├── .env                # Stores environment variables (API Key) - **DO NOT COMMIT**
├── .gitignore          # Specifies intentionally untracked files
├── backend/
│   └── app.py          # Flask backend logic (game engine, LLM calls)
├── frontend/
│   ├── index.html      # Main HTML page
│   └── static/
│       ├── script.js   # Frontend JavaScript logic (UI updates, backend calls)
│       └── style.css   # Basic CSS styling
├── requirements.txt    # Python dependencies (Flask, chess, Pillow, openai, anthropic, ...)
└── README.md           # This file
```
