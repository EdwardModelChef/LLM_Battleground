# LLM Battleground

A web application that allows different Large Language Models (LLMs) to play chess against each other.

## Description

This project sets up a simple Flask web server that serves a chessboard interface. Users can select from available LLMs (currently supporting OpenAI's GPT-4o/O1 and Anthropic's Claude 3.5 Sonnet) to play as White and Black. The application handles game state, generates board images, sends prompts to the selected LLMs (including the board image and game history), parses their moves (using tool calling/function calling), and updates the board.

## Features

*   Select different LLM providers and models for White and Black pieces.
*   Customize system prompts for each player.
*   Visual chessboard display using PIL (Pillow) image generation.
*   Game state management using `python-chess`.
*   LLM move generation via API calls (OpenAI & Anthropic).
*   Tool/Function calling for structured LLM move output (Move + Reasoning).
*   Game history tracking (SAN notation).
*   Displays LLM reasoning (if provided) and raw output for the last move.
*   Automatic game end detection (checkmate, stalemate, etc.).
*   Simple web interface using Flask, HTML, CSS, and JavaScript.

## Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/EdwardModelChef/LLM_Battleground.git
    cd LLM_Battleground
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Create `.env` file:**
    Create a file named `.env` in the project root directory (`LLM_Battleground/`). Add your API keys:
    ```dotenv
    OPENAI_API_KEY="your_openai_api_key_here"
    ANTHROPIC_API_KEY="your_anthropic_api_key_here"
    # GEMINI_API_KEY="your_gemini_api_key_here" # (Currently disabled in backend)
    ```
5.  **Run the Flask application:**
    ```bash
    cd backend
    python app.py
    ```
    The server will typically start on `http://0.0.0.0:5001/`.

## Usage

1.  Open your web browser and navigate to `http://localhost:5001` (or the address shown in the terminal).
2.  Select the desired LLM for the White player and the Black player from the dropdown menus.
3.  (Optional) Modify the system prompts for each player.
4.  Click "Start Game".
5.  Click the "Make LLM Move" button to have the current player's selected LLM make a move.
6.  The board will update, and the status message will indicate the last move and whose turn it is next. The raw output and reasoning from the last LLM move will also be displayed.
7.  Continue clicking "Make LLM Move" until the game ends.

## Technologies Used

*   **Backend:** Python, Flask
*   **LLM Interaction:** OpenAI API, Anthropic API
*   **Chess Logic:** `python-chess`
*   **Image Generation:** Pillow (PIL Fork)
*   **Frontend:** HTML, CSS, JavaScript
*   **Environment Variables:** `python-dotenv`

## Future Plans

*   Implement more robust error handling and move validation.
*   Add support for more LLM providers/models.
*   Improve frontend UI/UX (e.g., highlighting last move, clearer status).
*   Allow human players to play against an LLM.
*   More sophisticated board image generation (e.g., using SVG).
*   Potential deployment options (e.g., Docker, cloud platforms).
*   GitHub Pages integration for project showcase.
