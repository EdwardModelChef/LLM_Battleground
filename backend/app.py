import os
from flask import Flask, send_from_directory, jsonify, request
import chess
import chess.svg
# import google.generativeai as genai # Remove Gemini import
from openai import OpenAI
import anthropic
# Pillow is needed for image manipulation
from PIL import Image, ImageDraw, ImageFont # Import ImageDraw and ImageFont
import io
import logging
from dotenv import load_dotenv
import time # Import time for potential delays
import base64 # Needed for OpenAI image encoding
import json # Needed for parsing tool arguments

# Configure logging (Moved up)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
load_dotenv(dotenv_path='../.env') # Load environment variables from .env file at the project root
DEFAULT_WHITE_SYSTEM_PROMPT = "You are an expert AI chess player controlling the White pieces. Analyze the board and choose the best move."
DEFAULT_BLACK_SYSTEM_PROMPT = "You are an expert AI chess player controlling the Black pieces. Analyze the board and choose the best move."

# Google Gemini - REMOVED
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# if not GEMINI_API_KEY:
#     logger.warning("GEMINI_API_KEY not found in .env file. Gemini models will not be available.")
#     gemini_model = None
# else:
#     try:
#         genai.configure(api_key=GEMINI_API_KEY)
#         # Use the appropriate Gemini model that supports vision input
#         gemini_model = genai.GenerativeModel('gemini-1.5-flash') # Or 'gemini-pro-vision'
#         logger.info("Gemini model initialized.")
#     except Exception as e:
#         logger.error(f"Failed to initialize Gemini model. Check API Key and model name. Error: {e}")
#         gemini_model = None

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not found in .env file. OpenAI models will not be available.")
    openai_client = None
else:
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        logger.info("OpenAI client initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        openai_client = None

# Anthropic
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    logger.warning("ANTHROPIC_API_KEY not found in .env file. Anthropic models will not be available.")
    anthropic_client = None
else:
    try:
        anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        logger.info("Anthropic client initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize Anthropic client: {e}")
        anthropic_client = None


# Use relative paths carefully based on where app.py is run from
# This assumes app.py is run from within the 'backend' directory
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))
STATIC_DIR = os.path.join(FRONTEND_DIR, 'static') # Static files are inside frontend/static

app = Flask(__name__, static_folder=STATIC_DIR) # Serve static files from frontend/static

# --- Global Game State (Simple approach for MVP) ---
# WARNING: This is NOT suitable for production (multiple users/games).
game_board = None
game_active = False
white_model_selection = None # Store selected model for White
black_model_selection = None # Store selected model for Black
white_base_prompt = DEFAULT_WHITE_SYSTEM_PROMPT # Store current prompt for White
black_base_prompt = DEFAULT_BLACK_SYSTEM_PROMPT # Store current prompt for Black
game_history_san = [] # Store game history as list of SAN moves
available_models = { # Define available models and their clients/names
    # "gemini-1.5-flash": gemini_model, # REMOVED Gemini
    "gpt-4o": openai_client,
    "o1": openai_client, # Add o1 back
    "claude-3-7-sonnet-20250219": anthropic_client
}


# --- Routes ---
@app.route('/')
def index():
    """Serve the main index.html file."""
    logger.info(f"Serving index.html from: {FRONTEND_DIR}")
    # Ensure the path is correct relative to the app's root or use absolute paths
    # send_from_directory needs the directory *containing* the file
    return send_from_directory(FRONTEND_DIR, 'index.html')

# Flask automatically serves files from the 'static_folder' at the '/static' URL path by default.
# So, <link rel="stylesheet" href="/static/style.css"> in HTML will work correctly.
# No explicit '/static/<path:path>' route is needed unless customizing the URL path.

@app.route('/start_game', methods=['POST'])
def start_game():
    """Initializes a new chess game with selected models and system prompts."""
    global game_board, game_active, white_model_selection, black_model_selection
    global white_base_prompt, black_base_prompt, game_history_san # Allow modification

    data = request.get_json()
    white_model = data.get('white_model')
    black_model = data.get('black_model')
    # Get prompts from request, fallback to defaults if missing/empty
    req_white_prompt = data.get('white_prompt')
    req_black_prompt = data.get('black_prompt')

    logger.info(f"Received start game request with White: {white_model}, Black: {black_model}")
    logger.info(f"Received White Prompt: {'<provided>' if req_white_prompt else '<default>'}")
    logger.info(f"Received Black Prompt: {'<provided>' if req_black_prompt else '<default>'}")

    # Validate models
    if white_model not in available_models or available_models[white_model] is None:
         logger.error(f"Invalid or unavailable model selected for White: {white_model}")
         return jsonify({"error": f"Invalid or unavailable model selected for White: {white_model}"}), 400
    if black_model not in available_models or available_models[black_model] is None:
         logger.error(f"Invalid or unavailable model selected for Black: {black_model}")
         return jsonify({"error": f"Invalid or unavailable model selected for Black: {black_model}"}), 400

    game_board = chess.Board()
    game_history_san = [] # Reset history on new game
    game_active = True
    white_model_selection = white_model
    black_model_selection = black_model

    # Update global prompts if provided, otherwise reset to default
    white_base_prompt = req_white_prompt if req_white_prompt else DEFAULT_WHITE_SYSTEM_PROMPT
    black_base_prompt = req_black_prompt if req_black_prompt else DEFAULT_BLACK_SYSTEM_PROMPT
    logger.info(f"Game starting with prompts:\n  White: {white_base_prompt}\n  Black: {black_base_prompt}")

    return jsonify({
        "status": f"Game started. White ({white_model_selection})'s Turn.",
        "fen": game_board.fen(),
        "turn": "white",
        "game_over": False,
        "result": None,
        "active_white_prompt": white_base_prompt, # Return the active prompts
        "active_black_prompt": black_base_prompt
    })

# --- Helper Functions ---

def generate_board_image_pil(board):
    """Generates a PNG image of the current board state using Pillow."""
    size = 400  # Image size
    square_size = size // 8
    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)

    light_square_color = (240, 217, 181)  # Light beige
    dark_square_color = (181, 136, 99)   # Brown

    # Define simple text representations for pieces (adjust font path if needed)
    try:
        # Try loading a common system font; adjust path if necessary
        # common_font_paths = ["/System/Library/Fonts/Supplemental/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "arial.ttf"]
        # font_path = next((p for p in common_font_paths if os.path.exists(p)), None)
        # if not font_path:
        #    logger.warning("Could not find Arial or DejaVuSans font. Using default PIL font.")
        #    font = ImageFont.load_default()
        # else:
        #    font = ImageFont.truetype(font_path, int(square_size * 0.7))
         # Let's just try the default font for simplicity now
         font = ImageFont.load_default(size=int(square_size * 0.7)) # Default font has size parameter
    except Exception as e:
        logger.error(f"Error loading font: {e}. Using default PIL font.")
        font = ImageFont.load_default()


    piece_symbols_map = { # Map chess.Piece symbols to display characters
        'P': 'P', 'N': 'N', 'B': 'B', 'R': 'R', 'Q': 'Q', 'K': 'K',
        'p': 'p', 'n': 'n', 'b': 'b', 'r': 'r', 'q': 'q', 'k': 'k'
    }

    for i in range(8):  # Rows (ranks), 0 to 7
        for j in range(8):  # Columns (files), 0 to 7
            # Determine square color
            color = light_square_color if (i + j) % 2 == 0 else dark_square_color
            # Draw the square
            draw.rectangle(
                [(j * square_size, i * square_size), ((j + 1) * square_size, (i + 1) * square_size)],
                fill=color
            )

            # Get piece at the square
            # chess.Board squares are indexed 0-63, A1=0, H8=63
            # PIL draws from top-left (0,0)
            # Map rank i (0-7 from top) and file j (0-7 from left) to chess square index
            rank_index = 7 - i # chess ranks are 1-8 bottom-up
            file_index = j     # chess files are a-h left-right
            square = chess.square(file_index, rank_index)
            piece = board.piece_at(square)

            if piece:
                piece_symbol = piece.symbol()
                display_char = piece_symbols_map.get(piece_symbol, '?') # Get text char
                piece_color = "black" if piece.color == chess.BLACK else "white"

                # Calculate text size and position to center it
                # Use textbbox for potentially better centering with variable-width fonts
                try:
                     bbox = draw.textbbox((0, 0), display_char, font=font)
                     text_width = bbox[2] - bbox[0]
                     text_height = bbox[3] - bbox[1]
                     text_x = j * square_size + (square_size - text_width) / 2
                     text_y = i * square_size + (square_size - text_height) / 2 - (bbox[1]) # Adjust for baseline
                     draw.text((text_x, text_y), display_char, fill=piece_color, font=font, stroke_width=1, stroke_fill="gray") # Add outline
                except Exception as e:
                     logger.error(f"Error drawing text for piece {piece_symbol}: {e}")
                     # Fallback drawing if textbbox fails
                     draw.text((j * square_size + square_size//3, i * square_size + square_size//3),
                              display_char, fill="red", font=ImageFont.load_default())


    # Save image to a bytes buffer
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_bytes = img_byte_arr.getvalue()
    logger.info(f"Generated board image using Pillow ({len(img_bytes)} bytes).")
    return img_bytes


# --- LLM Move Generation Functions ---

# Define the tool schema for submitting a chess move
submit_move_tool_schema = {
    "name": "submit_chess_move",
    "description": "Submits the next chess move to be played, including reasoning.",
    "input_schema": {
        "type": "object",
        "properties": {
            "move": {
                "type": "string",
                "description": "The chess move in Standard Algebraic Notation (SAN, e.g., 'Nf3', 'e4', 'O-O') or UCI notation (e.g., 'g1f3', 'e2e4').",
            },
            "reasoning": {
                "type": "string",
                "description": "A brief explanation (1-2 sentences) for why this move was chosen."
            }
        },
        "required": ["move"], # Only move is strictly required
    },
}

# REMOVED get_gemini_move function
# def get_gemini_move(board, board_image, player_color, model_name):
#     """Gets the next move from a Gemini model."""
#     # ... (entire function removed)


def _construct_llm_messages(board, game_history, base_system_prompt, color_str, model_name, base64_image):
    """Constructs the messages list for the OpenAI Chat Completions API call, including legal moves and requesting reasoning."""
    # Combine base prompt with specific instructions for the turn

    # Get legal moves
    legal_moves = list(board.legal_moves)
    legal_move_sans = sorted([board.san(move) for move in legal_moves])
    legal_moves_str = ", ".join(legal_move_sans)

    # Format game history
    history_str = " ".join(game_history)
    if not history_str:
        history_str = "[No moves yet]"

    # Assemble the structured prompt text
    full_prompt_text = (
        f"{base_system_prompt}\n\n"
        f"**Game History (SAN):**\n{history_str}\n\n"
        f"**Your Task:**\n"
        f"- You are playing {color_str} ({model_name}). It is your turn.\n"
        f"- Analyze the board image provided.\n"
        f"- Determine the best legal move.\n"
        f"- Submit the move and your reasoning using the 'submit_chess_move' tool.\n\n"
        f"**Constraints:**\n"
        f"- You MUST use the 'submit_chess_move' tool.\n"
        f"- Your chosen move MUST be one of the following legal moves: [{legal_moves_str}].\n"
        f"- Provide the chosen move EXACTLY as it appears in the list in the 'move' parameter.\n"
        f"- Provide a brief (1-2 sentence) explanation in the 'reasoning' parameter."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": full_prompt_text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}",
                        "detail": "auto"
                    },
                },
            ],
        }
    ]
    return messages

def get_openai_move(board, board_image, player_color, model_name):
    """Gets the next move from an OpenAI model using function calling.
    Returns a dictionary: {'move': move_string, 'raw_output': raw_output_representation}
    or None if an error occurs.
    """
    global white_base_prompt, black_base_prompt, game_history_san # Access globals
    if not openai_client:
        logger.error("OpenAI client not initialized.")
        return None

    color_str = "White" if player_color == chess.WHITE else "Black"
    base_prompt = white_base_prompt if player_color == chess.WHITE else black_base_prompt
    base64_image = base64.b64encode(board_image).decode('utf-8')

    # Construct messages using helper
    messages = _construct_llm_messages(board, game_history_san, base_prompt, color_str, model_name, base64_image)
    logger.info(f"Sending prompt to OpenAI ({model_name}) for {color_str} with tool calling...")

    try:
        if model_name == 'o1':
            # --- Use responses.create for o1 --- 
            logger.info(f"Using responses.create endpoint for o1 model.")

            # Get legal moves for o1 prompt
            legal_moves_o1 = list(board.legal_moves)
            legal_move_sans_o1 = sorted([board.san(move) for move in legal_moves_o1])
            legal_moves_str_o1 = ", ".join(legal_move_sans_o1)

            # Format game history for o1
            history_str_o1 = " ".join(game_history_san)
            if not history_str_o1:
                history_str_o1 = "[No moves yet]"

            # Assemble structured prompt for o1
            o1_prompt_text = (
                f"{base_prompt}\n\n"
                f"**Game History (SAN):**\n{history_str_o1}\n\n"
                f"**Your Task:**\n"
                f"- You are playing {color_str} ({model_name}). It is your turn.\n"
                f"- Analyze the board image provided.\n"
                f"- Determine the best legal move.\n"
                f"- Submit the move and your reasoning using the 'submit_chess_move' tool.\n\n"
                f"**Constraints:**\n"
                f"- You MUST use the 'submit_chess_move' tool.\n"
                f"- Your chosen move MUST be one of the following legal moves: [{legal_moves_str_o1}].\n"
                f"- Provide the chosen move EXACTLY as it appears in the list in the 'move' parameter.\n"
                f"- Provide a brief (1-2 sentence) explanation in the 'reasoning' parameter."
            )

            # Construct input based on responses.create vision example structure
            o1_input = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            # Combine base prompt + turn instructions + legal moves + reasoning request for text part
                            "text": o1_prompt_text
                        },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{base64_image}"
                        }
                    ]
                }
            ]

            # Define tool for responses.create (assuming similar structure)
            o1_tool = {
                "type": "function",
                "name": submit_move_tool_schema["name"],
                "description": submit_move_tool_schema["description"],
                "parameters": submit_move_tool_schema["input_schema"] # Schema now includes reasoning
            }

            # Call responses.create
            response = openai_client.responses.create(
                model="o1",
                input=o1_input,
                text={"format": {"type": "text"}}, # Request text output (might contain tool call)
                reasoning={"effort": "medium"}, # Or "low"/"high"
                tools=[o1_tool]
                # Note: No direct equivalent to tool_choice is documented for responses.create
            )
            # logger.debug(f"Raw OpenAI Response (o1) object: {response}")

            # --- Response Parsing for o1 (Confirmed Structure) ---
            raw_output_repr = {"endpoint": "responses.create"} # Initialize raw output dict
            extracted_move = None
            extracted_reasoning = None
            tool_call_found = False

            # Iterate through the output list to find the function call
            if hasattr(response, 'output') and isinstance(response.output, list):
                raw_output_repr["output_list"] = [] # Store raw output items
                for item in response.output:
                    item_dict = {} # Create dict representation of item
                    if hasattr(item, 'type'): item_dict['type'] = item.type
                    if hasattr(item, 'name'): item_dict['name'] = item.name
                    if hasattr(item, 'arguments'): item_dict['arguments_str'] = item.arguments # Store raw arguments string
                    if hasattr(item, 'text'): item_dict['text'] = item.text # For potential text outputs
                    raw_output_repr["output_list"].append(item_dict)

                    # Check if this item is the tool call we want
                    if hasattr(item, 'type') and item.type == 'function_call' and hasattr(item, 'name') and item.name == submit_move_tool_schema["name"]:
                        tool_call_found = True
                        if hasattr(item, 'arguments') and isinstance(item.arguments, str):
                            try:
                                arguments_dict = json.loads(item.arguments) # Parse the JSON string
                                extracted_move = arguments_dict.get("move")
                                extracted_reasoning = arguments_dict.get("reasoning", "[No reasoning provided]") # Get reasoning
                                if extracted_move:
                                     logger.info(f"o1 responses.create - Extracted move: '{extracted_move}', Reasoning: '{extracted_reasoning}'")
                                     # Add parsed args to raw output for clarity
                                     item_dict['parsed_arguments'] = arguments_dict
                                     raw_output_repr['reasoning'] = extracted_reasoning # Add reasoning to main raw output
                                else:
                                     logger.error(f"o1 responses.create - 'move' missing in parsed arguments: {arguments_dict}")
                            except json.JSONDecodeError:
                                logger.error(f"o1 responses.create - Failed to parse arguments JSON: {item.arguments}")
                        else:
                             logger.error(f"o1 responses.create - Tool call arguments missing or not a string: {item}")
                        break # Found the correct tool call, no need to check further

            # Fallback if the expected tool call wasn't found
            if not tool_call_found:
                 logger.warning(f"o1 responses.create - Did not find expected tool call '{submit_move_tool_schema['name']}' in response output.")
                 # Check for plain text output as a last resort
                 if hasattr(response, 'output_text') and response.output_text:
                      extracted_move = response.output_text.strip()
                      logger.warning(f"o1 responses.create - Using output_text as fallback move: '{extracted_move}'")
                      raw_output_repr["fallback_text"] = extracted_move
                      raw_output_repr['reasoning'] = extracted_reasoning # Add reasoning to main raw output
                 else:
                      extracted_move = None

            return {"move": extracted_move, "raw_output": raw_output_repr}
            # --- End o1 specific logic ---

        else: # Assume gpt-4o or other models using chat.completions
            # --- Use chat.completions.create for gpt-4o etc. --- 
            logger.info(f"Using chat.completions.create endpoint for {model_name} model.")
            # Construct messages using the updated helper function (now includes history)
            messages = _construct_llm_messages(board, game_history_san, base_prompt, color_str, model_name, base64_image)
            
            # Define tool for chat.completions (schema now includes reasoning)
            chat_tool_definition = {
                "type": "function",
                "function": {
                    "name": submit_move_tool_schema["name"],
                    "description": submit_move_tool_schema["description"],
                    "parameters": submit_move_tool_schema["input_schema"] # Schema now includes reasoning
                }
            }

            response = openai_client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=[chat_tool_definition],
                tool_choice={"type": "function", "function": {"name": submit_move_tool_schema["name"]}}
            )

            message = response.choices[0].message

            # Parse response (existing logic)
            raw_output_repr = None

            if message.tool_calls:
                tool_call = message.tool_calls[0]
                raw_output_repr = {"endpoint": "chat.completions", "tool_name": tool_call.function.name, "arguments": tool_call.function.arguments}
                if tool_call.function.name == submit_move_tool_schema["name"]:
                    try:
                        arguments = json.loads(tool_call.function.arguments)
                        move_str = arguments.get("move")
                        reasoning_str = arguments.get("reasoning", "[No reasoning provided]") # Get reasoning
                        if move_str:
                            logger.info(f"ChatCompletions ({color_str}, {model_name}) - Extracted move: '{move_str}', Reasoning: '{reasoning_str}'")
                            raw_output_repr['parsed_reasoning'] = reasoning_str # Add reasoning to raw output
                            return {"move": move_str.strip(), "raw_output": raw_output_repr}
                        else:
                            logger.error(f"ChatCompletions ({color_str}, {model_name}) - 'move' missing: {arguments}")
                            return {"move": None, "raw_output": raw_output_repr}
                    except Exception as e:
                        logger.error(f"ChatCompletions ({color_str}, {model_name}) - Error processing args: {e}", exc_info=True)
                        return {"move": None, "raw_output": raw_output_repr}
                else:
                    logger.warning(f"ChatCompletions ({color_str}, {model_name}) called unexpected tool: {tool_call.function.name}")
                    return {"move": None, "raw_output": raw_output_repr}
            else:
                llm_text = message.content.strip() if message.content else ""
                raw_output_repr = {"endpoint": "chat.completions", "text_response": llm_text}
                logger.warning(f"ChatCompletions ({color_str}, {model_name}) did not use tool. Response: '{llm_text}'")
                return {"move": llm_text, "raw_output": raw_output_repr}
            # --- End chat.completions logic ---

    except Exception as e:
        logger.error(f"Error during OpenAI API call ({model_name}): {e}", exc_info=True)
        return None


def get_anthropic_move(board, board_image, player_color, model_name):
    """Gets the next move from an Anthropic model using tool calling.
    Returns a dictionary: {'move': move_string, 'raw_output': raw_output_representation}
    or None if an error occurs.
    """
    global white_base_prompt, black_base_prompt, game_history_san # Access globals
    if not anthropic_client:
        logger.error("Anthropic client not initialized.")
        return None

    color_str = "White" if player_color == chess.WHITE else "Black"
    base_prompt = white_base_prompt if player_color == chess.WHITE else black_base_prompt
    base64_image = base64.b64encode(board_image).decode('utf-8')

    # Format game history for Anthropic
    history_str_claude = " ".join(game_history_san)
    if not history_str_claude:
        history_str_claude = "[No moves yet]"

    # --- Add this block to get legal moves ---
    legal_moves = list(board.legal_moves)
    legal_moves_str = ', '.join([board.san(move) for move in legal_moves])
    if not legal_moves_str:
        legal_moves_str = "[No legal moves available]"
    # --- End added block ---

    # Construct messages using the structured format
    full_prompt_text = (
        f"{base_prompt}\n\n"
        f"**Game History (SAN):**\n{history_str_claude}\n\n"
        f"**Your Task:**\n"
        f"- You are playing {color_str} ({model_name}). It is your turn.\n"
        f"- Analyze the board image provided.\n"
        f"- Determine the best legal move.\n"
        f"- Submit the move and your reasoning using the 'submit_chess_move' tool.\n\n"
        f"**Constraints:**\n"
        f"- You MUST use the 'submit_chess_move' tool.\n"
        f"- Your chosen move MUST be one of the following legal moves: [{legal_moves_str}].\n"
        f"- Provide the chosen move EXACTLY as it appears in the list in the 'move' parameter.\n"
        f"- Provide a brief (1-2 sentence) explanation in the 'reasoning' parameter."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64_image,
                    },
                },
                {"type": "text", "text": full_prompt_text}
            ],
        }
    ]
    logger.info(f"Sending prompt to Anthropic ({model_name}) for {color_str} with tool calling...")

    try:
        response = anthropic_client.messages.create( # Correct endpoint
            model=model_name,
            max_tokens=100,
            tools=[submit_move_tool_schema],
            tool_choice={"type": "tool", "name": submit_move_tool_schema["name"]},
            messages=messages
        )

        move_str = None
        raw_output_repr = None
        reasoning_str = None
        text_response = ""

        if response.content and isinstance(response.content, list):
            for block in response.content:
                if block.type == 'tool_use' and block.name == submit_move_tool_schema["name"]:
                    raw_output_repr = {"tool_name": block.name, "input": block.input} # Store raw tool use
                    move_str = block.input.get("move")
                    reasoning_str = block.input.get("reasoning", "[No reasoning provided]") # Get reasoning
                    if move_str:
                        logger.info(f"Anthropic Tool Use ({color_str}, {model_name}) - Extracted move: '{move_str}', Reasoning: '{reasoning_str}'")
                        move_str = move_str.strip()
                        raw_output_repr['parsed_reasoning'] = reasoning_str # Add reasoning
                        # Keep processing other blocks in case there's text too
                    else:
                        logger.error(f"Anthropic Tool Use ({color_str}, {model_name}) - 'move' missing: {block.input}")
                        # Move extraction failed, but we store raw output
                elif block.type == 'text':
                    text_response += block.text

        # Prioritize move from tool call if found
        if move_str:
            return {"move": move_str, "raw_output": raw_output_repr}

        # If no successful tool call for move, check if there was other output
        if raw_output_repr: # A tool was called but it wasn't the right one or move was missing
             return {"move": None, "raw_output": raw_output_repr}
        elif text_response:
            # Fallback if only text was returned
            raw_output_repr = {"text_response": text_response.strip()}
            logger.warning(f"Anthropic ({color_str}, {model_name}) did not use tool. Response: '{text_response.strip()}'")
            return {"move": text_response.strip(), "raw_output": raw_output_repr}
        else:
            # No usable output at all
            logger.error(f"Anthropic ({color_str}, {model_name}) - No tool use or text content found in response.")
            return {"move": None, "raw_output": {"error": "No content found"}}

    except Exception as e:
        logger.error(f"Error during Anthropic API call ({model_name}): {e}", exc_info=True)
        return None

def parse_llm_response_for_move(board, extracted_move_str, color_str, model_name):
    """Parses the extracted move string (from tool call or text) to find a legal move."""
    if not extracted_move_str:
         logger.error(f"No move string provided to parse ({color_str}, {model_name}).")
         return None

    logger.info(f"Attempting to parse extracted move from {model_name}: '{extracted_move_str}'")
    cleaned_move_str = extracted_move_str.strip('.,!?;:"\'()[]{} ')

    # --- Pre-processing: Handle common non-standard notations --- START ---
    # Remove 'P-' prefix sometimes added by models for pawn moves
    if cleaned_move_str.startswith("P-") and len(cleaned_move_str) > 2:
        potential_san = cleaned_move_str[2:]
        logger.info(f"Removed 'P-' prefix, attempting to parse '{potential_san}' as SAN.")
        cleaned_move_str = potential_san
    # Add other pre-processing rules here if needed
    # --- Pre-processing: Handle common non-standard notations --- END ---

    if not cleaned_move_str:
        logger.error(f"Cleaned move string is empty ({color_str}, {model_name}). Original: '{extracted_move_str}'")
        return None

    # Try parsing as SAN first (most common)
    try:
        move = board.parse_san(cleaned_move_str)
        logger.info(f"Successfully parsed SAN move: {move.uci()} for '{cleaned_move_str}' ({model_name})")
        # Check legality (parse_san doesn't guarantee legality if ambiguous like 'e4')
        if move in board.legal_moves:
             return move
        else:
             # Try resolving ambiguity if needed (e.g., pawn captures)
             # This part might need refinement depending on how ambiguous SAN is handled
             logger.warning(f"Parsed SAN move {move.uci()} for '{cleaned_move_str}' ({model_name}) is ambiguous or illegal. Checking legal moves.")
             # If SAN was ambiguous (like Ne5 when two knights can go there),
             # python-chess might return one possibility. Check if *any* legal move
             # results in the same SAN.
             for legal_move in board.legal_moves:
                  if board.san(legal_move) == cleaned_move_str:
                       logger.info(f"Resolved ambiguous SAN '{cleaned_move_str}' to legal move: {legal_move.uci()}")
                       return legal_move
             logger.error(f"Parsed SAN move {move.uci()} for '{cleaned_move_str}' ({model_name}) is illegal and ambiguity resolution failed.")
             return None # Explicitly illegal even after parsing

    except ValueError as e_san:
        logger.warning(f"Parsing '{cleaned_move_str}' as SAN failed ({model_name}): {e_san}. Trying UCI...")
        # Try parsing as UCI if SAN failed
        try:
            # Ensure UCI is lowercase for parsing
            uci_move_str = cleaned_move_str.lower()
            # Basic UCI format check (e.g., length 4 or 5 for promotion)
            if len(uci_move_str) < 4 or len(uci_move_str) > 5:
                 raise ValueError("Invalid UCI length")

            move = board.parse_uci(uci_move_str)
            # parse_uci checks for legality implicitly
            logger.info(f"Successfully parsed UCI move: {move.uci()} for '{cleaned_move_str}' ({model_name})")
            return move
        except ValueError as e_uci:
            logger.error(f"Parsing '{cleaned_move_str}' as UCI also failed ({model_name}): {e_uci}")
            return None

    # Should not be reached if SAN or UCI parsing succeeded or failed cleanly
    logger.error(f"Failed to parse '{cleaned_move_str}' as either SAN or UCI ({model_name}).")
    return None


# --- Main Move Logic ---

def get_llm_move(board, board_image, player_color):
    """Dispatcher function to get the move from the LLM.
    Returns a dictionary: {'parsed_move': chess.Move | None, 'raw_output': raw_output_representation | None}
    """
    global white_model_selection, black_model_selection

    if player_color == chess.WHITE:
        model_name = white_model_selection
    else: # Black's turn
        model_name = black_model_selection

    color_str = "White" if player_color == chess.WHITE else "Black"
    llm_response_dict = None # Will hold {'move': str|None, 'raw_output': dict|None}

    logger.info(f"Requesting move for {color_str} using model: {model_name}")

    # Dispatch based on model name - o1 now goes to get_openai_move
    if model_name in ["gpt-4o", "o1"]:
        llm_response_dict = get_openai_move(board, board_image, player_color, model_name)
    elif model_name == "claude-3-7-sonnet-20250219":
        llm_response_dict = get_anthropic_move(board, board_image, player_color, model_name)
    else:
        logger.error(f"Unknown or unsupported model selected: {model_name}")
        return {'parsed_move': None, 'raw_output': {"error": f"Unknown model: {model_name}"}} # Return error info

    if llm_response_dict is None:
        logger.error(f"LLM API call failed for {model_name} ({color_str}).")
        return {'parsed_move': None, 'raw_output': {"error": "API call failed"}}

    extracted_move_str = llm_response_dict.get('move')
    raw_output = llm_response_dict.get('raw_output')

    if extracted_move_str is None and raw_output and raw_output.get('tool_name') == submit_move_tool_schema["name"]:
         logger.warning(f"Tool {submit_move_tool_schema['name']} called by {model_name} but move extraction failed.")
         # Proceed to parse with None move string, parse function will handle it

    # Parse the extracted move string (or None if extraction failed)
    parsed_move = parse_llm_response_for_move(board, extracted_move_str, color_str, model_name)

    if parsed_move is None:
        logger.error(f"Failed to parse valid/legal move from {model_name} ({color_str}). Extracted: '{extracted_move_str}'")
        # Return failure, but include the raw output for debugging/display
        return {'parsed_move': None, 'raw_output': raw_output}

    # Return the validated move and the raw output
    return {'parsed_move': parsed_move, 'raw_output': raw_output}


@app.route('/make_move', methods=['POST'])
def make_move():
    """Handles the request to make the next LLM move."""
    global game_board, game_active, white_model_selection, black_model_selection, game_history_san

    if not game_active or game_board is None:
        logger.warning("Make move request received but game is not active.")
        return jsonify({"error": "Game not started or already finished"}), 400

    if game_board.is_game_over():
        logger.info("Make move request received but game is already over.")
        return jsonify({"error": "Game is already over"}), 400

    player_color = game_board.turn
    color_str = "White" if player_color == chess.WHITE else "Black"
    current_model = white_model_selection if player_color == chess.WHITE else black_model_selection

    logger.info(f"--- {color_str}'s Turn ({current_model}) ---")

    # 1. Generate Board Image
    try:
        board_image = generate_board_image_pil(game_board)
        if not board_image:
             raise RuntimeError("Pillow image generation returned empty data.")
        logger.info("Board image generated successfully using Pillow.")
    except Exception as e:
        logger.error(f"Failed to generate board image: {e}", exc_info=True)
        game_active = False
        return jsonify({"error": f"Failed to generate board image: {e}", "game_over": True, "result": "Error - Image Generation Failed"}), 500

    # 2. Get Move from LLM (returns dict with 'parsed_move' and 'raw_output')
    move_info = get_llm_move(game_board, board_image, player_color)
    llm_move = move_info.get('parsed_move')
    raw_llm_output = move_info.get('raw_output') # Get the raw output

    # Add a small delay
    time.sleep(1)

    if llm_move is None:
        logger.error(f"LLM ({current_model}) failed to provide a valid/legal move for {color_str}.")
        game_active = False
        # Include raw output in error response if available
        error_payload = {
            "status": f"Game Halted: LLM ({current_model}, {color_str}) failed to provide a valid/legal move.",
            "fen": game_board.fen(),
            "turn": color_str.lower(),
            "game_over": True,
            "result": "Halted - Invalid LLM Move",
            "llm_output": raw_llm_output # Send raw output back
        }
        return jsonify(error_payload), 200

    # 3. Apply Move (Validation check removed as parse_llm_response handles it)
    move_san = game_board.san(llm_move)
    logger.info(f"Applying valid move for {color_str} ({current_model}): {llm_move.uci()} ({move_san})")
    game_board.push(llm_move)

    # --- Append move to history --- START ---
    # Format move number and SAN (e.g., "1.", "1...", "2.")
    move_num_str = ""
    if player_color == chess.WHITE: # White's move, add move number
        move_num_str = f"{game_board.fullmove_number}."
    else: # Black's move, just add ellipsis if needed (or nothing if first move)
        # Check if it's the very first half-move (Black on move 1)
        if game_board.fullmove_number == 1 and len(game_history_san) == 1:
             move_num_str = ""
        else:
             move_num_str = "" # Don't add number for black, just the move
        # Alternative: Add ellipsis for Black's move like "1... e5"
        # if game_board.fullmove_number >= 1 and len(game_history_san) > 0 :
        #      move_num_str = f"{game_board.fullmove_number}..."

    # Add move number (if white) and the move SAN
    history_entry = f"{move_num_str} {move_san}".strip() # Remove leading space if move_num_str is empty
    game_history_san.append(history_entry)
    logger.info(f"Updated game history: {' '.join(game_history_san)}")
    # --- Append move to history --- END ---

    # 4. Check Game End
    game_over = game_board.is_game_over()
    result = None
    status_message = ""
    if game_over:
        game_active = False
        outcome = game_board.outcome()
        if outcome:
            winner_color = "White" if outcome.winner == chess.WHITE else "Black" if outcome.winner == chess.BLACK else "Nobody"
            termination = outcome.termination.name.title().replace("_", " ")
            if outcome.winner is not None:
                 result = f"{winner_color} Wins - {termination}"
            else: # Draw
                 result = f"Draw - {termination}"
            logger.info(f"Game Over. Result: {result}")
            status_message = f"Game Over: {result}"
        else:
             result = "Game Over - Unknown Outcome"
             logger.warning("Game is over, but outcome object is None.")
             status_message = "Game Over: Unknown Outcome"
    else:
        next_turn_color = "White" if game_board.turn == chess.WHITE else "Black"
        next_model = white_model_selection if game_board.turn == chess.WHITE else black_model_selection
        status_message = f"Move: {move_san}. {next_turn_color}'s Turn ({next_model})."


    # 5. Return new state, including the raw LLM output for the last move
    return jsonify({
        "status": status_message,
        "fen": game_board.fen(),
        "turn": "white" if game_board.turn == chess.WHITE else "black",
        "last_move_san": move_san,
        "game_over": game_over,
        "result": result,
        "llm_output": raw_llm_output # Add raw LLM output to the response
    })


# --- Main Execution ---
if __name__ == '__main__':
    # Reminders:
    # 1. Run `pip install -r requirements.txt` (from the project root)
    # 2. Create a `.env` file in the *project root* (LLM_Battleground)
    # 3. Add `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` to the `.env` file. (Removed GEMINI_API_KEY)
    # 4. Make sure you have access to the respective model APIs.
    # 5. Ensure Pillow can find a font or adjust font path in generate_board_image_pil.
    logger.info("Starting Flask server...")
    # Use host='0.0.0.0' to make it accessible on your network if needed
    app.run(debug=True, port=5001, host='0.0.0.0') # Running on port 5001
