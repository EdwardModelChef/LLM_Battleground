// Global variables
let board = null; // Chessboard object
let gameActive = false;
let currentTurn = 'white'; // Track whose turn it is

// --- DOM Elements ---
const boardElement = document.getElementById('game-board');
const statusElement = document.getElementById('status');
const startButton = document.getElementById('start-button');
const whiteModelSelect = document.getElementById('white-model'); // Get white model select
const blackModelSelect = document.getElementById('black-model'); // Get black model select
const whiteOutputElement = document.getElementById('white-output');
const blackOutputElement = document.getElementById('black-output');
const whitePromptTextarea = document.getElementById('white-prompt');
const blackPromptTextarea = document.getElementById('black-prompt');
const stopButton = document.getElementById('stop-button');

// Default prompts (can match backend defaults or be simpler)
const DEFAULT_WHITE_PROMPT_FRONTEND = "You are an expert AI chess player controlling the White pieces.";
const DEFAULT_BLACK_PROMPT_FRONTEND = "You are an expert AI chess player controlling the Black pieces.";

// --- Chessboard Configuration ---
const boardConfig = {
    draggable: false, // Pieces are not draggable by the user in MVP
    position: 'start', // Start with the initial setup
    pieceTheme: 'static/img/chesspieces/wikipedia/{piece}.png' // Use default images provided by chessboard.js CDN path or local
};

// --- Initialization ---
$(document).ready(function() {
    // Clear output boxes on load
    clearOutputHistory();
    // Set default prompt text
    whitePromptTextarea.value = DEFAULT_WHITE_PROMPT_FRONTEND;
    blackPromptTextarea.value = DEFAULT_BLACK_PROMPT_FRONTEND;
    board = Chessboard('game-board', boardConfig);
    updateStatus("Waiting to start...");

    // Event listener for the start button
    startButton.addEventListener('click', handleStartGame);
    // Event listener for the stop button
    stopButton.addEventListener('click', handleStopGame);
});

// --- Game Logic Functions ---

async function handleStartGame() {
    console.log("Start Game button clicked");
    // Disable controls
    startButton.disabled = true;
    stopButton.disabled = false; // Enable stop button
    whiteModelSelect.disabled = true;
    blackModelSelect.disabled = true;
    whitePromptTextarea.disabled = true;
    blackPromptTextarea.disabled = true;
    updateStatus("Initializing game...");

    const selectedWhiteModel = whiteModelSelect.value;
    const selectedBlackModel = blackModelSelect.value;
    const whitePrompt = whitePromptTextarea.value;
    const blackPrompt = blackPromptTextarea.value;

    console.log(`Selected Models - White: ${selectedWhiteModel}, Black: ${selectedBlackModel}`);
    console.log(`Sending Prompts - White: ${whitePrompt.substring(0,30)}..., Black: ${blackPrompt.substring(0,30)}...`);

    try {
        const response = await fetch('/start_game', {
             method: 'POST',
             headers: {
                 'Content-Type': 'application/json',
             },
             body: JSON.stringify({
                  white_model: selectedWhiteModel,
                  black_model: selectedBlackModel,
                  white_prompt: whitePrompt, // Send prompts
                  black_prompt: blackPrompt
             })
        });
        if (!response.ok) {
             const errorData = await response.json().catch(() => ({ error: `HTTP error! status: ${response.status}` })); // Try to get JSON error
             throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
        }
        const data = await response.json();

        // Update text areas with the prompts actually used by the backend
        if (data.active_white_prompt) {
            whitePromptTextarea.value = data.active_white_prompt;
        }
        if (data.active_black_prompt) {
            blackPromptTextarea.value = data.active_black_prompt;
        }

        console.log("Game started response:", data);
        gameActive = true;
        currentTurn = data.turn;
        board.position(data.fen, false); // Update board visually (false = no animation)
        updateStatus(data.status);

        // Start the game loop by triggering the first move (White)
        if (currentTurn === 'white') {
            // Slight delay to allow UI update
            setTimeout(makeNextMove, 500);
        } else {
             console.warn("Game started, but it's not White's turn?", data.turn);
             updateStatus("Error: Game started with unexpected turn.");
             gameActive = false; // Stop if state is weird
             enableControls(); // Re-enable controls
        }

    } catch (error) {
        console.error('Error starting game:', error);
        updateStatus(`Error starting game: ${error.message}`);
        enableControls(); // Re-enable controls on error
    }
}

async function makeNextMove() {
    // Store whose turn it is *before* making the move request
    const movingPlayer = currentTurn;
    if (!gameActive) {
        console.log("makeNextMove called but game is not active.");
        return; // Stop if game ended or halted
    }

    const thinkingColor = currentTurn.charAt(0).toUpperCase() + currentTurn.slice(1);
    updateStatus(`${thinkingColor} is thinking...`);

    try {
        const response = await fetch('/make_move', { method: 'POST' });

        // Check for non-JSON responses (e.g., 500 errors might return HTML)
        const contentType = response.headers.get("content-type");
        if (!gameActive) {
            console.log("Game stopped while waiting for move response.");
            return;
        }

        if (!response.ok) {
            let errorText = `HTTP error! status: ${response.status}`;
            if (contentType && contentType.indexOf("application/json") === -1) {
                errorText += ` - Server returned non-JSON response. Check server logs.`;
                // Potentially read response.text() here for more clues if needed, but careful with large HTML errors
            }
            throw new Error(errorText);
        }

        if (!(contentType && contentType.indexOf("application/json") !== -1)) {
            throw new Error("Server returned non-JSON response, even with status 200 OK. Check server.")
        }

        const data = await response.json();

        if (!gameActive) {
            console.log("Game stopped just after receiving move response.");
            return;
        }

        console.log("Make move response:", data);

        // --- Display LLM's output --- START ---
        let outputToDisplay = "[No output received]"; // Default message
        if (data.llm_output) {
            try {
                let attemptedMove = "N/A";
                let reasoning = "[No reasoning provided]";

                if (data.llm_output.tool_name && data.llm_output.arguments) {
                    // It's likely an OpenAI tool call (chat.completions)
                    const args = JSON.parse(data.llm_output.arguments);
                    attemptedMove = args.move || "N/A";
                    reasoning = args.reasoning || reasoning; // Use default if missing
                    outputToDisplay = `Tool: ${data.llm_output.tool_name}\nMove: ${attemptedMove}\nReasoning: ${reasoning}`;
                } else if (data.llm_output.tool_name && data.llm_output.input) {
                    // It's likely an Anthropic tool use
                    attemptedMove = data.llm_output.input.move || "N/A";
                    reasoning = data.llm_output.input.reasoning || reasoning; // Use default if missing
                    outputToDisplay = `Tool: ${data.llm_output.tool_name}\nMove: ${attemptedMove}\nReasoning: ${reasoning}`;
                } else if (data.llm_output.endpoint === "responses.create" && data.llm_output.output_list) {
                    // It's likely an OpenAI o1 call (responses.create)
                    // Find the tool call in the output list
                    let toolCallArgs = null;
                    for (const item of data.llm_output.output_list) {
                        if (item.type === 'function_call' && item.name === 'submit_chess_move') {
                            toolCallArgs = item.parsed_arguments; // Use pre-parsed args from backend
                            break;
                        }
                    }
                    if (toolCallArgs) {
                        attemptedMove = toolCallArgs.move || "N/A";
                        reasoning = toolCallArgs.reasoning || reasoning;
                        outputToDisplay = `Tool: submit_chess_move\nMove: ${attemptedMove}\nReasoning: ${reasoning}`;
                    } else if (data.llm_output.fallback_text) {
                         outputToDisplay = `Text Fallback: ${data.llm_output.fallback_text}`;
                    } else {
                        outputToDisplay = `[Tool call expected but not found for o1]`
                    }
                } else if (data.llm_output.text_response) {
                    // It's a text fallback
                    outputToDisplay = `Text Response: ${data.llm_output.text_response}`;
                } else if (data.llm_output.error) {
                    // Error reported from backend raw_output
                    outputToDisplay = `Error: ${data.llm_output.error}`;
                } else {
                    // Fallback to raw JSON if structure is unexpected
                    outputToDisplay = "[Unrecognized Output Structure]\n" + JSON.stringify(data.llm_output, null, 2);
                }
            } catch (e) {
                console.error("Error parsing llm_output:", e);
                outputToDisplay = "[Error parsing output Frontend]\n" + JSON.stringify(data.llm_output, null, 2);
            }
        }

        // Append to the correct player's output box history
        const outputElement = (movingPlayer === 'white') ? whiteOutputElement : blackOutputElement;
        // Add a separator if there's existing content
        if (outputElement.textContent !== '-') {
            outputElement.textContent += "\n---\n";
        }
        outputElement.textContent += `Turn ${Math.ceil(board.fullmove_number)} (${movingPlayer}):\n${outputToDisplay}`;
        outputElement.scrollTop = outputElement.scrollHeight; // Scroll to bottom
        // --- Display LLM's output --- END ---

        // --- Process game state response --- START ---
        if (data.result === "Halted - Invalid LLM Move" || data.error) {
             // Handle game halt / backend error
             console.error("Game Halted or Backend Error:", data.status || data.error);
             updateStatus(data.status || `Error: ${data.error}`); // Display halt/error status
             gameActive = false;
             enableControls(); // Re-enable controls on error/halt
             // Update board to the state *before* the failed move attempt
             if (data.fen) board.position(data.fen, false);
             return; // Stop processing further
        }

        // --- Update board and status for successful move --- START ---
        board.position(data.fen, true); // Update board visually (true = animate)
        updateStatus(data.status);
        currentTurn = data.turn;
        // --- Update board and status for successful move --- END ---

        // --- Check if game is over after successful move --- START ---
        if (data.game_over) {
            console.log("Game Over. Result:", data.result);
            updateStatus(`Game Over: ${data.result}`); // Display final result
            gameActive = false;
            enableControls(); // Re-enable controls on game over
        } else {
            // If game is not over, schedule the next move
            // Add a slight delay between moves for better visualization
            setTimeout(makeNextMove, 1000); // e.g., wait 1 second before next move
        }
        // --- Check if game is over after successful move --- END ---
        // --- Process game state response --- END ---

    } catch (error) {
        console.error('Error making move:', error);
        updateStatus(`Error making move: ${error.message}. Check console and backend logs.`);
        gameActive = false; // Stop the game loop on fetch/network error
        enableControls(); // Re-enable controls on error
    }
}

// --- Utility Functions ---

function updateStatus(message) {
    console.log("Status Update:", message);
    statusElement.textContent = `Game Status: ${message}`;
}

// Helper to enable/disable controls
function enableControls() {
    startButton.disabled = false;
    stopButton.disabled = true; // Disable stop button when game is not running
    whiteModelSelect.disabled = false;
    blackModelSelect.disabled = false;
    whitePromptTextarea.disabled = false;
    blackPromptTextarea.disabled = false;
}

// Helper to clear output history
function clearOutputHistory() {
    whiteOutputElement.textContent = '-';
    blackOutputElement.textContent = '-';
}

// --- Stop Game Handler ---
function handleStopGame() {
    console.log("Stop Game button clicked");
    if (gameActive) {
        gameActive = false;
        updateStatus("Game stopped by user.");
        enableControls(); // Re-enable controls
    }
}
