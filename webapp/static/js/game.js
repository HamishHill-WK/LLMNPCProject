// DOM elements
const outputDiv = document.getElementById('game-output');
const inputForm = document.getElementById('input-form');
const playerInput = document.getElementById('player-input');
const inputFormNPC = document.getElementById('input-form-npc');
const simulationInput = document.getElementById('simulation-input');
const npcDropdownA = document.getElementById('npc-dropdown-A');
const npcDropdownB = document.getElementById('npc-dropdown-B');

// AI config elements
const aiProviderDropdown = document.getElementById('ai-provider-dropdown');
const modelDropdown = document.getElementById('model-dropdown');
const apiKeyInput = document.getElementById('api-key-input');
const openaiApiKeyDiv = document.getElementById('openai-api-key');
const updateAiConfigButton = document.getElementById('update-ai-config');

// Inference time tracking variables
let requestStartTime = 0;

// Flag to track if a request is in progress
let isRequestInProgress = false;

// Auto-scroll to bottom of output
function scrollToBottom() {
    outputDiv.scrollTop = outputDiv.scrollHeight;
}

// Add message to the game output
function addMessage(message, className) {
    const messageDiv = document.createElement('div');
    messageDiv.className = className;
    messageDiv.textContent = message;
    outputDiv.appendChild(messageDiv);
    scrollToBottom();
}

// Update model dropdown options based on selected provider
function updateModelDropdown() {
    const provider = aiProviderDropdown.value;
    const models = availableModels[provider] || [];
    
    // Show/hide API key input for OpenAI
    openaiApiKeyDiv.style.display = provider === 'openai' ? 'block' : 'none';
    
    // Clear existing options
    modelDropdown.innerHTML = '';
    
    // Add new options
    models.forEach(model => {
        const option = document.createElement('option');
        option.value = model;
        option.textContent = model;
        modelDropdown.appendChild(option);
    });
}

function changeNPC(npc) {
    fetch('/api/change_npc', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ npc_id: npc }),
    })
    .then(response => response.json())
    .then(data => {
        // Add NPC response
        addMessage(`NPC changed to ${npc}`, 'system-message');
    })
    .catch(error => {
        console.error('Error:', error);
        addMessage('Error communicating with the server.', 'system-message');
    });
}

function changeSimNPC(npc, target) {
    fetch('/api/change_sim_npc', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ npc_id: npc, target_id: target }),
    })
    .then(response => response.json())
    .then(data => {
        // Add NPC response
        addMessage(`NPC ${target} changed to ${npc}`, 'system-message');
    })
    .catch(error => {
        console.error('Error:', error);
        addMessage('Error communicating with the server.', 'system-message');
    });
}

// Initialize event listeners
function initializeEventListeners() {
    // Handle form submission with AJAX
    inputForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Prevent multiple submissions while a request is in progress
        if (isRequestInProgress) {
            console.log('Request already in progress, ignoring submission');
            return;
        }
        
        const input = playerInput.value.trim();
        if (!input) return;
        
        // Add player input to display
        addMessage('> ' + input, 'player-input');
        
        // Clear input field
        playerInput.value = '';
        
        // Record start time
        requestStartTime = performance.now();
        
        // Set flag to indicate request is in progress
        isRequestInProgress = true;
        
        // Disable the input and submit button during request
        playerInput.disabled = true;
        const submitButton = inputForm.querySelector('button[type="submit"]');
        if (submitButton) submitButton.disabled = true;
        
        // Send request to server
        fetch('/api/interact', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                player_input: input,
                // Add a unique request ID to prevent duplicates
                request_id: Date.now().toString() + Math.random().toString(36).substr(2, 9)
            }),
        })
        .then(response => response.json())
        .then(data => {
            // Calculate response time
            const responseTime = performance.now() - requestStartTime;
            
            // Add NPC response
            addMessage(data.response, 'npc-response');
            
            // Add timing info as a system message
            addMessage(`Response generated in ${responseTime.toFixed(2)}ms`, 'system-message timing-info');
        })
        .catch(error => {
            console.error('Error:', error);
            addMessage('Error communicating with the server.', 'system-message');
        })
        .finally(() => {
            // Reset flag and re-enable inputs regardless of success/failure
            isRequestInProgress = false;
            playerInput.disabled = false;
            if (submitButton) submitButton.disabled = false;
            playerInput.focus();
        });
    });

    inputFormNPC.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const input = simulationInput.value.trim();
        if (!input) return;
        
        // Add simulation input to display
        addMessage('> ' + input, 'player-input');
        
        // Clear input field
        simulationInput.value = '';
        
        // Close any existing EventSource
        if (window.eventSource) {
            window.eventSource.close();
        }
        
        // Record start time for simulation
        const simulationStartTime = performance.now();
        let messageCount = 0;
        
        // Create a new EventSource for streaming
        console.log('NPC A:', npcDropdownA.value);
        console.log('NPC B:', npcDropdownB.value);
        console.log('Input:', input);
        const url = `/api/simulate_stream?simulation_input=${encodeURIComponent(input)}&npc_a=${encodeURIComponent(npcDropdownA.value)}&npc_b=${encodeURIComponent(npcDropdownB.value)}`;
        const eventSource = new EventSource(url);
        window.eventSource = eventSource;
        
        // Handle incoming message events
        eventSource.onmessage = function(event) {
            try {
                const data = JSON.parse(event.data);
                // Add the appropriate NPC's message to the display
                addMessage(`${data.speaker}: ${data.message}`, 'npc-response');
                messageCount++;
            } catch (error) {
                console.error('Error parsing SSE message:', error);
                addMessage(event.data, 'npc-response'); // Fallback to raw data
            }
        };
        
        // Handle end of conversation
        eventSource.addEventListener('end', function() {
            console.log('Conversation complete');
            const simulationTime = performance.now() - simulationStartTime;
            addMessage(`Simulation completed in ${simulationTime.toFixed(2)}ms (${messageCount} messages)`, 'system-message timing-info');
            eventSource.close();
        });
        
        // Handle errors
        eventSource.onerror = function() {
            console.error('SSE connection error');
            addMessage('Connection to server lost.', 'system-message');
            eventSource.close();
        };
    });

    // AI provider change event
    aiProviderDropdown.addEventListener('change', updateModelDropdown);
    
    // Update AI configuration
    updateAiConfigButton.addEventListener('click', function() {
        const provider = aiProviderDropdown.value;
        const model = modelDropdown.value;
        const apiKey = apiKeyInput.value.trim();
        
        // Show updating message
        addMessage('Updating AI configuration...', 'system-message');
        
        // Send request to update configuration
        fetch('/api/change_ai_config', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                provider: provider,
                model: model,
                api_key: apiKey
            }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Update available models in case they changed
                if (data.available_models) {
                    availableModels.ollama = data.available_models.ollama || [];
                    availableModels.openai = data.available_models.openai || [];
                    updateModelDropdown();
                }
                
                addMessage(`AI configuration updated. Using ${data.ai_provider} with model ${data.selected_model}`, 'system-message');
            } else {
                addMessage(`Error updating AI configuration: ${data.error || 'Unknown error'}`, 'system-message');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            addMessage(`Error communicating with the server: ${error}`, 'system-message');
        });
    });

    // Clean up when leaving the page
    window.addEventListener('beforeunload', function() {
        if (window.eventSource) {
            window.eventSource.close();
        }
    });
    
    // Clear Memories button handler
    document.getElementById('clear-memories').addEventListener('click', function(e) {
        e.preventDefault();
        
        if (confirm('Are you sure you want to clear all character memories? This action cannot be undone.')) {
            // Show clearing message
            addMessage('Clearing character memories...', 'system-message');
            
            // Send request to clear memories
            fetch('/api/clear_memories', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    addMessage('All character memories have been cleared successfully.', 'system-message');
                } else {
                    addMessage(`Error clearing memories: ${data.error || 'Unknown error'}`, 'system-message');
                }
            })
            .catch(error => {
                console.error('Error:', error);
                addMessage('Error communicating with the server.', 'system-message');
            });
        }
    });
    
    // Clear Knowledge button handler
    document.getElementById('clear-knowledge').addEventListener('click', function(e) {
        e.preventDefault();
        
        if (confirm('Are you sure you want to reset game knowledge to defaults? This action cannot be undone.')) {
            // Show clearing message
            addMessage('Resetting game knowledge to defaults...', 'system-message');
            
            // Send request to clear knowledge
            fetch('/api/clear_knowledge', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    addMessage('Game knowledge has been reset to defaults successfully.', 'system-message');
                } else {
                    addMessage(`Error resetting knowledge: ${data.error || 'Unknown error'}`, 'system-message');
                }
            })
            .catch(error => {
                console.error('Error:', error);
                addMessage('Error communicating with the server.', 'system-message');
            });
        }
    });
}

// Initialize the application
function initializeApp() {
    // Initialize the AI model dropdown
    updateModelDropdown();
    
    // Initialize event listeners
    initializeEventListeners();
    
    // Initialize the UI
    scrollToBottom();
    playerInput.focus();
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

// Clean up on page unload
window.addEventListener('beforeunload', function() {
    if (window.eventSource) {
        window.eventSource.close();
    }
});