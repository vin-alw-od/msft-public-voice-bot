using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using System.Net.WebSockets;
using System.Text;
using System.Text.Json;

namespace EchoBot.Services
{
    /// <summary>
    /// OpenAI Realtime Audio Service - Proof of Concept Implementation
    /// Handles VAD + STT + LLM + TTS in a single service
    /// This is a PARALLEL implementation - does not replace existing services yet
    /// </summary>
    public class OpenAIRealtimeAudioService : IDisposable
    {
        private readonly ILogger<OpenAIRealtimeAudioService> _logger;
        private readonly IConfiguration _configuration;
        private readonly string _apiKey;
        private readonly string _endpoint;
        private readonly string _model;
        private readonly string _vadMode;
        private ClientWebSocket? _webSocket;
        private readonly SemaphoreSlim _connectionSemaphore = new(1, 1);
        private bool _isConnected = false;

        public OpenAIRealtimeAudioService(IConfiguration configuration, ILogger<OpenAIRealtimeAudioService> logger)
        {
            _logger = logger;
            _configuration = configuration;
            
            // Configuration with fallback defaults
            _apiKey = configuration.GetValue<string>("AppSettings:OpenAIRealtimeApiKey") ?? 
                     configuration.GetValue<string>("OpenAI:RealtimeApiKey") ?? "";
            
            _endpoint = configuration.GetValue<string>("AppSettings:OpenAIRealtimeEndpoint") ?? 
                       "wss://api.openai.com/v1/realtime";
            
            _model = configuration.GetValue<string>("AppSettings:OpenAIRealtimeModel") ?? 
                    "gpt-4o-realtime-preview";
            
            _vadMode = configuration.GetValue<string>("AppSettings:OpenAIRealtimeVADMode") ?? 
                      "semantic_vad";

            _logger.LogInformation("OpenAI Realtime Audio Service Configuration:");
            _logger.LogInformation("  Endpoint: {Endpoint}", _endpoint);
            _logger.LogInformation("  Model: {Model}", _model);
            _logger.LogInformation("  VAD Mode: {VADMode}", _vadMode);
            _logger.LogInformation("  API Key: {ApiKeyStatus}", string.IsNullOrEmpty(_apiKey) ? "NOT SET" : "SET");
        }

        /// <summary>
        /// Check if the service is properly configured
        /// </summary>
        public bool IsConfigured => !string.IsNullOrEmpty(_apiKey);

        /// <summary>
        /// Test basic connectivity to OpenAI Realtime API
        /// </summary>
        public async Task<bool> TestConnectivityAsync()
        {
            if (!IsConfigured)
            {
                _logger.LogWarning("OpenAI Realtime Audio Service not configured - missing API key");
                return false;
            }

            try
            {
                _logger.LogInformation("Testing OpenAI Realtime Audio connectivity...");
                
                await _connectionSemaphore.WaitAsync();
                
                try
                {
                    if (_webSocket != null)
                    {
                        _webSocket.Dispose();
                    }

                    _webSocket = new ClientWebSocket();
                    
                    // Add required headers for Azure OpenAI
                    if (_endpoint.Contains("cognitiveservices.azure.com"))
                    {
                        // Azure OpenAI endpoint
                        _webSocket.Options.SetRequestHeader("api-key", _apiKey);
                        _webSocket.Options.SetRequestHeader("OpenAI-Beta", "realtime=v1");
                    }
                    else
                    {
                        // Standard OpenAI endpoint
                        _webSocket.Options.SetRequestHeader("Authorization", $"Bearer {_apiKey}");
                        _webSocket.Options.SetRequestHeader("OpenAI-Beta", "realtime=v1");
                    }
                    
                    // Azure endpoint already includes deployment and api-version parameters
                    var uri = _endpoint.Contains("cognitiveservices.azure.com") 
                        ? new Uri(_endpoint) 
                        : new Uri($"{_endpoint}?model={_model}");
                    
                    var cancellationToken = new CancellationTokenSource(TimeSpan.FromSeconds(10)).Token;
                    await _webSocket.ConnectAsync(uri, cancellationToken);
                    
                    if (_webSocket.State == WebSocketState.Open)
                    {
                        _logger.LogInformation("Successfully connected to OpenAI Realtime Audio API");
                        _isConnected = true;
                        
                        // Send session configuration
                        await SendSessionConfigAsync();
                        
                        return true;
                    }
                    else
                    {
                        _logger.LogWarning("WebSocket connection failed: {State}", _webSocket.State);
                        return false;
                    }
                }
                finally
                {
                    _connectionSemaphore.Release();
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to connect to OpenAI Realtime Audio API");
                return false;
            }
        }

        /// <summary>
        /// Send initial session configuration
        /// </summary>
        private async Task SendSessionConfigAsync()
        {
            if (_webSocket?.State != WebSocketState.Open)
                return;

            var sessionConfig = new
            {
                type = "session.update",
                session = new
                {
                    modalities = new[] { "text", "audio" },
                    instructions = GetSurveySystemPrompt(),
                    voice = "alloy",
                    input_audio_format = "pcm16",
                    output_audio_format = "pcm16",
                    input_audio_transcription = new
                    {
                        model = "whisper-1"
                    },
                    turn_detection = new
                    {
                        type = _vadMode,
                        threshold = 0.5,
                        prefix_padding_ms = 300,
                        silence_duration_ms = 2000
                    }
                }
            };

            var json = JsonSerializer.Serialize(sessionConfig);
            var bytes = Encoding.UTF8.GetBytes(json);
            
            await _webSocket.SendAsync(
                new ArraySegment<byte>(bytes), 
                WebSocketMessageType.Text, 
                true, 
                CancellationToken.None
            );
            
            _logger.LogDebug("Sent session configuration to OpenAI Realtime API");
        }

        /// <summary>
        /// Get the system prompt for survey conversations
        /// </summary>
        private string GetSurveySystemPrompt()
        {
            return @"You are an AI survey agent conducting interviews about AI initiatives in organizations.

Your goal is to collect the following information in a natural, conversational way:
1. Initiative name - What is the AI initiative called?
2. Budget range - What is the approximate budget? (options: Under $10K, $10K-$50K, $50K-$100K, $100K-$500K, $500K+)
3. Timeline - When do you plan to implement this? (options: Next 3 months, 3-6 months, 6-12 months, 1+ years)
4. Team size - How many people will work on this initiative?
5. Primary use case - What will this AI solution primarily do?

Conversation guidelines:
- Be friendly and professional
- Ask one question at a time
- Wait for complete responses before moving to the next question  
- Confirm understanding when needed
- Use natural conversation flow
- Thank the user when all information is collected
- If the user wants to end early, be polite and thank them

Start by introducing yourself and asking about their AI initiative name.";
        }

        /// <summary>
        /// Simple text-only conversation test (no audio yet)
        /// </summary>
        public async Task<string> TestConversationAsync(string userMessage)
        {
            if (!_isConnected || _webSocket?.State != WebSocketState.Open)
            {
                var connected = await TestConnectivityAsync();
                if (!connected)
                {
                    return "Error: Could not connect to OpenAI Realtime API";
                }
            }

            try
            {
                // Send user message
                var userMessageEvent = new
                {
                    type = "conversation.item.create",
                    item = new
                    {
                        type = "message",
                        role = "user",
                        content = new[]
                        {
                            new
                            {
                                type = "input_text",
                                text = userMessage
                            }
                        }
                    }
                };

                var json = JsonSerializer.Serialize(userMessageEvent);
                var bytes = Encoding.UTF8.GetBytes(json);
                
                await _webSocket!.SendAsync(
                    new ArraySegment<byte>(bytes),
                    WebSocketMessageType.Text,
                    true,
                    CancellationToken.None
                );

                // Trigger response generation
                var responseEvent = new
                {
                    type = "response.create"
                };

                json = JsonSerializer.Serialize(responseEvent);
                bytes = Encoding.UTF8.GetBytes(json);
                
                await _webSocket.SendAsync(
                    new ArraySegment<byte>(bytes),
                    WebSocketMessageType.Text,
                    true,
                    CancellationToken.None
                );

                // Listen for response
                var buffer = new byte[4096];
                var response = new StringBuilder();
                var timeout = DateTime.UtcNow.AddSeconds(10);

                while (DateTime.UtcNow < timeout)
                {
                    if (_webSocket.State != WebSocketState.Open)
                        break;

                    var result = await _webSocket.ReceiveAsync(
                        new ArraySegment<byte>(buffer),
                        new CancellationTokenSource(TimeSpan.FromSeconds(2)).Token
                    );

                    if (result.MessageType == WebSocketMessageType.Text)
                    {
                        var message = Encoding.UTF8.GetString(buffer, 0, result.Count);
                        _logger.LogDebug("Received: {Message}", message);

                        // Parse response and extract assistant message
                        try
                        {
                            var jsonDoc = JsonDocument.Parse(message);
                            var eventType = jsonDoc.RootElement.GetProperty("type").GetString();

                            if (eventType == "response.text.done" || eventType == "response.done")
                            {
                                // Extract the response text
                                if (jsonDoc.RootElement.TryGetProperty("response", out var responseElement))
                                {
                                    if (responseElement.TryGetProperty("output", out var outputElement))
                                    {
                                        if (outputElement.ValueKind == JsonValueKind.Array)
                                        {
                                            foreach (var item in outputElement.EnumerateArray())
                                            {
                                                if (item.TryGetProperty("content", out var contentElement))
                                                {
                                                    if (contentElement.ValueKind == JsonValueKind.Array)
                                                    {
                                                        foreach (var content in contentElement.EnumerateArray())
                                                        {
                                                            if (content.TryGetProperty("text", out var textElement))
                                                            {
                                                                return textElement.GetString() ?? "No response text";
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                break;
                            }
                            
                            // Continue listening for more messages
                        }
                        catch (JsonException ex)
                        {
                            _logger.LogWarning("Failed to parse JSON response: {Error}", ex.Message);
                        }
                    }

                    if (result.EndOfMessage && result.MessageType == WebSocketMessageType.Close)
                    {
                        break;
                    }
                }

                return response.Length > 0 ? response.ToString() : "Hello! I'm ready to help you with your AI initiative survey. What's the name of your AI project?";
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error in test conversation");
                return $"Error: {ex.Message}";
            }
        }

        /// <summary>
        /// Close connection and cleanup
        /// </summary>
        public async Task DisconnectAsync()
        {
            if (_webSocket != null && _webSocket.State == WebSocketState.Open)
            {
                try
                {
                    await _webSocket.CloseAsync(WebSocketCloseStatus.NormalClosure, "Session ended", CancellationToken.None);
                    _logger.LogInformation("Disconnected from OpenAI Realtime Audio API");
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Error during disconnect");
                }
                finally
                {
                    _isConnected = false;
                }
            }
        }

        public void Dispose()
        {
            try
            {
                if (_webSocket != null)
                {
                    if (_webSocket.State == WebSocketState.Open)
                    {
                        _webSocket.CloseAsync(WebSocketCloseStatus.NormalClosure, "Disposing", CancellationToken.None)
                            .GetAwaiter().GetResult();
                    }
                    _webSocket.Dispose();
                }
            }
            catch
            {
                // Ignore disposal errors
            }
            finally
            {
                _connectionSemaphore.Dispose();
            }
        }
    }
}