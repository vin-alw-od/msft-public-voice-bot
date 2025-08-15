using Microsoft.CognitiveServices.Speech;
using Microsoft.CognitiveServices.Speech.Audio;
using Microsoft.Skype.Bots.Media;
using System.Runtime.InteropServices;
using EchoBot.Bot;
using EchoBot.Services;
using EchoBot.Util;
using System.Collections.Concurrent;

namespace EchoBot.Media
{
    /// <summary>
    /// Enhanced Speech Service that integrates with Python LLM API instead of just echoing
    /// </summary>
    public class LLMSpeechService
    {
        private readonly ILLMService _llmService;
        private readonly ISpeechService _speechService;
        private readonly ConcurrentDictionary<string, string> _callSessions = new ConcurrentDictionary<string, string>();
        private readonly ILogger _logger;
        private string _currentCallId;

        /// <summary>
        /// Event raised when audio response is ready to be played
        /// </summary>
        public event EventHandler<EchoBot.Bot.MediaStreamEventArgs>? AudioResponse;

        public LLMSpeechService(ILLMService llmService, ISpeechService speechService, ILogger<LLMSpeechService> logger)
        {
            _llmService = llmService;
            _speechService = speechService;
            _logger = logger;
        }

        public void SetCurrentCallId(string callId)
        {
            _currentCallId = callId;
            _logger.LogInformation("LLM Speech Service initialized for call: {CallId}", callId);
        }

        /// <summary>
        /// Process audio buffer from Teams call
        /// </summary>
        public async Task ProcessAudioAsync(string callId, AudioMediaBuffer audioBuffer)
        {
            try
            {
                // Set the current call context
                SetCurrentCallId(callId);
                
                // Convert audio buffer to stream and process via speech service interface
                var audioData = new byte[audioBuffer.Length];
                Marshal.Copy(audioBuffer.Data, audioData, 0, (int)audioBuffer.Length);
                
                using var audioStream = new MemoryStream(audioData);
                var recognizedText = await _speechService.SpeechToTextAsync(audioStream);
                
                if (!string.IsNullOrEmpty(recognizedText))
                {
                    _logger.LogInformation("Recognized speech: {Text}", recognizedText);
                    await ProcessRecognizedSpeechAsync(recognizedText);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error processing audio for call {CallId}", callId);
            }
        }

        /// <summary>
        /// Process recognized speech through LLM API
        /// </summary>
        public async Task ProcessRecognizedSpeechAsync(string recognizedText)
        {
            try
            {
                _logger.LogInformation("Processing recognized speech through LLM: {Text}", recognizedText);

                // Get or create session for this call
                var sessionId = await GetOrCreateSessionAsync(_currentCallId);

                if (string.IsNullOrEmpty(sessionId))
                {
                    _logger.LogWarning("No session available for call {CallId}", _currentCallId);
                    return;
                }

                // Process through LLM
                var llmResponse = await _llmService.ProcessUserInputAsync(sessionId, recognizedText);

                if (!string.IsNullOrEmpty(llmResponse.Message))
                {
                    _logger.LogInformation("LLM Response: {Message}, Status: {Status}", llmResponse.Message, llmResponse.Status);

                    // Convert LLM response to speech using the speech service
                    await SynthesizeTextAsync(llmResponse.Message);

                    // If conversation is completed, cleanup session
                    if (llmResponse.IsCompleted)
                    {
                        _logger.LogInformation("Conversation completed for session: {SessionId}", sessionId);
                        await CleanupSessionAsync(_currentCallId);
                    }
                }
                else
                {
                    _logger.LogWarning("Empty response from LLM service");
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error processing speech through LLM");
                
                // Fallback response
                var fallbackMessage = "I'm sorry, I'm having trouble processing that. Could you please try again?";
                await SynthesizeTextAsync(fallbackMessage);
            }
        }

        /// <summary>
        /// Get or create LLM session for the current call
        /// </summary>
        private async Task<string> GetOrCreateSessionAsync(string callId)
        {
            if (string.IsNullOrEmpty(callId))
                return null;

            if (_callSessions.TryGetValue(callId, out string existingSessionId))
            {
                return existingSessionId;
            }

            try
            {
                // Start new survey session
                var response = await _llmService.StartSurveyAsync($"teams_call_{callId}");
                
                // Extract session ID from response (format: "SESSION_ID:xxx|message")
                if (response.StartsWith("SESSION_ID:"))
                {
                    var parts = response.Split('|', 2);
                    var sessionId = parts[0].Substring("SESSION_ID:".Length);
                    var initialMessage = parts.Length > 1 ? parts[1] : "Hello! I'm here to collect information about your AI initiatives.";

                    _callSessions.TryAdd(callId, sessionId);

                    // Speak the initial greeting
                    await SynthesizeTextAsync(initialMessage);

                    _logger.LogInformation("Created new LLM session {SessionId} for call {CallId}", sessionId, callId);
                    return sessionId;
                }
                else
                {
                    _logger.LogError("Unexpected response format from LLM service: {Response}", response);
                    return null;
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to create LLM session for call {CallId}", callId);
                return null;
            }
        }

        /// <summary>
        /// Cleanup session when call ends
        /// </summary>
        public async Task CleanupSessionAsync(string callId)
        {
            if (string.IsNullOrEmpty(callId))
                return;

            if (_callSessions.TryRemove(callId, out string sessionId))
            {
                try
                {
                    await _llmService.EndSessionAsync(sessionId);
                    _logger.LogInformation("Cleaned up LLM session {SessionId} for call {CallId}", sessionId, callId);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error cleaning up session {SessionId}", sessionId);
                }
            }
        }

        /// <summary>
        /// Start the LLM speech service with initial greeting
        /// </summary>
        public async Task StartAsync()
        {
            // Start with an initial greeting when the call begins
            if (!string.IsNullOrEmpty(_currentCallId))
            {
                await GetOrCreateSessionAsync(_currentCallId);
            }
        }

        /// <summary>
        /// Stop the LLM speech service and cleanup sessions
        /// </summary>
        public async Task StopAsync()
        {
            if (!string.IsNullOrEmpty(_currentCallId))
            {
                await CleanupSessionAsync(_currentCallId);
            }
        }

        /// <summary>
        /// Synthesize text and play it through the media stream
        /// </summary>
        private async Task SynthesizeTextAsync(string text)
        {
            try
            {
                _logger.LogInformation("Synthesizing text: {Text}", text);
                
                // Get audio bytes from the speech service
                var audioBytes = await _speechService.TextToSpeechBytesAsync(text);
                
                if (audioBytes != null && audioBytes.Length > 0)
                {
                    _logger.LogInformation("Generated {Length} bytes of audio", audioBytes.Length);
                    
                    // Convert audio bytes to AudioMediaBuffers
                    var currentTick = DateTime.Now.Ticks;
                    var audioMediaBuffers = Utilities.CreateAudioMediaBuffers(audioBytes, currentTick, _logger);
                    
                    // Trigger the AudioResponse event
                    if (audioMediaBuffers?.Count > 0)
                    {
                        var eventArgs = new EchoBot.Bot.MediaStreamEventArgs
                        {
                            AudioMediaBuffers = audioMediaBuffers
                        };
                        
                        AudioResponse?.Invoke(this, eventArgs);
                        _logger.LogInformation("Audio response event triggered with {Count} buffers", audioMediaBuffers.Count);
                    }
                }
                else
                {
                    _logger.LogWarning("No audio generated for text: {Text}", text);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to synthesize text: {Text}", text);
            }
        }

        public void Dispose()
        {
            // Cleanup all sessions
            foreach (var kvp in _callSessions)
            {
                try
                {
                    _llmService.EndSessionAsync(kvp.Value).Wait(TimeSpan.FromSeconds(5));
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error disposing session {SessionId}", kvp.Value);
                }
            }
            _callSessions.Clear();
        }
    }
}