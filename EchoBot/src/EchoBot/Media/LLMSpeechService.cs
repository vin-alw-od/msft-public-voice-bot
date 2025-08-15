using Microsoft.CognitiveServices.Speech;
using Microsoft.CognitiveServices.Speech.Audio;
using Microsoft.Graph.Communications.Calls.Media;
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
            
            // Trigger initial greeting immediately after call ID is set
            _ = Task.Run(async () =>
            {
                try
                {
                    await Task.Delay(3000); // Wait 3 seconds for call to be ready
                    _logger.LogError("🔥 Triggering initial greeting for call: {CallId}", callId);
                    await GetOrCreateSessionAsync(callId);
                    _logger.LogError("🔥 Initial greeting completed for call: {CallId}", callId);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "🔥 Failed to trigger initial greeting for call: {CallId}", callId);
                }
            });
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
            _logger.LogInformation("GetOrCreateSessionAsync called for call ID: {CallId}", callId ?? "NULL");
            
            if (string.IsNullOrEmpty(callId))
            {
                _logger.LogError("Call ID is null or empty in GetOrCreateSessionAsync");
                return null;
            }

            if (_callSessions.TryGetValue(callId, out string existingSessionId))
            {
                _logger.LogInformation("Found existing session {SessionId} for call {CallId}", existingSessionId, callId);
                return existingSessionId;
            }

            try
            {
                _logger.LogInformation("Starting new survey session via LLM service for call: {CallId}", callId);
                // Start new survey session
                var response = await _llmService.StartSurveyAsync($"teams_call_{callId}");
                
                _logger.LogInformation("LLM service response: {Response}", response);
                
                // Extract session ID from response (format: "SESSION_ID:xxx|message")
                if (response.StartsWith("SESSION_ID:"))
                {
                    var parts = response.Split('|', 2);
                    var sessionId = parts[0].Substring("SESSION_ID:".Length);
                    var initialMessage = parts.Length > 1 ? parts[1] : "Hello! I'm here to collect information about your AI initiatives.";

                    _logger.LogInformation("Parsed session ID: {SessionId}, initial message: {Message}", sessionId, initialMessage);

                    _callSessions.TryAdd(callId, sessionId);

                    // Speak the initial greeting
                    _logger.LogInformation("About to synthesize initial greeting for call: {CallId}", callId);
                    await SynthesizeTextAsync(initialMessage);

                    _logger.LogInformation("Successfully created LLM session {SessionId} for call {CallId}", sessionId, callId);
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
            _logger.LogError("🔥 CRITICAL DEBUG: StartAsync called for call ID: {CallId}", _currentCallId ?? "NULL");
            
            // Start with an initial greeting when the call begins
            if (!string.IsNullOrEmpty(_currentCallId))
            {
                _logger.LogError("🔥 CRITICAL DEBUG: Creating session and triggering initial greeting for call: {CallId}", _currentCallId);
                try
                {
                    await GetOrCreateSessionAsync(_currentCallId);
                    _logger.LogError("🔥 CRITICAL DEBUG: StartAsync completed successfully for call: {CallId}", _currentCallId);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "🔥 CRITICAL DEBUG: Error in StartAsync for call: {CallId}", _currentCallId);
                }
            }
            else
            {
                _logger.LogError("🔥 CRITICAL DEBUG: Cannot start LLM speech service - current call ID is null or empty");
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
                    
                    // Convert audio bytes to AudioMediaBuffers using the FIXED chunking method
                    var currentTick = DateTime.Now.Ticks;
                    var audioMediaBuffers = CreateAudioMediaBuffersFromBytes(audioBytes, currentTick);
                    
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

        /// <summary>
        /// Create audio media buffers from byte array, properly chunking into 20ms packets
        /// Based on the working AudioDataStream method
        /// </summary>
        private List<AudioMediaBuffer> CreateAudioMediaBuffersFromBytes(byte[] buffer, long currentTick)
        {
            var audioMediaBuffers = new List<AudioMediaBuffer>();
            var referenceTime = currentTick;
            var numberOfTicksInOneAudioBuffers = 20 * 10000; // 20ms
            var chunkSize = 640; // 20ms of PCM 16kHz audio

            _logger.LogInformation($"Creating audio buffers from {buffer.Length} bytes of audio data");
            _logger.LogInformation($"Audio chunking: Will create {Math.Ceiling((double)buffer.Length / chunkSize)} buffers from {buffer.Length} bytes");

            // Split the large buffer into 640-byte chunks (20ms each)
            for (int offset = 0; offset < buffer.Length; offset += chunkSize)
            {
                var bytesToCopy = Math.Min(chunkSize, buffer.Length - offset);
                
                // Allocate memory for this chunk
                IntPtr unmanagedBuffer = Marshal.AllocHGlobal(bytesToCopy);
                
                try
                {
                    // Copy the chunk data
                    Marshal.Copy(buffer, offset, unmanagedBuffer, bytesToCopy);
                    
                    // Create audio buffer for this chunk
                    var audioBuffer = new AudioSendBuffer(unmanagedBuffer, (uint)bytesToCopy, AudioFormat.Pcm16K, referenceTime);
                    audioMediaBuffers.Add(audioBuffer);
                    
                    // Advance reference time by 20ms
                    referenceTime += numberOfTicksInOneAudioBuffers;
                }
                catch (Exception ex)
                {
                    // Clean up memory if something goes wrong
                    Marshal.FreeHGlobal(unmanagedBuffer);
                    _logger.LogError(ex, "Error creating audio buffer at offset {Offset}", offset);
                    throw;
                }
            }

            _logger.LogInformation($"Created {audioMediaBuffers.Count} AudioMediaBuffers from {buffer.Length} bytes");
            return audioMediaBuffers;
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