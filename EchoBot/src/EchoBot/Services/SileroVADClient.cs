using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Configuration;

namespace EchoBot.Services
{
    /// <summary>
    /// Configuration for Silero VAD integration
    /// </summary>
    public class SileroVADConfig
    {
        public bool Enabled { get; set; } = false;
        public string ServiceUrl { get; set; } = "http://localhost:8001";
        public float Threshold { get; set; } = 0.5f;
        public int MinSpeechDuration { get; set; } = 250;
        public int MinSilenceDuration { get; set; } = 2000;
        public int SpeechPadMs { get; set; } = 500;
        public int SampleRate { get; set; } = 16000;
        public int TimeoutMs { get; set; } = 5000;
        public bool FallbackToAzure { get; set; } = true;
    }

    /// <summary>
    /// Request/Response models for Silero VAD API
    /// </summary>
    public class VADRequest
    {
        public string audio_data { get; set; } = string.Empty;
        public string? session_id { get; set; }
        public VADConfigRequest? config { get; set; }
    }

    public class VADConfigRequest
    {
        public float threshold { get; set; }
        public int min_speech_duration { get; set; }
        public int min_silence_duration { get; set; }
        public int speech_pad_ms { get; set; }
        public int sample_rate { get; set; }
    }

    public class VADResponse
    {
        public bool is_speech { get; set; }
        public float speech_probability { get; set; }
        public float? speech_start { get; set; }
        public float? speech_end { get; set; }
        public float? speech_duration { get; set; }
        public string session_state { get; set; } = string.Empty;
        public float processing_time_ms { get; set; }
    }

    public class VADHealthResponse
    {
        public string status { get; set; } = string.Empty;
        public bool model_loaded { get; set; }
        public float uptime_seconds { get; set; }
        public int total_requests { get; set; }
        public float average_processing_time_ms { get; set; }
    }

    /// <summary>
    /// Client for integrating with Silero VAD service
    /// Provides fallback to Azure VAD if service is unavailable
    /// </summary>
    public interface ISileroVADClient
    {
        Task<bool> IsServiceHealthyAsync();
        Task<VADResponse?> DetectSpeechAsync(byte[] audioData, string sessionId);
        Task<bool> UpdateConfigAsync(SileroVADConfig config);
        bool IsEnabled { get; }
    }

    public class SileroVADClient : ISileroVADClient, IDisposable
    {
        private readonly HttpClient _httpClient;
        private readonly ILogger<SileroVADClient> _logger;
        private readonly SileroVADConfig _config;
        private bool _isServiceHealthy = false;
        private DateTime _lastHealthCheck = DateTime.MinValue;
        private readonly TimeSpan _healthCheckInterval = TimeSpan.FromMinutes(5);

        public bool IsEnabled => _config.Enabled;

        public SileroVADClient(IConfiguration configuration, ILogger<SileroVADClient> logger)
        {
            _logger = logger;
            
            // Load configuration
            _config = new SileroVADConfig();
            configuration.GetSection("SileroVAD").Bind(_config);
            
            // Log configuration
            _logger.LogInformation("Silero VAD Client Configuration:");
            _logger.LogInformation("  Enabled: {Enabled}", _config.Enabled);
            _logger.LogInformation("  Service URL: {ServiceUrl}", _config.ServiceUrl);
            _logger.LogInformation("  Threshold: {Threshold}", _config.Threshold);
            _logger.LogInformation("  Min Silence Duration: {Duration}ms", _config.MinSilenceDuration);
            _logger.LogInformation("  Fallback to Azure: {Fallback}", _config.FallbackToAzure);

            // Initialize HTTP client
            _httpClient = new HttpClient();
            _httpClient.BaseAddress = new Uri(_config.ServiceUrl);
            _httpClient.Timeout = TimeSpan.FromMilliseconds(_config.TimeoutMs);
            
            // Add headers
            _httpClient.DefaultRequestHeaders.Add("User-Agent", "EchoBot-SileroVAD/1.0");
            
            if (_config.Enabled)
            {
                _logger.LogInformation("✅ Silero VAD Client initialized and enabled");
                
                // Perform initial health check
                Task.Run(async () => await IsServiceHealthyAsync());
            }
            else
            {
                _logger.LogInformation("⚪ Silero VAD Client initialized but disabled (using Azure VAD)");
            }
        }

        public async Task<bool> IsServiceHealthyAsync()
        {
            if (!_config.Enabled)
                return false;

            // Check if we need to perform health check
            if (_isServiceHealthy && DateTime.UtcNow - _lastHealthCheck < _healthCheckInterval)
            {
                return _isServiceHealthy;
            }

            try
            {
                _logger.LogDebug("Performing Silero VAD health check...");
                
                var response = await _httpClient.GetAsync("/health");
                
                if (response.IsSuccessStatusCode)
                {
                    var jsonString = await response.Content.ReadAsStringAsync();
                    var healthData = JsonSerializer.Deserialize<VADHealthResponse>(jsonString, new JsonSerializerOptions
                    {
                        PropertyNameCaseInsensitive = true
                    });
                    
                    _isServiceHealthy = healthData?.model_loaded == true && healthData.status == "healthy";
                    _lastHealthCheck = DateTime.UtcNow;
                    
                    if (_isServiceHealthy)
                    {
                        _logger.LogDebug("✅ Silero VAD service healthy - model loaded, uptime: {Uptime}s", healthData?.uptime_seconds);
                    }
                    else
                    {
                        _logger.LogWarning("⚠️ Silero VAD service unhealthy - model not loaded or status not healthy");
                    }
                }
                else
                {
                    _logger.LogWarning("⚠️ Silero VAD health check failed with status: {StatusCode}", response.StatusCode);
                    _isServiceHealthy = false;
                }
            }
            catch (TaskCanceledException)
            {
                _logger.LogWarning("⚠️ Silero VAD health check timed out");
                _isServiceHealthy = false;
            }
            catch (Exception ex)
            {
                _logger.LogWarning("⚠️ Silero VAD health check error: {Error}", ex.Message);
                _isServiceHealthy = false;
            }

            return _isServiceHealthy;
        }

        public async Task<VADResponse?> DetectSpeechAsync(byte[] audioData, string sessionId)
        {
            if (!_config.Enabled || !await IsServiceHealthyAsync())
            {
                return null; // Fall back to Azure VAD
            }

            try
            {
                _logger.LogDebug("Sending audio to Silero VAD: {Bytes} bytes, session: {SessionId}", audioData.Length, sessionId);
                
                // Convert audio to base64
                var audioBase64 = Convert.ToBase64String(audioData);
                
                // Prepare request
                var request = new VADRequest
                {
                    audio_data = audioBase64,
                    session_id = sessionId,
                    config = new VADConfigRequest
                    {
                        threshold = _config.Threshold,
                        min_speech_duration = _config.MinSpeechDuration,
                        min_silence_duration = _config.MinSilenceDuration,
                        speech_pad_ms = _config.SpeechPadMs,
                        sample_rate = _config.SampleRate
                    }
                };
                
                var requestJson = JsonSerializer.Serialize(request);
                var content = new StringContent(requestJson, Encoding.UTF8, "application/json");
                
                // Send request
                var response = await _httpClient.PostAsync("/vad/detect", content);
                
                if (response.IsSuccessStatusCode)
                {
                    var responseJson = await response.Content.ReadAsStringAsync();
                    var vadResponse = JsonSerializer.Deserialize<VADResponse>(responseJson, new JsonSerializerOptions
                    {
                        PropertyNameCaseInsensitive = true
                    });
                    
                    _logger.LogDebug("Silero VAD result: speech={Speech}, prob={Probability:F3}, state={State}, processing={ProcessingMs:F1}ms", 
                        vadResponse?.is_speech, vadResponse?.speech_probability, vadResponse?.session_state, vadResponse?.processing_time_ms);
                    
                    return vadResponse;
                }
                else
                {
                    _logger.LogWarning("Silero VAD detection failed with status: {StatusCode}, response: {Response}", 
                        response.StatusCode, await response.Content.ReadAsStringAsync());
                    return null;
                }
            }
            catch (TaskCanceledException)
            {
                _logger.LogWarning("Silero VAD detection timed out for session: {SessionId}", sessionId);
                return null;
            }
            catch (Exception ex)
            {
                _logger.LogError("Silero VAD detection error for session {SessionId}: {Error}", sessionId, ex.Message);
                return null;
            }
        }

        public async Task<bool> UpdateConfigAsync(SileroVADConfig config)
        {
            if (!_config.Enabled || !await IsServiceHealthyAsync())
            {
                return false;
            }

            try
            {
                var configRequest = new VADConfigRequest
                {
                    threshold = config.Threshold,
                    min_speech_duration = config.MinSpeechDuration,
                    min_silence_duration = config.MinSilenceDuration,
                    speech_pad_ms = config.SpeechPadMs,
                    sample_rate = config.SampleRate
                };
                
                var requestJson = JsonSerializer.Serialize(configRequest);
                var content = new StringContent(requestJson, Encoding.UTF8, "application/json");
                
                var response = await _httpClient.PostAsync("/vad/config", content);
                
                if (response.IsSuccessStatusCode)
                {
                    _logger.LogInformation("✅ Silero VAD config updated successfully");
                    
                    // Update local config
                    _config.Threshold = config.Threshold;
                    _config.MinSpeechDuration = config.MinSpeechDuration;
                    _config.MinSilenceDuration = config.MinSilenceDuration;
                    _config.SpeechPadMs = config.SpeechPadMs;
                    _config.SampleRate = config.SampleRate;
                    
                    return true;
                }
                else
                {
                    _logger.LogWarning("Silero VAD config update failed with status: {StatusCode}", response.StatusCode);
                    return false;
                }
            }
            catch (Exception ex)
            {
                _logger.LogError("Silero VAD config update error: {Error}", ex.Message);
                return false;
            }
        }

        public void Dispose()
        {
            _httpClient?.Dispose();
        }
    }
}