using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;

namespace EchoBot.Services
{
    public interface ILLMService
    {
        Task<string> StartSurveyAsync(string userId = null);
        Task<LLMResponse> ProcessUserInputAsync(string sessionId, string userInput);
        Task<SessionStatus> GetSessionStatusAsync(string sessionId);
        Task EndSessionAsync(string sessionId);
    }

    public class LLMService : ILLMService
    {
        private readonly HttpClient _httpClient;
        private readonly ILogger<LLMService> _logger;
        private readonly string _apiBaseUrl;

        public LLMService(HttpClient httpClient, IConfiguration configuration, ILogger<LLMService> logger)
        {
            _httpClient = httpClient;
            _logger = logger;
            
            // Get AppSettings which includes Key Vault values
            var appSettings = new AppSettings();
            configuration.Bind(appSettings);
            
            // Try Key Vault value first, then fallback to appsettings LLMApi section, then default
            _apiBaseUrl = appSettings.LLMApiBaseUrl ?? 
                         configuration.GetValue<string>("LLMApi:BaseUrl") ?? 
                         "http://localhost:8000";
            
            _logger.LogDebug("LLM API Base URL configured as: {ApiBaseUrl}", _apiBaseUrl);
            _logger.LogDebug("Key Vault LLMApiBaseUrl: {KeyVaultUrl}", appSettings.LLMApiBaseUrl);
            
            // Set default timeout for LLM API calls
            _httpClient.Timeout = TimeSpan.FromSeconds(30);
        }

        public async Task<string> StartSurveyAsync(string userId = null)
        {
            try
            {
                _logger.LogDebug("Starting new survey session for user: {UserId}", userId ?? "anonymous");

                var requestBody = new
                {
                    user_id = userId,
                    context = new { source = "teams_voice_bot" }
                };

                var json = JsonSerializer.Serialize(requestBody);
                var content = new StringContent(json, Encoding.UTF8, "application/json");

                var response = await _httpClient.PostAsync($"{_apiBaseUrl}/start-survey", content);
                response.EnsureSuccessStatusCode();

                var responseContent = await response.Content.ReadAsStringAsync();
                var startResponse = JsonSerializer.Deserialize<StartSurveyResponse>(responseContent, new JsonSerializerOptions
                {
                    PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower
                });

                _logger.LogInformation("Survey started successfully. Session ID: {SessionId}", startResponse.SessionId);
                
                // Store session ID for this user (you might want to use a proper session store)
                // For now, we'll return it as part of the message
                return $"SESSION_ID:{startResponse.SessionId}|{startResponse.Message}";
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to start survey session");
                throw new InvalidOperationException("Failed to start survey session", ex);
            }
        }

        public async Task<LLMResponse> ProcessUserInputAsync(string sessionId, string userInput)
        {
            try
            {
                _logger.LogDebug("Processing user input for session: {SessionId}", sessionId);

                var requestBody = new
                {
                    session_id = sessionId,
                    user_input = userInput
                };

                var json = JsonSerializer.Serialize(requestBody);
                var content = new StringContent(json, Encoding.UTF8, "application/json");

                var response = await _httpClient.PostAsync($"{_apiBaseUrl}/process-input", content);
                response.EnsureSuccessStatusCode();

                var responseContent = await response.Content.ReadAsStringAsync();
                var processResponse = JsonSerializer.Deserialize<ProcessInputResponse>(responseContent, new JsonSerializerOptions
                {
                    PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower
                });

                _logger.LogInformation("User input processed. Status: {Status}", processResponse.Status);

                return new LLMResponse
                {
                    SessionId = processResponse.SessionId,
                    Message = processResponse.Message,
                    Status = processResponse.Status,
                    IsCompleted = processResponse.Status == "completed"
                };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to process user input for session: {SessionId}", sessionId);
                throw new InvalidOperationException($"Failed to process user input for session: {sessionId}", ex);
            }
        }

        public async Task<SessionStatus> GetSessionStatusAsync(string sessionId)
        {
            try
            {
                var response = await _httpClient.GetAsync($"{_apiBaseUrl}/session/{sessionId}/status");
                response.EnsureSuccessStatusCode();

                var responseContent = await response.Content.ReadAsStringAsync();
                var statusResponse = JsonSerializer.Deserialize<SessionStatusResponse>(responseContent, new JsonSerializerOptions
                {
                    PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower
                });

                return new SessionStatus
                {
                    SessionId = statusResponse.SessionId,
                    Status = statusResponse.Status,
                    MissingFields = statusResponse.MissingFields?.Count ?? 0
                };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to get session status for: {SessionId}", sessionId);
                throw new InvalidOperationException($"Failed to get session status for: {sessionId}", ex);
            }
        }

        public async Task EndSessionAsync(string sessionId)
        {
            try
            {
                _logger.LogDebug("Ending session: {SessionId}", sessionId);
                
                var response = await _httpClient.DeleteAsync($"{_apiBaseUrl}/session/{sessionId}");
                response.EnsureSuccessStatusCode();
                
                _logger.LogInformation("Session ended successfully: {SessionId}", sessionId);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to end session: {SessionId}", sessionId);
                // Don't throw here - ending session is cleanup, shouldn't fail the main flow
            }
        }
    }

    // Response models
    public class StartSurveyResponse
    {
        public string SessionId { get; set; }
        public string Message { get; set; }
        public string Status { get; set; }
    }

    public class ProcessInputResponse
    {
        public string SessionId { get; set; }
        public string Message { get; set; }
        public string Status { get; set; }
        public Dictionary<string, object> CollectedData { get; set; }
        public List<string> MissingFields { get; set; }
    }

    public class SessionStatusResponse
    {
        public string SessionId { get; set; }
        public string Status { get; set; }
        public Dictionary<string, object> CollectedData { get; set; }
        public List<string> MissingFields { get; set; }
    }

    // Return models
    public class LLMResponse
    {
        public string SessionId { get; set; }
        public string Message { get; set; }
        public string Status { get; set; }
        public bool IsCompleted { get; set; }
    }

    public class SessionStatus
    {
        public string SessionId { get; set; }
        public string Status { get; set; }
        public int MissingFields { get; set; }
    }
}