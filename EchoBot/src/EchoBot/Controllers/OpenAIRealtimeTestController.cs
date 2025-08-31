using Microsoft.AspNetCore.Mvc;
using EchoBot.Services;

namespace EchoBot.Controllers
{
    /// <summary>
    /// Test controller for OpenAI Realtime Audio Service
    /// This is for testing only - does not affect production functionality
    /// </summary>
    [ApiController]
    [Route("api/[controller]")]
    public class OpenAIRealtimeTestController : ControllerBase
    {
        private readonly OpenAIRealtimeAudioService _realtimeService;
        private readonly ILogger<OpenAIRealtimeTestController> _logger;

        public OpenAIRealtimeTestController(
            OpenAIRealtimeAudioService realtimeService, 
            ILogger<OpenAIRealtimeTestController> logger)
        {
            _realtimeService = realtimeService;
            _logger = logger;
        }

        /// <summary>
        /// Test connectivity to OpenAI Realtime API
        /// GET: api/openairealtime/test-connection
        /// </summary>
        [HttpGet("test-connection")]
        public async Task<IActionResult> TestConnection()
        {
            try
            {
                var isConnected = await _realtimeService.TestConnectivityAsync();
                
                if (isConnected)
                {
                    return Ok(new { 
                        status = "success", 
                        message = "Successfully connected to OpenAI Realtime API",
                        configured = _realtimeService.IsConfigured
                    });
                }
                else
                {
                    return BadRequest(new { 
                        status = "error", 
                        message = "Failed to connect to OpenAI Realtime API",
                        configured = _realtimeService.IsConfigured
                    });
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error testing OpenAI Realtime connection");
                return StatusCode(500, new { 
                    status = "error", 
                    message = $"Connection test failed: {ex.Message}",
                    configured = _realtimeService.IsConfigured
                });
            }
        }

        /// <summary>
        /// Test simple text conversation
        /// POST: api/openairealtime/test-conversation
        /// </summary>
        [HttpPost("test-conversation")]
        public async Task<IActionResult> TestConversation([FromBody] TestConversationRequest request)
        {
            try
            {
                if (string.IsNullOrEmpty(request.Message))
                {
                    return BadRequest(new { status = "error", message = "Message is required" });
                }

                var response = await _realtimeService.TestConversationAsync(request.Message);
                
                return Ok(new { 
                    status = "success", 
                    userMessage = request.Message,
                    assistantResponse = response
                });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error in test conversation");
                return StatusCode(500, new { 
                    status = "error", 
                    message = $"Conversation test failed: {ex.Message}"
                });
            }
        }

        /// <summary>
        /// Get service status and configuration
        /// GET: api/openairealtime/status
        /// </summary>
        [HttpGet("status")]
        public IActionResult GetStatus()
        {
            return Ok(new { 
                configured = _realtimeService.IsConfigured,
                service = "OpenAI Realtime Audio Service",
                version = "1.0.0-poc",
                description = "Proof of concept implementation - parallel to existing services"
            });
        }
    }

    /// <summary>
    /// Request model for test conversation
    /// </summary>
    public class TestConversationRequest
    {
        public string Message { get; set; } = "";
    }
}