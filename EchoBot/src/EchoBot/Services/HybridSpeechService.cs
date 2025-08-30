using Microsoft.CognitiveServices.Speech;
using Microsoft.CognitiveServices.Speech.Audio;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using System.Text;

namespace EchoBot.Services
{
    /// <summary>
    /// Enhanced Speech Service that supports both Azure VAD and Silero VAD
    /// Provides seamless fallback and A/B testing capabilities
    /// </summary>
    public class HybridSpeechService : ISpeechService
    {
        private readonly SpeechConfig _speechConfig;
        private readonly ILogger<HybridSpeechService> _logger;
        private readonly ISileroVADClient _sileroVADClient;
        private readonly string _voiceName;
        private readonly bool _useHybridVAD;
        private readonly bool _logVADComparisons;

        public HybridSpeechService(
            IConfiguration configuration, 
            ILogger<HybridSpeechService> logger,
            ISileroVADClient sileroVADClient)
        {
            _logger = logger;
            _sileroVADClient = sileroVADClient;
            
            var speechKey = configuration.GetValue<string>("AppSettings:SpeechConfigKey");
            var speechRegion = configuration.GetValue<string>("AppSettings:SpeechConfigRegion");
            var botLanguage = configuration.GetValue<string>("AppSettings:BotLanguage") ?? "en-US";
            
            // Hybrid VAD configuration
            _useHybridVAD = configuration.GetValue<bool>("SileroVAD:Enabled", false);
            _logVADComparisons = configuration.GetValue<bool>("SileroVAD:LogComparisons", false);
            
            _logger.LogInformation("Hybrid Speech Service Configuration:");
            _logger.LogInformation("  Azure Speech Key: {Key}", string.IsNullOrEmpty(speechKey) ? "NULL/EMPTY" : "SET");
            _logger.LogInformation("  Azure Speech Region: {Region}", speechRegion);
            _logger.LogInformation("  Bot Language: {Language}", botLanguage);
            _logger.LogInformation("  Hybrid VAD Enabled: {HybridVAD}", _useHybridVAD);
            _logger.LogInformation("  Silero VAD Available: {SileroAvailable}", _sileroVADClient.IsEnabled);
            
            if (string.IsNullOrEmpty(speechKey) || string.IsNullOrEmpty(speechRegion))
            {
                throw new InvalidOperationException("Azure Speech service configuration is required. SpeechConfigKey and SpeechConfigRegion must be set.");
            }

            _speechConfig = SpeechConfig.FromSubscription(speechKey, speechRegion);
            _speechConfig.SpeechRecognitionLanguage = botLanguage;
            
            // Configure Azure VAD timeouts (will be used as fallback)
            _speechConfig.SetProperty(PropertyId.Speech_SegmentationSilenceTimeoutMs, "1200");
            _speechConfig.SetProperty(PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs, "1500");
            _speechConfig.SetProperty(PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, "3000");
            
            // Set voice for TTS
            _voiceName = GetVoiceForLanguage(botLanguage);
            _speechConfig.SpeechSynthesisVoiceName = _voiceName;
            
            _logger.LogInformation("✅ Hybrid Speech Service initialized with voice: {Voice}", _voiceName);
            
            if (_useHybridVAD && _sileroVADClient.IsEnabled)
            {
                _logger.LogInformation("🔀 Hybrid VAD mode activated - will use Silero VAD with Azure fallback");
            }
            else
            {
                _logger.LogInformation("🔵 Azure VAD mode - using standard Azure Speech Service VAD");
            }
        }

        public async Task<string> SpeechToTextAsync(Stream audioStream)
        {
            try
            {
                _logger.LogDebug("Starting speech-to-text conversion...");
                
                // Convert stream to byte array for potential Silero VAD processing
                byte[] audioBytes;
                using (var memoryStream = new MemoryStream())
                {
                    await audioStream.CopyToAsync(memoryStream);
                    audioBytes = memoryStream.ToArray();
                }
                
                // Generate session ID for VAD tracking
                string sessionId = $"stt_{DateTime.UtcNow:yyyyMMdd_HHmmss}_{Guid.NewGuid():N}";
                
                // Try Silero VAD first if enabled and healthy
                if (_useHybridVAD && _sileroVADClient.IsEnabled)
                {
                    var vadStartTime = DateTime.UtcNow;
                    var vadResult = await _sileroVADClient.DetectSpeechAsync(audioBytes, sessionId);
                    var vadProcessingTime = (DateTime.UtcNow - vadStartTime).TotalMilliseconds;
                    
                    if (vadResult != null)
                    {
                        _logger.LogInformation("🟢 Silero VAD result: speech={Speech}, prob={Probability:F3}, state={State} ({ProcessingMs:F0}ms)",
                            vadResult.is_speech, vadResult.speech_probability, vadResult.session_state, vadProcessingTime);
                        
                        // If Silero VAD says no speech, we can potentially skip Azure STT
                        if (!vadResult.is_speech && vadResult.speech_probability < 0.3)
                        {
                            _logger.LogInformation("⚡ Silero VAD detected silence - skipping Azure STT for efficiency");
                            
                            if (_logVADComparisons)
                            {
                                // Still send to Azure STT for comparison logging, but don't wait for it
                                _ = Task.Run(async () => await ProcessWithAzureSTT(audioBytes, sessionId, "comparison_only"));
                            }
                            
                            return string.Empty;
                        }
                    }
                    else
                    {
                        _logger.LogWarning("⚠️ Silero VAD failed - falling back to Azure VAD");
                    }
                }
                
                // Process with Azure STT (either as primary method or fallback)
                return await ProcessWithAzureSTT(audioBytes, sessionId, "primary");
            }
            catch (Exception ex)
            {
                _logger.LogError("❌ Speech-to-text error: {Error}", ex.Message);
                throw;
            }
        }

        private async Task<string> ProcessWithAzureSTT(byte[] audioBytes, string sessionId, string purpose)
        {
            try
            {
                _logger.LogDebug("Processing with Azure STT (purpose: {Purpose}, session: {SessionId})", purpose, sessionId);
                
                using var audioStream = new MemoryStream(audioBytes);
                using var audioConfig = AudioConfig.FromStreamInput(AudioInputStream.CreatePushStream());
                using var recognizer = new SpeechRecognizer(_speechConfig, audioConfig);
                
                var azureStartTime = DateTime.UtcNow;
                var result = await recognizer.RecognizeOnceAsync();
                var azureProcessingTime = (DateTime.UtcNow - azureStartTime).TotalMilliseconds;
                
                switch (result.Reason)
                {
                    case ResultReason.RecognizedSpeech:
                        _logger.LogInformation("🔵 Azure STT result: '{Text}' ({ProcessingMs:F0}ms, session: {SessionId})", 
                            result.Text, azureProcessingTime, sessionId);
                        
                        if (purpose == "comparison_only")
                        {
                            _logger.LogInformation("📊 VAD Comparison - Azure found speech: '{Text}' (Silero said silence)", result.Text);
                        }
                        
                        return result.Text;
                        
                    case ResultReason.NoMatch:
                        _logger.LogDebug("🔵 Azure STT: No speech detected ({ProcessingMs:F0}ms)", azureProcessingTime);
                        
                        if (purpose == "comparison_only")
                        {
                            _logger.LogInformation("📊 VAD Comparison - Both Silero and Azure detected no speech ✅");
                        }
                        
                        return string.Empty;
                        
                    case ResultReason.Canceled:
                        var cancellation = CancellationDetails.FromResult(result);
                        
                        if (cancellation.Reason == CancellationReason.Error)
                        {
                            _logger.LogError("❌ Azure STT error: {ErrorCode} - {ErrorDetails}", 
                                cancellation.ErrorCode, cancellation.ErrorDetails);
                            throw new InvalidOperationException($"Azure STT error: {cancellation.ErrorDetails}");
                        }
                        else
                        {
                            _logger.LogWarning("⚠️ Azure STT cancelled: {Reason}", cancellation.Reason);
                            return string.Empty;
                        }
                        
                    default:
                        _logger.LogWarning("⚠️ Azure STT unexpected result: {Reason}", result.Reason);
                        return string.Empty;
                }
            }
            catch (Exception ex)
            {
                if (purpose == "primary")
                {
                    _logger.LogError("❌ Azure STT processing error: {Error}", ex.Message);
                    throw;
                }
                else
                {
                    _logger.LogWarning("⚠️ Azure STT comparison processing error: {Error}", ex.Message);
                    return string.Empty;
                }
            }
        }

        public async Task<Stream> TextToSpeechAsync(string text)
        {
            try
            {
                _logger.LogDebug("Converting text to speech: '{Text}'", text);
                
                using var synthesizer = new SpeechSynthesizer(_speechConfig);
                using var result = await synthesizer.SpeakTextAsync(text);
                
                if (result.Reason == ResultReason.SynthesizingAudioCompleted)
                {
                    _logger.LogDebug("✅ Text-to-speech completed successfully");
                    return new MemoryStream(result.AudioData);
                }
                else if (result.Reason == ResultReason.Canceled)
                {
                    var cancellation = SpeechSynthesisCancellationDetails.FromResult(result);
                    _logger.LogError("❌ TTS error: {ErrorCode} - {ErrorDetails}", 
                        cancellation.ErrorCode, cancellation.ErrorDetails);
                    throw new InvalidOperationException($"TTS error: {cancellation.ErrorDetails}");
                }
                else
                {
                    _logger.LogWarning("⚠️ TTS unexpected result: {Reason}", result.Reason);
                    return Stream.Null;
                }
            }
            catch (Exception ex)
            {
                _logger.LogError("❌ Text-to-speech error: {Error}", ex.Message);
                throw;
            }
        }

        public async Task<byte[]> TextToSpeechBytesAsync(string text)
        {
            try
            {
                using var stream = await TextToSpeechAsync(text);
                using var memoryStream = new MemoryStream();
                await stream.CopyToAsync(memoryStream);
                return memoryStream.ToArray();
            }
            catch (Exception ex)
            {
                _logger.LogError("❌ Text-to-speech bytes error: {Error}", ex.Message);
                throw;
            }
        }

        private string GetVoiceForLanguage(string language)
        {
            return language.ToLower() switch
            {
                "en-us" => "en-US-JennyNeural",
                "en-gb" => "en-GB-SoniaNeural", 
                "es-es" => "es-ES-ElviraNeural",
                "fr-fr" => "fr-FR-DeniseNeural",
                "de-de" => "de-DE-KatjaNeural",
                "it-it" => "it-IT-ElsaNeural",
                "pt-br" => "pt-BR-FranciscaNeural",
                "ja-jp" => "ja-JP-NanamiNeural",
                "ko-kr" => "ko-KR-SunHiNeural",
                "zh-cn" => "zh-CN-XiaoxiaoNeural",
                _ => "en-US-JennyNeural"  // Default fallback
            };
        }
    }
}