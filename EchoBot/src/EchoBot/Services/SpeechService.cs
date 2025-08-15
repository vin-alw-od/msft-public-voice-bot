using Microsoft.CognitiveServices.Speech;
using Microsoft.CognitiveServices.Speech.Audio;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using System.Text;

namespace EchoBot.Services
{
    public interface ISpeechService
    {
        Task<string> SpeechToTextAsync(Stream audioStream);
        Task<Stream> TextToSpeechAsync(string text);
        Task<byte[]> TextToSpeechBytesAsync(string text);
    }

    public class SpeechService : ISpeechService
    {
        private readonly SpeechConfig _speechConfig;
        private readonly ILogger<SpeechService> _logger;
        private readonly string _voiceName;

        public SpeechService(IConfiguration configuration, ILogger<SpeechService> logger)
        {
            _logger = logger;
            
            var speechKey = configuration.GetValue<string>("SpeechConfigKey");
            var speechRegion = configuration.GetValue<string>("SpeechConfigRegion");
            var botLanguage = configuration.GetValue<string>("BotLanguage") ?? "en-US";
            
            if (string.IsNullOrEmpty(speechKey) || string.IsNullOrEmpty(speechRegion))
            {
                _logger.LogWarning("Speech service not configured. SpeechConfigKey and SpeechConfigRegion are required.");
                return;
            }

            _speechConfig = SpeechConfig.FromSubscription(speechKey, speechRegion);
            _speechConfig.SpeechRecognitionLanguage = botLanguage;
            
            // Set appropriate voice based on language
            _voiceName = botLanguage switch
            {
                "en-US" => "en-US-JennyNeural",
                "en-GB" => "en-GB-LibbyNeural", 
                "es-ES" => "es-ES-ElviraNeural",
                "fr-FR" => "fr-FR-DeniseNeural",
                _ => "en-US-JennyNeural"
            };
            
            _speechConfig.SpeechSynthesisVoiceName = _voiceName;
            
            _logger.LogInformation("Speech service initialized with language: {Language}, voice: {Voice}", botLanguage, _voiceName);
        }

        public async Task<string> SpeechToTextAsync(Stream audioStream)
        {
            if (_speechConfig == null)
            {
                throw new InvalidOperationException("Speech service is not configured");
            }

            try
            {
                _logger.LogInformation("Starting speech-to-text conversion");

                // Convert stream to the format expected by Speech SDK
                using var audioConfig = AudioConfig.FromStreamInput(AudioInputStream.CreatePushStream());
                using var speechRecognizer = new SpeechRecognizer(_speechConfig, audioConfig);

                // Push audio data to the recognizer
                var pushStream = AudioInputStream.CreatePushStream();
                audioConfig.Dispose();
                
                using var newAudioConfig = AudioConfig.FromStreamInput(pushStream);
                using var recognizer = new SpeechRecognizer(_speechConfig, newAudioConfig);

                // Read audio stream and push to recognizer
                var buffer = new byte[1024];
                int bytesRead;
                while ((bytesRead = await audioStream.ReadAsync(buffer, 0, buffer.Length)) > 0)
                {
                    pushStream.Write(buffer, bytesRead);
                }
                pushStream.Close();

                // Recognize speech
                var result = await recognizer.RecognizeOnceAsync();

                switch (result.Reason)
                {
                    case ResultReason.RecognizedSpeech:
                        _logger.LogInformation("Speech recognized: {Text}", result.Text);
                        return result.Text;
                    
                    case ResultReason.NoMatch:
                        _logger.LogWarning("No speech could be recognized");
                        return string.Empty;
                    
                    case ResultReason.Canceled:
                        var cancellation = CancellationDetails.FromResult(result);
                        _logger.LogError("Speech recognition canceled: {Reason}, {Details}", cancellation.Reason, cancellation.ErrorDetails);
                        throw new InvalidOperationException($"Speech recognition canceled: {cancellation.ErrorDetails}");
                    
                    default:
                        _logger.LogError("Unexpected speech recognition result: {Reason}", result.Reason);
                        throw new InvalidOperationException($"Unexpected speech recognition result: {result.Reason}");
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to convert speech to text");
                throw new InvalidOperationException("Failed to convert speech to text", ex);
            }
        }

        public async Task<Stream> TextToSpeechAsync(string text)
        {
            var audioBytes = await TextToSpeechBytesAsync(text);
            return new MemoryStream(audioBytes);
        }

        public async Task<byte[]> TextToSpeechBytesAsync(string text)
        {
            if (_speechConfig == null)
            {
                throw new InvalidOperationException("Speech service is not configured");
            }

            if (string.IsNullOrWhiteSpace(text))
            {
                return Array.Empty<byte>();
            }

            try
            {
                _logger.LogInformation("Starting text-to-speech conversion for text length: {Length}", text.Length);

                using var speechSynthesizer = new SpeechSynthesizer(_speechConfig, null);
                
                // Generate SSML for better voice control
                var ssml = GenerateSSML(text);
                
                var result = await speechSynthesizer.SpeakSsmlAsync(ssml);

                switch (result.Reason)
                {
                    case ResultReason.SynthesizingAudioCompleted:
                        _logger.LogInformation("Text-to-speech completed successfully. Audio length: {Length} bytes", result.AudioData.Length);
                        return result.AudioData;
                    
                    case ResultReason.Canceled:
                        var cancellation = SpeechSynthesisCancellationDetails.FromResult(result);
                        _logger.LogError("Text-to-speech canceled: {Reason}, {Details}", cancellation.Reason, cancellation.ErrorDetails);
                        throw new InvalidOperationException($"Text-to-speech canceled: {cancellation.ErrorDetails}");
                    
                    default:
                        _logger.LogError("Unexpected text-to-speech result: {Reason}", result.Reason);
                        throw new InvalidOperationException($"Unexpected text-to-speech result: {result.Reason}");
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to convert text to speech");
                throw new InvalidOperationException("Failed to convert text to speech", ex);
            }
        }

        private string GenerateSSML(string text)
        {
            // Create SSML with appropriate voice settings for conversational speech
            var ssml = new StringBuilder();
            ssml.AppendLine("<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='en-US'>");
            ssml.AppendLine($"<voice name='{_voiceName}'>");
            ssml.AppendLine("<prosody rate='medium' pitch='medium'>");
            
            // Clean and escape the text
            var cleanText = System.Security.SecurityElement.Escape(text);
            ssml.AppendLine(cleanText);
            
            ssml.AppendLine("</prosody>");
            ssml.AppendLine("</voice>");
            ssml.AppendLine("</speak>");

            return ssml.ToString();
        }
    }
}