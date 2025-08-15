using Microsoft.Skype.Bots.Media;

namespace EchoBot.Bot
{
    public class MediaStreamEventArgs : EventArgs
    {
        public List<AudioMediaBuffer> AudioMediaBuffers { get; set; }
    }
}
