import yt_dlp

# URL of the YouTube video
url = "https://www.youtube.com/watch?v=299TLCmymys"

# yt-dlp options
ydl_opts = {
    'format': 'bestaudio/best',
    'outtmpl': 'video_audio.%(ext)s',  # save as video_audio.mp3 or .m4a
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav',       # convert to wav
        'preferredquality': '192',     # bitrate for mp3 if you use mp3
    }],
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])