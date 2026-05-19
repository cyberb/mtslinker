import os

from mtslinker.ffmpeg import FFmpegRunner
from mtslinker.prober import MediaProber


class SegmentBuilder:
    """Builds, normalizes, and deduplicates video segments."""

    def __init__(self, ffmpeg: FFmpegRunner, prober: MediaProber):
        self.ffmpeg = ffmpeg
        self.prober = prober

    def generate_black(self, output_path: str, duration: float,
                       width: int = 1920, height: int = 1080,
                       pix_fmt: str = 'yuv420p') -> str:
        self.ffmpeg.run(
            [
                'ffmpeg', '-y', '-v', 'error',
                '-f', 'lavfi', '-i', f'color=c=black:s={width}x{height}:d={duration}:r=25',
                '-f', 'lavfi', '-i', f'anullsrc=r=44100:cl=stereo',
                '-t', str(duration),
                *self.ffmpeg.get_video_encoder_fast(),
                '-pix_fmt', pix_fmt,
                '-c:a', 'aac', '-b:a', '128k',
                '-shortest',
                output_path,
            ],
            description=f'generate black segment ({duration:.1f}s)',
        )
        return output_path

    def ensure_audio(self, input_path: str, output_path: str) -> str:
        info = self.prober.probe_streams(input_path)
        has_audio = any(
            s.get('codec_type') == 'audio' for s in info.get('streams', [])
        )
        if has_audio:
            return input_path
        # With -c:v copy the muxer gets every video packet almost instantly,
        # while anullsrc audio is produced lazily. To interleave correctly the
        # muxer buffers the copied video in RAM waiting for audio, which on a
        # long segment grows until "av_interleaved_write_frame: Cannot allocate
        # memory". -max_interleave_delta 0 makes it write packets immediately
        # instead of buffering to interleave. We also bound the anullsrc input
        # itself to the measured duration so it is finite, not infinite.
        duration = self.prober.get_duration(input_path)
        silent_in = (
            ['-f', 'lavfi', '-t', str(duration), '-i', 'anullsrc=r=44100:cl=stereo']
            if duration > 0
            else ['-f', 'lavfi', '-i', 'anullsrc=r=44100:cl=stereo']
        )
        self.ffmpeg.run(
            [
                'ffmpeg', '-y', '-v', 'error',
                '-i', input_path,
                *silent_in,
                '-c:v', 'copy', '-c:a', 'aac', '-b:a', '128k',
                '-shortest', '-max_interleave_delta', '0',
                output_path,
            ],
            description='add silent audio stream',
        )
        return output_path

    def normalize(self, input_path: str, output_path: str,
                  width: int, height: int, pix_fmt: str,
                  max_duration: float = 0, seek: float = 0) -> str:
        seek_args = ['-ss', str(seek)] if seek > 0 else []
        duration_args = ['-t', str(max_duration)] if max_duration > 0 else []
        self.ffmpeg.run(
            [
                'ffmpeg', '-y', '-v', 'error',
                *seek_args,
                '-i', input_path,
                '-vf', f'scale={width}:{height}:force_original_aspect_ratio=decrease,'
                       f'pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,setsar=1',
                '-pix_fmt', pix_fmt,
                *self.ffmpeg.get_video_encoder_fast(),
                '-c:a', 'aac', '-b:a', '128k', '-ar', '44100', '-ac', '2',
                '-r', '25',
                *duration_args,
                output_path,
            ],
            description=f'normalize segment {os.path.basename(input_path)}',
        )
        return output_path


