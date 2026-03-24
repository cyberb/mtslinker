import json
import logging
import os
import shutil
import subprocess
from typing import Dict, List, Tuple, Union


def _check_ffmpeg():
    """Check that ffmpeg and ffprobe are available."""
    for tool in ('ffmpeg', 'ffprobe'):
        if not shutil.which(tool):
            raise RuntimeError(
                f'{tool} is not installed. '
                'Install it with: apt install ffmpeg / brew install ffmpeg'
            )


def _ffprobe_streams(file_path: str) -> dict:
    """Return ffprobe stream info for a file."""
    result = subprocess.run(
        [
            'ffprobe', '-v', 'quiet',
            '-print_format', 'json',
            '-show_streams',
            '-show_format',
            file_path,
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return {}
    return json.loads(result.stdout)


def _has_video_stream(file_path: str) -> bool:
    """Check if a file contains a video stream."""
    info = _ffprobe_streams(file_path)
    for stream in info.get('streams', []):
        if stream.get('codec_type') == 'video':
            return True
    return False


def _get_duration(file_path: str) -> float:
    """Get duration of a media file in seconds."""
    info = _ffprobe_streams(file_path)
    fmt = info.get('format', {})
    if 'duration' in fmt:
        return float(fmt['duration'])
    for stream in info.get('streams', []):
        if 'duration' in stream:
            return float(stream['duration'])
    return 0.0


def _get_video_params(file_path: str) -> Tuple[int, int, str]:
    """Get width, height, and pixel format of the first video stream."""
    info = _ffprobe_streams(file_path)
    for stream in info.get('streams', []):
        if stream.get('codec_type') == 'video':
            w = int(stream.get('width', 1920))
            h = int(stream.get('height', 1080))
            pix_fmt = stream.get('pix_fmt', 'yuv420p')
            return w, h, pix_fmt
    return 1920, 1080, 'yuv420p'


def _generate_black_segment(output_path: str, duration: float,
                            width: int = 1920, height: int = 1080,
                            pix_fmt: str = 'yuv420p') -> str:
    """Generate a short black video+silent audio segment with ffmpeg."""
    subprocess.run(
        [
            'ffmpeg', '-y', '-v', 'error',
            '-f', 'lavfi', '-i', f'color=c=black:s={width}x{height}:d={duration}:r=25',
            '-f', 'lavfi', '-i', f'anullsrc=r=44100:cl=stereo',
            '-t', str(duration),
            '-c:v', 'libx264', '-preset', 'ultrafast', '-tune', 'stillimage',
            '-pix_fmt', pix_fmt,
            '-c:a', 'aac', '-b:a', '128k',
            '-shortest',
            output_path,
        ],
        capture_output=True, check=True,
    )
    return output_path


def _ensure_audio_stream(input_path: str, output_path: str) -> str:
    """If the file has no audio stream, add a silent one so concat works."""
    info = _ffprobe_streams(input_path)
    has_audio = any(
        s.get('codec_type') == 'audio' for s in info.get('streams', [])
    )
    if has_audio:
        return input_path

    subprocess.run(
        [
            'ffmpeg', '-y', '-v', 'error',
            '-i', input_path,
            '-f', 'lavfi', '-i', 'anullsrc=r=44100:cl=stereo',
            '-c:v', 'copy', '-c:a', 'aac', '-b:a', '128k',
            '-shortest',
            output_path,
        ],
        capture_output=True, check=True,
    )
    return output_path


def _normalize_segment(input_path: str, output_path: str,
                       width: int, height: int, pix_fmt: str) -> str:
    """Re-encode a segment to a common format for reliable concatenation."""
    subprocess.run(
        [
            'ffmpeg', '-y', '-v', 'error',
            '-i', input_path,
            '-vf', f'scale={width}:{height}:force_original_aspect_ratio=decrease,'
                   f'pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,'
                   f'setsar=1',
            '-pix_fmt', pix_fmt,
            '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '18',
            '-c:a', 'aac', '-b:a', '128k', '-ar', '44100', '-ac', '2',
            '-r', '25',
            output_path,
        ],
        capture_output=True, check=True,
    )
    return output_path


def process_and_download_clips(
    directory: str, json_data: Dict
) -> Tuple[float, List[Tuple[str, float]]]:
    """Extract chunk URLs and start times from the API JSON data.

    Returns:
        (total_duration, [(url, start_time), ...])
    """
    total_duration = float(json_data.get('duration', 0))
    if not total_duration:
        raise ValueError('Duration not found in JSON data.')

    chunks = []
    for event in json_data.get('eventLogs', []):
        if isinstance(event, dict):
            data = event.get('data', {})
            if isinstance(data, dict) and 'url' in data:
                url = data['url']
                start_time = event.get('relativeTime', 0)
                chunks.append((url, start_time))

    return total_duration, chunks


def compile_final_video(
    total_duration: float,
    downloaded_files: List[Tuple[str, float]],
    directory: str,
    output_path: str,
    max_duration: Union[int, None],
):
    """Concatenate downloaded segments using ffmpeg (no MoviePy re-encoding).

    Steps:
      1. Separate video and audio-only files.
      2. Normalize video segments to a common resolution.
      3. Generate black gap segments where needed.
      4. Concatenate with ffmpeg concat demuxer (-c copy).
      5. Merge audio-only tracks on top.
    """
    _check_ffmpeg()

    video_files = []  # (path, start_time)
    audio_files = []  # (path, start_time)

    for file_path, start_time in downloaded_files:
        if _has_video_stream(file_path):
            video_files.append((file_path, start_time))
        else:
            audio_files.append((file_path, start_time))

    logging.info(f'Segments: {len(video_files)} video, {len(audio_files)} audio-only')

    if not video_files:
        logging.error('No video segments found.')
        return

    # Determine target resolution from the first video segment
    first_video = video_files[0][0]
    target_w, target_h, target_pix_fmt = _get_video_params(first_video)
    logging.info(f'Target resolution: {target_w}x{target_h}, pix_fmt={target_pix_fmt}')

    # Sort by start time
    video_files.sort(key=lambda x: x[1])

    # Build the list of segments (normalized videos + gap fillers)
    tmp_dir = os.path.join(directory, '_tmp_ffmpeg')
    os.makedirs(tmp_dir, exist_ok=True)

    concat_segments = []
    current_time = 0.0

    for i, (vpath, start_time) in enumerate(video_files):
        # Insert black gap if needed
        gap = start_time - current_time
        if gap > 0.1:  # skip tiny gaps < 100ms
            gap_path = os.path.join(tmp_dir, f'gap_{i}.mp4')
            _generate_black_segment(gap_path, gap, target_w, target_h, target_pix_fmt)
            concat_segments.append(gap_path)
            logging.info(f'Generated {gap:.1f}s black gap before segment {i}')

        # Normalize the segment
        norm_path = os.path.join(tmp_dir, f'norm_{i}.mp4')
        _normalize_segment(vpath, norm_path, target_w, target_h, target_pix_fmt)
        # Ensure it has an audio stream
        with_audio_path = os.path.join(tmp_dir, f'norma_{i}.mp4')
        final_seg = _ensure_audio_stream(norm_path, with_audio_path)
        concat_segments.append(final_seg)

        seg_dur = _get_duration(final_seg)
        current_time = start_time + seg_dur

    # Trailing gap
    if current_time < total_duration - 0.1:
        gap_path = os.path.join(tmp_dir, 'gap_end.mp4')
        _generate_black_segment(gap_path, total_duration - current_time,
                                target_w, target_h, target_pix_fmt)
        concat_segments.append(gap_path)

    # Write concat list
    concat_list_path = os.path.join(tmp_dir, 'concat.txt')
    with open(concat_list_path, 'w') as f:
        for seg in concat_segments:
            f.write(f"file '{os.path.abspath(seg)}'\n")

    # Concatenate with ffmpeg concat demuxer (stream copy - no re-encoding)
    video_only_path = os.path.join(tmp_dir, 'video_concat.mp4')
    logging.info(f'Concatenating {len(concat_segments)} segments...')

    concat_cmd = [
        'ffmpeg', '-y', '-v', 'warning',
        '-f', 'concat', '-safe', '0',
        '-i', concat_list_path,
        '-c', 'copy',
    ]

    if max_duration:
        concat_cmd.extend(['-t', str(max_duration)])

    concat_cmd.append(video_only_path)
    subprocess.run(concat_cmd, check=True)

    # If there are audio-only tracks, overlay them
    if audio_files:
        logging.info(f'Merging {len(audio_files)} audio-only tracks...')
        result_path = _merge_audio_tracks(video_only_path, audio_files, tmp_dir, output_path)
    else:
        # Just move/copy the result
        shutil.move(video_only_path, output_path)
        result_path = output_path

    # Cleanup temp files
    shutil.rmtree(tmp_dir, ignore_errors=True)
    logging.info(f'Final video saved to {result_path}')


def _merge_audio_tracks(
    video_path: str,
    audio_files: List[Tuple[str, float]],
    tmp_dir: str,
    output_path: str,
) -> str:
    """Merge audio-only tracks on top of the concatenated video using ffmpeg."""
    # Build a complex filter to mix audio tracks with delays
    inputs = ['-i', video_path]
    filter_parts = []
    audio_labels = ['[0:a]']  # existing audio from video

    for i, (apath, start_time) in enumerate(audio_files):
        inputs.extend(['-i', apath])
        input_idx = i + 1
        delay_ms = int(start_time * 1000)
        filter_parts.append(
            f'[{input_idx}:a]adelay={delay_ms}|{delay_ms}[a{input_idx}]'
        )
        audio_labels.append(f'[a{input_idx}]')

    mix_filter = ';'.join(filter_parts)
    if mix_filter:
        mix_filter += ';'
    mix_filter += ''.join(audio_labels) + f'amix=inputs={len(audio_labels)}:normalize=0[aout]'

    subprocess.run(
        [
            'ffmpeg', '-y', '-v', 'warning',
            *inputs,
            '-filter_complex', mix_filter,
            '-map', '0:v', '-map', '[aout]',
            '-c:v', 'copy',
            '-c:a', 'aac', '-b:a', '192k',
            output_path,
        ],
        check=True,
    )
    return output_path
