import json
import logging
import os
import shutil
import subprocess
from typing import Dict, List, Tuple, Union

AUDIO_MERGE_BATCH_SIZE = 8


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


def _run_ffmpeg(cmd: list, description: str = 'ffmpeg'):
    """Run an ffmpeg command, logging stderr on failure."""
    logging.debug(f'Running {description}: {" ".join(cmd[:6])}...')
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip() if result.stderr else '(no stderr)'
        logging.error(f'{description} failed (exit {result.returncode}):\n{stderr}')
        raise subprocess.CalledProcessError(
            result.returncode, cmd[0], result.stdout, result.stderr
        )
    if result.stderr and result.stderr.strip():
        logging.debug(f'{description} stderr: {result.stderr.strip()[:500]}')
    return result


def _generate_black_segment(output_path: str, duration: float,
                            width: int = 1920, height: int = 1080,
                            pix_fmt: str = 'yuv420p') -> str:
    """Generate a short black video+silent audio segment with ffmpeg."""
    _run_ffmpeg(
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
        description=f'generate black segment ({duration:.1f}s)',
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

    _run_ffmpeg(
        [
            'ffmpeg', '-y', '-v', 'error',
            '-i', input_path,
            '-f', 'lavfi', '-i', 'anullsrc=r=44100:cl=stereo',
            '-c:v', 'copy', '-c:a', 'aac', '-b:a', '128k',
            '-shortest',
            output_path,
        ],
        description='add silent audio stream',
    )
    return output_path


def _normalize_segment(input_path: str, output_path: str,
                       width: int, height: int, pix_fmt: str) -> str:
    """Re-encode a segment to a common format for reliable concatenation."""
    _run_ffmpeg(
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
        description=f'normalize segment {os.path.basename(input_path)}',
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
      5. Merge audio-only tracks on top (in batches to avoid OOM).
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
    _run_ffmpeg(concat_cmd, description=f'concat {len(concat_segments)} segments')

    # If there are audio-only tracks, overlay them
    if audio_files:
        logging.info(f'Merging {len(audio_files)} audio-only tracks...')
        result_path = _merge_audio_tracks(
            video_only_path, audio_files, tmp_dir, output_path
        )
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
    """Merge audio-only tracks on top of the concatenated video.

    To avoid OOM from passing dozens of inputs to a single ffmpeg amix,
    we pre-mix all audio tracks into one WAV in batches, then overlay
    that single track onto the video.

    Strategy:
      1. Each audio file is individually converted to a delayed WAV
         (silence-padded to its start_time offset).
      2. WAVs are mixed in batches of AUDIO_MERGE_BATCH_SIZE using amix.
      3. Batch results are mixed together (tree reduction) until one
         remains.
      4. The final mixed audio is overlaid onto the video.
    """
    video_duration = _get_duration(video_path)
    batch_size = AUDIO_MERGE_BATCH_SIZE

    # Step 1: Convert each audio track to a delayed mono/stereo WAV.
    # We use adelay + apad + atrim so each file is positioned at its
    # correct offset and truncated to video duration. This way amix
    # inputs are all the same length and ffmpeg doesn't need to buffer
    # indefinitely.
    delayed_paths = []
    for i, (apath, start_time) in enumerate(audio_files):
        delayed_path = os.path.join(tmp_dir, f'audio_delayed_{i}.wav')
        delay_ms = int(start_time * 1000)
        _run_ffmpeg(
            [
                'ffmpeg', '-y', '-v', 'error',
                '-i', apath,
                '-af', (
                    f'adelay={delay_ms}|{delay_ms},'
                    f'apad=whole_dur={video_duration},'
                    f'atrim=0:{video_duration},'
                    f'asetpts=PTS-STARTPTS'
                ),
                '-ar', '44100', '-ac', '2',
                delayed_path,
            ],
            description=f'delay audio track {i}/{len(audio_files)} '
                        f'(offset {start_time:.1f}s)',
        )
        delayed_paths.append(delayed_path)
        logging.debug(f'Prepared delayed audio {i+1}/{len(audio_files)}')

    # Step 2+3: Tree-reduce via amix in batches.
    round_num = 0
    current_paths = delayed_paths
    while len(current_paths) > 1:
        round_num += 1
        next_paths = []
        for batch_start in range(0, len(current_paths), batch_size):
            batch = current_paths[batch_start:batch_start + batch_size]
            if len(batch) == 1:
                next_paths.append(batch[0])
                continue

            batch_out = os.path.join(
                tmp_dir, f'audio_mix_r{round_num}_b{batch_start}.wav'
            )
            inputs = []
            for bp in batch:
                inputs.extend(['-i', bp])

            labels = ''.join(f'[{j}:a]' for j in range(len(batch)))
            amix_filter = (
                f'{labels}amix=inputs={len(batch)}'
                f':duration=longest:normalize=0'
            )

            _run_ffmpeg(
                [
                    'ffmpeg', '-y', '-v', 'error',
                    *inputs,
                    '-filter_complex', amix_filter,
                    '-ar', '44100', '-ac', '2',
                    batch_out,
                ],
                description=f'amix round {round_num}, batch {batch_start} '
                            f'({len(batch)} tracks)',
            )
            next_paths.append(batch_out)

            # Clean up consumed intermediate files (not the originals
            # from round 0 — those are the delayed WAVs we still need
            # if something goes wrong, but they're in tmp_dir anyway).
            if round_num > 1:
                for bp in batch:
                    try:
                        os.remove(bp)
                    except OSError:
                        pass

        logging.info(
            f'Audio mix round {round_num}: {len(current_paths)} -> '
            f'{len(next_paths)} tracks'
        )
        current_paths = next_paths

    mixed_audio_path = current_paths[0]

    # Step 4: Overlay the single mixed audio track onto the video.
    logging.info('Overlaying mixed audio onto video...')
    _run_ffmpeg(
        [
            'ffmpeg', '-y', '-v', 'warning',
            '-i', video_path,
            '-i', mixed_audio_path,
            '-filter_complex',
            '[0:a][1:a]amix=inputs=2:duration=first:normalize=0[aout]',
            '-map', '0:v', '-map', '[aout]',
            '-c:v', 'copy',
            '-c:a', 'aac', '-b:a', '192k',
            output_path,
        ],
        description='overlay mixed audio onto video',
    )
    return output_path
