import json
import logging
import os
import shutil
import subprocess
from typing import Dict, List, Tuple, Union

AUDIO_MERGE_BATCH_SIZE = 8


def _has_nvenc() -> bool:
    """Check if NVIDIA NVENC hardware encoder is available."""
    try:
        result = subprocess.run(
            ['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'nullsrc=s=64x64:d=0.1',
             '-c:v', 'h264_nvenc', '-f', 'null', '-'],
            capture_output=True, timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


# Detected once at import time
_NVENC_AVAILABLE = None


def _get_video_encoder() -> list:
    """Return ffmpeg video encoder args, preferring NVENC if available."""
    global _NVENC_AVAILABLE
    if _NVENC_AVAILABLE is None:
        _NVENC_AVAILABLE = _has_nvenc()
        if _NVENC_AVAILABLE:
            logging.info('NVENC GPU encoder detected, using h264_nvenc')
        else:
            logging.info('No NVENC, using libx264 CPU encoder')
    if _NVENC_AVAILABLE:
        return ['-c:v', 'h264_nvenc', '-preset', 'p4', '-cq', '23']
    return ['-c:v', 'libx264', '-preset', 'fast', '-crf', '23']


def _get_video_encoder_fast() -> list:
    """Return fast encoder args for simple content (slides, gaps)."""
    global _NVENC_AVAILABLE
    if _NVENC_AVAILABLE is None:
        _NVENC_AVAILABLE = _has_nvenc()
    if _NVENC_AVAILABLE:
        return ['-c:v', 'h264_nvenc', '-preset', 'p1', '-cq', '28']
    return ['-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23']


def _check_ffmpeg():
    """Check that ffmpeg and ffprobe are available."""
    for tool in ('ffmpeg', 'ffprobe'):
        if not shutil.which(tool):
            raise RuntimeError(
                f'{tool} is not installed. '
                'Install it with: apt install ffmpeg / brew install ffmpeg'
            )


def _is_valid_media(file_path: str) -> bool:
    """Check if a media file is valid (not corrupt / has moov atom)."""
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'csv=p=0', file_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        logging.warning(f'Corrupt/invalid file: {file_path}: {result.stderr.strip()[:200]}')
        return False
    return True


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
            *_get_video_encoder_fast(),
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
            *_get_video_encoder_fast(),
            '-c:a', 'aac', '-b:a', '128k', '-ar', '44100', '-ac', '2',
            '-r', '25',
            output_path,
        ],
        description=f'normalize segment {os.path.basename(input_path)}',
    )
    return output_path


def process_and_download_clips(
    directory: str, json_data: Dict
) -> Tuple[float, List[Tuple[str, float]], List[Dict]]:
    """Extract chunk URLs, start times, and presentation slides from the API JSON.

    Returns:
        (total_duration, [(url, start_time), ...], [slide_event, ...])

    Each slide_event is:
        {"time": float, "slide_number": int, "slide_url": str}
    """
    total_duration = float(json_data.get('duration', 0))
    if not total_duration:
        raise ValueError('Duration not found in JSON data.')

    chunks = []
    slide_events = []
    for event in json_data.get('eventLogs', []):
        if not isinstance(event, dict):
            continue
        data = event.get('data', {})
        if not isinstance(data, dict):
            continue

        # Media chunks (video/audio)
        if 'url' in data:
            stream = data.get('stream', {})
            conf_id = None
            if isinstance(stream, dict):
                conf = stream.get('conference', {})
                if isinstance(conf, dict):
                    conf_id = conf.get('id')
            chunks.append((data['url'], event.get('relativeTime', 0), conf_id))

        # Presentation slide changes
        if event.get('module') == 'presentation.update':
            fr = data.get('fileReference', {})
            if not isinstance(fr, dict):
                continue
            slide = fr.get('slide', {})
            if not isinstance(slide, dict) or not slide.get('url'):
                continue
            slide_url = slide['url']
            slide_events.append({
                'time': event.get('relativeTime', 0),
                'slide_number': slide.get('number', 0),
                'slide_url': slide_url,
            })

    # Deduplicate consecutive identical slides
    deduped_slides = []
    for se in slide_events:
        if not deduped_slides or se['slide_url'] != deduped_slides[-1]['slide_url']:
            deduped_slides.append(se)

    if deduped_slides:
        logging.info(f'Found {len(deduped_slides)} presentation slide changes')

    return total_duration, chunks, deduped_slides


def _deduplicate_overlapping(
    video_files: list,
) -> List[Tuple[str, float]]:
    """Remove overlapping video segments, keeping the best per time window.

    Recordings often have parallel streams (multiple webcams) at the
    same timestamp. Laying them out sequentially inflates the duration.
    This keeps only one segment per overlapping group.

    Priority: prefer the conference (user) with the most total segments
    (likely the presenter), then fall back to longest segment.
    """
    if not video_files:
        return video_files

    # Count segments per conf_id to identify the "main" user
    from collections import Counter
    conf_counts = Counter()
    for item in video_files:
        conf_id = item[2] if len(item) > 2 else None
        if conf_id:
            conf_counts[conf_id] += 1

    # Annotate with duration and conf_id
    annotated = []  # (path, start, duration, end, conf_id)
    for item in video_files:
        path, start = item[0], item[1]
        conf_id = item[2] if len(item) > 2 else None
        dur = _get_duration(path)
        annotated.append((path, start, dur, start + dur, conf_id))

    def _score(seg):
        """Higher score = more preferred. Prefer main user, then longer."""
        _, _, dur, _, conf_id = seg
        conf_rank = conf_counts.get(conf_id, 0) if conf_id else 0
        return (conf_rank, dur)

    # Sort by start time, then best score first
    annotated.sort(key=lambda x: (x[1], -_score(x)[0], -_score(x)[1]))

    kept = []
    for seg in annotated:
        path, start, dur, end, conf_id = seg
        if not kept:
            kept.append(seg)
            continue

        _, prev_start, _, prev_end, _ = kept[-1]

        if start >= prev_end - 0.5:
            kept.append(seg)
        else:
            # Overlaps — keep the one with better score
            if _score(seg) > _score(kept[-1]):
                logging.debug(
                    f'Dedup: replacing {kept[-1][2]:.1f}s segment '
                    f'(conf={kept[-1][4]}) with {dur:.1f}s segment '
                    f'(conf={conf_id}, score={_score(seg)})'
                )
                kept[-1] = seg

    original = len(video_files)
    deduped = len(kept)
    if original != deduped:
        logging.info(f'Dedup: {original} -> {deduped} segments '
                     f'(removed {original - deduped} overlapping)')

    return [(path, start) for path, start, dur, end, conf_id in kept]


def _composite_slides(
    video_path: str,
    slide_events: List[Dict],
    output_path: str,
    tmp_dir: str,
    total_duration: float,
) -> str:
    """Composite presentation slides with webcam video.

    Layout (1280x720):
      - Left 960px: presentation slide
      - Right 320px, top: webcam (320x180)
      - Right 320px, below webcam: black
    """
    CANVAS_W, CANVAS_H = 1280, 720
    SLIDE_W, SLIDE_H = 960, 720
    CAM_W, CAM_H = 320, 180

    logging.info(f'Compositing {len(slide_events)} slide changes onto video...')

    # Step 1: Create a slide video track using concat demuxer.
    # Each slide becomes a segment of the right duration.
    slides_dir = os.path.join(tmp_dir, 'slide_segments')
    os.makedirs(slides_dir, exist_ok=True)

    slide_segments = []
    for i, se in enumerate(slide_events):
        t_start = se['time']
        t_end = slide_events[i + 1]['time'] if i + 1 < len(slide_events) else total_duration
        duration = t_end - t_start
        if duration <= 0:
            continue

        seg_path = os.path.join(slides_dir, f'seg_{i}.mp4')
        # Use -frames:v to strictly limit frame count at 1fps.
        # This avoids runaway encoding for long durations.
        n_frames = max(1, int(duration))
        _run_ffmpeg(
            [
                'ffmpeg', '-y', '-v', 'error',
                '-loop', '1', '-framerate', '1',
                '-i', se['local_path'],
                '-vf', f'scale={SLIDE_W}:{SLIDE_H}:force_original_aspect_ratio=decrease,'
                       f'pad={SLIDE_W}:{SLIDE_H}:(ow-iw)/2:(oh-ih)/2:white',
                *_get_video_encoder_fast(),
                '-pix_fmt', 'yuv420p',
                '-r', '1', '-frames:v', str(n_frames),
                seg_path,
            ],
            description=f'slide segment {i+1}/{len(slide_events)} '
                        f'({n_frames} frames, {duration:.0f}s)',
        )
        slide_segments.append(seg_path)

    # Add black leader if first slide starts after 0
    first_time = slide_events[0]['time'] if slide_events else 0
    if first_time > 0.5:
        leader_path = os.path.join(slides_dir, 'leader.mp4')
        _run_ffmpeg(
            [
                'ffmpeg', '-y', '-v', 'error',
                '-f', 'lavfi', '-i',
                f'color=c=black:s={SLIDE_W}x{SLIDE_H}:d={first_time}:r=1',
                *_get_video_encoder_fast(),
                '-pix_fmt', 'yuv420p',
                leader_path,
            ],
            description='slide leader (black)',
        )
        slide_segments.insert(0, leader_path)

    # Concatenate slide segments into one track
    slide_track_path = os.path.join(tmp_dir, 'slide_track.mp4')
    concat_list = os.path.join(slides_dir, 'concat.txt')
    with open(concat_list, 'w') as f:
        for seg in slide_segments:
            f.write(f"file '{os.path.abspath(seg)}'\n")

    _run_ffmpeg(
        [
            'ffmpeg', '-y', '-v', 'error',
            '-f', 'concat', '-safe', '0', '-i', concat_list,
            '-c', 'copy', slide_track_path,
        ],
        description='concat slide track',
    )
    logging.info('Slide track created')

    # Step 2: Combine slide track + webcam into final layout.
    # Simple 2-input overlay: slide on left, webcam scaled to top-right.
    filter_graph = (
        f'[1:v]scale={CAM_W}:{CAM_H}:force_original_aspect_ratio=decrease,'
        f'pad={CAM_W}:{CAM_H}:(ow-iw)/2:(oh-ih)/2:black[webcam];'
        f'[0:v]pad={CANVAS_W}:{CANVAS_H}:0:0:black[padded];'
        f'[padded][webcam]overlay={SLIDE_W}:0[out]'
    )

    cmd = [
        'ffmpeg', '-y', '-v', 'warning',
        '-i', slide_track_path,
        '-i', video_path,
        '-filter_complex', filter_graph,
        '-map', '[out]', '-map', '1:a?',
        *_get_video_encoder(),
        '-c:a', 'copy',
        '-r', '25',
        '-shortest',
        output_path,
    ]

    _run_ffmpeg(cmd, description='composite slides + webcam')
    logging.info('Slide compositing complete')

    # Cleanup slide segments
    shutil.rmtree(slides_dir, ignore_errors=True)
    return output_path


def compile_final_video(
    total_duration: float,
    downloaded_files: List[Tuple[str, float]],
    directory: str,
    output_path: str,
    max_duration: Union[int, None],
    slide_events: List[Dict] = None,
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

    video_files = []  # (path, start_time, conf_id)
    audio_files = []  # (path, start_time)
    skipped = 0

    for item in downloaded_files:
        file_path, start_time = item[0], item[1]
        conf_id = item[2] if len(item) > 2 else None
        if not _is_valid_media(file_path):
            skipped += 1
            continue
        if _has_video_stream(file_path):
            video_files.append((file_path, start_time, conf_id))
        else:
            audio_files.append((file_path, start_time))

    if skipped:
        logging.warning(f'Skipped {skipped} corrupt/invalid files')
    logging.info(f'Segments: {len(video_files)} video, {len(audio_files)} audio-only')

    if not video_files:
        logging.error('No video segments found.')
        return

    # Determine target resolution from the first video segment
    # Pick the largest resolution among video segments (some may be tiny thumbnails)
    target_w, target_h, target_pix_fmt = 0, 0, 'yuv420p'
    for vpath, *_ in video_files:
        w, h, pf = _get_video_params(vpath)
        if w * h > target_w * target_h:
            target_w, target_h, target_pix_fmt = w, h, pf
    # Minimum 640x360 for reasonable quality
    if target_w * target_h < 640 * 360:
        target_w, target_h = 640, 360
    logging.info(f'Target resolution: {target_w}x{target_h}, pix_fmt={target_pix_fmt}')

    # Sort by start time
    video_files.sort(key=lambda x: x[1])

    # Deduplicate overlapping segments: recordings often have parallel
    # streams (webcam + screen share) running at the same time. If we
    # concatenate them all sequentially the output is 3-4x too long.
    # Strategy: walk through sorted segments; when a new segment starts
    # before the current winner ends, keep whichever is longer and
    # discard the shorter one.
    video_files = _deduplicate_overlapping(video_files)
    logging.info(f'After dedup: {len(video_files)} non-overlapping video segments')

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

    # Composite slides if presentation data exists
    if slide_events:
        composited_path = os.path.join(tmp_dir, 'video_composited.mp4')
        _composite_slides(
            video_only_path, slide_events, composited_path,
            tmp_dir, total_duration,
        )
        video_only_path = composited_path

    # If there are audio-only tracks, overlay them
    if audio_files:
        logging.info(f'Merging {len(audio_files)} audio-only tracks...')
        result_path = _merge_audio_tracks(
            video_only_path, audio_files, tmp_dir, output_path,
            total_duration,
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
    total_duration: float = 0,
) -> str:
    """Merge audio-only tracks on top of the concatenated video.

    To avoid OOM (too many ffmpeg inputs) and disk exhaustion (huge WAVs),
    we mix audio in small batches directly with adelay inside the filter,
    outputting compressed m4a. Each batch produces one small file, and
    consumed intermediates are deleted immediately.

    Strategy:
      1. Mix audio files in batches of AUDIO_MERGE_BATCH_SIZE, applying
         adelay inside the filter graph (no pre-materialized delayed files).
      2. Tree-reduce batch outputs until one mixed track remains.
      3. Overlay the single mixed audio onto the video.
    """
    video_duration = total_duration or _get_duration(video_path)
    batch_size = AUDIO_MERGE_BATCH_SIZE

    # Step 1: Mix in batches with inline adelay (no intermediate WAVs).
    # Each batch takes up to batch_size original audio files, applies
    # adelay per track, mixes them, and writes a compressed m4a.
    batch_outputs = []  # (path, effective_start=0 since delay is baked in)
    for batch_idx, batch_start in enumerate(
        range(0, len(audio_files), batch_size)
    ):
        batch = audio_files[batch_start:batch_start + batch_size]
        batch_out = os.path.join(tmp_dir, f'audio_batch_{batch_idx}.m4a')

        inputs = []
        filter_parts = []
        mix_labels = []
        for j, (apath, start_time) in enumerate(batch):
            inputs.extend(['-i', apath])
            delay_ms = int(start_time * 1000)
            label = f'a{j}'
            filter_parts.append(
                f'[{j}:a]adelay={delay_ms}|{delay_ms},'
                f'apad=whole_dur={video_duration},'
                f'atrim=0:{video_duration},'
                f'asetpts=PTS-STARTPTS[{label}]'
            )
            mix_labels.append(f'[{label}]')

        if len(batch) == 1:
            # Single track — just delay and encode, no amix needed
            filter_graph = filter_parts[0].rsplit('[', 1)[0]  # strip label
        else:
            filter_graph = (
                ';'.join(filter_parts) + ';'
                + ''.join(mix_labels)
                + f'amix=inputs={len(batch)}:duration=longest:normalize=0'
            )

        try:
            _run_ffmpeg(
                [
                    'ffmpeg', '-y', '-v', 'error',
                    *inputs,
                    '-filter_complex', filter_graph,
                    '-c:a', 'aac', '-b:a', '128k',
                    '-ar', '44100', '-ac', '2',
                    batch_out,
                ],
                description=f'amix batch {batch_idx+1} '
                            f'({len(batch)} tracks, offset {batch[0][1]:.0f}-'
                            f'{batch[-1][1]:.0f}s)',
            )
            batch_outputs.append(batch_out)
        except subprocess.CalledProcessError:
            logging.warning(
                f'Audio batch {batch_idx+1} failed, skipping '
                f'{len(batch)} tracks'
            )
        logging.info(
            f'Audio batch {batch_idx+1}/'
            f'{(len(audio_files) + batch_size - 1) // batch_size} done'
        )

    # If no batches succeeded, skip audio overlay entirely.
    if not batch_outputs:
        logging.warning('All audio batches failed, skipping audio overlay')
        shutil.move(video_path, output_path)
        return output_path

    # Step 2: Tree-reduce batch outputs until one remains.
    round_num = 0
    current_paths = batch_outputs
    while len(current_paths) > 1:
        round_num += 1
        next_paths = []
        for batch_start in range(0, len(current_paths), batch_size):
            batch = current_paths[batch_start:batch_start + batch_size]
            if len(batch) == 1:
                next_paths.append(batch[0])
                continue

            batch_out = os.path.join(
                tmp_dir, f'audio_reduce_r{round_num}_b{batch_start}.m4a'
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
                    '-c:a', 'aac', '-b:a', '128k',
                    '-ar', '44100', '-ac', '2',
                    batch_out,
                ],
                description=f'amix reduce round {round_num}, '
                            f'{len(batch)} tracks',
            )
            next_paths.append(batch_out)

            # Delete consumed intermediates
            for bp in batch:
                try:
                    os.remove(bp)
                except OSError:
                    pass

        logging.info(
            f'Audio reduce round {round_num}: {len(current_paths)} -> '
            f'{len(next_paths)} tracks'
        )
        current_paths = next_paths

    mixed_audio_path = current_paths[0]

    # Step 3: Overlay the single mixed audio track onto the video.
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
