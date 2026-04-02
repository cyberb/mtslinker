import json
import logging
import math
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


def _has_cuda_overlay() -> bool:
    """Check if ffmpeg has CUDA overlay filter support."""
    try:
        result = subprocess.run(
            ['ffmpeg', '-v', 'error', '-filters'],
            capture_output=True, text=True, timeout=10,
        )
        return 'overlay_cuda' in result.stdout
    except Exception:
        return False


# Detected once at import time
_NVENC_AVAILABLE = None
_CUDA_OVERLAY_AVAILABLE = None


def _detect_gpu():
    """Detect GPU capabilities once."""
    global _NVENC_AVAILABLE, _CUDA_OVERLAY_AVAILABLE
    if _NVENC_AVAILABLE is None:
        _NVENC_AVAILABLE = _has_nvenc()
        _CUDA_OVERLAY_AVAILABLE = _has_cuda_overlay() if _NVENC_AVAILABLE else False
        if _CUDA_OVERLAY_AVAILABLE:
            logging.info('CUDA overlay + NVENC detected, using full GPU pipeline')
        elif _NVENC_AVAILABLE:
            logging.info('NVENC detected (no CUDA overlay), using GPU encoder only')
        else:
            logging.info('No GPU support, using CPU pipeline')


def _get_video_encoder() -> list:
    """Return ffmpeg video encoder args, preferring NVENC if available."""
    _detect_gpu()
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


def _compute_grid(n: int) -> Tuple[int, int]:
    """Return (cols, rows) for a grid holding n items."""
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return cols, rows


def _composite_grid(
    active_segments: list,
    duration: float,
    output_path: str,
    target_w: int,
    target_h: int,
) -> str:
    """Create a grid video from multiple simultaneous webcam segments.

    Args:
        active_segments: list of (path, offset_within_file) tuples
        duration: length of this time window
        output_path: where to write
        target_w, target_h: output resolution
    """
    n = len(active_segments)
    cols, rows = _compute_grid(n)
    cell_w = target_w // cols
    cell_h = target_h // rows

    inputs = []
    filter_parts = []
    labels = []

    for i, (path, offset) in enumerate(active_segments):
        inputs.extend(['-ss', str(offset), '-i', path])
        label = f'v{i}'
        filter_parts.append(
            f'[{i}:v]scale={cell_w}:{cell_h}:force_original_aspect_ratio=decrease,'
            f'pad={cell_w}:{cell_h}:(ow-iw)/2:(oh-ih)/2:black,setsar=1[{label}]'
        )
        labels.append(f'[{label}]')

    # Pad with black cells if needed
    total_cells = cols * rows
    for i in range(n, total_cells):
        inputs.extend(['-f', 'lavfi', '-i',
                       f'color=black:s={cell_w}x{cell_h}:d={duration}:r=25'])
        label = f'v{i}'
        labels.append(f'[{len(active_segments) + i - n}:v]')
        # lavfi inputs don't need scaling, rename
        idx = len(active_segments) + i - n
        labels[-1] = f'[{idx}:v]'

    # Build xstack layout string: x_y positions
    layout_parts = []
    for i in range(total_cells):
        c = i % cols
        r = i // cols
        layout_parts.append(f'{c * cell_w}_{r * cell_h}')
    layout = '|'.join(layout_parts)

    filter_graph = ';'.join(filter_parts)
    if filter_graph:
        filter_graph += ';'
    filter_graph += (
        ''.join(labels)
        + f'xstack=inputs={total_cells}:layout={layout}[out]'
    )

    # Audio is handled separately by _merge_audio_tracks — output video only.
    cmd = [
        'ffmpeg', '-y', '-v', 'error',
        *inputs,
        '-t', str(duration),
        '-filter_complex', filter_graph,
        '-map', '[out]',
        '-an',
        *_get_video_encoder_fast(),
        '-pix_fmt', 'yuv420p',
        '-r', '25',
        output_path,
    ]
    _run_ffmpeg(cmd, description=f'grid {n} webcams, {duration:.0f}s')
    return output_path


def _build_grid_segments(
    video_files: list,
    tmp_dir: str,
    target_w: int,
    target_h: int,
    total_duration: float,
) -> List[Tuple[str, float]]:
    """Build grid-composited segments from overlapping webcam streams.

    Instead of deduplicating, composites all concurrent webcams into a grid.
    Returns a list of (segment_path, start_time) ready for concatenation.
    """
    # Annotate with duration
    annotated = []
    for item in video_files:
        path, start = item[0], item[1]
        dur = _get_duration(path)
        if dur > 0:
            annotated.append((path, start, dur, start + dur))

    if not annotated:
        return []

    # Collect all event times (segment starts and ends)
    events = set()
    for _, start, _, end in annotated:
        events.add(start)
        events.add(end)
    events.add(total_duration)
    event_times = sorted(events)

    # Merge adjacent windows with the same active set
    grid_dir = os.path.join(tmp_dir, 'grid_segments')
    os.makedirs(grid_dir, exist_ok=True)

    result = []
    prev_active = None
    window_start = None

    for i in range(len(event_times) - 1):
        t_start = event_times[i]
        t_end = event_times[i + 1]
        if t_end - t_start < 0.1:
            continue

        # Find active segments in this window
        active = []
        for path, seg_start, dur, seg_end in annotated:
            if seg_start < t_end and seg_end > t_start:
                offset = max(0, t_start - seg_start)
                active.append((path, offset))

        active_key = tuple(a[0] for a in active)

        # Merge with previous window if same active set
        if active_key == prev_active and window_start is not None:
            continue  # will be handled when active set changes

        # Emit previous merged window
        if prev_active is not None and window_start is not None:
            merged_end = t_start
            merged_dur = merged_end - window_start
            if merged_dur > 0.1:
                _emit_grid_window(
                    prev_active_segs, merged_dur, window_start,
                    grid_dir, len(result), target_w, target_h, result,
                )

        window_start = t_start
        prev_active = active_key
        prev_active_segs = active

    # Emit final window
    if prev_active is not None and window_start is not None:
        merged_end = event_times[-1]
        merged_dur = merged_end - window_start
        if merged_dur > 0.1:
            _emit_grid_window(
                prev_active_segs, merged_dur, window_start,
                grid_dir, len(result), target_w, target_h, result,
            )

    logging.info(f'Grid: built {len(result)} segments from '
                 f'{len(annotated)} webcam streams')
    return result


def _emit_grid_window(active, duration, start_time, grid_dir, idx,
                      target_w, target_h, result):
    """Helper to emit a single grid window segment."""
    if not active:
        return
    seg_path = os.path.join(grid_dir, f'grid_{idx}.mp4')
    if len(active) == 1:
        # Single webcam — just extract the slice
        path, offset = active[0]
        _run_ffmpeg(
            [
                'ffmpeg', '-y', '-v', 'error',
                '-ss', str(offset), '-i', path,
                '-t', str(duration),
                '-vf', f'scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,'
                       f'pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black,setsar=1',
                *_get_video_encoder_fast(),
                '-pix_fmt', 'yuv420p',
                '-c:a', 'aac', '-b:a', '128k', '-ar', '44100', '-ac', '2',
                '-r', '25',
                seg_path,
            ],
            description=f'grid window {idx} (1 webcam, {duration:.0f}s)',
        )
    else:
        _composite_grid(active, duration, seg_path, target_w, target_h)
    result.append((seg_path, start_time))


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
    """Remove overlapping video segments, keeping one per time window.

    Recordings often have parallel streams (multiple webcams) at the
    same timestamp. Laying them out sequentially inflates the duration.

    Strategy:
      1. Pick the "main" conference — the one with the most total recorded
         duration (the presenter typically has a few long segments, while
         participants toggle cameras creating many short ones).
      2. Keep only segments from the main conference.
      3. For time gaps where the main conference has no video, fill in
         with the longest available segment from any other conference.
    """
    if not video_files:
        return video_files

    # Annotate with duration and conf_id
    from collections import defaultdict
    annotated = []  # (path, start, duration, end, conf_id)
    for item in video_files:
        path, start = item[0], item[1]
        conf_id = item[2] if len(item) > 2 else None
        dur = _get_duration(path)
        annotated.append((path, start, dur, start + dur, conf_id))

    # Score each conference by total duration (not segment count)
    conf_total_dur = defaultdict(float)
    for _, _, dur, _, conf_id in annotated:
        if conf_id:
            conf_total_dur[conf_id] += dur

    if conf_total_dur:
        main_conf = max(conf_total_dur, key=conf_total_dur.get)
        logging.info(
            f'Dedup: main conference {main_conf} '
            f'({conf_total_dur[main_conf]:.0f}s total from '
            f'{sum(1 for x in annotated if x[4] == main_conf)} segments)'
        )
    else:
        main_conf = None

    # Separate main conference segments from others
    main_segs = sorted(
        [s for s in annotated if s[4] == main_conf],
        key=lambda x: x[1],
    )
    other_segs = sorted(
        [s for s in annotated if s[4] != main_conf],
        key=lambda x: x[1],
    )

    # Build timeline from main conference segments (merge overlapping)
    kept = []
    for seg in main_segs:
        if not kept or seg[1] >= kept[-1][3] - 0.5:
            kept.append(seg)
        else:
            # Overlapping main segments — keep the longer one
            if seg[2] > kept[-1][2]:
                kept[-1] = seg

    # Fill gaps with best available from other conferences
    filled = []
    for i, seg in enumerate(kept):
        gap_start = kept[i - 1][3] if i > 0 else 0
        gap_end = seg[1]
        if gap_end - gap_start > 1.0:
            # Find the longest other-conference segment covering this gap
            best = None
            for other in other_segs:
                # Segment must overlap the gap
                if other[3] <= gap_start or other[1] >= gap_end:
                    continue
                if best is None or other[2] > best[2]:
                    best = other
            if best:
                filled.append(best)
        filled.append(seg)

    original = len(video_files)
    deduped = len(filled)
    if original != deduped:
        logging.info(f'Dedup: {original} -> {deduped} segments '
                     f'(removed {original - deduped} overlapping)')

    return [(path, start) for path, start, dur, end, conf_id in filled]


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
    # Webcam is input 0 (25fps, drives output frame rate).
    # Slide track is input 1 (1fps, overlaid on left).
    _detect_gpu()
    if _CUDA_OVERLAY_AVAILABLE:
        filter_graph = (
            f'[0:v]hwupload_cuda,scale_cuda={CAM_W}:{CAM_H}[webcam_gpu];'
            f'[1:v]hwupload_cuda,scale_cuda={CANVAS_W}:{CANVAS_H}[slide_gpu];'
            f'[slide_gpu][webcam_gpu]overlay_cuda={SLIDE_W}:0[out]'
        )
    else:
        filter_graph = (
            f'[0:v]scale={CAM_W}:{CAM_H}:force_original_aspect_ratio=decrease,'
            f'pad={CANVAS_W}:{CANVAS_H}:{SLIDE_W}:0:black[base];'
            f'[1:v]scale={SLIDE_W}:{SLIDE_H}:force_original_aspect_ratio=decrease,'
            f'pad={SLIDE_W}:{SLIDE_H}:(ow-iw)/2:(oh-ih)/2:white[slide];'
            f'[base][slide]overlay=0:0[out]'
        )

    cmd = [
        'ffmpeg', '-y', '-v', 'warning',
        '-i', video_path,
        '-i', slide_track_path,
        '-filter_complex', filter_graph,
        '-map', '[out]', '-map', '0:a?',
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

    # Build the list of segments (normalized videos + gap fillers)
    tmp_dir = os.path.join(directory, '_tmp_ffmpeg')
    os.makedirs(tmp_dir, exist_ok=True)

    # Check for overlapping segments (multiple concurrent webcams)
    has_overlaps = False
    annotated_check = sorted(
        [(item[0], item[1], _get_duration(item[0])) for item in video_files],
        key=lambda x: x[1],
    )
    for i in range(1, len(annotated_check)):
        prev_end = annotated_check[i-1][1] + annotated_check[i-1][2]
        if annotated_check[i][1] < prev_end - 0.5:
            has_overlaps = True
            break

    if has_overlaps and not slide_events:
        # Grid layout: composite all concurrent webcams into a grid
        logging.info('Multiple concurrent webcams detected, building grid layout')
        video_files = _build_grid_segments(
            video_files, tmp_dir, target_w, target_h, total_duration,
        )
    else:
        # Deduplicate overlapping segments (keep presenter for slide videos)
        video_files = _deduplicate_overlapping(video_files)
    logging.info(f'After dedup/grid: {len(video_files)} segments')

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
