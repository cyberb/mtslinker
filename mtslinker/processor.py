"""Video processor — two-phase pipeline: analyze then execute.

Uses composition of specialized classes:
  FFmpegRunner  — ffmpeg execution, GPU detection
  MediaProber   — file probing, duration, streams
  SegmentBuilder — normalize, gaps, dedup
  GridCompositor — multi-webcam grid layout
  SlideCompositor — presentation slide overlay
  AudioMerger   — batched audio mixing
"""
import json
import logging
import os
import shutil
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Union

from mtslinker.ffmpeg import FFmpegRunner
from mtslinker.prober import MediaProber
from mtslinker.segments import SegmentBuilder
from mtslinker.grid import GridCompositor
from mtslinker.slides import SlideCompositor
from mtslinker.audio import AudioMerger


class VideoProcessor:
    """Orchestrates the two-phase video processing pipeline."""

    def __init__(self):
        self.ffmpeg = FFmpegRunner()
        self.prober = MediaProber()
        self.segments = SegmentBuilder(self.ffmpeg, self.prober)
        self.grid = GridCompositor(self.ffmpeg)
        self.slides = SlideCompositor(self.ffmpeg)
        self.audio = AudioMerger(self.ffmpeg, self.prober)

    def analyze(self, total_duration, downloaded_files, directory,
                output_path, max_duration=None, slide_events=None):
        """Phase 1: Probe all files, make all decisions, write manifest."""
        logging.info('Phase 1: Analyzing files...')

        all_files = self.prober.probe_all_files(downloaded_files)

        video_files = []
        audio_files = []
        skipped = 0
        errors = []

        for f in all_files:
            if not f['valid']:
                skipped += 1
                errors.append(f'Corrupt/invalid: {f["path"]}')
                continue
            if f['has_video']:
                video_files.append(f)
            else:
                audio_files.append(f)

        if skipped:
            logging.warning(f'Skipped {skipped} corrupt/invalid files')
        logging.info(f'Segments: {len(video_files)} video, {len(audio_files)} audio-only')

        if not video_files:
            raise ValueError('No valid video segments found.')

        # Determine target resolution
        target_w, target_h, target_pix_fmt = 0, 0, 'yuv420p'
        for f in video_files:
            if f['width'] * f['height'] > target_w * target_h:
                target_w, target_h, target_pix_fmt = f['width'], f['height'], f['pix_fmt']
        if target_w * target_h < 640 * 360:
            target_w, target_h = 640, 360
        MAX_H = 720
        if target_h > MAX_H:
            target_w = int(target_w * MAX_H / target_h)
            target_w -= target_w % 2
            target_h = MAX_H
        logging.info(f'Target resolution: {target_w}x{target_h}, pix_fmt={target_pix_fmt}')

        # Detect overlaps
        has_overlaps = False
        for i in range(1, len(video_files)):
            prev_dur = video_files[i-1].get('api_duration', 0) or video_files[i-1]['duration']
            prev_end = video_files[i-1]['start_time'] + prev_dur
            if video_files[i]['start_time'] < prev_end - 0.5:
                has_overlaps = True
                break

        # Decide strategy and build segment plan
        overlap_strategy = 'none'
        kept_video = []
        extra_audio = []
        segments = []

        if has_overlaps and slide_events:
            overlap_strategy = 'dedup'
            vf_tuples = [(f['path'], f['start_time'], f.get('conf_id'),
                           f.get('is_admin', False)) for f in video_files]
            deduped = self.segments.deduplicate(vf_tuples)
            kept_paths = {v[0] for v in deduped}

            dedup_map = {f['path']: f for f in video_files}
            for path, start_time in deduped:
                if path in dedup_map:
                    kept_video.append(dedup_map[path])

            for f in video_files:
                if f['path'] not in kept_paths and f['has_audio']:
                    extra_audio.append(f)

            logging.info(f'Dedup: kept {len(kept_video)}, '
                         f'{len(extra_audio)} extra audio from webcams')

        elif has_overlaps:
            overlap_strategy = 'grid'
            annotated = []
            for f in video_files:
                dur = f.get('api_duration', 0) or f['duration']
                annotated.append((f['path'], f['start_time'], dur,
                                  f['start_time'] + dur, f['has_audio']))

            event_times_set = set()
            for _, start, _, end, _ in annotated:
                event_times_set.add(start)
                event_times_set.add(end)
            event_times_set.add(total_duration)
            event_times = sorted(event_times_set)

            grid_windows = []
            prev_active_key = None
            window_start = None
            window_active = None
            for ei in range(len(event_times) - 1):
                t_start = event_times[ei]
                t_end = event_times[ei + 1]
                if t_end - t_start < 0.1:
                    continue
                active = []
                for path, seg_start, dur, seg_end, has_audio in annotated:
                    if seg_start < t_end and seg_end > t_start:
                        offset = max(0, t_start - seg_start)
                        remaining = dur - offset
                        if remaining > 0.5:
                            active.append({'path': path, 'offset': offset,
                                           'remaining': remaining,
                                           'has_audio': has_audio})
                active_key = tuple(a['path'] for a in active)
                if active_key == prev_active_key and window_start is not None:
                    continue
                if prev_active_key is not None and window_start is not None:
                    grid_windows.append({
                        'start_time': window_start,
                        'duration': t_start - window_start,
                        'sources': window_active,
                    })
                window_start = t_start
                prev_active_key = active_key
                window_active = active
            if window_start is not None and window_active:
                grid_windows.append({
                    'start_time': window_start,
                    'duration': event_times[-1] - window_start,
                    'sources': window_active,
                })

            for gw in grid_windows:
                if not gw['sources']:
                    segments.append({
                        'type': 'gap',
                        'duration': gw['duration'],
                        'start_time': gw['start_time'],
                    })
                else:
                    segments.append({
                        'type': 'grid',
                        'start_time': gw['start_time'],
                        'planned_duration': gw['duration'],
                        'sources': gw['sources'],
                    })

            # Grid composite includes audio from input 0 (sorted by has_audio).
            # Extract audio from OTHER webcams (not input 0) to avoid echo
            # but still capture all voices.
            grid_input0_paths = set()
            for seg in segments:
                if seg['type'] == 'grid' and seg.get('sources'):
                    # Input 0 = first source sorted by has_audio (same as execute)
                    sorted_sources = sorted(seg['sources'],
                                            key=lambda s: not s.get('has_audio', False))
                    if sorted_sources:
                        grid_input0_paths.add(sorted_sources[0]['path'])

            for f in video_files:
                if f['has_audio'] and f['path'] not in grid_input0_paths:
                    extra_audio.append(f)
            if extra_audio:
                logging.info(f'Grid: {len(extra_audio)} non-primary webcam audio to extract')

        else:
            vf_tuples = [(f['path'], f['start_time'], f.get('conf_id'),
                           f.get('is_admin', False)) for f in video_files]
            deduped = self.segments.deduplicate(vf_tuples)
            dedup_map = {f['path']: f for f in video_files}
            for path, start_time in deduped:
                if path in dedup_map:
                    kept_video.append(dedup_map[path])

        # Build segment plan for non-grid strategies
        if overlap_strategy != 'grid':
            segments = []
        current_time = 0.0
        for i, f in enumerate(kept_video):
            start_time = f['start_time']
            if start_time < current_time - 0.5:
                errors.append(f'Segment {i} overlaps at {start_time:.1f}s (current={current_time:.1f}s)')
                continue

            gap = start_time - current_time
            if gap > 0.1:
                segments.append({
                    'type': 'gap', 'duration': gap, 'start_time': current_time,
                })

            max_dur = 0
            for j in range(i + 1, len(kept_video)):
                ns = kept_video[j]['start_time']
                if ns > start_time + 0.5:
                    max_dur = ns - start_time
                    break

            file_dur = f.get('api_duration', 0) or f['duration']
            planned_dur = max_dur if max_dur > 0 else file_dur
            segments.append({
                'type': 'video',
                'source_path': f['path'],
                'start_time': start_time,
                'source_duration': f['duration'],
                'max_duration': max_dur,
                'planned_duration': planned_dur,
                'has_audio': f['has_audio'],
                'width': f['width'],
                'height': f['height'],
            })
            current_time = start_time + planned_dur

        # Fill grid timeline gaps
        if overlap_strategy == 'grid' and segments:
            filled = []
            current_time = 0.0
            for s in segments:
                start = s.get('start_time', 0)
                if start - current_time > 0.1:
                    filled.append({
                        'type': 'gap',
                        'duration': start - current_time,
                        'start_time': current_time,
                    })
                filled.append(s)
                current_time = start + s.get('planned_duration', s.get('duration', 0))
            segments = filled

        # Trailing gap
        if overlap_strategy == 'grid':
            current_time = 0
            for s in segments:
                end = s.get('start_time', 0) + s.get('planned_duration', s.get('duration', 0))
                if end > current_time:
                    current_time = end
        if current_time < total_duration - 0.1:
            segments.append({
                'type': 'gap',
                'duration': total_duration - current_time,
                'start_time': current_time,
            })

        # Audio tracks
        all_audio = [{'path': f['path'], 'start_time': f['start_time'],
                      'origin': 'original'} for f in audio_files]
        for f in extra_audio:
            all_audio.append({
                'path': f['path'],
                'start_time': f['start_time'],
                'origin': 'webcam_extract',
            })

        # Slide compositing plan
        slide_plan = None
        if slide_events:
            CHUNK_SECS = 1800
            n_chunks = max(1, int(total_duration + CHUNK_SECS - 1) // CHUNK_SECS)
            slide_plan = {
                'events': slide_events,
                'chunk_seconds': CHUNK_SECS,
                'num_chunks': n_chunks,
            }

        # GPU detection
        self.ffmpeg.detect_gpu()
        gpu = {
            'nvenc': bool(self.ffmpeg.nvenc_available),
            'cuda_overlay': bool(self.ffmpeg.cuda_overlay_available),
        }

        # Validation warnings
        warnings = []
        for seg in segments:
            if seg['type'] != 'video':
                continue
            src_dur = seg.get('source_duration', 0)
            planned = seg.get('planned_duration', 0)
            if src_dur and planned and src_dur < planned - 1.0:
                warnings.append(
                    f'Segment at {seg["start_time"]:.0f}s: source={src_dur:.0f}s '
                    f'but planned={planned:.0f}s (will pad {planned-src_dur:.0f}s black)'
                )

        manifest = {
            'version': 1,
            'total_duration': total_duration,
            'directory': directory,
            'output_path': output_path,
            'max_duration': max_duration,
            'target': {'width': target_w, 'height': target_h, 'pix_fmt': target_pix_fmt},
            'overlap_strategy': overlap_strategy,
            'segments': segments,
            'audio_tracks': all_audio,
            'slide_compositing': slide_plan,
            'gpu': gpu,
            'errors': errors,
            'warnings': warnings,
            'stats': {
                'total_files': len(all_files),
                'video_files': len(video_files),
                'audio_files': len(audio_files),
                'kept_video': len(kept_video),
                'extra_audio': len(extra_audio),
                'skipped': skipped,
                'segments': len(segments),
                'gaps': sum(1 for s in segments if s['type'] == 'gap'),
            },
        }

        manifest_path = os.path.join(directory, 'manifest.json')
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        logging.info(f'Manifest written to {manifest_path}')
        logging.info(f'Plan: {manifest["stats"]["segments"]} segments '
                     f'({manifest["stats"]["gaps"]} gaps), '
                     f'{len(all_audio)} audio tracks, '
                     f'strategy={overlap_strategy}')

        if errors:
            for e in errors:
                logging.warning(f'Analyze: {e}')
        if warnings:
            for w in warnings:
                logging.warning(f'Analyze: {w}')

        return manifest

    def execute(self, manifest):
        """Phase 2: Execute the plan from the manifest."""
        logging.info('Phase 2: Executing plan...')

        directory = manifest['directory']
        output_path = manifest['output_path']
        total_duration = manifest['total_duration']
        target = manifest['target']
        target_w, target_h = target['width'], target['height']
        target_pix_fmt = target['pix_fmt']

        tmp_dir = os.path.join(directory, '_tmp_ffmpeg')
        os.makedirs(tmp_dir, exist_ok=True)

        # Extract audio from dropped webcam files
        audio_files = []
        extra_dir = os.path.join(tmp_dir, 'extra_audio')
        os.makedirs(extra_dir, exist_ok=True)
        extract_count = 0
        for track in manifest['audio_tracks']:
            if track['origin'] == 'webcam_extract':
                audio_path = os.path.join(extra_dir, f'extra_{extract_count}.m4a')
                try:
                    self.ffmpeg.run(
                        ['ffmpeg', '-y', '-v', 'error',
                         '-i', track['path'], '-vn',
                         '-c:a', 'aac', '-b:a', '128k',
                         '-ar', '44100', '-ac', '2',
                         audio_path],
                        description=f'extract audio from webcam {extract_count}',
                    )
                    audio_files.append((audio_path, track['start_time']))
                    extract_count += 1
                except subprocess.CalledProcessError:
                    logging.warning(f'Failed to extract audio from {track["path"]}')
            else:
                audio_files.append((track['path'], track['start_time']))
        if extract_count:
            logging.info(f'Extracted audio from {extract_count} webcam files')

        # Process segments
        segments = manifest['segments']
        grid_dir = os.path.join(tmp_dir, 'grid_segments')
        concat_segments = []

        for i, seg in enumerate(segments):
            if seg['type'] == 'gap':
                gap_path = os.path.join(tmp_dir, f'gap_{i}.mp4')
                self.segments.generate_black(gap_path, seg['duration'],
                                             target_w, target_h, target_pix_fmt)
                concat_segments.append(gap_path)
                logging.info(f'Generated {seg["duration"]:.1f}s black gap')

            elif seg['type'] == 'grid':
                os.makedirs(grid_dir, exist_ok=True)
                seg_path = os.path.join(grid_dir, f'grid_{i}.mp4')
                duration = seg['planned_duration']
                sources = sorted(seg['sources'],
                                 key=lambda s: not s.get('has_audio', False))
                active = [(s['path'], s['offset']) for s in sources]
                try:
                    if len(active) == 1:
                        path, offset = active[0]
                        self.ffmpeg.run(
                            ['ffmpeg', '-y', '-v', 'error',
                             '-ss', str(offset), '-i', path,
                             '-t', str(duration),
                             '-vf', f'scale={target_w}:{target_h}:'
                                    f'force_original_aspect_ratio=decrease,'
                                    f'pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black,setsar=1',
                             *self.ffmpeg.get_video_encoder_fast(),
                             '-pix_fmt', 'yuv420p',
                             '-c:a', 'aac', '-b:a', '128k', '-ar', '44100', '-ac', '2',
                             '-r', '25', seg_path],
                            description=f'grid {i} (1 webcam, {duration:.0f}s)',
                        )
                    else:
                        self.grid.composite(active, duration, seg_path, target_w, target_h)
                    with_audio = os.path.join(tmp_dir, f'grida_{i}.mp4')
                    final_seg = self.segments.ensure_audio(seg_path, with_audio)
                    concat_segments.append(final_seg)
                except subprocess.CalledProcessError:
                    logging.warning(f'Grid segment {i} failed, using black gap')
                    gap_path = os.path.join(tmp_dir, f'gridfail_{i}.mp4')
                    self.segments.generate_black(gap_path, duration,
                                                 target_w, target_h, target_pix_fmt)
                    concat_segments.append(gap_path)

            elif seg['type'] == 'video':
                norm_path = os.path.join(tmp_dir, f'norm_{i}.mp4')
                self.segments.normalize(seg['source_path'], norm_path,
                                        target_w, target_h, target_pix_fmt,
                                        max_duration=seg.get('max_duration', 0))
                with_audio_path = os.path.join(tmp_dir, f'norma_{i}.mp4')
                final_seg = self.segments.ensure_audio(norm_path, with_audio_path)

                actual_dur = self.prober.get_duration(final_seg)
                planned_dur = seg.get('planned_duration', 0)
                if planned_dur > 0 and actual_dur < planned_dur - 1.0:
                    shortfall = planned_dur - actual_dur
                    logging.warning(
                        f'Segment {i} is {actual_dur:.1f}s but planned '
                        f'{planned_dur:.1f}s — padding {shortfall:.1f}s'
                    )
                    pad_path = os.path.join(tmp_dir, f'pad_{i}.mp4')
                    self.segments.generate_black(pad_path, shortfall,
                                                 target_w, target_h, target_pix_fmt)
                    concat_segments.append(final_seg)
                    concat_segments.append(pad_path)
                else:
                    concat_segments.append(final_seg)

        # Concatenate
        concat_list_path = os.path.join(tmp_dir, 'concat.txt')
        with open(concat_list_path, 'w') as f:
            for seg in concat_segments:
                f.write(f"file '{os.path.abspath(seg)}'\n")

        video_only_path = os.path.join(tmp_dir, 'video_concat.mp4')
        logging.info(f'Concatenating {len(concat_segments)} segments...')

        cap = manifest.get('max_duration') or total_duration
        concat_cmd = [
            'ffmpeg', '-y', '-v', 'warning',
            '-f', 'concat', '-safe', '0',
            '-i', concat_list_path, '-c', 'copy',
        ]
        if cap:
            concat_cmd.extend(['-t', str(cap)])
        concat_cmd.append(video_only_path)
        self.ffmpeg.run(concat_cmd, description=f'concat {len(concat_segments)} segments')

        # Composite slides
        slide_plan = manifest.get('slide_compositing')
        if slide_plan and slide_plan.get('events'):
            composited_path = os.path.join(tmp_dir, 'video_composited.mp4')
            self.slides.composite(
                video_only_path, slide_plan['events'], composited_path,
                tmp_dir, total_duration,
            )
            video_only_path = composited_path

        # Merge audio
        if audio_files:
            logging.info(f'Merging {len(audio_files)} audio tracks...')
            result_path = self.audio.merge(
                video_only_path, audio_files, tmp_dir, output_path,
                total_duration,
            )
        else:
            shutil.move(video_only_path, output_path)
            result_path = output_path

        shutil.rmtree(tmp_dir, ignore_errors=True)
        logging.info(f'Final video saved to {result_path}')
        return result_path

    def process(self, total_duration, downloaded_files, directory,
                output_path, max_duration=None, slide_events=None):
        """Full pipeline: analyze then execute."""
        manifest = self.analyze(
            total_duration, downloaded_files, directory, output_path,
            max_duration, slide_events,
        )
        return self.execute(manifest)


# --- Backward-compatible module-level functions ---

def process_and_download_clips(directory, json_data):
    """Parse API JSON for chunks, slides, admin IDs."""
    segments_builder = SegmentBuilder(FFmpegRunner.__new__(FFmpegRunner), MediaProber())
    total_duration = float(json_data.get('duration', 0))
    if not total_duration:
        raise ValueError('Duration not found in JSON data.')

    admin_conf_ids = segments_builder.extract_admin_conf_ids(json_data)

    mediasession_adds = {}
    chunks = []
    slide_events = []
    for event in json_data.get('eventLogs', []):
        if not isinstance(event, dict):
            continue
        module = event.get('module', '')
        data = event.get('data', {})
        if not isinstance(data, dict):
            continue

        if module == 'mediasession.add' and 'url' in data:
            ms_id = data.get('id')
            stream = data.get('stream', {})
            conf_id = None
            if isinstance(stream, dict):
                conf = stream.get('conference', {})
                if isinstance(conf, dict):
                    conf_id = conf.get('id')
            mediasession_adds[ms_id] = {
                'url': data['url'],
                'start': event.get('relativeTime', 0),
                'conf_id': conf_id,
                'api_duration': 0,
            }

        if module == 'mediasession.update':
            ms_id = data.get('id')
            if ms_id in mediasession_adds:
                mediasession_adds[ms_id]['api_duration'] = data.get('duration', 0)

        elif 'url' in data and module not in ('mediasession.add',):
            stream = data.get('stream', {})
            conf_id = None
            if isinstance(stream, dict):
                conf = stream.get('conference', {})
                if isinstance(conf, dict):
                    conf_id = conf.get('id')
            url = data['url']
            if not any(ms['url'] == url for ms in mediasession_adds.values()):
                chunks.append((url, event.get('relativeTime', 0), conf_id, 0))

        if module == 'presentation.update':
            fr = data.get('fileReference', {})
            if not isinstance(fr, dict):
                continue
            slide = fr.get('slide', {})
            if not isinstance(slide, dict) or not slide.get('url'):
                continue
            slide_events.append({
                'time': event.get('relativeTime', 0),
                'slide_number': slide.get('number', 0),
                'slide_url': slide['url'],
            })

    tagged_chunks = []
    for ms in mediasession_adds.values():
        tagged_chunks.append((
            ms['url'], ms['start'], ms['conf_id'],
            ms['conf_id'] in admin_conf_ids,
            ms['api_duration'],
        ))
    for url, start, conf_id, api_dur in chunks:
        tagged_chunks.append((url, start, conf_id, conf_id in admin_conf_ids, api_dur))

    deduped_slides = []
    for se in slide_events:
        if not deduped_slides or se['slide_url'] != deduped_slides[-1]['slide_url']:
            deduped_slides.append(se)

    if deduped_slides:
        logging.info(f'Found {len(deduped_slides)} presentation slide changes')

    return total_duration, tagged_chunks, deduped_slides


def compile_final_video(total_duration, downloaded_files, directory,
                        output_path, max_duration, slide_events=None):
    """Backward-compatible entry point."""
    processor = VideoProcessor()
    processor.process(total_duration, downloaded_files, directory,
                      output_path, max_duration, slide_events)


# Keep old names for backward compatibility with webinar.py
def analyze_video(*args, **kwargs):
    return VideoProcessor().analyze(*args, **kwargs)

def execute_video(manifest):
    return VideoProcessor().execute(manifest)
