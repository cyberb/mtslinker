import json
import logging
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple


class MediaProber:
    """Probes media files for stream info, duration, resolution, audio levels."""

    def probe_streams(self, file_path: str) -> dict:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json',
             '-show_streams', '-show_format', file_path],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            return {}
        return json.loads(result.stdout)

    def is_valid(self, file_path: str) -> bool:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'csv=p=0', file_path],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            logging.warning(f'Corrupt/invalid file: {file_path}: {result.stderr.strip()[:200]}')
            return False
        return True

    def has_video(self, file_path: str) -> bool:
        info = self.probe_streams(file_path)
        return any(s.get('codec_type') == 'video' for s in info.get('streams', []))

    def has_audio(self, file_path: str) -> bool:
        info = self.probe_streams(file_path)
        return any(s.get('codec_type') == 'audio' for s in info.get('streams', []))

    def get_duration(self, file_path: str) -> float:
        info = self.probe_streams(file_path)
        fmt = info.get('format', {})
        if 'duration' in fmt:
            return float(fmt['duration'])
        for stream in info.get('streams', []):
            if 'duration' in stream:
                return float(stream['duration'])
        return 0.0

    def get_video_params(self, file_path: str) -> Tuple[int, int, str]:
        info = self.probe_streams(file_path)
        for stream in info.get('streams', []):
            if stream.get('codec_type') == 'video':
                w = int(stream.get('width', 1920))
                h = int(stream.get('height', 1080))
                pix_fmt = stream.get('pix_fmt', 'yuv420p')
                return w, h, pix_fmt
        return 1920, 1080, 'yuv420p'

    def is_silent(self, file_path: str, threshold: float = -88.0) -> bool:
        result = subprocess.run(
            ['ffmpeg', '-v', 'error', '-i', file_path,
             '-t', '10', '-af', 'volumedetect', '-f', 'null', '-'],
            capture_output=True, text=True,
        )
        for line in result.stderr.splitlines():
            if 'mean_volume' in line:
                try:
                    vol = float(line.split('mean_volume:')[1].strip().split()[0])
                    return vol < threshold
                except (ValueError, IndexError):
                    pass
        return True

    def analyze_audio_levels(self, file_path: str, window_sec: float = 2.0,
                             sample_rate: int = 44100) -> List[Tuple[float, float]]:
        reset_samples = int(sample_rate * window_sec)
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            tmp_path = tmp.name
        try:
            result = subprocess.run(
                ['ffmpeg', '-v', 'error', '-i', file_path,
                 '-af', f'astats=metadata=1:reset={reset_samples},'
                        f'ametadata=print:key=lavfi.astats.Overall.RMS_level'
                        f':file={tmp_path}',
                 '-f', 'null', '-'],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                return []
            levels = []
            current_time = None
            with open(tmp_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('frame:'):
                        for part in line.split():
                            if part.startswith('pts_time:'):
                                try:
                                    current_time = float(part.split(':')[1])
                                except (ValueError, IndexError):
                                    pass
                    elif 'RMS_level' in line and current_time is not None:
                        try:
                            val = float(line.split('=')[1])
                            levels.append((current_time, val))
                        except (ValueError, IndexError):
                            pass
            return levels
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def probe_file(self, file_path: str) -> dict:
        info = self.probe_streams(file_path)
        streams = info.get('streams', [])
        fmt = info.get('format', {})

        has_video = any(s.get('codec_type') == 'video' for s in streams)
        has_audio = any(s.get('codec_type') == 'audio' for s in streams)

        width, height, pix_fmt = 0, 0, 'yuv420p'
        for s in streams:
            if s.get('codec_type') == 'video':
                width = int(s.get('width', 0))
                height = int(s.get('height', 0))
                pix_fmt = s.get('pix_fmt', 'yuv420p')
                break

        duration = 0.0
        if 'duration' in fmt:
            duration = float(fmt['duration'])
        else:
            for s in streams:
                if 'duration' in s:
                    duration = float(s['duration'])
                    break

        valid = bool(streams) and duration > 0
        return {
            'path': file_path,
            'valid': valid,
            'has_video': has_video,
            'has_audio': has_audio,
            'width': width,
            'height': height,
            'pix_fmt': pix_fmt,
            'duration': duration,
        }

    def probe_all_files(self, downloaded_files: list) -> List[dict]:
        results = []
        items = [
            (item[0], item[1],
             item[2] if len(item) > 2 else None,
             item[3] if len(item) > 3 else False,
             item[4] if len(item) > 4 else 0)
            for item in downloaded_files
        ]
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {}
            for path, start_time, conf_id, is_admin, api_duration in items:
                fut = pool.submit(self.probe_file, path)
                futures[fut] = (start_time, conf_id, is_admin, api_duration)
            for fut in as_completed(futures):
                start_time, conf_id, is_admin, api_duration = futures[fut]
                info = fut.result()
                info['start_time'] = start_time
                info['conf_id'] = conf_id
                info['is_admin'] = is_admin
                if api_duration > 0:
                    info['api_duration'] = api_duration
                results.append(info)
        results.sort(key=lambda x: x['start_time'])
        return results
