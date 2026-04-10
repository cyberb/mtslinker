import logging
import os
from collections import defaultdict
from typing import Dict, List, Tuple

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
        self.ffmpeg.run(
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

    def normalize(self, input_path: str, output_path: str,
                  width: int, height: int, pix_fmt: str,
                  max_duration: float = 0) -> str:
        duration_args = ['-t', str(max_duration)] if max_duration > 0 else []
        self.ffmpeg.run(
            [
                'ffmpeg', '-y', '-v', 'error',
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

    def deduplicate(self, video_files: list) -> List[Tuple[str, float]]:
        if not video_files:
            return video_files

        annotated = []
        for item in video_files:
            path, start = item[0], item[1]
            conf_id = item[2] if len(item) > 2 else None
            is_admin = item[3] if len(item) > 3 else False
            dur = self.prober.get_duration(path)
            annotated.append((path, start, dur, start + dur, conf_id, is_admin))

        conf_total_dur = defaultdict(float)
        conf_is_admin = {}
        for _, _, dur, _, conf_id, is_admin in annotated:
            if conf_id:
                conf_total_dur[conf_id] += dur
                if is_admin:
                    conf_is_admin[conf_id] = True

        admin_confs = {c for c in conf_total_dur if conf_is_admin.get(c)}
        if admin_confs:
            main_conf = max(admin_confs, key=conf_total_dur.get)
            logging.info(
                f'Dedup: main conference {main_conf} (ADMIN, '
                f'{conf_total_dur[main_conf]:.0f}s total from '
                f'{sum(1 for x in annotated if x[4] == main_conf)} segments)'
            )
        elif conf_total_dur:
            main_conf = max(conf_total_dur, key=conf_total_dur.get)
            logging.info(
                f'Dedup: main conference {main_conf} '
                f'({conf_total_dur[main_conf]:.0f}s total from '
                f'{sum(1 for x in annotated if x[4] == main_conf)} segments)'
            )
        else:
            main_conf = None

        main_segs = sorted(
            [s for s in annotated if s[4] == main_conf], key=lambda x: x[1])
        other_segs = sorted(
            [s for s in annotated if s[4] != main_conf], key=lambda x: x[1])

        kept = []
        for seg in main_segs:
            if not kept or seg[1] >= kept[-1][3] - 0.5:
                kept.append(seg)
            else:
                if seg[2] > kept[-1][2]:
                    kept[-1] = seg

        filled = []
        for i, seg in enumerate(kept):
            gap_start = kept[i - 1][3] if i > 0 else 0
            gap_end = seg[1]
            if gap_end - gap_start > 1.0:
                best = None
                for other in other_segs:
                    if other[3] <= gap_start or other[1] >= gap_end:
                        continue
                    overlap_start = max(other[1], gap_start)
                    overlap_end = min(other[3], gap_end)
                    overlap_dur = overlap_end - overlap_start
                    if overlap_dur <= 0:
                        continue
                    if best is None or overlap_dur > best[2]:
                        best = (other[0], overlap_start, overlap_dur,
                                overlap_end, other[4], other[5])
                if best:
                    filled.append(best)
            filled.append(seg)

        result = [(path, start) for path, start, dur, end, conf_id, is_admin in filled]
        logging.info(f'Dedup: {len(annotated)} -> {len(result)} segments '
                     f'(removed {len(annotated) - len(result)} overlapping)')
        return result

    def extract_admin_conf_ids(self, json_data: Dict) -> set:
        admin_user_ids = set()
        user_to_conf = {}

        for event in json_data.get('eventLogs', []):
            if not isinstance(event, dict):
                continue
            module = event.get('module', '')
            data_list = event.get('data', [])
            if isinstance(data_list, dict):
                data_list = [data_list]
            if not isinstance(data_list, list):
                continue

            for d in data_list:
                if not isinstance(d, dict):
                    continue
                if 'userlist' in module:
                    role = d.get('role', '')
                    user = d.get('user', {})
                    if isinstance(user, dict) and role == 'ADMIN':
                        uid = user.get('id')
                        if uid:
                            admin_user_ids.add(uid)
                if module == 'conference.add':
                    user = d.get('user', {})
                    if isinstance(user, dict):
                        uid = user.get('id')
                        cid = d.get('id')
                        if uid and cid:
                            user_to_conf.setdefault(uid, set()).add(cid)

        admin_confs = set()
        for uid in admin_user_ids:
            admin_confs.update(user_to_conf.get(uid, set()))

        if admin_confs:
            logging.info(f'Found {len(admin_confs)} conference(s) from ADMIN users')
        return admin_confs
