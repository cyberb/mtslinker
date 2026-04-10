import math
from typing import List, Union

from mtslinker.ffmpeg import FFmpegRunner
from mtslinker.timeline import GridSource


class GridCompositor:
    """Composites multiple webcam streams into a grid layout."""

    def __init__(self, ffmpeg: FFmpegRunner):
        self.ffmpeg = ffmpeg

    def compute_layout(self, n: int):
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        return cols, rows

    def even(self, x: int) -> int:
        return x & ~1

    def composite(self, active_segments: List[Union[GridSource, tuple]],
                  duration: float, output_path: str,
                  target_w: int, target_h: int,
                  mix_all_audio: bool = False) -> str:
        """Composite multiple webcam streams into a grid.

        Args:
            active_segments: List of GridSource objects or legacy (path, offset)
                tuples.
            mix_all_audio: When True, amix audio from all inputs. When False,
                only map audio from input 0 (legacy behavior).
        """
        # Normalize to GridSource objects
        sources = []
        for seg in active_segments:
            if isinstance(seg, GridSource):
                sources.append(seg)
            elif len(seg) == 3:
                sources.append(GridSource(path=seg[0], offset=seg[1], has_audio=seg[2]))
            else:
                sources.append(GridSource(path=seg[0], offset=seg[1]))

        n = len(sources)
        cols, rows = self.compute_layout(n)
        cell_w = self.even(target_w // cols)
        cell_h = self.even(target_h // rows)

        inputs = []
        filter_parts = []
        labels = []

        for i, src in enumerate(sources):
            inputs.extend(['-ss', str(src.offset), '-i', src.path])
            label = f'v{i}'
            filter_parts.append(
                f'[{i}:v]scale=w=min({cell_w}\\,iw):h=min({cell_h}\\,ih)'
                f':force_original_aspect_ratio=decrease,'
                f'pad={cell_w}:{cell_h}:(ow-iw)/2:(oh-ih)/2:black,setsar=1[{label}]'
            )
            labels.append(f'[{label}]')

        total_cells = cols * rows
        for i in range(n, total_cells):
            idx = len(sources) + i - n
            inputs.extend(['-f', 'lavfi', '-i',
                           f'color=black:s={cell_w}x{cell_h}:d={duration}:r=25'])
            labels.append(f'[{idx}:v]')

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

        # Audio mixing
        if mix_all_audio and n > 1:
            audio_labels = []
            for i, src in enumerate(sources):
                alabel = f'a{i}'
                if src.has_audio:
                    filter_graph += (
                        f';[{i}:a]apad=whole_dur={duration},'
                        f'atrim=0:{duration}[{alabel}]'
                    )
                else:
                    filter_graph += (
                        f';anullsrc=r=44100:cl=stereo[null{i}]'
                        f';[null{i}]atrim=0:{duration}[{alabel}]'
                    )
                audio_labels.append(f'[{alabel}]')
            filter_graph += (
                ';' + ''.join(audio_labels)
                + f'amix=inputs={n}:duration=longest:normalize=0[aout]'
            )
            audio_map = ['-map', '[aout]']
        else:
            audio_map = ['-map', '0:a?']

        cmd = [
            'ffmpeg', '-y', '-v', 'error',
            *inputs,
            '-t', str(duration),
            '-filter_complex', filter_graph,
            '-map', '[out]', *audio_map,
            *self.ffmpeg.get_video_encoder_fast(),
            '-pix_fmt', 'yuv420p',
            '-c:a', 'aac', '-b:a', '128k', '-ar', '44100', '-ac', '2',
            '-r', '25',
            output_path,
        ]
        self.ffmpeg.run(cmd, description=f'grid {n} webcams, {duration:.0f}s')
        return output_path
