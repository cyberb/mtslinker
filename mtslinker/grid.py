import math
import os

from mtslinker.ffmpeg import FFmpegRunner


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

    def composite(self, active_segments: list, duration: float,
                  output_path: str, target_w: int, target_h: int) -> str:
        n = len(active_segments)
        cols, rows = self.compute_layout(n)
        cell_w = self.even(target_w // cols)
        cell_h = self.even(target_h // rows)

        inputs = []
        filter_parts = []
        labels = []

        for i, (path, offset) in enumerate(active_segments):
            inputs.extend(['-ss', str(offset), '-i', path])
            label = f'v{i}'
            filter_parts.append(
                f'[{i}:v]scale=w=min({cell_w}\\,iw):h=min({cell_h}\\,ih)'
                f':force_original_aspect_ratio=decrease,'
                f'pad={cell_w}:{cell_h}:(ow-iw)/2:(oh-ih)/2:black,setsar=1[{label}]'
            )
            labels.append(f'[{label}]')

        total_cells = cols * rows
        for i in range(n, total_cells):
            idx = len(active_segments) + i - n
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

        cmd = [
            'ffmpeg', '-y', '-v', 'error',
            *inputs,
            '-t', str(duration),
            '-filter_complex', filter_graph,
            '-map', '[out]', '-map', '0:a?',
            *self.ffmpeg.get_video_encoder_fast(),
            '-pix_fmt', 'yuv420p',
            '-c:a', 'aac', '-b:a', '128k', '-ar', '44100', '-ac', '2',
            '-r', '25',
            output_path,
        ]
        self.ffmpeg.run(cmd, description=f'grid {n} webcams, {duration:.0f}s')
        return output_path
