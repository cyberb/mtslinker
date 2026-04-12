import math
from typing import List, Optional, Union

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
            + f'xstack=inputs={total_cells}:layout={layout}[stacked]'
            + f';[stacked]scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,'
            + f'pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black,setsar=1[out]'
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

    def presenter_composite(self, main: GridSource,
                            overlay: Optional[GridSource],
                            extra_audio: List[GridSource],
                            duration: float, output_path: str,
                            target_w: int, target_h: int,
                            mix_all_audio: bool = True) -> str:
        """Composite presenter layout: main full-screen, overlay PIP top-right.

        Output is always exactly target_w x target_h.
        """
        pip_w = self.even(target_w // 4)
        pip_h = self.even(target_h // 4)
        margin = 16

        inputs = ['-ss', str(main.offset), '-i', main.path]
        all_sources = [main]

        if overlay:
            inputs.extend(['-ss', str(overlay.offset), '-i', overlay.path])
            all_sources.append(overlay)

        for src in extra_audio:
            inputs.extend(['-ss', str(src.offset), '-i', src.path])
            all_sources.append(src)

        # Video filter
        vfilter = (
            f'[0:v]scale={target_w}:{target_h}:'
            f'force_original_aspect_ratio=decrease,'
            f'pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black,setsar=1[main]'
        )

        if overlay:
            vfilter += (
                f';[1:v]scale={pip_w}:{pip_h}:'
                f'force_original_aspect_ratio=decrease,'
                f'pad={pip_w}:{pip_h}:(ow-iw)/2:(oh-ih)/2:black,setsar=1[pip]'
                f';[main][pip]overlay=x={target_w - pip_w - margin}:y={margin}[out]'
            )
        else:
            vfilter += ';[main]copy[out]'

        # Audio filter — mix all sources that have audio
        audio_labels = []
        afilter = ''
        for i, src in enumerate(all_sources):
            if not src.has_audio:
                continue
            alabel = f'a{i}'
            afilter += (
                f';[{i}:a]apad=whole_dur={duration},'
                f'atrim=0:{duration}[{alabel}]'
            )
            audio_labels.append(f'[{alabel}]')

        if len(audio_labels) > 1:
            afilter += (
                ';' + ''.join(audio_labels)
                + f'amix=inputs={len(audio_labels)}:duration=longest:normalize=0[aout]'
            )
            audio_map = ['-map', '[aout]']
        elif len(audio_labels) == 1:
            # Single audio, keep the label and map it
            afilter += f';{audio_labels[0]}acopy[aout]'
            audio_map = ['-map', '[aout]']
        else:
            afilter = ''
            audio_map = []

        filter_graph = vfilter + afilter

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
        desc = 'presenter'
        if overlay:
            desc += ' + PIP'
        if extra_audio:
            desc += f' + {len(extra_audio)} audio'
        self.ffmpeg.run(cmd, description=f'{desc}, {duration:.0f}s')
        return output_path
