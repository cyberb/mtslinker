import os
from mtslinker.grid import GridCompositor
from tests.conftest import make_test_video


def test_compute_layout_single(ffmpeg):
    grid = GridCompositor(ffmpeg)
    assert grid.compute_layout(1) == (1, 1)


def test_compute_layout_two(ffmpeg):
    grid = GridCompositor(ffmpeg)
    assert grid.compute_layout(2) == (2, 1)


def test_compute_layout_four(ffmpeg):
    grid = GridCompositor(ffmpeg)
    assert grid.compute_layout(4) == (2, 2)


def test_compute_layout_five(ffmpeg):
    grid = GridCompositor(ffmpeg)
    assert grid.compute_layout(5) == (3, 2)


def test_even(ffmpeg):
    grid = GridCompositor(ffmpeg)
    assert grid.even(640) == 640
    assert grid.even(641) == 640
    assert grid.even(1) == 0


def test_composite_two_webcams(ffmpeg, tmp_dir):
    grid = GridCompositor(ffmpeg)
    v1 = os.path.join(tmp_dir, 'v1.mp4')
    v2 = os.path.join(tmp_dir, 'v2.mp4')
    out = os.path.join(tmp_dir, 'grid.mp4')
    make_test_video(v1, duration=2)
    make_test_video(v2, duration=2)
    grid.composite([(v1, 0), (v2, 0)], 2.0, out, 640, 360)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0
