import os
from tests.conftest import make_test_video


def test_generate_black(segments, tmp_dir):
    path = os.path.join(tmp_dir, 'black.mp4')
    segments.generate_black(path, 2.0, 320, 240)
    assert os.path.exists(path)
    assert os.path.getsize(path) > 0


def test_ensure_audio_keeps_existing(segments, tmp_dir):
    vid = os.path.join(tmp_dir, 'with.mp4')
    out = os.path.join(tmp_dir, 'out.mp4')
    make_test_video(vid, with_audio=True)
    result = segments.ensure_audio(vid, out)
    assert result == vid


def test_ensure_audio_adds_silent(segments, tmp_dir):
    vid = os.path.join(tmp_dir, 'without.mp4')
    out = os.path.join(tmp_dir, 'out.mp4')
    make_test_video(vid, with_audio=False)
    result = segments.ensure_audio(vid, out)
    assert result == out
    assert os.path.exists(out)


def test_normalize(segments, tmp_dir):
    vid = os.path.join(tmp_dir, 'input.mp4')
    out = os.path.join(tmp_dir, 'norm.mp4')
    make_test_video(vid)
    segments.normalize(vid, out, 640, 360, 'yuv420p')
    assert os.path.exists(out)


def test_normalize_with_max_duration(segments, tmp_dir):
    vid = os.path.join(tmp_dir, 'input.mp4')
    out = os.path.join(tmp_dir, 'norm.mp4')
    make_test_video(vid, duration=5)
    segments.normalize(vid, out, 640, 360, 'yuv420p', max_duration=2)
    from mtslinker.prober import MediaProber
    dur = MediaProber().get_duration(out)
    assert dur < 3.0


def test_deduplicate_empty(segments):
    assert segments.deduplicate([]) == []
