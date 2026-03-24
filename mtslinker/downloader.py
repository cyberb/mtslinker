import os
from typing import Dict, List, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
import tqdm
import logging

TIMEOUT_SETTINGS = httpx.Timeout(None, connect=None)
CHUNK_SIZE = 1024 * 1024  # 1 MB
MAX_PARALLEL_DOWNLOADS = 4


def construct_json_data_url(event_session_id: str, recording_id: str) -> str:
    if not event_session_id:
        raise ValueError('Missing webinar event session ID.')

    if not recording_id:
        return f'https://my.mts-link.ru/api/eventsessions/{event_session_id}/record?withoutCuts=false'
    return f'https://my.mts-link.ru/api/event-sessions/{event_session_id}/record-files/{recording_id}/flow?withoutCuts=false'


def fetch_json_data(url: str, session_id: Union[str, None]) -> Dict:
    cookies = {}
    if session_id:
        cookies['sessionId'] = session_id

    with httpx.Client(timeout=TIMEOUT_SETTINGS) as client:
        response = client.get(
            url,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0',
            },
            cookies=cookies
        )

    try:
        error_data = response.json()
        if error_data.get("error", {}).get("code") == 403:
            logging.error(
                'Access denied: session_id token is required. '
                'Provide it using the "--session-id" parameter.'
            )
            return
    except Exception:
        logging.warning('Server response does not contain JSON.')

    response.raise_for_status()
    return response.json()


def download_video_chunk(video_url: str, save_directory: str) -> str:
    filename = os.path.basename(video_url)
    file_path = os.path.join(save_directory, filename)

    if not os.path.exists(file_path):
        with open(file_path, 'wb') as file:
            with httpx.Client(timeout=TIMEOUT_SETTINGS) as client:
                with client.stream('GET', video_url) as response:
                    response.raise_for_status()
                    total_size = int(response.headers.get('content-length', 0))
                    with tqdm.tqdm(total=total_size, unit='B', unit_scale=True,
                                   desc=f'Downloading {filename}') as progress:
                        for chunk in response.iter_bytes(chunk_size=CHUNK_SIZE):
                            file.write(chunk)
                            progress.update(len(chunk))
    return file_path


def download_chunks_parallel(
    chunks: List[Tuple[str, float]],
    save_directory: str,
    max_workers: int = MAX_PARALLEL_DOWNLOADS,
) -> List[Tuple[str, float]]:
    """Download multiple chunks in parallel.

    Args:
        chunks: List of (url, start_time) tuples.
        save_directory: Directory to save files to.
        max_workers: Maximum number of parallel downloads.

    Returns:
        List of (file_path, start_time) tuples in original order.
    """
    results = [None] * len(chunks)

    def _download(index, url, start_time):
        path = download_video_chunk(url, save_directory)
        return index, path, start_time

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_download, i, url, start_time)
            for i, (url, start_time) in enumerate(chunks)
        ]
        for future in as_completed(futures):
            idx, path, start_time = future.result()
            results[idx] = (path, start_time)

    return results
