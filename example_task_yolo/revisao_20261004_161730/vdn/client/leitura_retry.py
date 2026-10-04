
import os
import time
import cv2
import ultralytics.data.base as base

_original_imread = base.imread

def imread_retry(filename, flags=cv2.IMREAD_COLOR):
    for tentativa in range(1, 4):
        try:
            imagem = _original_imread(filename, flags=flags)
            motivo = "leitor retornou None"
        except (OSError, cv2.error) as erro:
            imagem = None
            motivo = repr(erro)

        if imagem is not None:
            if tentativa > 1:
                print(
                    f"[IMAGE RECOVERED] "
                    f"client={os.environ.get('CLIENT_ID', 'unknown')} "
                    f"tentativa={tentativa} arquivo={filename}",
                    flush=True,
                )
            return imagem

        print(
            f"[IMAGE READ FAILED] "
            f"client={os.environ.get('CLIENT_ID', 'unknown')} "
            f"tentativa={tentativa}/3 arquivo={filename} "
            f"motivo={motivo}",
            flush=True,
        )
        if tentativa < 3:
            time.sleep(0.2 * tentativa)

    return None

base.imread = imread_retry
