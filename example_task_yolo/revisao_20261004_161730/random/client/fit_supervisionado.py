import os
import pickle
import signal
import subprocess
import sys
import tempfile
from pathlib import Path


def executar_fit(parameters):
    timeout = float(os.environ.get("FIT_TIMEOUT_SECONDS", "120"))
    attempts = int(os.environ.get("FIT_ATTEMPTS", "2"))
    if timeout <= 0 or attempts < 1:
        raise ValueError("Timeout e número de tentativas devem ser positivos")

    client_id = os.environ.get("CLIENT_ID", "unknown")
    error = "sem resultado"

    with tempfile.TemporaryDirectory(prefix="yolo_fit_") as directory:
        input_path = Path(directory) / "input.pkl"
        output_path = Path(directory) / "output.pkl"

        with input_path.open("wb") as stream:
            pickle.dump(parameters, stream, protocol=pickle.HIGHEST_PROTOCOL)

        for attempt in range(1, attempts + 1):
            output_path.unlink(missing_ok=True)
            print(
                f"[FIT START] client={client_id} tentativa={attempt}/{attempts}",
                flush=True,
            )

            process = subprocess.Popen(
                [
                    sys.executable, "-u", str(Path(__file__).resolve()),
                    str(input_path), str(output_path),
                ],
                start_new_session=True,
                env={**os.environ, "PYTHONFAULTHANDLER": "1"},
            )

            try:
                code = process.wait(timeout=timeout)
                if code != 0:
                    raise RuntimeError(f"worker exitcode={code}")

                with output_path.open("rb") as stream:
                    message = pickle.load(stream)

                if message[0] != "ok":
                    raise RuntimeError(str(message[1]))

                return message

            except (
                subprocess.TimeoutExpired, OSError, EOFError,
                pickle.UnpicklingError, RuntimeError,
            ) as exc:
                error = repr(exc)
                print(
                    f"[FIT FAILED] client={client_id} "
                    f"tentativa={attempt} erro={error}",
                    flush=True,
                )

            finally:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=10)

    raise RuntimeError(
        f"Fit falhou depois de {attempts} tentativas: {error}"
    )


class FileConnection:
    def __init__(self, path):
        self.path = Path(path)

    def send(self, message):
        temporary = self.path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            pickle.dump(message, stream, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, self.path)

    def close(self):
        pass


if __name__ == "__main__":
    from client import _fit_worker

    with open(sys.argv[1], "rb") as stream:
        parameters = pickle.load(stream)

    _fit_worker(parameters, FileConnection(sys.argv[2]))
