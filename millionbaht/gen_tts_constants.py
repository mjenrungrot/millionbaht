from typing import Optional
from gtts import gTTS
from pathlib import Path

from millionbaht.constants import Constants


def gen_tts_constants(statements: list[str], outdir: Path, **kwargs) -> None:
    del kwargs
    outdir.mkdir(exist_ok=True)
    for i, statement in enumerate(statements):
        del i
        hash_statement = hash(statement)
        gTTS(statement, lang="th").save(outdir / f"{hash_statement}.wav")


if __name__ == "__main__":
    for statements, outdir in Constants.ALL_STATEMENTS:
        gen_tts_constants(statements, outdir)
