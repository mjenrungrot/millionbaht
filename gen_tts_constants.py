from gtts import gTTS
from constants import Constants

for statements, outdir in Constants.ALL_STATEMENTS:
    outdir.mkdir(exist_ok=True)
    for i, statement in enumerate(statements):
        gTTS(statement, lang="th").save(outdir / f"{i}.mp3")