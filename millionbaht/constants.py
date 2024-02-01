from pathlib import Path
import os
from dotenv import load_dotenv
from gtts import gTTS


PROJECT_PATH = Path(__file__).parent.parent

load_dotenv()


class Constants:
    OPENING_STATEMENTS = [
        "สวัสดีครับ",
        "สวัสดีค่ะ มาเจอกันอีกทีที่สถานีวิทยุของเรา",
        "สวัสดีทุกคนครับ พร้อมที่จะทำให้วันนี้เต็มไปด้วยเพลงเพราะ",
        "สวัสดีค่ะทุกคน มีความสุขมากที่ได้เจอกันในที่นี้",
        "สวัสดีครับทุกคน มีความยินดีที่ได้เป็นส่วนหนึ่งของวงการวิทยุ",
        "สวัสดีค่ะทุกคน พร้อมที่จะเปิดเสียงดนตรีสำหรับคุณ เพลง",
        "สวัสดีครับทุกคน มีเพลงเพราะๆ รอคุณอยู่",
        "สวัสดีค่ะทุกคน มาพร้อมกับเพลงที่ทำให้คุณลืมทุกวัน",
    ]
    OPENNING_OUTDIR = PROJECT_PATH / "dl" / "openning"

    ENDING_STATEMENTS = [
        "ขอบคุณทุกคนที่ร่วมฟังเราวันนี้ หวังว่าคุณจะมีวันที่สดใส และเราจะพบกันใหม่ในครั้งถัดไปค่ะ",
        "ขอบคุณทุกคนที่ทำให้วันนี้เต็มไปด้วยความสุข ลาก่อนค่ะ หวังว่าเราจะได้เจอกันอีกในเร็วๆ นี้",
        "ถึงตรงนี้แล้วนะค่ะ ขอบคุณทุกคนที่ร่วมสนุกกับเรา ลาก่อนและหวังว่าคุณจะมีคืนที่สุข",
        "ขอบคุณทุกคนที่ทำให้วันนี้เป็นวันที่น่าจดจำ เราจะไปพบกับคุณในรอบต่อไป ลาก่อนค่ะ",
        "มาถึงจุดสุดท้ายของรายการวันนี้แล้วค่ะ ขอบคุณทุกคนที่ร่วมฟัง ลาก่อนและหวังว่าคุณจะมีคืนที่สนุกสุข",
    ]
    ENDING_OUTDIR = PROJECT_PATH / "dl" / "ending"

    EMPTY_STATEMENTS = [
        "เพลงหมดแล้ว ขอเพลงเพิ่มด่วน",
    ]
    EMPTY_OUTDIR = PROJECT_PATH / "dl" / "empty"

    ALL_STATEMENTS = [
        (OPENING_STATEMENTS, OPENNING_OUTDIR),
        (ENDING_STATEMENTS, ENDING_OUTDIR),
        (EMPTY_STATEMENTS, EMPTY_OUTDIR),
    ]

    SONGDIR = PROJECT_PATH / "dl" / "songs"
    YTDL_FORMAT = os.getenv("YTDL_FORMAT", "worstaudio")
    BOT_TOKEN = os.getenv("BOT_TOKEN")
    BOT_PREFIX = os.getenv("BOT_PREFIX", ".")
    PRINT_STACK_TRACE = os.getenv("PRINT_STACK_TRACE", "1").lower() in ("true", "1", "t")
    BOT_REPORT_COMMAND_NOT_FOUND = os.getenv("BOT_REPORT_COMMAND_NOT_FOUND", "1") in ("true", "1", "t")
    BOT_REPORT_DL_ERROR = os.getenv("BOT_REPORT_DL_ERROR", "0").lower() in (
        "true",
        "t",
        "1",
    )
    CF_API_TOKEN = os.environ["CLOUDFLARE_API_KEY"]
    CF_ACCOUNT_ID = os.environ["CLOUDFLARE_ACCOUNT_ID"]
    COLOR = int(os.getenv("BOT_COLOR", "ff0000"), 16)


def gen_tts_constants(statements: list[str], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for i, statement in enumerate(statements):
        del i
        hash_statement = hash(statement)
        gTTS(statement, lang="th").save(outdir / f"{hash_statement}.wav")


Constants.SONGDIR.mkdir(parents=True, exist_ok=True)


for statements, outdir in Constants.ALL_STATEMENTS:
    if not outdir.exists() or len(list(outdir.iterdir())) != len(statements):
        gen_tts_constants(statements, outdir)
