from pathlib import Path
import os
from dotenv import load_dotenv
from gtts import gTTS
from numpy import save
from millionbaht.tiktoktts import tiktok_tts, save_audio_file


PROJECT_PATH = Path(__file__).parent.parent

load_dotenv()


class Constants:
    OPENING_STATEMENTS = [
        "สวัสดีคุณทุกคนที่รักการฟังวิทยุ! ยินดีต้อนรับสู่สถานีวิทยุ millionbaht ที่จะทำให้คุณเพลิดเพลินกับเสียงเพลงที่โดดเด่นและเนื้อหาที่น่าสนใจ",
        "สวัสดีคุณสาวกับหนุ่มทุกคน! มาร่วมเป็นส่วนหนึ่งของสถานีวิทยุ millionbaht ที่จะสร้างประสบการณ์การฟังที่ทันสมัยและน่าตื่นเต้น",
        "สวัสดีคุณทุกคนที่รักการฟังวิทยุ! เชื่อว่าคุณจะหลงใหลในเสียงเพลงที่เป็นที่ฮิตและความบันเทิงที่คุณต้องการได้ที่สถานีวิทยุ millionbaht",
        "สวัสดีคุณทุกคนที่รักการฟังวิทยุ! มาร่วมเป็นส่วนหนึ่งของสถานีวิทยุ millionbaht ที่จะให้คุณได้พบกับเพลงสุดฮิตและความสนุกที่ไม่มีวันหมด",
        "สวัสดีคุณทุกคนที่รักการฟังวิทยุ! มาพบกับสถานีวิทยุ millionbaht ที่จะทำให้คุณได้สัมผัสกับเสียงเพลงที่เป็นเอกลักษณ์และเนื้อหาที่คุณต้องการ",
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

    SOUNDEFFECTS_PROBABILITY = 1.0
    SOUNDEFFECTS_STATEMENTS = [
        {
            "text": "wow",
            "voice": "en_female_f08_warmy_breeze",
        }
    ]
    SOUNDEFFECTS_OUTDIR = PROJECT_PATH / "dl" / "soundeffects"

    ALL_STATEMENTS = [
        (OPENING_STATEMENTS, OPENNING_OUTDIR),
        (ENDING_STATEMENTS, ENDING_OUTDIR),
        (EMPTY_STATEMENTS, EMPTY_OUTDIR),
        (SOUNDEFFECTS_STATEMENTS, SOUNDEFFECTS_OUTDIR),
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
    for statement in statements:
        hash_statement = hash(statement)
        file = outdir / f"tts_{hash_statement}.wav"
        if not file.exists():
            gTTS(statement, lang="th").save(file)


Constants.SONGDIR.mkdir(parents=True, exist_ok=True)
Constants.SOUNDEFFECTS_OUTDIR.mkdir(parents=True, exist_ok=True)

for statements, outdir in [
    (Constants.OPENING_STATEMENTS, Constants.OPENNING_OUTDIR),
    (Constants.ENDING_STATEMENTS, Constants.ENDING_OUTDIR),
    (Constants.EMPTY_STATEMENTS, Constants.EMPTY_OUTDIR),
]:
    gen_tts_constants(statements, outdir)


def gen_soundeffects_constants() -> None:
    # cleanup
    for file in Constants.SOUNDEFFECTS_OUTDIR.glob("tts_*.wav"):
        file.unlink()

    # gen sound effects
    for statement in Constants.SOUNDEFFECTS_STATEMENTS:
        text = statement["text"]
        voice = statement["voice"]
        hash_statement = hash(text + voice)
        file = Constants.SOUNDEFFECTS_OUTDIR / f"tts_{hash_statement}.wav"
        if not file.exists():
            audio_bytes = tiktok_tts(text, voice)
            assert audio_bytes is not None
            save_audio_file(audio_bytes, str(file))


gen_soundeffects_constants()
