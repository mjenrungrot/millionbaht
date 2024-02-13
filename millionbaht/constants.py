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
        {"text": "Welcome, everyone, to an unforgettable night of beats and rhythms!", "voice": "en_us_001"},
        {"text": "Let's kick things off with some vibes that'll get your heart racing!", "voice": "en_us_002"},
        {
            "text": "Get ready to be swept away by the next track, a tune that will take you on a journey!",
            "voice": "en_us_006",
        },
        {"text": "Here's a fresh beat hot off the press, guaranteed to make you move!", "voice": "en_us_007"},
        {"text": "I want to see everyone moving! Let the rhythm take control!", "voice": "en_us_009"},
        {"text": "Are you ready to turn it up? Let's make this a night to remember!", "voice": "en_us_010"},
        {"text": "Hold tight, we're about to dive into the deep end of bass!", "voice": "en_us_001"},
        {"text": "Feel the build-up? Get ready for the drop that will shake the ground!", "voice": "en_us_002"},
        {"text": "This track is on fire! Let's see those dance moves!", "voice": "en_us_006"},
        {"text": "Keep the energy up! This beat is not slowing down!", "voice": "en_us_007"},
        {"text": "Let's take a moment to vibe with a smoother, chill track.", "voice": "en_us_009"},
        {"text": "Time to catch your breath with a tune that'll touch your soul.", "voice": "en_us_010"},
        {"text": "I want to hear you! Let your voices be part of this track!", "voice": "en_us_001"},
        {
            "text": "Put your hands up and wave them to the beat! You are the energy of this night!",
            "voice": "en_us_002",
        },
        {
            "text": "Thank you for being an amazing crowd tonight. Let's close with a track that'll leave us on a high note!",
            "voice": "en_us_006",
        },
        {
            "text": "As we come to the end, remember, the music never really stops. Keep the vibe alive!",
            "voice": "en_us_007",
        },
        {"text": "Dedicated to all the lovers in the house tonight, this tune is for you.", "voice": "en_us_009"},
        {
            "text": "Celebrating life and music - this next beat is for everyone making memories tonight!",
            "voice": "en_us_010",
        },
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
    # gen sound effects
    good_files = set()
    for statement in Constants.SOUNDEFFECTS_STATEMENTS:
        text = statement["text"]
        voice = statement["voice"]
        hash_statement = hash(text + voice)
        filename = f"tts_{hash_statement}.mp3"
        file = Constants.SOUNDEFFECTS_OUTDIR / filename
        good_files.add(filename)
        if not file.exists():
            audio_bytes = tiktok_tts(text, voice)
            assert audio_bytes is not None
            save_audio_file(audio_bytes, str(file))

    # cleanup
    for file in Constants.SOUNDEFFECTS_OUTDIR.glob("*"):
        if file.name not in good_files:
            file.unlink()


gen_soundeffects_constants()
