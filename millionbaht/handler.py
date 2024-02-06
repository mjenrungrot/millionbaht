import logging
import os
import random
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import re
from typing import Optional, Literal, cast
import traceback
from io import BytesIO
import unicodedata
import discord
import torch
import yt_dlp
from discord import FFmpegOpusAudio, VoiceClient
from discord.ext import commands
from pydantic import BaseModel, ConfigDict
import torchaudio
from gtts import gTTS
import math
import icu

from millionbaht.constants import Constants
from millionbaht.typedef import MessageableChannel, YoutubeEntry


logging.basicConfig(format="[%(asctime)s] [%(levelname)-8s] %(name)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


class ProcRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str
    title: str = ""
    semitone: float = 0
    speed: float = 1
    fast: bool = False
    force_tts: bool = False


class ProcResponse(BaseModel):
    success: bool
    path: Path
    url: str = ""
    message: str = ""

    @classmethod
    def fail(cls, err: Exception) -> "ProcResponse":
        print(traceback.format_exc())
        return ProcResponse(
            success=False,
            path=Path(),
            message=f"{type(err).__name__}: {str(err)}",
        )

    @classmethod
    def ok(cls, path: Path, url: str) -> "ProcResponse":
        return ProcResponse(success=True, path=path, url=url)


_DUMMY_PROC_REQUEST = ProcRequest(query="")
_DUMMY_PROC_RESPONSE = ProcResponse(success=True, path=Path())


_LANGS = ["en", "th", "ko", "ja"]
_CHARSETS = [
    set(icu.LocaleData(lang).getExemplarSet(icu.USET_ADD_CASE_MAPPINGS, icu.ULocaleDataExemplarSetType.ES_STANDARD))  # type: ignore
    for lang in _LANGS
]


def _split_lang(s: str) -> list[tuple[str, str]]:
    s = unicodedata.normalize("NFC", s)
    res: list[tuple[str, str]] = []
    last_lang = "unk"
    stk = ""
    for c in s:
        cur_lang = "unk"
        for lang, charset in zip(_LANGS, _CHARSETS):
            if c in charset:
                cur_lang = lang
                break
        if cur_lang == "unk" or cur_lang == last_lang:
            stk += c
        else:
            res.append((last_lang if last_lang != "unk" else "en", stk.strip()))
            last_lang = cur_lang
            stk = c
    res.append((last_lang if last_lang != "unk" else "en", stk.strip()))
    return res[1:]


def _transform_title(
    audio: torch.Tensor,
    orig_freq: int,
    title: str,
    max_duration: float = 10.0,
    force_tts: bool = False,
    message_prefix: str = "เพลง ",
) -> tuple[torch.Tensor, int]:
    # if query is hard to understand, use title instead
    logger.info(f"_transform_title Input: {audio.shape}")

    if audio.ndim == 1:
        audio = audio.unsqueeze(0)

    # preprocess message
    message = f"{message_prefix}{title}"
    message = message.lower()
    message = re.sub("([\(\[]).*?([\)\]])", "\g<1>\g<2>", message)
    message = message.replace("official music video", "")
    message = message.replace("official mv", "")
    message = message.replace("official m/v", "")
    message = message.replace("official audio", "")
    message = message.replace("music video", "")
    message = message.replace("[", "").replace("]", "")
    message = message.replace("(", "").replace(")", "")
    message = message.replace("【", "").replace("】", "")
    message = message.replace("（", "").replace("）", "")
    message = message.replace("「", "").replace("」", "")
    message = message.replace("|", "")

    ttss = []
    for lang, word in _split_lang(message):
        audio_fd = BytesIO()
        gTTS(word, lang=lang).write_to_fp(audio_fd)
        audio_fd.seek(0)
        audio_tts, tts_rate = torchaudio.load(audio_fd, format="mp3")  # type: ignore
        audio_fd.close()
        audio_tts = torchaudio.functional.resample(audio_tts, tts_rate, orig_freq)
        ttss.append(audio_tts)
    audio_tts = torch.cat(ttss, dim=-1)

    if force_tts or audio_tts.shape[-1] / orig_freq < max_duration:
        if audio_tts.shape[0] != audio.shape[0]:
            audio_tts = audio_tts.repeat(audio.shape[0], 1)
        out = torch.cat([audio_tts, audio], dim=-1)
        return out, orig_freq

    return audio, orig_freq


def _transform_strip(
    audio: torch.Tensor,
    orig_freq: int,
    threshold_sec_left: float = 30.0,
    threshold_sec_right: float = 30.0,
    buffer_left_sec: float = 3.0,
    buffer_right_sec: float = 3.0,
    db_threshold: float = -24,
    db_sec: float = 0.2,
) -> tuple[torch.Tensor, int]:
    # find the first frame such that the loudness is above the db_threshold for at least db_sec
    N = audio.shape[-1]
    n_frame_left = min(round(threshold_sec_left * orig_freq), N // 2)
    n_frame_right = min(round(threshold_sec_right * orig_freq), N // 2)
    db_frames = min(round(db_sec * orig_freq), N // 2)
    buff_left = round(buffer_left_sec * orig_freq)
    buff_right = round(buffer_right_sec * orig_freq)
    hop = db_frames // 2

    logger.info(f"_transform_strip Input: {audio.shape}")

    x = audio.reshape(-1, N).mean(dim=-2)
    x = x.unfold(-1, db_frames, hop)
    db = ((x.max(dim=-1).values - x.min(dim=-1).values) / math.sqrt(8)).log10() * 20
    left_blk = next(x[0] for x in enumerate(db) if x[1] > db_threshold)
    right_blk = db.shape[0] - next(x[0] for x in enumerate(reversed(db)) if x[1] > db_threshold)
    left_i = max(min(left_blk * hop - buff_left, n_frame_left), 0)
    right_i = min(max(right_blk * hop + buff_right, N - n_frame_right), N)

    audio = audio[..., left_i:right_i]
    logger.info(f"_transform_strip Output: {audio.shape}")
    return audio, orig_freq


def _transform_speed(audio: torch.Tensor, orig_freq: int, speed: float) -> tuple[torch.Tensor, int]:
    if speed == 1:
        return audio, orig_freq
    if speed < 0:
        raise ValueError("speed must be positive")

    logger.info(f"_transform_speed: {speed} Input: {audio.shape}")
    out, _ = torchaudio.functional.speed(audio, orig_freq=orig_freq, factor=speed)
    logger.info(f"_transform_speed: {speed} Output: {out.shape}")
    return out, orig_freq


def _transform_semitone(audio: torch.Tensor, orig_freq: int, semitone: float) -> tuple[torch.Tensor, int]:
    if semitone == 0:
        return audio, orig_freq

    if round(semitone) != semitone:
        raise ValueError("semitone must be integer")

    logger.info(f"_transform_semitone: {semitone} Input: {audio.shape}")
    out = torchaudio.functional.pitch_shift(
        audio,
        sample_rate=orig_freq,
        n_steps=round(semitone),
        bins_per_octave=12,
    )
    logger.info(f"_transform_semitone: {semitone} Output: {out.shape}")
    return out, orig_freq


def _transform_volume(
    audio: torch.Tensor,
    orig_freq: int,
    target_db: float = -20,
) -> tuple[torch.Tensor, int]:
    logger.info(f"_transform_volume Input: {audio.shape}")
    loudness = torchaudio.functional.loudness(audio, orig_freq).item()
    gain = target_db - loudness
    out = torchaudio.functional.gain(audio, gain)
    logger.info(f"_transform_volume Output: {out.shape}")
    return out, orig_freq


def _transform_fade(
    audio: torch.Tensor,
    orig_freq: int,
    fade_in_sec: float = 4.0,
    fade_out_sec: float = 4.0,
    fade_shape: Literal["linear", "exponential"] = "exponential",
) -> tuple[torch.Tensor, int]:
    logger.info(f"_transform_fade Input: {audio.shape}")
    fade_in = torch.linspace(0, 1, int(fade_in_sec * orig_freq))
    ones_in = torch.ones(audio.shape[-1] - len(fade_in))
    fade_out = torch.linspace(1, 0, int(fade_out_sec * orig_freq))
    ones_out = torch.ones(audio.shape[-1] - len(fade_out))
    if fade_shape == "exponential":
        fade_in = torch.pow(2, (fade_in - 1)) * fade_in
        fade_out = torch.pow(2, (fade_out - 1)) * fade_out

    fade_in = torch.cat([fade_in, ones_in])
    fade_out = torch.cat([ones_out, fade_out])
    output = audio * fade_in * fade_out
    logger.info(f"_transform_fade Output: {audio.shape}")
    return output, orig_freq


def transform_song(path: Path, req: ProcRequest, outpath: Path) -> Path:
    logger.info(f"start transform_song: {req} -> {outpath}")
    x, rate = torchaudio.load(path, normalize=True)  # type: ignore
    x = cast(torch.Tensor, x)
    rate = cast(int, rate)
    x, rate = _transform_volume(x, rate)
    if not req.fast:
        x, rate = _transform_strip(x, rate)
    x, rate = _transform_speed(x, rate, req.speed)
    if not req.fast:
        x, rate = _transform_semitone(x, rate, req.semitone)
    x, rate = _transform_fade(x, rate)
    x, rate = _transform_title(x, rate, req.title, force_tts=req.force_tts)
    torchaudio.save(outpath, x, rate)  # type: ignore
    logger.info(f"end transform_song: {req} -> {outpath}")
    return outpath


def _get_mod_name(req: ProcRequest) -> str:
    mod_name = req.query
    for k in req.model_fields.keys():
        v = getattr(req, k)
        if k not in ("query", "title") and v != getattr(_DUMMY_PROC_REQUEST, k):
            mod_name += f"--{k}:{v}"
    mod_name += f".mod.m4a"
    return mod_name


# runs on a different process
def process_song(req: ProcRequest) -> ProcResponse:
    info_path = Constants.SONGDIR / f"{req.query}.json"
    mod_path = Constants.SONGDIR / _get_mod_name(req)
    if not info_path.exists():
        ydl = get_ydl()
        try:
            info = ydl.extract_info(req.query, download=False)
        except yt_dlp.utils.DownloadError as err:
            return ProcResponse.fail(err)

        assert isinstance(info, dict)
        if "entries" in info:
            info = info["entries"][0]
        try:
            info = YoutubeEntry.model_validate(info)
            if info.is_live:
                raise ValueError("cannot process a live video")
        except Exception as e:
            return ProcResponse.fail(e)

        try:
            ydl.download([req.query])
        except yt_dlp.utils.DownloadError as err:
            return ProcResponse.fail(err)

        try:
            path = Constants.SONGDIR / f"{req.query}.{info.ext}"
            info_path.write_text(info.model_dump_json())
        except Exception as e:
            return ProcResponse.fail(e)
    else:
        info = YoutubeEntry.model_validate_json(info_path.read_text())
        path = Constants.SONGDIR / f"{req.query}.{info.ext}"

    if mod_path.exists():
        return ProcResponse.ok(mod_path, f"https://youtu.be/{info.id}")
    else:
        assert path.exists()
        path = transform_song(path, req, mod_path)
        return ProcResponse.ok(path, f"https://youtu.be/{info.id}")


#################


class State(Enum):
    AwaitingProcessing = "Awaiting Processing"
    Processing = "Processing"
    AwaitingPlay = "Awaiting Play"
    Playing = "Now Playing"
    Done = "Done"


@dataclass
class SongRequest:
    proc_request: ProcRequest
    proc_response: Optional[ProcResponse] = None
    state: State = State.AwaitingProcessing
    next: Optional["SongRequest"] = None
    channel: Optional[MessageableChannel] = None


if os.getenv("DEBUG", False):
    pool = ThreadPoolExecutor(1)
else:
    pool = ProcessPoolExecutor(1)


class SongQueue:
    # [Done, AwaitingPlay, Processing, AwatingProcessing]

    def __init__(self, bot: commands.Bot, guild_id: int):
        self.bot = bot
        self.loop = bot.loop
        self.guild_id = guild_id
        self.head = SongRequest(_DUMMY_PROC_REQUEST, _DUMMY_PROC_RESPONSE, state=State.Done)
        self.last_processed = self.head  # last AwaitingPlay or Processing
        self.last_played = self.head  # last Done
        self.last = self.head
        self.queue_empty_played = True
        self.size = 0
        self.is_auto = False  # auto play next song
        self.current_context: Optional[commands.Context] = None

        if True:
            voice_client = self.get_voice_client()
            if voice_client is not None and not voice_client.is_playing():
                opening_audio = random.choice(list(Constants.OPENNING_OUTDIR.iterdir()))
                voice_client.play(
                    FFmpegOpusAudio(str(opening_audio)),
                    after=lambda x: self.validate(),
                )

    def set_is_auto_state(self, state: bool):
        self.is_auto = state

    def set_context(self, ctx: commands.Context):
        self.current_context = ctx

    async def process_song(self, song: SongRequest):
        # do non-intensive thing here
        if song.channel is not None:
            if len(song.proc_request.title) > 0:
                print(song.proc_request.title)
                await song.channel.send(f'Processing "{song.proc_request.title}" ...')
        song.proc_response = await self.loop.run_in_executor(pool, process_song, song.proc_request)
        song.state = State.AwaitingPlay
        self.validate()

    def put(self, request: ProcRequest, channel: Optional[MessageableChannel]):
        new_request = SongRequest(request, channel=channel)
        self.last.next = new_request
        self.last = new_request
        self.queue_empty_played = False
        self.size += 1
        self.validate()

    def skip(self, n_skip: int):
        if n_skip < 0:
            n_skip = self.size
        it = self.last_played
        for _ in range(n_skip):
            it.state = State.Done
            if it.next is not None:
                it = it.next
            else:
                break
        voice_client = self.get_voice_client()
        if voice_client:
            voice_client.stop()

    def get_queue(self) -> str:
        it = self.last_played
        res: list[str] = []
        i = 1
        while it is not None:
            if it.state != State.Done:
                res.append(f"{i}. {it.proc_request.title} ({it.state.value})")
            it = it.next
            i += 1
        if res:
            return "\n".join(res)
        return ":: empty ::"

    def leave(self):
        it = self.last_played
        while it != None:
            it.state = State.Done
            it = it.next
        voice_client = self.get_voice_client()
        if voice_client:
            voice_client.stop()

            def after(err):
                del err
                self.loop.create_task(voice_client.disconnect())

            ending_audio = random.choice(list(Constants.ENDING_OUTDIR.iterdir()))
            voice_client.play(
                FFmpegOpusAudio(str(ending_audio)),
                after=after,
            )

    def get_voice_client(self) -> Optional[VoiceClient]:
        for voice_client in self.bot.voice_clients:
            if isinstance(voice_client, VoiceClient):
                if voice_client.guild.id == self.guild_id:
                    return voice_client
        return None

    def validate(self):
        while (
            self.last_processed.state in [State.Done, State.Playing, State.AwaitingPlay]
            and self.last_processed.next is not None
        ):
            self.last_processed = self.last_processed.next
        if self.last_processed.state == State.AwaitingProcessing:
            self.last_processed.state = State.Processing
            self.loop.create_task(self.process_song(self.last_processed))

        while self.last_played.state == State.Done and self.last_played.next is not None:
            self.last_played = self.last_played.next

        voice_client = self.get_voice_client()
        if voice_client is not None and not voice_client.is_playing():
            if self.last_played.state == State.AwaitingPlay:
                to_play = self.last_played
                response = to_play.proc_response
                assert response is not None
                to_play.state = State.Playing

                if response.success:
                    assert to_play.channel is not None
                    if len(to_play.proc_request.title) > 0:
                        self.loop.create_task(to_play.channel.send(f'Now playing: "{to_play.proc_request.title}"'))

                    def after(exception):
                        del exception
                        to_play.state = State.Done
                        self.validate()

                    voice_client.play(discord.FFmpegOpusAudio(str(response.path)), after=after)
                else:
                    assert to_play.channel is not None
                    self.loop.create_task(to_play.channel.send(f"Error: ```{response.message}```"))
            elif self.last_played.state == State.Done:
                if not self.queue_empty_played:
                    if self.is_auto:
                        assert self.current_context is not None
                        try:
                            random_song_path = random.choice(
                                list(filter(lambda x: x.stem.endswith(".mod"), Constants.SONGDIR.iterdir()))
                            )
                            sid, *kws = random_song_path.stem.split(".")[0].split("--")
                            kwdict = {}
                            for kw in kws:
                                k, v = kw.split(":")
                                kwdict[k] = v
                            info = YoutubeEntry.model_validate_json((Constants.SONGDIR / f"{sid}.json").read_text())
                            req = ProcRequest(query=info.id, title=info.title, **kwdict)
                            channel = self.current_context.channel
                            logger.info(f"Before {self.get_queue()=}")
                            self.put(req, channel)
                            logger.info(f"After {self.get_queue()=}")
                        except Exception as e:
                            logger.info(f"{list(Constants.SONGDIR.iterdir())=}")
                            logger.info("No song found")
                    else:
                        empty_audio = random.choice(list(Constants.EMPTY_OUTDIR.iterdir()))
                        voice_client.play(
                            FFmpegOpusAudio(str(empty_audio)),
                            after=lambda x: self.validate(),
                        )
                        self.queue_empty_played = True


def get_ydl() -> yt_dlp.YoutubeDL:
    return yt_dlp.YoutubeDL(
        {
            "format": Constants.YTDL_FORMAT,
            "source_address": "0.0.0.0",
            "default_search": "ytsearch",
            "outtmpl": "%(id)s.%(ext)s",
            "noplaylist": True,
            "allow_playlist_files": False,
            # 'progress_hooks': [lambda info, ctx=ctx: video_progress_hook(ctx, info)],
            # 'match_filter': lambda info, incomplete, will_need_search=will_need_search, ctx=ctx: start_hook(ctx, info, incomplete, will_need_search),
            "paths": {"home": str(Constants.SONGDIR)},
        }
    )
