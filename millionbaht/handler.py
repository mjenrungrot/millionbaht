import logging
import os
import random
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Literal

from io import BytesIO
import discord
import torch
import yt_dlp
from discord import FFmpegOpusAudio, VoiceClient
from discord.ext import commands
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict
import torchaudio
from gtts import gTTS

from millionbaht import gen_tts_constants
from millionbaht.constants import Constants
from millionbaht.typedef import MessageableChannel
from millionbaht.gen_tts_constants import gen_tts_constants


logging.basicConfig(format="[%(asctime)s] [%(levelname)-8s] %(name)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


class ProcRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str
    title: str = ""
    path: Optional[Path] = None
    semitone: float = 0
    speed: float = 1
    fast: bool = False
    force_tts: bool = False
    skip_tts: bool = False
    ## Internal
    is_auto: bool = False


class ProcResponse(BaseModel):
    success: bool
    path: Path
    url: str = ""
    message: str = ""

    @classmethod
    def fail(cls, err: Exception) -> "ProcResponse":
        return ProcResponse(
            success=False,
            path=Path(),
            message=f"{type(err)}: {str(err)}",
        )

    @classmethod
    def ok(cls, path: Path, url: str) -> "ProcResponse":
        return ProcResponse(success=True, path=path, url=url)


_DUMMY_PROC_REQUEST = ProcRequest(query="")
_DUMMY_PROC_RESPONSE = ProcResponse(success=True, path=Path())

load_dotenv()
_YTDL_FORMAT = os.getenv("YTDL_FORMAT", "worstaudio")


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
    message = message.replace("official mv", "")

    audio_fd = BytesIO()
    gTTS(message, lang="th").write_to_fp(audio_fd)
    audio_fd.seek(0)

    audio_tts, tts_rate = torchaudio.load(audio_fd, format="mp3")  # type: ignore
    audio_fd.close()
    if force_tts or audio_tts.shape[-1] / tts_rate < max_duration:
        audio_tts = torchaudio.functional.resample(audio_tts, tts_rate, orig_freq)
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
    buffer_left_sec: float = 2.0,
    buffer_right_sec: float = 2.0,
    pitch_threshold: float = 700.0,
    pitch_duration: int = 100,
) -> tuple[torch.Tensor, int]:
    # find the first frame such that the pitch is below the threshold for at least pitch_duration frames

    logger.info(f"_transform_strip Input: {audio.shape}")

    # LEFT
    considered_length = max(audio.shape[-1], orig_freq * threshold_sec_left)
    pitch = torchaudio.functional.detect_pitch_frequency(audio[..., :considered_length], orig_freq)
    t_axis = torch.linspace(0, considered_length, pitch.shape[-1])
    is_singing = pitch < pitch_threshold
    for i, t in enumerate(t_axis):
        left = i
        right = min(pitch.shape[-1], i + pitch_duration)
        if is_singing[..., left:right].all():
            audio = audio[..., round((t_axis[i] + buffer_left_sec / orig_freq).item()) :]
            break
    logger.info(f"_transform_strip Output: {audio.shape}")

    # RIGHT
    considered_length = max(audio.shape[-1], orig_freq * threshold_sec_right)
    pitch = torchaudio.functional.detect_pitch_frequency(audio[..., -considered_length:], orig_freq)
    t_axis = torch.linspace(audio.shape[-1] - considered_length, audio.shape[-1], pitch.shape[-1])
    is_singing = pitch < pitch_threshold
    for i in range(pitch.shape[-1] - pitch_duration, -1, -1):
        t = t_axis[i]
        left = i
        right = i + pitch_duration
        if is_singing[..., left:right].all():
            audio = audio[..., : round((t_axis[i] + buffer_right_sec / orig_freq).item())]
            break
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
    return audio, orig_freq


def _transform_volume(
    audio: torch.Tensor,
    orig_freq: int,
    target_db: float = 0.0,
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
    fade_out = torch.linspace(0, 1, int(fade_out_sec * orig_freq))
    ones_out = torch.ones(audio.shape[-1] - len(fade_out))
    if fade_shape == "exponential":
        fade_in = torch.pow(2, (fade_in - 1)) * fade_in
        fade_out = torch.pow(2, (fade_out - 1)) * fade_out

    fade_in = torch.cat([fade_in, ones_in])
    fade_out = torch.cat([ones_out, fade_out])
    output = audio * fade_in * fade_out
    logger.info(f"_transform_fade Output: {audio.shape}")
    return output, orig_freq


def transform_song(path: Path, req: ProcRequest) -> Path:
    new_path = path.parent / f"{path.stem}_mod{path.suffix}"

    logger.info(f"start transform_song: {req} -> {new_path}")
    x, rate = torchaudio.load(path, normalize=True)  # type: ignore
    if not req.fast:
        x, rate = _transform_strip(x, rate)
    x, rate = _transform_speed(x, rate, req.speed)
    if not req.fast:
        x, rate = _transform_semitone(x, rate, req.semitone)
    x, rate = _transform_volume(x, rate)
    x, rate = _transform_fade(x, rate)
    if not req.skip_tts:
        x, rate = _transform_title(x, rate, req.title)
    torchaudio.save(new_path, x, rate)  # type: ignore
    logger.info(f"end transform_song: {req} -> {new_path}")
    return new_path


# runs on a different process
def process_song(req: ProcRequest) -> ProcResponse:
    is_auto = req.is_auto
    if not is_auto:
        ydl = get_ydl()
        try:
            info = ydl.extract_info(req.query, download=False)
        except yt_dlp.utils.DownloadError as err:
            return ProcResponse.fail(err)

        assert isinstance(info, dict)
        if "entries" in info:
            info = info["entries"][0]
        assert isinstance(info, dict)

        try:
            ydl.download([req.query])
        except yt_dlp.utils.DownloadError as err:
            return ProcResponse.fail(err)

        # only transform if not auto
        if not req.is_auto:
            try:
                path = Constants.SONGDIR / f'{info["id"]}.{info["ext"]}'
                path = transform_song(path, req)
            except Exception as e:
                return ProcResponse.fail(e)

        try:
            return ProcResponse.ok(path, f'https://youtu.be/{info["id"]}')
        except Exception as e:
            return ProcResponse.fail(e)
    else:
        assert req.path is not None
        response = ProcResponse.ok(req.path, f"https://youtu.be/{req.query}")
        return response


pool = ProcessPoolExecutor(1)


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
                try:
                    opening_audio = random.choice(
                        list(
                            filter(
                                lambda x: not x.name.endswith(".gitignore"), list(Constants.OPENNING_OUTDIR.iterdir())
                            )
                        )
                    )
                except IndexError:
                    gen_tts_constants(Constants.OPENING_STATEMENTS, Constants.OPENNING_OUTDIR)
                finally:
                    opening_audio = random.choice(
                        list(
                            filter(
                                lambda x: not x.name.endswith(".gitignore"), list(Constants.OPENNING_OUTDIR.iterdir())
                            )
                        )
                    )
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
        voice_client = self.get_voice_client()
        if voice_client:
            voice_client.stop()

            def after(err):
                del err
                self.loop.create_task(voice_client.disconnect())

            try:
                ending_audio = random.choice(
                    list(filter(lambda x: not x.name.endswith(".gitignore"), list(Constants.ENDING_OUTDIR.iterdir())))
                )
            except IndexError:
                gen_tts_constants(Constants.ENDING_STATEMENTS, Constants.ENDING_OUTDIR)
            finally:
                ending_audio = random.choice(
                    list(filter(lambda x: not x.name.endswith(".gitignore"), list(Constants.ENDING_OUTDIR.iterdir())))
                )
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
                if self.is_auto:
                    assert self.current_context is not None
                    try:
                        random_song_path = random.choice(
                            list(
                                filter(
                                    lambda x: (not x.name.endswith(".gitignore")) and (x.name.endswith("_mod.mp4")),
                                    list(Constants.SONGDIR.iterdir()),
                                )
                            )
                        )
                        req = ProcRequest(
                            query=str(random_song_path.stem.removeprefix("_mod")),
                            title="",
                            path=random_song_path,
                            semitone=0,
                            speed=1,
                            skip_tts=True,
                            is_auto=True,
                        )
                        channel = self.current_context.channel
                        self.put(req, channel)
                        return
                    except Exception as e:
                        print(list(Constants.SONGDIR.iterdir()))
                        print("No song found")

                if not self.queue_empty_played:
                    try:
                        empty_audio = random.choice(
                            list(
                                filter(
                                    lambda x: not x.name.endswith(".gitignore"),
                                    list(Constants.EMPTY_OUTDIR.iterdir()),
                                )
                            )
                        )
                    except IndexError:
                        gen_tts_constants(Constants.EMPTY_STATEMENTS, Constants.EMPTY_OUTDIR)
                    finally:
                        empty_audio = random.choice(
                            list(
                                filter(
                                    lambda x: not x.name.endswith(".gitignore"),
                                    list(Constants.EMPTY_OUTDIR.iterdir()),
                                )
                            )
                        )
                        voice_client.play(
                            FFmpegOpusAudio(str(empty_audio)),
                            after=lambda x: self.validate(),
                        )
                    self.queue_empty_played = True


def get_ydl() -> yt_dlp.YoutubeDL:
    return yt_dlp.YoutubeDL(
        {
            "format": _YTDL_FORMAT,
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
