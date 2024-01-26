import os
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import discord
import yt_dlp
from discord import FFmpegOpusAudio, VoiceClient
from discord.ext import commands
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

from millionbaht.constants import Constants
from millionbaht.typedef import MessageableChannel


class ProcRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str
    title: str = ""
    semitone: float = 0
    speed: float = 1


class ProcResponse(BaseModel):
    success: bool
    path: Path
    url: str = ""
    message: str = ""

    @classmethod
    def fail(cls, err: Exception) -> "ProcResponse":
        return ProcResponse(
            success=False, path=Path(), message=f"{type(err)}: {str(err)}"
        )

    @classmethod
    def ok(cls, path: Path, url: str) -> "ProcResponse":
        return ProcResponse(success=True, path=path, url=url)


_DUMMY_PROC_REQUEST = ProcRequest(query="")
_DUMMY_PROC_RESPONSE = ProcResponse(success=True, path=Path())

load_dotenv()
_YTDL_FORMAT = os.getenv("YTDL_FORMAT", "worstaudio")


# runs on a different process
def process_song(req: ProcRequest) -> ProcResponse:
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

    try:
        path = Constants.SONGDIR / f'{info["id"]}.{info["ext"]}'
        return ProcResponse.ok(path, f'https://youtu.be/{info["id"]}')
    except Exception as e:
        return ProcResponse.fail(e)


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


pool = ProcessPoolExecutor(1)


class SongQueue:
    # [Done, AwaitingPlay, Processing, AwatingProcessing]

    def __init__(self, bot: commands.Bot, guild_id: int):
        self.bot = bot
        self.loop = bot.loop
        self.guild_id = guild_id
        self.head = SongRequest(
            _DUMMY_PROC_REQUEST, _DUMMY_PROC_RESPONSE, state=State.Done
        )
        self.last_processed = self.head  # last AwaitingPlay or Processing
        self.last_played = self.head  # last Done
        self.last = self.head
        self.queue_empty_played = True
        self.size = 0

        if True:
            voice_client = self.get_voice_client()
            if voice_client is not None and not voice_client.is_playing():
                try:
                    opening_audio = random.choice(list(filter(lambda x: not x.name.endswith(".gitignore"), list(Constants.OPENNING_OUTDIR.iterdir()))))
                    voice_client.play(
                        FFmpegOpusAudio(str(opening_audio)),
                        after=lambda x: self.validate(),
                    )
                except IndexError:
                    # TODO: fix when there is no opening audio
                    pass

    async def process_song(self, song: SongRequest):
        # do non-intensive thing here
        if song.channel is not None:
            await song.channel.send(f'Processing "{song.proc_request.title}" ...')
        song.proc_response = await self.loop.run_in_executor(
            pool, process_song, song.proc_request
        )
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
                ending_audio = random.choice(list(filter(lambda x: not x.name.endswith(".gitignore"), list(Constants.ENDING_OUTDIR.iterdir()))))
                voice_client.play(
                    FFmpegOpusAudio(str(ending_audio)),
                    after=after,
                )
            except IndexError:
                # TODO: fix when there is no ending audio
                pass
        

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

        while (
            self.last_played.state == State.Done and self.last_played.next is not None
        ):
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
                    self.loop.create_task(
                        to_play.channel.send(
                            f'Now playing: "{to_play.proc_request.title}"'
                        )
                    )

                    def after(exception):
                        del exception
                        to_play.state = State.Done
                        self.validate()

                    voice_client.play(
                        discord.FFmpegOpusAudio(str(response.path)), after=after
                    )
                else:
                    assert to_play.channel is not None
                    self.loop.create_task(
                        to_play.channel.send(f"Error: ```{response.message}```")
                    )
            elif self.last_played.state == State.Done:
                if not self.queue_empty_played:
                    try:
                        empty_audio = random.choice(list(filter(lambda x: not x.name.endswith(".gitignore"), list(Constants.EMPTY_OUTDIR.iterdir()))))
                        voice_client.play(
                            FFmpegOpusAudio(str(empty_audio)),
                            after=lambda x: self.validate(),
                        )
                    except IndexError:
                        # TODO: fix when there is no empty audio
                        pass
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
