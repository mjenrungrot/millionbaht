from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pipe, Process, Queue as ProcQueue, Value
from multiprocessing.connection import Connection
from multiprocessing.sharedctypes import Synchronized
import time
import os
from urllib import request
from dotenv import load_dotenv
import discord
from discord.ext import commands, tasks
from discord import VoiceState, VoiceClient
import asyncio
from queue import Queue
from typing import Any, Optional, Union, Literal

from src.handler import ProcRequest, SongQueue


# Load Env
load_dotenv()
TOKEN = os.getenv("BOT_TOKEN")
PREFIX = os.getenv("BOT_PREFIX", ".")
YTDL_FORMAT = os.getenv("YTDL_FORMAT", "worstaudio")
PRINT_STACK_TRACE = os.getenv("PRINT_STACK_TRACE", "1").lower() in ("true", "t", "1")
BOT_REPORT_COMMAND_NOT_FOUND = os.getenv("BOT_REPORT_COMMAND_NOT_FOUND", "1").lower() in ("true", "t", "1")
BOT_REPORT_DL_ERROR = os.getenv("BOT_REPORT_DL_ERROR", "0").lower() in (
    "true",
    "t",
    "1",
)
try:
    COLOR = int(os.getenv("BOT_COLOR", "ff0000"), 16)
except ValueError:
    print("the BOT_COLOR in .env is not a valid hex color")
    print("using default color ff0000")
    COLOR = 0xFF0000


BOT = commands.Bot(
    command_prefix=PREFIX,
    intents=discord.Intents(voice_states=True, guilds=True, guild_messages=True, message_content=True),
)


def parse_request(args: tuple[str, ...]) -> ProcRequest:
    new_args: list[str] = []
    kwargs: dict[str, Any] = {}
    for arg in args:
        if arg.startswith("--"):
            k, w = arg[2:].split("=")
            kwargs[k] = w
        else:
            new_args.append(arg)
    return ProcRequest(query=" ".join(new_args), **kwargs)


QUEUES: dict[int, SongQueue] = {}  # guild_id -> SongQueue


# @tasks.loop(seconds=3)
# async def validate_queue():
#     for k, v in QUEUES.items():
#         v.validate()


@BOT.event
async def on_ready():
    assert BOT.user is not None
    print(f"logged in successfully as {BOT.user.name}")


@BOT.command(name="play", aliases=["p", "pleng", ""])
async def play(ctx: commands.Context, *args: str):
    voice_client = await enter_voice_channel(ctx)
    if voice_client is None:
        return
    queue = get_queue(ctx)
    queue.validate()
    request = parse_request(args)
    queue.put(request, ctx.channel)


@BOT.command(name="skip")
async def skip(ctx: commands.Context, n_skip: Union[int, Literal['all']] = 1):
    if not usr_in_same_voice_room(ctx):
        await ctx.send("you have to be in the same voice channel as the bot")
        return
    await ctx.send(f"skipping {n_skip} songs")
    if n_skip == "all":
        n_skip = -1
    queue = get_queue(ctx)
    queue.skip(n_skip)

    
    

def get_queue(ctx: commands.Context) -> SongQueue:
    assert ctx.guild is not None
    guild_id = ctx.guild.id
    if not guild_id in QUEUES:
        queue = SongQueue(BOT, guild_id)
        QUEUES[guild_id] = queue
    else:
        queue = QUEUES[guild_id]
    return queue



# BOT.voice_clients[0].channel.connect


async def enter_voice_channel(ctx: commands.Context) -> Optional[VoiceClient]:
    voice_state = ctx.author.voice or None  # type: ignore

    if not isinstance(voice_state, VoiceState):
        await ctx.send("you have to be in a voice channel to use this command")
        return None

    # if (
    #     BOT.user.id not in [member.id for member in voice_state.channel.members]  # type: ignore
    #     # and ctx.guild.id in queues.keys()
    # ):
    #     await ctx.send(
    #         "you have to be in the same voice channel as the bot to use this command"
    #     )
    #     return None
    assert voice_state.channel is not None
    try:
        return await voice_state.channel.connect()
    except discord.ClientException:
        assert ctx.guild is not None
        guild_id = ctx.guild.id
        for voice_client in BOT.voice_clients:
            if isinstance(voice_client, VoiceClient):
                if voice_client.guild.id == guild_id:
                    return voice_client
    return None


def usr_in_same_voice_room(ctx: commands.Context) -> bool:
    voice_state = ctx.author.voice or None  # type: ignore
    if not isinstance(voice_state, VoiceState):
        return False
    if BOT.user.id not in [member.id for member in voice_state.channel.members]:  # type: ignore
        return False
    return True


def main():
    if TOKEN is None:
        return (
            "no token provided. Please create a .env file containing the token.\n"
            "for more information view the README.md"
        )
    try:
        BOT.run(TOKEN)
    except discord.PrivilegedIntentsRequired as error:
        return error


if __name__ == "__main__":
    try:
        main()
    except SystemError as error:
        if PRINT_STACK_TRACE:
            raise
        else:
            print(error)
