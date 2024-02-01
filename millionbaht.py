#!/usr/bin/env python3
from io import BytesIO
import subprocess as sp
from typing import Any, Literal, Optional, Union
from pathlib import Path
import json
import discord
import yt_dlp
from discord import VoiceClient, VoiceState
from discord.ext import commands
from litellm import acompletion
import requests
import asyncio


from millionbaht.constants import Constants
from millionbaht.handler import ProcRequest, SongQueue, get_ydl, State


BOT = commands.Bot(
    command_prefix=Constants.BOT_PREFIX,
    intents=discord.Intents(voice_states=True, guilds=True, guild_messages=True, message_content=True),
)


def parse_request(args: tuple[str, ...]) -> ProcRequest:
    new_args: list[str] = []
    kwargs: dict[str, Any] = {}
    for arg in args:
        if arg.startswith("--"):
            kw = arg[2:].split("=")
            kwargs[kw[0]] = kw[1] if len(kw) > 1 else True
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


@BOT.command(name="play", aliases=["p", "pleng", ""], brief="Play a song")
async def play(ctx: commands.Context, *args: str):
    try:
        req = parse_request(args)
    except Exception as e:
        await send_error(ctx, e)
        return
    voice_client = await enter_voice_channel(ctx)
    if voice_client is None:
        return
    queue = get_queue(ctx)
    queue.validate()
    await ctx.send(f'Looking for "{req.query}" ...')
    ydl = get_ydl()
    try:
        info = ydl.extract_info(req.query, download=False)
    except yt_dlp.utils.DownloadError as err:
        await send_error(ctx, err)
        return

    assert isinstance(info, dict)
    if "entries" in info:
        info = info["entries"][0]
    assert isinstance(info, dict)

    req.query = info["id"]
    req.title = info["title"]
    await ctx.send(f"Adding: https://youtu.be/{req.query}")
    queue.put(req, ctx.channel)


@BOT.command(name="remove", brief="Remove current song from the database")
async def remove(ctx: commands.Context):
    if not usr_in_same_voice_room(ctx):
        await ctx.send("You have to be in the same voice channel as the bot")
        return
    queue = get_queue(ctx)
    if queue.last_played.state == State.Playing:
        try:
            req = queue.last_played.proc_request
            queue.skip(1)
            files_to_remove: list[Path] = []
            for file in Constants.SONGDIR.iterdir():
                if req.query in file.stem:
                    files_to_remove.append(file)

            for file in files_to_remove:
                embed = discord.Embed(description=f"Removed {file} from the database", color=Constants.COLOR)
                await ctx.send(embed=embed)
                file.unlink()

        except Exception as e:
            print("[remove] Error: ", e)


@BOT.command(name="code", brief="Prompt with codellama-7b-instruct-awq")
async def llm_code(ctx: commands.Context, *args: str):
    try:
        req = parse_request(args)
        query = req.query
        assert isinstance(query, str)
        response = await acompletion(
            model="cloudflare/@hf/thebloke/codellama-7b-instruct-awq",
            messages=[{"role": "user", "content": query}],
            stream=True,
        )
        outputs = ""
        async for chunk in response:  # type: ignore
            outputs += str(chunk["choices"][0]["delta"]["content"])
            if len(outputs) > 1 and "\n" in outputs:
                outputs = "".join(outputs)
                outputs.replace("\n", "")
                embed = discord.Embed(title="Code", description=outputs, color=Constants.COLOR)
                await ctx.send(embed=embed)
                outputs = ""
        if len(outputs) > 0:
            embed = discord.Embed(title="Code", description=outputs, color=Constants.COLOR)
            await ctx.send(embed=embed)
    except Exception as e:
        await send_error(ctx, e)
        return


@BOT.command(name="chat", brief="Prompt with llama-2-7b-chat-fp16")
async def llm_chat(ctx: commands.Context, *args: str):
    try:
        req = parse_request(args)
        query = req.query
        assert isinstance(query, str)
        response = await acompletion(
            model="cloudflare/@cf/meta/llama-2-7b-chat-fp16",
            messages=[{"role": "user", "content": query}],
            stream=True,
        )
        outputs = ""
        async for chunk in response:  # type: ignore
            outputs += str(chunk["choices"][0]["delta"]["content"])
            if len(outputs) > 1 and "\n" in outputs:
                outputs = "".join(outputs)
                outputs.replace("\n", "")
                embed = discord.Embed(title="Chat", description=outputs, color=Constants.COLOR)
                await ctx.send(embed=embed)
                outputs = ""
        if len(outputs) > 0:
            embed = discord.Embed(title="Chat", description=outputs, color=Constants.COLOR)
            await ctx.send(embed=embed)
    except Exception as e:
        await send_error(ctx, e)
        return


@BOT.command(name="roop", brief="Prompt with stable-diffusion-xl-base-1.0")
async def roop(ctx: commands.Context, *args: str):
    headers = {
        "Authorization": f"Bearer {Constants.CF_API_TOKEN}",
        "Content-Type": "application/x-www-form-urlencoded",
    }

    req = parse_request(args)
    query = req.query
    data = json.dumps({"prompt": query})

    response = await asyncio.to_thread(
        requests.post,
        f"https://api.cloudflare.com/client/v4/accounts/{Constants.CF_ACCOUNT_ID}/ai/run/@cf/stabilityai/stable-diffusion-xl-base-1.0",
        headers=headers,
        data=data,
    )

    f = BytesIO(response.content)
    f.seek(0)
    filename = f"sdxl-{query}.png"
    picture = discord.File(f, filename=filename)
    embed = discord.Embed()
    embed.set_image(url=f"attachment://{filename}")
    await ctx.send(file=picture, embed=embed)


@BOT.command(name="skip", brief="Skip a song")
async def skip(ctx: commands.Context, n_skip: Union[int, Literal["all"]] = 1):
    if not usr_in_same_voice_room(ctx):
        await ctx.send("You have to be in the same voice channel as the bot")
        return
    await ctx.send(f"Skipping {n_skip} songs")
    if n_skip == "all":
        n_skip = -1
    queue = get_queue(ctx)
    queue.skip(n_skip)


@BOT.command(name="shutdown", brief="Shutdown the bot")
async def shutdown(ctx: commands.Context):
    await ctx.send("Shutting down...")
    await BOT.close()


@BOT.command(name="queue", aliases=["q"], brief="Show the queue")
async def queue(ctx: commands.Context):
    if not usr_in_same_voice_room(ctx):
        await ctx.send("You have to be in the same voice channel as the bot")
        return
    queue = get_queue(ctx)
    await ctx.send("Queue:\n" + queue.get_queue())


@BOT.command(name="leave", brief="Leave the voice channel")
async def leave(ctx: commands.Context):
    if not usr_in_same_voice_room(ctx):
        await ctx.send("You have to be in the same voice channel as the bot")
        return
    get_queue(ctx).leave()
    assert ctx.guild is not None
    guild_id = ctx.guild.id
    del QUEUES[guild_id]


@BOT.command(name="restart", brief="Restart the bot [all servers]")
async def restart(ctx: commands.Context, *args):
    del args
    if not usr_in_same_voice_room(ctx):
        await ctx.send("You have to be in the same voice channel as the bot")
        return
    await ctx.send("Restarting...")
    sp.run(["./restart"])


@BOT.command(name="update", brief="Update the bot [all servers]")
async def update(ctx: commands.Context, *args):
    del args
    if not usr_in_same_voice_room(ctx):
        await ctx.send("You have to be in the same voice channel as the bot")
        return
    await ctx.send("Updating...")
    command = ["git", "pull"]
    output = sp.check_output(command, stderr=sp.STDOUT)
    embed = discord.Embed(description=output.decode("utf-8"), color=Constants.COLOR)
    await ctx.send(embed=embed)


@BOT.command(name="auto", brief="Turn on autoplay mode. Usage: auto [true/false]")
async def auto(ctx: commands.Context, state: bool = True):
    queue = get_queue(ctx)
    queue.set_is_auto_state(state)
    await ctx.send(f"auto mode is now {queue.is_auto}")


def get_queue(ctx: commands.Context) -> SongQueue:
    assert ctx.guild is not None
    guild_id = ctx.guild.id
    if not guild_id in QUEUES:
        queue = SongQueue(BOT, guild_id)
        queue.set_context(ctx)
        QUEUES[guild_id] = queue
    else:
        queue = QUEUES[guild_id]
    return queue


# BOT.voice_clients[0].channel.connect


async def enter_voice_channel(ctx: commands.Context) -> Optional[VoiceClient]:
    voice_state = ctx.author.voice or None  # type: ignore

    if not isinstance(voice_state, VoiceState):
        await ctx.send("You have to be in a voice channel to use this command")
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


async def send_error(ctx: commands.Context, err: Exception):
    await ctx.send(f"Error: ```{type(err).__name__}: {err}```")


def main():
    if Constants.BOT_TOKEN is None:
        return (
            "no token provided. Please create a .env file containing the token.\n"
            "for more information view the README.md"
        )
    try:
        BOT.run(Constants.BOT_TOKEN)
    except discord.PrivilegedIntentsRequired as error:
        return error


if __name__ == "__main__":
    try:
        main()
    except SystemError as error:
        if Constants.PRINT_STACK_TRACE:
            raise
        else:
            print(error)
