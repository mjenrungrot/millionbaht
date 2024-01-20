#!/usr/bin/env python3
import re

import random
import discord
from discord.ext import commands
import yt_dlp
import urllib
import asyncio
import threading
import os
import shutil
import sys
import subprocess as sp
from dotenv import load_dotenv
import torchaudio
import torch
from torch_pitch_shift import pitch_shift
import inflect
import time
from gtts import gTTS

number_engine = inflect.engine()
random.seed(time.perf_counter_ns())

radio_opening_statements = [
    "สวัสดีครับ วันนี้มีคนขอเพลง",
    "สวัสดีค่ะ มาเจอกันอีกทีที่สถานีวิทยุของเรา",
    "สวัสดีทุกคนครับ พร้อมที่จะทำให้วันนี้เต็มไปด้วยเพลงเพราะ",
    "สวัสดีค่ะทุกคน มีความสุขมากที่ได้เจอกันในที่นี้",
    "สวัสดีครับทุกคน มีความยินดีที่ได้เป็นส่วนหนึ่งของวงการวิทยุ",
    "สวัสดีค่ะทุกคน พร้อมที่จะเปิดเสียงดนตรีสำหรับคุณ",
    "สวัสดีครับทุกคน มีเพลงเพราะๆ รอคุณอยู่",
    "สวัสดีค่ะทุกคน มาพร้อมกับเพลงที่ทำให้คุณลืมทุกวันเครียด",
]

radio_ending_statements = [
    "ขอบคุณทุกคนที่ร่วมฟังเราวันนี้ หวังว่าคุณจะมีวันที่สดใส และเราจะพบกันใหม่ในครั้งถัดไปค่ะ",
    "ขอบคุณทุกคนที่ทำให้วันนี้เต็มไปด้วยความสุข ลาก่อนค่ะ หวังว่าเราจะได้เจอกันอีกในเร็วๆ นี้",
    "ถึงตรงนี้แล้วนะค่ะ ขอบคุณทุกคนที่ร่วมสนุกกับเรา ลาก่อนและหวังว่าคุณจะมีคืนที่สุข",
    "ขอบคุณทุกคนที่ทำให้วันนี้เป็นวันที่น่าจดจำ เราจะไปพบกับคุณในรอบต่อไป ลาก่อนค่ะ",
    "มาถึงจุดสุดท้ายของรายการวันนี้แล้วค่ะ ขอบคุณทุกคนที่ร่วมฟัง ลาก่อนและหวังว่าคุณจะมีคืนที่สนุกสุข"
]

sponsors = [
    "บริษัท ไทยแอร์เวย์ จำกัด",
    "ธนาคารกรุงเทพ จำกัด",
    "โรบินสัน กรุ๊ป",
    "แอสิกส์ ไอเอ็มเอ็ม",
    "แมคโดนัลด์ ไทย",
    "ทีโอที กรุ๊ป",
    "ซีพี แอ็กพลอร์",
    "แมนพาวเวอร์ กรุ๊ป",
    "ไทยโออิล จำกัด",
    "พีทีที กรุ๊ป",
    "กรมการขนส่งทางบก",
    "กระทรวงการคลัง",
    "กรมพัฒนาธุรกิจการค้า",
    "กรมทรัพยากรธรรมชาติและสิ่งแวดล้อม",
    "กรมสรรพากร",
    "กรมการปกครอง",
    "กรมทางหลวงชนบท",
    "กรมวิทยุโทรทัศน์และสื่อสาร",
    "กรมการแพทย์",
    "กรมประชาสัมพันธ์",
    "กรมวิทยุยานเกราะ",
    "บริษัท ไทยประกันชีวิต",
    "สหกรณ์การเกษตร",
    "โรงพยาบาลแม็กซ์เวลล์",
    "บริษัท ไทยโปรประจำ",
    "ธนาคารกรุงไทย",
    "โรงเรียนสอนภาษาอังกฤษ",
    "บริษัท ไทยโลจิสติกส์",
    "โรงแรมแกรนด์ ไทย",
    "ศูนย์การค้าเซ็นทรัล",
    "บริษัท ไทยโฟร์ยู",
    "บริษัท ไทยโทเทิ่ล",
    "โรงงานผลิตอาหารสัตว์",
    "สำนักงานทนายความ",
    "บริษัท ไทยแมร์ชั่นท์",
    "สถาบันการเงิน",
    "โรงเรียนสอนดนตรี",
    "บริษัท ไทยเทรดดิ้ง",
    "สำนักงานการท่องเที่ยวแห่งประเทศไทย",
    "โรงงานผลิตเสื้อผ้า",
    "บริษัท ไทยโมบาย",
    "สถานีวิทยุท้องถิ่น",
    "ศูนย์ฝึกอาชีพ",
    "บริษัท ไทยแพ็ค",
    "สถาบันการศึกษา",
    "โรงงานผลิตรถยนต์",
    "สมาคมธุรกิจ",
    "บริษัท ไทยเทค",
    "สำนักงานภาษี",
    "โรงเรียนสอนศิลปะ",
    "บริษัท ไทยไฮโด",
    "สำนักงานทรัพย์สิน",
    "โรงเรียนสอนคอมพิวเตอร์",
    "บริษัท ไทยโฮเต็ล",
    "สำนักงานโฆษณา",
    "ศูนย์ศิลปวัฒนธรรม",
    "บริษัท ไทยเซอร์วิส",
]



load_dotenv()
TOKEN = os.getenv('BOT_TOKEN', )
PREFIX = os.getenv('BOT_PREFIX', '.')
YTDL_FORMAT = os.getenv('YTDL_FORMAT', 'worstaudio')
PRINT_STACK_TRACE = os.getenv('PRINT_STACK_TRACE', '1').lower() in ('true', 't', '1')
BOT_REPORT_COMMAND_NOT_FOUND = os.getenv('BOT_REPORT_COMMAND_NOT_FOUND', '1').lower() in ('true', 't', '1')
BOT_REPORT_DL_ERROR = os.getenv('BOT_REPORT_DL_ERROR', '0').lower() in ('true', 't', '1')
try:
    COLOR = int(os.getenv('BOT_COLOR', 'ff0000'), 16)
except ValueError:
    print('the BOT_COLOR in .env is not a valid hex color')
    print('using default color ff0000')
    COLOR = 0xff0000

bot = commands.Bot(command_prefix=PREFIX, intents=discord.Intents(voice_states=True, guilds=True, guild_messages=True, message_content=True))
queues = {} # {server_id: [(vid_file, info), ...]}

def main():
    if TOKEN is None:
        return ("no token provided. Please create a .env file containing the token.\n"
                "for more information view the README.md")
    try: bot.run(TOKEN)
    except discord.PrivilegedIntentsRequired as error:
        return error

@bot.command(name='queue', aliases=['q'])
async def queue(ctx: commands.Context, *args):
    try: queue = queues[ctx.guild.id]
    except KeyError: queue = None
    if queue == None:
        await ctx.send('the bot isn\'t playing anything')
    else:
        title_str = lambda val: '‣ %s\n\n' % val[1] if val[0] == 0 else '**%2d:** %s\n' % val
        queue_str = ''.join(map(title_str, enumerate([i[1]["title"] for i in queue])))
        embedVar = discord.Embed(color=COLOR)
        embedVar.add_field(name='Now playing:', value=queue_str)
        await ctx.send(embed=embedVar)
    if not await sense_checks(ctx):
        return

@bot.command(name='skip', aliases=['s'])
async def skip(ctx: commands.Context, *args):
    try: queue_length = len(queues[ctx.guild.id])
    except KeyError: queue_length = 0
    if queue_length <= 0:
        await ctx.send('the bot isn\'t playing anything')
    if not await sense_checks(ctx):
        return

    try: n_skips = int(args[0])
    except IndexError:
        n_skips = 1
    except ValueError:
        if args[0] == 'all': n_skips = queue_length
        else: n_skips = 1
    if n_skips == 1:
        message = 'skipping track'
    elif n_skips < queue_length:
        message = f'skipping `{n_skips}` of `{queue_length}` tracks'
    else:
        message = 'skipping all tracks'
        n_skips = queue_length
    await ctx.send(message)

    voice_client = get_voice_client_from_channel_id(ctx.author.voice.channel.id)
    for _ in range(n_skips - 1):
        queues[ctx.guild.id].pop(0)
    voice_client.stop()


def convert_to_thai_layout(input_text: str) -> str:
    # Define mapping of characters from default layout to Thai layout
    eng_keys = ["`1234567890-=", "qwertyuiop[]\\", "asdfghjkl;'", "zxcvbnm,./", "~!@#$%^&*()_+", "QWERTYUIOP{}|", "ASDFGHJKL:\"", "ZXCVBNM<>?"]
    tha_keys = ["_ๅ/-ภถุึคตจขช", "ๆไำพะัีรนยบลฃ", "ฟหกดเ้่าสวง", "ผปแอิืทมใฝ", "%+๑๒๓๔ู฿๕๖๗๘๙", "๐\"ฎฑธํ๊ณฯญฐ,ฅ", "ฤฆฏโฌ็๋ษศซ.", "()ฉฮฺ์?ฒฬฦ"]
    for en, th in zip(eng_keys, tha_keys):
        assert len(en) == len(th)
    eng_keys = "".join(eng_keys)
    tha_keys = "".join(tha_keys)
    layout_mapping = dict(zip(eng_keys, tha_keys))

    # Convert each character in the input text
    thai_text = ''.join(layout_mapping.get(char, char) for char in input_text)

    return thai_text


def transform_fn(audio_path: str, info: dict, **kwargs) -> (str, dict):
    semitone = kwargs.get('semitone', 0)
    speed = kwargs.get('speed', 1)

    base_path, filename = os.path.split(audio_path)
    basename, ext = os.path.splitext(filename)
    new_audio_path = os.path.join(base_path, f"{basename}.mod{ext}")

    x, rate = torchaudio.load(audio_path)
    message = f"Input: {x.shape}, {rate} Hz"
    print(message)

    if semitone != 0:
        new_rate = 16000
        x = torchaudio.functional.resample(x, rate, new_rate)
        message = f"Input: {x.shape}, {new_rate} Hz"
        print(message)
        outs = []
        x = x.unsqueeze(1).to("cuda:0")
        print(x.shape)
        for x_ in x.split(1000000, dim=-1):
            print("in=>", x_.shape)
            if x_.shape[-1] < 1000:
                break
            out = pitch_shift(x_, semitone, sample_rate=new_rate).detach().cpu()
            print("out=>", out.shape)
            outs.append(out)
        out = torch.cat(outs, dim=-1).squeeze(1)
        print(out.shape)
    else:
        new_rate = 44100
        x = torchaudio.functional.resample(x, rate, new_rate)
        message = f"Input: {x.shape}, {new_rate} Hz"
        print(message)
        out = x

    message = f"Output: {out.shape}, {new_rate} Hz"
    print(message)

    ## adjust speed

    if speed != 1:
        out, _ = torchaudio.functional.speed(out, orig_freq=new_rate, factor=speed)


    ### TTS1
    semitone_text = " " if semitone == 0 else f" ที่ {number_engine.number_to_words(semitone)} semitone "
    text = random.choice(radio_opening_statements) + info["title"] + semitone_text + "สนับสนุนโดย " + random.choice(sponsors)
    tts1 = gTTS(text, lang="th")
    tts1.save(audio_path + ".tts1.mp3")
    audio_tts1, vocoder_rate = torchaudio.load(audio_path + ".tts1.mp3")
    message = f"TTS Output: {audio_tts1.shape}, {vocoder_rate} Hz"
    print(message)

    if audio_tts1.shape[0] < out.shape[0]:
        audio_tts1 = audio_tts1[:1, ...].repeat(out.shape[0], 1)

    audio_tts1 = torchaudio.functional.resample(audio_tts1, vocoder_rate, new_rate)
    message = f"TTS Output: {audio_tts1.shape}, {new_rate} Hz"
    print(message)

    ### TTS2
    text = random.choice(radio_ending_statements)
    tts2 = gTTS(text, lang="th")
    tts2.save(audio_path + ".tts2.mp3")
    audio_tts2, vocoder_rate = torchaudio.load(audio_path + ".tts2.mp3")
    message = f"TTS2 Output: {audio_tts2.shape}, {vocoder_rate} Hz"
    print(message)

    if audio_tts2.shape[0] < out.shape[0]:
        audio_tts2 = audio_tts2[:1, ...].repeat(out.shape[0], 1)

    audio_tts2 = torchaudio.functional.resample(audio_tts2, vocoder_rate, new_rate)
    message = f"TTS2 Output: {audio_tts2.shape}, {new_rate} Hz"
    print(message)

    pause = torch.zeros(out.shape[0], int(0.1*new_rate))
    out = torch.cat([audio_tts1, pause, out, audio_tts2], dim=-1)
    message = f"Output: {out.shape}, {new_rate} Hz"
    print(message)

    torchaudio.save(new_audio_path, out, sample_rate=new_rate)
    return (new_audio_path, info)


@bot.command(name='restart')
async def restart(ctx: commands.Context, *args):
    voice_state = ctx.author.voice
    if not await sense_checks(ctx, voice_state=voice_state):
        return
    await ctx.send('restarting...')
    sp.run(['./restart'])


@bot.command(name='update')
async def restart(ctx: commands.Context, *args):
    voice_state = ctx.author.voice
    if not await sense_checks(ctx, voice_state=voice_state):
        return
    await ctx.send('updating...')
    sp.run(['git', 'pull'])


@bot.command(name='play', aliases=['p', 'pleng', ''])
async def play(ctx: commands.Context, *args):
    voice_state = ctx.author.voice
    if not await sense_checks(ctx, voice_state=voice_state):
        return

    new_args = []
    semitone = 0
    speed = 1
    convert_to_thai = False
    for arg in args:
        if arg.startswith('--semitone='):
            semitone = int(arg.split('=')[1])
            print(f'semitone: {semitone}')
        elif arg.startswith('--speed='):
            speed = float(arg.split('=')[1])
            print(f'speed: {speed}')
        elif arg.startswith("--th"):
            convert_to_thai = True
        else:
            new_args.append(arg)

    query = ' '.join(new_args)
    if convert_to_thai:
        query = convert_to_thai_layout(query)

    # this is how it's determined if the url is valid (i.e. whether to search or not) under the hood of yt-dlp
    will_need_search = not urllib.parse.urlparse(query).scheme

    server_id = ctx.guild.id

    # source address as 0.0.0.0 to force ipv4 because ipv6 breaks it for some reason
    # this is equivalent to --force-ipv4 (line 312 of https://github.com/yt-dlp/yt-dlp/blob/master/yt_dlp/options.py)
    await ctx.send(f'looking for `{query}`...')
    with yt_dlp.YoutubeDL({'format': YTDL_FORMAT,
                           'source_address': '0.0.0.0',
                           'default_search': 'ytsearch',
                           'outtmpl': '%(id)s.%(ext)s',
                           'noplaylist': True,
                           'allow_playlist_files': False,
                           # 'progress_hooks': [lambda info, ctx=ctx: video_progress_hook(ctx, info)],
                           # 'match_filter': lambda info, incomplete, will_need_search=will_need_search, ctx=ctx: start_hook(ctx, info, incomplete, will_need_search),
                           'paths': {'home': f'./dl/{server_id}'}}) as ydl:
        try:
            info = ydl.extract_info(query, download=False)
        except yt_dlp.utils.DownloadError as err:
            await notify_about_failure(ctx, err)
            return

        if 'entries' in info:
            info = info['entries'][0]
        # send link if it was a search, otherwise send title as sending link again would clutter chat with previews
        await ctx.send('downloading ' + (f'https://youtu.be/{info["id"]}' if will_need_search else f'`{info["title"]}`'))
        try:
            ydl.download([query])
        except yt_dlp.utils.DownloadError as err:
            await notify_about_failure(ctx, err)
            return
        
        try:
            path = f'./dl/{server_id}/{info["id"]}.{info["ext"]}'
            path, info = transform_fn(path, info, semitone=semitone, speed=speed)
        except Exception as e:
            await ctx.send(f'failed to transform audio with {str(e)}')
            pass

        try: queues[server_id].append((path, info))
        except KeyError: # first in queue
            queues[server_id] = [(path, info)]
            try: connection = await voice_state.channel.connect()
            except discord.ClientException: connection = get_voice_client_from_channel_id(voice_state.channel.id)
            connection.play(discord.FFmpegOpusAudio(path), after=lambda error=None, connection=connection, server_id=server_id:
                                                             after_track(error, connection, server_id))

def get_voice_client_from_channel_id(channel_id: int):
    for voice_client in bot.voice_clients:
        if voice_client.channel.id == channel_id:
            return voice_client

def after_track(error, connection, server_id):
    if error is not None:
        print(error)
    try: path = queues[server_id].pop(0)[0]
    except KeyError: return # probably got disconnected
    if path not in [i[0] for i in queues[server_id]]: # check that the same video isn't queued multiple times
        try: os.remove(path)
        except FileNotFoundError: pass
    try: connection.play(discord.FFmpegOpusAudio(queues[server_id][0][0]), after=lambda error=None, connection=connection, server_id=server_id:
                                                                          after_track(error, connection, server_id))
    except IndexError: # that was the last item in queue
        queues.pop(server_id) # directory will be deleted on disconnect
        asyncio.run_coroutine_threadsafe(safe_disconnect(connection), bot.loop).result()

async def safe_disconnect(connection):
    if not connection.is_playing():
        pass
        # await connection.disconnect()
        
async def sense_checks(ctx: commands.Context, voice_state=None) -> bool:
    if voice_state is None: voice_state = ctx.author.voice 
    if voice_state is None:
        await ctx.send('you have to be in a voice channel to use this command')
        return False

    if bot.user.id not in [member.id for member in ctx.author.voice.channel.members] and ctx.guild.id in queues.keys():
        await ctx.send('you have to be in the same voice channel as the bot to use this command')
        return False
    return True

@bot.event
async def on_voice_state_update(member: discord.User, before: discord.VoiceState, after: discord.VoiceState):
    if member != bot.user:
        return
    if before.channel is None and after.channel is not None: # joined vc
        return
    if before.channel is not None and after.channel is None: # disconnected from vc
        # clean up
        server_id = before.channel.guild.id
        try: queues.pop(server_id)
        except KeyError: pass
        try: shutil.rmtree(f'./dl/{server_id}/')
        except FileNotFoundError: pass

@bot.event
async def on_command_error(ctx: discord.ext.commands.Context, err: discord.ext.commands.CommandError):
    # now we can handle command errors
    if isinstance(err, discord.ext.commands.errors.CommandNotFound):
        if BOT_REPORT_COMMAND_NOT_FOUND:
            await ctx.send("command not recognized. To see available commands type {}help".format(PREFIX))
        return

    # we ran out of handlable exceptions, re-start. type_ and value are None for these
    sys.stderr.write(f'unhandled command error raised, {err=}')
    sys.stderr.flush()
    sp.run(['./restart'])


@bot.event
async def on_ready():
    print(f'logged in successfully as {bot.user.name}')
async def notify_about_failure(ctx: commands.Context, err: yt_dlp.utils.DownloadError):
    if BOT_REPORT_DL_ERROR:
        # remove shell colors for discord message
        sanitized = re.compile(r'\x1b[^m]*m').sub('', err.msg).strip()
        if sanitized[0:5].lower() == "error":
            # if message starts with error, strip it to avoid being redundant
            sanitized = sanitized[5:].strip(" :")
        await ctx.send('failed to download due to error: {}'.format(sanitized))
    else:
        await ctx.send('sorry, failed to download this video')
    return

if __name__ == '__main__':
    try:
        sys.exit(main())
    except SystemError as error:
        if PRINT_STACK_TRACE:
            raise
        else:
            print(error)
