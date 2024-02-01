from typing import Optional, Union

from discord import (
    DMChannel,
    GroupChannel,
    PartialMessageable,
    StageChannel,
    TextChannel,
    Thread,
    VoiceChannel,
)
from pydantic import BaseModel

PartialMessageableChannel = Union[TextChannel, VoiceChannel, StageChannel, Thread, DMChannel, PartialMessageable]
MessageableChannel = Union[PartialMessageableChannel, GroupChannel]


class YoutubeEntry(BaseModel):
    id: str
    title: str
    is_live: bool
    ext: str