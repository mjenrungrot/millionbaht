from typing import Union
from discord import TextChannel, VoiceChannel, StageChannel, Thread, DMChannel, PartialMessageable, GroupChannel

PartialMessageableChannel = Union[TextChannel, VoiceChannel, StageChannel, Thread, DMChannel, PartialMessageable]
MessageableChannel = Union[PartialMessageableChannel, GroupChannel]
