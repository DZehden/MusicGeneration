from mido import MidiFile, MidiTrack
from pypianoroll import Track


class MidiEvent:
    def __init__(self, event_type, velocity, duration, start_tick, note):
        self.event_type = event_type
        self.velocity = velocity
        self.duration = duration
        self.start_tick = start_tick
        self.note = note


class MidiRepExtractor:
    NOTE_ON = 'note_on'
    NOTE_OFF = 'note_off'

    def __init__(self, filename):
        self.file = MidiFile(filename)

    def print_messages(self):
        """
        Displays all messages in a midi
        :return:
        """
        mid = MidiFile(self.fn)
        for msg in mid:
            print(msg)

    def get_event_array(self, track=None):
        """
        Extracts note on and off events from a midi file

        :param track: Track index to get messages from
        :return: array of MidiEvents
        """
        data = []
        to_parse = self.file
        if (track is not None) and isinstance(track, int):
            to_parse = self.file.tracks[track]
        start_tick = 0
        parse_dur = len(to_parse) - 1
        for i in range(parse_dur):
            msg = to_parse[i]
            if msg.is_meta:
                continue
            if msg.type == MidiRepExtractor.NOTE_ON or msg.type == MidiRepExtractor.NOTE_OFF:
                event = MidiEvent(msg.type, msg.velocity, to_parse[i+1].time, start_tick, msg.note)
                data.append(event)
                start_tick += msg.time
        return data

    def get_temporal_image(self, early_stop = -1):
        """
        Returns an array of frequencies at the specified time resolution
        :return:
        """
        parse_dur = early_stop * self.file.ticks_per_beat
        image = []


    def get_midi_obj(self):
        """
        Returns a mido MidiFile object
        :return:
        """
        return MidiFile(self.fn)

