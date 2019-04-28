from mido import MidiFile, MidiTrack

class MidiEvent:
    def __init__(self, type, velocity, duration, start_tick, note):
        self.type = type
        self.velocity = velocity
        self.duration = duration
        self.start_tick = start_tick
        self.note = note

class MidiRepExtractor:
    NOTE_ON = 'note_on'
    NOTE_OFF = 'note_off'
    def __init__(self, filename):
        self.fn = filename

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
        to_parse = MidiFile(self.fn)
        if track != None and isinstance(track, int):
            to_parse = to_parse.tracks[track]
        start_tick = 0
        for i in range(len(to_parse) - 1):
            msg = to_parse[i]
            if msg.is_meta:
                continue
            if msg.type == MidiRepExtractor.NOTE_ON or msg.type == MidiRepExtractor.NOTE_OFF:
                event = MidiEvent(msg.type, msg.velocity, to_parse[i+1].time, start_tick, msg.note)
                data.append(event)
                start_tick += msg.time
        return data

    def get_temporal_image(self, resolution=None):
        """
        Returns an array of frequencies at the specified time resolution
        :return:
        """
        # TODO: Implement this method
        pass

    def get_midi_obj(self):
        """
        Returns a mido MidiFile object
        :return:
        """
        return MidiFile(self.fn)

