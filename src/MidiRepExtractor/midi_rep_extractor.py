from mido import MidiFile, MidiTrack, Message

class MidiEvent:
    def __init__(self, event_type, velocity, duration, start_tick, note):
        self.event_type = event_type
        self.velocity = velocity
        self.duration = duration
        self.start_tick = start_tick
        self.note = note

    def __str__(self):
        return 'Note: ' + str(self.note) +', Type: ' + self.event_type + ', Velocity: ' + str(self.velocity) + ', Start tick: ' + str(self.start_tick)

class Note:
    def __init__(self, note, velocity, duration, start_time):
        self.note = note
        self.velocity = velocity
        self.duration = duration
        self.start_time = start_time

    def __str__(self):
        return 'Note: ' + str(self.note) + ', Velocity: ' + str(self.velocity) + ', Duration: ' + str(self.duration) + ', Start time: ' + str(self.start_time)

    def __repr__(self):
        return 'Note: ' + str(self.note) + ', Velocity: ' + str(self.velocity) + ', Duration: ' + str(self.duration) + ', Start time: ' + str(self.start_time)

class MidiRepExtractor:
    NOTE_ON = 'note_on'
    NOTE_OFF = 'note_off'

    def __init__(self, filename):
        if filename != None:
            self.file = MidiFile(filename)

    def print_messages(self, track=0):
        """
        Displays all messages in a midi
        :return:
        """
        for msg in self.file.tracks[track]:
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
                start_tick += msg.time
                event = MidiEvent(msg.type, msg.velocity, -1, start_tick, msg.note)
                data.append(event)
        return data

    def get_note_array_from_track(self, track):
        notes = []
        on_notes = {}
        time = 0
        to_parse = self.file.tracks[track]
        for i in range(len(to_parse)):
            msg = to_parse[i]
            time += msg.time
            if msg.is_meta:
                continue
            if msg.type == MidiRepExtractor.NOTE_OFF or (msg.type == MidiRepExtractor.NOTE_ON and (msg.velocity == 0 or msg.note in on_notes.keys())):
                if msg.note not in on_notes.keys():
                    continue
                note_obj = on_notes[msg.note]
                note_obj.duration = time - note_obj.start_time
                del on_notes[msg.note]
            elif msg.type == MidiRepExtractor.NOTE_ON and msg.velocity != 0:
#                if msg.note in on_notes.keys():
#                    raise Exception('Note turned on twice')
                note_obj = Note(msg.note, msg.velocity, -1, time)
                on_notes[msg.note] = note_obj
                notes.append(note_obj)
        return notes

    def get_note_array(self, track=None):
        if track != None and isinstance(track, int):
            return self.get_note_array_from_track(track)
        else:
            if self.file.type == 2:
                raise Exception('Asynchronus track')
            note_list = []
            for i in range(self.get_num_tracks()):
                note_list = [*note_list, *self.get_note_array_from_track(i)]
            note_list.sort(key=lambda x: x.start_time)
            return note_list

    def get_ticks_per_beat(self):
        return self.file.ticks_per_beat

    def get_num_tracks(self):
        return len(self.file.tracks)

    def insert_chronologically_from_end(self, array, element, key):
        index = len(array) - 1
        while index > -1 and key(element) < key(array[index]):
            index -= 1
        array.insert(index + 1, element)

    def write_midi_file(self, note_array, file_name, ticks_per_beat):
        event_array = []
        key = lambda x: x.start_tick
        for note in note_array:
            on_event = MidiEvent(MidiRepExtractor.NOTE_ON, note.velocity, -1, note.start_time, note.note)
            off_event = MidiEvent(MidiRepExtractor.NOTE_OFF, note.velocity, -1, note.start_time + note.duration, note.note)
            self.insert_chronologically_from_end(event_array, on_event, key)
            self.insert_chronologically_from_end(event_array, off_event, key)

        mid = MidiFile()
        track = MidiTrack()
        mid.tracks.append(track)
        for i, event in enumerate(event_array):
            if i != 0:
                time = event.start_tick - event_array[i-1].start_tick
            else:
                time = event.start_tick
            track.append(Message(event.event_type, note=event.note, velocity=event.velocity, time=time))
        mid.ticks_per_beat = ticks_per_beat
        mid.save(file_name)

    def get_temporal_image(self, resolution=None):
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

