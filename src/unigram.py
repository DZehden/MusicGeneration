import numpy.random as rand

from ngram import get_all_note_lists, Distribution
from MidiRepExtractor import MidiRepExtractor, Note

print('Extracting note representation from MIDI files')
data = get_all_note_lists('../classical_piano/')

print('Parsing notes to make unigram model')
note_distribution = Distribution()
duration_distribution = Distribution()
for note_list in data:
    ticks_per_beat = note_list[0]
    for note in note_list[1]:
        note_distribution.add_occurrence(note.note)
        duration_distribution.add_occurrence(note.duration / ticks_per_beat)

print('Creating new song from unigram model')
note_support, note_probabilites = note_distribution.get_pdf()
duration_support, duration_probabilites = duration_distribution.get_pdf()
tpb = 480
note_list = []
for i in range(100):
    note = rand.choice(note_support, p=note_probabilites)
    duration = int(rand.choice(duration_support, p=duration_probabilites) * tpb)
    if i == 0:
        start_tick = 0
    else:
        # Immediately after last note is played
        start_tick = note_list[i - 1].start_time + note_list[i - 1].duration
    # Constant velocity for simplicity
    note_list.append(Note(note, 40, duration, start_tick))

print('Store song as MIDI file')
extractor = MidiRepExtractor(None)
extractor.write_midi_file(note_list, 'unigram_test.mid', 240)
print('Done!')
