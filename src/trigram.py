import numpy.random as rand

from ngram import get_all_note_lists, Distribution
from MidiRepExtractor import MidiRepExtractor, Note

print('Extracting note representation from MIDI files')
data = get_all_note_lists('../classical_piano/')

print('Parsing notes to make trigram model')
note_distributions = {}
duration_distributions = {}
hist_size = 2
for note_list in data:
    ticks_per_beat = note_list[0]
    note_hist = []
    duration_hist = []
    for note in note_list[1]:
        # Add current note to conditional distributions
        if (*note_hist,) not in note_distributions.keys():
            note_distributions[(*note_hist,)] = Distribution()
        note_distributions[(*note_hist,)].add_occurrence(note.note)
        duration = note.duration / ticks_per_beat
        if (*duration_hist,) not in duration_distributions.keys():
            duration_distributions[(*duration_hist,)] = Distribution()
        duration_distributions[(*duration_hist,)].add_occurrence(duration)

        # Edit Histories
        note_hist.append(note.note)
        if len(note_hist) > hist_size:
            del note_hist[0]
        duration_hist.append(duration)
        if len(duration_hist) > hist_size:
            del duration_hist[0]

print('Creating new song from trigram model')
note_hist = []
duration_hist = []
tpb = 480
note_list = []
for i in range(100):
    note_support, note_probabilites = note_distributions[(*note_hist,)].get_pdf()
    print("Note num " + str(len(note_list)))
    print("Support size " + str(len(note_support)))
    print("Total occurrences " + str(note_distributions[(*note_hist,)].total_occurrences()) + '\n')
    duration_support, duration_probabilites = duration_distributions[(*duration_hist,)].get_pdf()
    note = rand.choice(note_support, p=note_probabilites)
    raw_duration = rand.choice(duration_support, p=duration_probabilites)
    duration = int(raw_duration * tpb)
    if i == 0:
        start_tick = 0
    else:
        # Immediately after last note is played
        start_tick = note_list[i - 1].start_time + note_list[i - 1].duration
    # Constant velocity for simplicity
    note_list.append(Note(note, 40, duration, start_tick))

    # Edit Histories
    note_hist.append(note)
    if len(note_hist) > hist_size:
        del note_hist[0]
    duration_hist.append(raw_duration)
    if len(duration_hist) > hist_size:
        del duration_hist[0]

print('Store song as MIDI file')
extractor = MidiRepExtractor(None)
extractor.write_midi_file(note_list, 'trigram_0.mid', 240)
print('Done!')
