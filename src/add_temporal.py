import numpy.random as rand

from ngram import get_all_note_lists, Distribution
from MidiRepExtractor import MidiRepExtractor, Note

MAX_OUT = 10
f_name = 'total_batch_10'

print('Extracting note representation from MIDI files')
data = get_all_note_lists('../nottingham-dataset/MIDI/melody/')

print('Parsing notes to make model')
all_durations = set()
distributions = {}
hist_size = 1
for note_list in data:
    ticks_per_beat = note_list[0]
    note_hist = []
    duration_hist = []
    for note in note_list[1]:
        # Add current note to conditional distributions
        key = (*note_hist, note.note)
        duration = note.duration / ticks_per_beat
        if key not in distributions.keys():
            distributions[key] = Distribution()
        distributions[key].add_occurrence(duration)

        # Edit Histories
        note_hist.append((note.note, duration))
        if len(note_hist) > hist_size:
            del note_hist[0]

print('Adding temporal dimension to clips')
tpb = 480
lines = open(f_name + '.txt').readlines()
output = open(f_name + '_temporal.txt', 'w+')
extractor = MidiRepExtractor(None)
for i, line in enumerate(lines):
    if i >= MAX_OUT:
        break
    raw = [int(x) for x in line.split(' ')]
    note_list = []
    note_hist = []
    start_time = 0
    buff = ""
    for raw_note in raw:
        key = (*note_hist, raw_note)
        if key not in distributions.keys():
            potential = [x for x in distributions.keys() if (len(x) == 1 and x[0] == raw_note) or (len(x) == 2 and x[1] == raw_note)]
            key = potential[rand.choice(range(len(potential)))]
        conditional_support, conditional_probabilities = distributions[key].get_pdf()
        print("Condition: " + str(key))
        print("Support size: " + str(len(conditional_support)))
        print("Total occurrences: " + str(distributions[key].total_occurrences()) + '\n')
        raw_duration = rand.choice(conditional_support, p=conditional_probabilities)
        duration = int(raw_duration * tpb)
        if raw_note != 0:
            note_list.append(Note(raw_note, 80, duration, start_time, -1))
        start_time += duration

        buff += str(raw_note) + '_' + str(int(raw_duration * 48)) + ' '
        note_hist.append((raw_note, raw_duration))
        if len(note_hist) > hist_size:
            del note_hist[0]

    output.write(buff + '\n')
    extractor.write_midi_file(note_list, f_name + '_' + str(i) + '_temporal.mid', 240)

output.close()
print('Done!')
