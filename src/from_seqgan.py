from MidiRepExtractor import Note, MidiRepExtractor

f_name = 'pretrain_epoch_0'
#f_name = 'total_batch_10'
total = 10
lines = open(f_name + '.txt').readlines()

for line_num in range(total):
    raw_notes = [int(x) for x in lines[line_num].split(' ')]
    notes = []
    start_time = 0
    duration = 120
    for n in raw_notes:
        notes.append(Note(n, 80, 480, start_time, -1))
        start_time += duration

    ext = MidiRepExtractor(None)
    ext.write_midi_file(notes, f_name + '_' + str(line_num) + '.mid', 240)
