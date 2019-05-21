#!/usr/bin/env python3

import ngram

SEQ_LENGTH = 20

arr = ngram.get_all_note_lists('../nottingham-dataset/MIDI/melody/')

f = open('positive.txt', 'w+')

def write_buffer(buff, f):
    for i in range(SEQ_LENGTH - 1):
        f.write(buff[i] + ' ')
    f.write(buff[len(buff) - 1] + '\n')

note_lengths = set({0.5, 1.0, 2.0, 1.5, 4.0, 0.25, 0.125, 3.0, 0.75, 0.375, 2.5, 0.6669921875, 0.3330078125})
down_times = set()

num_valid = 0
overlap = 0
invalid_note_lengths = 0
for song in arr:
    buff = []
    skip = False
    tpb = song[0]
    for i, note in enumerate(song[1]):
        buff.append(str(note.note) + '_' + str(int(note.duration / tpb * 48)))
        if i < len(song[1]) - 1 and note.start_time + note.duration > song[1][i + 1].start_time:
            skip = True
            overlap += 1
            break
        elif i < len(song[1]) - 1 and int((song[1][i + 1].start_time - (note.start_time + note.duration)) / tpb * 48) >= 6:
            buff.append(str(0) + '_' + str(int((song[1][i + 1].start_time - (note.start_time + note.duration)) / tpb * 48)))
            down_times.add((song[1][i + 1].start_time - (note.start_time + note.duration)) / tpb)
        if note.duration / tpb not in note_lengths:
            skip = True
            invalid_note_lengths += 1
            break
        if len(buff) >= SEQ_LENGTH:
            write_buffer(buff, f)
            buff = []
    if not skip:
        num_valid += 1

print("Valid songs: " + str(num_valid))
print("Songs with overlap: " + str(overlap))
print("Songs with invalid note lengths: " + str(invalid_note_lengths))
print("Down times: " + str(down_times))
f.close()
