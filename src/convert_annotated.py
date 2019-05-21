#!/usr/bin/env python3

import ngram
from MidiRepExtractor import Chord
from os import listdir
from os.path import isfile, join

SEQ_LENGTH = 20
MELODY_DIR = '../nottingham-dataset/MIDI/melody/'
CHORDS_DIR = '../nottingham-dataset/MIDI/chords/'
files = [f for f in listdir(MELODY_DIR) if isfile(join(MELODY_DIR, f)) and isfile(join(CHORDS_DIR, f))]

pairs = [ngram.get_melody_harmony_pair(MELODY_DIR + f, CHORDS_DIR + f) for f in files]

f = open('positive_annotated.txt', 'w+')

def write_buffer(buff, f):
    for i in range(SEQ_LENGTH - 1):
        f.write(buff[i] + ' ')
    f.write(buff[len(buff) - 1] + '\n')

chord_enum = {}

num_valid = 0
overlap = 0
melody_chonks = []
chords_chonks = []
print(len(pairs))
for pair in pairs:
    skip = False
    melody = pair[0]
    chords = pair[1]
    safety_net = []
    safety_net2 = []
    melody_buff = []
    chords_buff = []
    start_times = []
    tpb = melody[0]
    for i, note in enumerate(melody[1]):
        # Check if first thing is a rest
        if i == 0 and note.start_time != 0:
            melody_buff.append(str(0) + '_' + str(int(melody[1][i].start_time / tpb * 48)))
        melody_buff.append(str(note.note) + '_' + str(int(note.duration / tpb * 48)))
        if i < len(melody[1]) - 1 and note.start_time + note.duration > melody[1][i + 1].start_time:
            skip = True
            overlap += 1
            break
        elif i < len(melody[1]) - 1 and int((melody[1][i + 1].start_time - (note.start_time + note.duration)) / tpb * 48) >= 6:
            melody_buff.append(str(0) + '_' + str(int((melody[1][i + 1].start_time - (note.start_time + note.duration)) / tpb * 48)))
        if len(melody_buff) >= SEQ_LENGTH:
            safety_net.append(melody_buff)
            melody_buff = []
            start_times.append(melody[1][i].start_time + melody[1][i].duration)
    tpb = chords[0]
    section = 0
    current_chord = None
    for i, note in enumerate(chords[1]):
        while section < len(start_times) and note.start_time >= start_times[section]:
            if chords_buff == []:
                if section == 0:
                    safety_net2.append([Chord([0], 0, int(start_times[section] / tpb * 48), 0)])
                else:
                    safety_net2.append([Chord([0], 0, int((start_times[section] - start_times[section - 1]) / tpb * 48), start_times[section - 1])])
            else:
                safety_net2.append(chords_buff)
            chords_buff = []
            section += 1
        if current_chord == None:
            current_chord = Chord([note.note], note.velocity, int(note.duration / tpb * 48), note.start_time)
        elif note.start_time == current_chord.start_time:
            current_chord.notes.append(note.note)
            if i < len(chords[1]) - 1 and int((chords[1][i + 1].start_time - (note.start_time + note.duration)) / tpb * 40) >= 6:
                chords_buff.append(current_chord)
                chords_buff.append(Chord([0], 0, int((chords[1][i + 1].start_time - (note.start_time + note.duration)) / tpb * 48), (note.start_time + note.duration) / tpb * 48))
                current_chord = None
        else:
            chords_buff.append(current_chord)
            current_chord = Chord([note.note], note.velocity, int(note.duration / tpb * 48), note.start_time)

    if len(safety_net) != len(safety_net2):
        skip = True
#    while len(chords_chonks) < len(melody_chonks):
#        print(section)
#        print(len(start_times))
#        if section == 0:
#            chords_chonks.append([Chord([0], 0, int((start_times[section] / tpb * 48), 0))])
#        else:
#            chords_chonks.append([Chord([0], 0, int((start_times[section] - start_times[section - 1]) / tpb * 48), 0)])
#        section += 1
    if not skip:
        num_valid += 1
        melody_chonks.extend(safety_net)
        chords_chonks.extend(safety_net2)
    else:
        continue

assert len(melody_chonks) == len(chords_chonks)
count = 0
for i in range(len(melody_chonks)):
    write_buffer(melody_chonks[i], f)
    for chord in chords_chonks[i]:
        note_tuple = (*chord.notes,)
        if note_tuple not in chord_enum.keys():
            chord_enum[note_tuple] = count
            count += 1
        f.write(str(chord_enum[note_tuple]) + '_' + str(chord.duration) + ' ')
    f.write('\n\n')


print("Valid songs: " + str(num_valid))
print("Songs with overlap: " + str(overlap))
f.close()
