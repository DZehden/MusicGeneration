import os

from MidiRepExtractor import MidiRepExtractor

def get_note_list(path):
    note_lists = []
    extractor = MidiRepExtractor(path)
    try:
        note_lists.append((extractor.get_ticks_per_beat(), extractor.get_note_array()))
    except Exception as e:
        print(e)
    return note_lists

def get_all_note_lists(root_dir):
    note_lists = []
    for r, d, f in os.walk(root_dir):
        for file in f:
            if '.mid' in file:
                path = os.path.join(r, file)
                note_lists = [*note_lists, *get_note_list(path)]
    return note_lists
