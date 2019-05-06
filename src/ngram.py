import os

from MidiRepExtractor import MidiRepExtractor

def get_note_list(path, combine=False):
    note_lists = []
    extractor = MidiRepExtractor(path)
    if combine == True:
        try:
            note_lists.append((extractor.get_ticks_per_beat(), extractor.get_note_array()))
        except Exception as e:
            print(e)
    else:
        for i in range(extractor.get_num_tracks()):
            try:
                note_lists.append((extractor.get_ticks_per_beat(), extractor.get_note_array(track=i)))
            except Exception as e:
                print(e)
    return note_lists

def get_all_note_lists(root_dir, combine=False):
    note_lists = []
    for r, d, f in os.walk(root_dir):
        for file in f:
            if '.mid' in file:
                path = os.path.join(r, file)
                note_lists = [*note_lists, *get_note_list(path, combine=combine)]
    return note_lists

class Distribution:
    def __init__(self, condition):
        self.condition = condition
        self.frequency = {}

    def add_occurrence(self, event):
        if event not in self.frequency.keys():
            self.frequency[event] = 0
        self.frequency[event] += 1

    def get_pdf(self):
        support = [*self.frequency.keys()]
        total = 0
        for key in support:
            total += self.frequency[key]
        probabilites = []
        for key in support:
            probabilites.append(self.frequency[key] / total)
        return (support, probabilites)
