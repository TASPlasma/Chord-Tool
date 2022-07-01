import tkinter as tk
import audio
from tkinter import ttk

lst = audio.chords['name'].tolist()



class ChordProgEditor():
    def __init__(self, window):
        self.window = window
        self.frame = tk.Frame(self.window, bg='green')
        self.title = tk.Label(self.frame, text='Chord Progression Editor', bg='green', anchor='w')
        self.title.grid(row=1, column=2, sticky='ew')

        self.frame.grid(row=0, column=1, sticky='nsew')
        self.frame.grid_columnconfigure(
            (1,2,3,4), 
            weight=1, 
            uniform='equal'
            )

        self.chord_txt_boxes = []

        for i in range(4):

            self.chord_txt = ttk.Combobox(self.frame)
            self.chord_txt['values'] = lst
            self.chord_txt.bind('<KeyRelease>', self.check_input)


            self.chord_txt.grid(row=2, column=i+1)
            self.chord_txt_boxes.append(self.chord_txt)

        self.play_btn = tk.Button(
            self.frame, 
            text='Play Chord Progression', 
            command=lambda:self.play_cmd()
            )
        self.play_btn.grid(row=4, column=4, sticky='ew')

        self.export_midi_btn = tk.Button(
            self.frame,
            text='Export Chord Progression To MIDI',
            command=lambda:self.export_cmd()
        )
        self.export_midi_btn.grid(row = 4, column=3, sticky='ew')

    def update_chord_prog(self):
        self.chord_prog = []
        for box in self.chord_txt_boxes:
            self.chord_prog.append(box.get())

        return self.chord_prog

    def play_cmd(self):
        self.prog = self.update_chord_prog()
        audio.play_chord_progression2(self.prog)

    def export_cmd(self):
        self.chord_prog = self.update_chord_prog()
        audio.midi_file_from_chord_prog(self.chord_prog)

    def check_input(self, event):
        value = event.widget.get()

        if value == '':
            self.chord_txt['values'] = audio.chords['name'].head().tolist()
        else:
            data = []
            for item in lst:
                if value.lower() in item.lower():
                    data.append(item)

            self.chord_txt['values'] = data

