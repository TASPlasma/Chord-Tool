import tkinter as tk
import audio

class RandomChord():
    def __init__(self, window):
        self.window = window
        self.frame = tk.Frame(self.window, bg='blue')
        self.frame.grid(row=0, column=1, sticky='nsew')
        self.frame.grid_columnconfigure(
            (1,2,3,4,5,6), 
            weight=1, 
            uniform='equal'
            )
        self.title = tk.Label(self.frame, text="Random Chord", bg='blue')
        self.title.grid(row=1, column=3, sticky='ew')

        self.cat1 = tk.Button(
            self.frame, 
            text="Degree 1", 
            command=lambda:self.chord_cat(1)
            )
        self.cat2 = tk.Button(
            self.frame, 
            text="Degree 2",
            command=lambda:self.chord_cat(2)
            )
        self.cat3 = tk.Button(
            self.frame, 
            text="Degree 3",
            command=lambda:self.chord_cat(3)
            )
        self.cat4 = tk.Button(
            self.frame, 
            text="Degree 4",
            command=lambda:self.chord_cat(4)
            )
        self.cat5 = tk.Button(
            self.frame, 
            text="Degree 5",
            command=lambda:self.chord_cat(5)
            )
        self.cat6 = tk.Button(
            self.frame, 
            text="Degree 6",
            command=lambda:self.chord_cat(6)
            )

        self.btns = [
            self.cat1, 
            self.cat2, 
            self.cat3, 
            self.cat4, 
            self.cat5, 
            self.cat6
            ]

        for i, btn in enumerate(self.btns):
            btn.grid(row=2, column = i+1, sticky='ew')

    def chord_cat(self, i):

        if i == 1:
            df = audio.chords[audio.chords['dissonance']==0]
            df = df.drop(['name', 'dissonance'], axis=1)
        elif i == 2:
            df = audio.chords[(audio.chords['dissonance'] >= 1) 
            & (audio.chords['dissonance'] <= 2)]
            df = df.drop(['name', 'dissonance'], axis=1)
        elif i == 3:
            df = audio.chords[(audio.chords['dissonance'] >= 3) 
            & (audio.chords['dissonance'] <= 7)]
            df = df.drop(['name', 'dissonance'], axis=1)
        elif i == 4:
            df = audio.chords[(audio.chords['dissonance'] >= 8) 
            & (audio.chords['dissonance'] <= 11)]
            df = df.drop(['name', 'dissonance'], axis=1)
        elif i == 5:
            df = audio.chords[(audio.chords['dissonance'] >= 12) 
            & (audio.chords['dissonance'] <= 15)]
            df = df.drop(['name', 'dissonance'], axis=1)
        else:
            df = audio.chords[(audio.chords['dissonance'] >= 16) 
            & (audio.chords['dissonance'] <= 19)]
            df = df.drop(['name', 'dissonance'], axis=1)
        
        

        self.chord = df.sample().values.flatten()

        text_var = tk.StringVar()
        text_var.set(audio.chord_name(self.chord))
        tk.Label(self.frame, text=text_var.get()).grid(
            row=5, 
            column=i, 
            sticky='ew'
            )
        audio.play_voicing(self.chord)
