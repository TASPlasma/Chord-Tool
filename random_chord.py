import tkinter as tk
import audio
import pygame
import os
from pathlib import Path

class RandomChord():
    def __init__(self, window, color):
        self.window = window
        self.frame = tk.Frame(self.window, bg=color)
        self.frame.grid(row=0, column=1, sticky='nsew')
        self.frame.grid_columnconfigure(
            (1,2,3,4,5,6), 
            weight=1, 
            uniform='equal'
            )
        self.title = tk.Label(self.frame, text="Random Chord", bg=color, fg='#ffffff')
        self.title.grid(row=1, column=3, sticky='ew')
        self.btn_color = '#1a1a1a'
        self.txt_color = '#ffffff'
        self.lbl_color = '#000000'

        folder = os.getcwd()
        folder = os.path.join(folder, "Rhodes Notes")
        clip_folder = Path(folder)
        self.file = f'{clip_folder}\\temp.wav'

        self.cat1 = tk.Button(
            self.frame, 
            text="Degree 1", 
            bg=self.btn_color,
            fg=self.txt_color,
            command=lambda:self.chord_cat(1, self.file)
            )
        self.cat2 = tk.Button(
            self.frame, 
            text="Degree 2",
            bg=self.btn_color,
            fg=self.txt_color,
            command=lambda:self.chord_cat(2, self.file)
            )
        self.cat3 = tk.Button(
            self.frame, 
            text="Degree 3",
            bg=self.btn_color,
            fg=self.txt_color,
            command=lambda:self.chord_cat(3, self.file)
            )
        self.cat4 = tk.Button(
            self.frame, 
            text="Degree 4",
            bg=self.btn_color,
            fg=self.txt_color,
            command=lambda:self.chord_cat(4, self.file)
            )
        self.cat5 = tk.Button(
            self.frame, 
            text="Degree 5",
            bg=self.btn_color,
            fg=self.txt_color,
            command=lambda:self.chord_cat(5, self.file)
            )
        self.cat6 = tk.Button(
            self.frame, 
            text="Degree 6",
            bg=self.btn_color,
            fg=self.txt_color,
            command=lambda:self.chord_cat(6, self.file)
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

    def chord_cat(self, i, filename):

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
            & (audio.chords['dissonance'] <= 100)]
            df = df.drop(['name', 'dissonance'], axis=1)
        

        self.chord = df.sample().values.flatten()

        text_var = tk.StringVar()
        text_var.set(audio.chord_name(self.chord))
        tk.Label(self.frame, text=text_var.get(), bg=self.btn_color, fg=self.txt_color).grid(
            row=5, 
            column=i, 
            sticky='ew'
            )

        file_exists = os.path.exists(filename)
        if file_exists:
            pygame.mixer.music.unload()
            os.remove(filename)
        
        segment = audio.play_chord_progression2([self.chord], voice_prog=[audio.basic_voicing(self.chord)])
        file = audio.file_from_audio_segment(segment, 'temp.wav')
        self.play_sound(file)

    def play_sound(self, filename):
        with open(filename) as f:
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()
            # pygame.mixer.music.get_endevent()
            # pygame.event.wait()
