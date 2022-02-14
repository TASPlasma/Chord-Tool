import math
import pandas as pd
import numpy as np
import wave
import os
import sys
import winsound
import pydub
import time
from pydub import AudioSegment
from pathlib import Path

# C = 0, C# = 1, etc.
roots = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# 1 |-> sus 2, 2 |-> minor 3rd, 3 |-> major 3rd, 4 |-> sus 4
thirds = [1,2,3,4]

# 1 |-> flat 5th, 2 |-> perfect 5th, 3 |-> sharp 5th
fifths = [1,2,3]

# 0|-> no 7th, 1 |-> minor 7th, 2 |-> major 7th
sevenths = [0,1,2]

# 0 |-> no 9th, 1 |-> flat 9th, 2 |-> natural 9th, 3 |-> sharp 9th
ninths = [0, 1, 2, 3]

# 0 |-> no 11th, 1 |-> natural 11th, 2 |-> sharp 11th
elevenths = [0, 1, 2]

# 0 |-> no 13th, 1 |-> flat 13th, 2 |-> natural 13th
thirteenths = [0, 1, 2]

def chord_name(chord, shift=0):
    # Inputs: chord; an array, shift, an integer
    # the chord array has a choice of root note, and the remaining elements
    # are the choice of extensions
    
    # Outputs: a string which is the name of the chord
    
    name = ""
    
    # Next we define a some dictionaries using our chord note conventions:
    root_dict = {
    0: 'C',
    1: 'C#',
    2: 'D',
    3: 'Eb',
    4: 'E',
    5: 'F',
    6: 'F#',
    7: 'G',
    8: 'G#',
    9: 'A',
    10: 'Bb',
    11: 'B'
    }
    
    third_dict = {
        1: 'sus2',
        2: 'm',
        3: '',
        4: 'sus4'
    }
    
    fifth_dict = {
        1: ' b5',
        2: '',
        3: ' #5'
    }
    
    seventh_dict = {
        0: '',
        1: ' 7',
        2: ' maj7'
    }
    
    ninth_dict = {
        0: '',
        1: ' b9',
        2: ' 9',
        3: ' #9'
    }
    
    eleventh_dict = {
        0: '',
        1: ' 11',
        2: ' #11'
    }
    
    thirteenth_dict = {
        0: '',
        1: ' b13',
        2: ' 13'
    }
    
    chord_array_dict = {
        0: root_dict,
        1: third_dict,
        2: fifth_dict,
        3: seventh_dict,
        4: ninth_dict,
        5: eleventh_dict,
        6: thirteenth_dict
    }
    
    for i in range(7):
        if i == 0:
            name += chord_array_dict[i][chord[i]+shift]
        else:
            name += chord_array_dict[i][chord[i]]
        
    return name

# chord |-> the elements of Z/12Z of a chord
# voicing |-> the elements of Z/12Z of a voicing of a chord
def keys_of_chord(chord):
    """
    Outputs the 'keys' of a chord.
    Example 1: Any octave of C is 0
    Example 2: Any octave of E is 4
    """
    chord = np.array(chord)
    offset = np.array([0, chord[0]+1, chord[0]+5, chord[0]+9, chord[0]+0, chord[0]+4, chord[0]+7])
    keys = (chord+offset)%12
    del_inds = [i for i in range(3,7) if chord[i] == 0]
    keys=np.delete(keys, del_inds)
    return keys

def keys_of_voicing(voicing):
    """
    Outputs the 'keys' of a voicing
    """
    voicing = np.array(voicing)
    keys = np.unique(voicing%12)
    return keys

# chord-tuple |-> voicing of chord such that lowest note is root, remaining notes are chosen randomly
def random_initial_root_voicing(chord):
    oct_cond = 0
    
    keys = keys_of_chord(chord=chord)
    
    if keys[0] <= 3:
        oct_cond = 1
    
    l = len(keys)
    g=[1+oct_cond if n == keys[0] else 3+oct_cond if n == keys[2] else np.random.randint(3, 5) for n in keys]
    g = np.concatenate(([0+oct_cond], g))
    keys = np.concatenate(([keys[0]], keys))
    voice = 12*g+keys
    
    return voice

def play_voicing(
    chord, 
    voice=np.array([]), 
    folder=r"C:\Users\Owner\Documents\Chord Project\Chord-Tool\Rhodes Notes"
    ):
    if not voice.any():
        voice = random_initial_root_voicing(chord)
    clip_folder = Path(folder)
    clips = [clip_folder / f'2RhodesNote-0{i}.wav' 
             if i < 10 else clip_folder / f'2RhodesNote-{i}.wav' 
             for i in voice]
    sounds = [AudioSegment.from_file(clip, format="wav")-20 
              for i, clip in enumerate(clips)]
    s = sounds[0]
    for i in range(len(sounds)-1):
        s=s.overlay(sounds[i+1]-2, position = i*50)
    
    combined = s+8
    
    combined.export(clip_folder / 'temp.wav', format = 'wav')
    
    winsound.PlaySound(str(clip_folder / 'temp.wav'), winsound.SND_FILENAME)