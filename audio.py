import pandas as pd
import numpy as np
import wave
import os
import sys
import pydub
import time
import pygame
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

folder = os.getcwd()
folder = os.path.join(folder, "chord_dataset.txt")
folder = Path(folder)

chords = pd.read_csv(
    folder, 
    sep=' '
)

# Utility functions used later

def note_dist(a,b):
    """
    Inputs: a (the first note)
    Inputs: b (the second note)
    Outputs the note 'distance' of two notes, not necessarily a metric
    for example, note_dist(0, 10)=2
    the distance will never exceed 6 (the 7 exception is a hack for voice leading)
    note_dist is a symmetric function; note_dist(a,b)=note_dist(b,a)
    """
    if a < 0 or b < 0:
        return 7
    c = ((a % 12)-(b % 12)) % 12
    d = ((b % 12)-(a % 12)) % 12
    return min(c, d)

def dissonance(a, b):
    """
    Inputs: a, b keys of a voicing, or notes of a chord

    Outputs: dissonance_value, 
    a subjective value for dissonance between a, b
    """
    dist = note_dist(a, b)
    dissonance_value = 0
    if dist == 1 or dist == 6:
        dissonance_value = 2
    elif dist == 2:
        dissonance_value = 1

    return dissonance_value

def chord_dissonance(chord):
    """Inputs: chord, a chord with which the subjective dissonance will
    be measured.

    Outputs: chord_dissonance_value, 
    a subjective value for the dissonance of an entire chord
    """

    notes = keys_of_chord(chord)
    n = len(notes)

    chord_dissonance_value = 0
    for i in range(n):
        for j in range(i, n):
            if j > i:
                diss = dissonance(notes[i], notes[j])
                chord_dissonance_value += diss

    return chord_dissonance_value
        

def sign(a):
    """sign function of a real number
    """
    if int(a) == 0:
        return 1
    else:
        return int(a/abs(a))

def cartesian_coord(*arrays):
    """
    Outputs the cartesian product of arrays
    """
    swapper = [j for j in arrays]
    a = swapper[0]
    b = swapper[1]
    swapper[0]=b
    swapper[1]=a
    swapped = tuple(swapper)
    grid = np.meshgrid(*swapped)        
    coord_list = [entry.ravel() for entry in grid]
    points = np.vstack(coord_list).T
    points[:, [1, 0]] = points[:, [0, 1]]
    return points

# chord |-> the elements of Z/12Z of a chord
# voicing |-> the elements of Z/12Z of a voicing of a chord
def keys_of_chord(chord):
    """
    Outputs the 'keys' of a chord.
    Example 1: Any octave of C is 0
    Example 2: Any octave of E is 4
    """
    chord = np.array(chord)
    offset = np.array([
        0, 
        chord[0]+1, 
        chord[0]+5, 
        chord[0]+9, 
        chord[0]+0, 
        chord[0]+4, 
        chord[0]+7
        ])
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
    voice = voice_correction(voice)
    
    return voice

def playSound(filename):
    f = open(filename)
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy(): # check if the file is playing
        pass
    pygame.mixer.quit()
    

def play_voicing(
    chord, 
    voice=np.array([])
    ):
    if not voice.any():
        voice = random_initial_root_voicing(chord)

    folder = os.getcwd()
    folder = os.path.join(folder, "Rhodes Notes")
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
    
    out = combined.export(clip_folder / 'temp.wav', format = 'wav')
    out.close()
    
    # winsound.PlaySound(str(clip_folder / 'temp.wav'), winsound.SND_FILENAME)
    playSound(str(clip_folder / 'temp.wav'))
    # os.remove(str(clip_folder / 'temp.wav'))

# Function that returns index of input chord name
def chord_finder(chord_name):
    df = chords.loc[:, ~chords.columns.isin(['name', 'dissonance'])]
    return df.loc[(chords['name']==chord_name)].values.flatten()

# if the root is below 4, raise octave up 1
def voice_correction(voicing):
    """
    Ensures voicing is within range of audio clips
    """
    if voicing[0] < 4:
        voicing[0]+=12
        voicing[1]+=12
    for i in range(len(voicing)):
        if voicing[i] < 27 and i > 1:
            voicing[i]+=12
        elif voicing[i] > 58:
            voicing[i]-=12
        if voicing[i] // 12 > 4:
            voicing[i] -= 12
        if voicing[i] // 12 < 2 and i > 2:
            voicing[i] += 12
    copy = sorted(voicing)
    if (copy[len(copy)-1] - copy[len(copy)-2]) > 8:
        voicing[len(voicing)-1]-=12
    return voicing

def high_notes(voicing):
    voicing_highs = []
    average = 0
    for note in voicing:
        if note >= 28:
            voicing_highs.append(note)
            average += note
    average = average/len(voicing_highs)
    if average == 0:
        average = max(voicing)
    return voicing_highs, average

def all_voicings(chord):
    """
    Returns a collection of reasonable rootless voicings for a chord
    """
    keys = keys_of_chord(chord)
    octaves = [3,4]
    n = len(keys)
    octave_arrays = [octaves]*(n-1)
    octave_arrays.insert(0, [1])
    octave_arrays = tuple(octave_arrays)

    voicings = cartesian_coord(*octave_arrays)
    voicings = [12*voice+keys for voice in voicings]
        
    return voicings

def conditional_voicing2(voicing1, chord2):
    """
    Computes the average note in voicing1 except any bass notes
    Considers possible voicings for chord2 and chooses the one
    whose average is closest to voicing1
    
    Inputs: voicing1, the voicing of the first chord
    Inputs: chord2, the chord following the first chord that needs
    a voicing
    
    Outputs: voicing2, a voicing for chord2
    """
    voicing1_highs, average1 = high_notes(voicing1)
    voicing2 = random_initial_root_voicing(chord2)
    voicing2_highs, average2 = high_notes(voicing2)
    # average1 = average1/len(voicing1)
    # average2 = average2/len(voicing2)

    for voice in all_voicings(chord2):
        root=voice[0]%12
        if root < 4:
            root += 12
        
        voice_highs, average = high_notes(voice)
        #average = average/len(voice)
        if abs(average1-average) < abs(average1-average2):
            voicing2 = np.concatenate(([root],voice[1:]))
            average2 = average
    return voice_correction(voicing2)

def play_chord_progression(chord_progression):
    n = len(chord_progression)
    chord1=chord_finder(chord_progression[0])
    voicing1 = random_initial_root_voicing(chord1)
    voicing_prev=voicing1
    for i in range(1):
        play_voicing(chord1, voice=voicing1)
        for j in range(n-1):
            chord_prev = chord_finder(chord_progression[j])
            chord_cur = chord_finder(chord_progression[j+1])
            voicing_cur = conditional_voicing2(voicing_prev, chord_cur)
            print(voicing_cur)
            play_voicing(chord_cur, voice = voicing_cur)
            voicing_prev = voicing_cur
        voicing1 = conditional_voicing2(voicing_cur, chord1)