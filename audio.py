import pandas as pd
import numpy as np
import os
import ffmpeg
import pygame
from PIL import Image, ImageDraw, ImageFont
from pydub import AudioSegment
from pathlib import Path
from mido import Message, MidiFile, MidiTrack

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

    #exception, augmented chords have dissonance
    if chord[1] == 3 and chord[2] == 3:
        chord_dissonance_value = 1
    
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
    
    if keys[0] <= 6:
        oct_cond = 1
    
    l = len(keys)
    if chord[2] != 2:
        g=[1+oct_cond if n == keys[0] else 4 if n == keys[2] else np.random.randint(3, 5) for n in keys]
    else:
        g=[1+oct_cond if n == keys[0] else 3+oct_cond if n == keys[2] else np.random.randint(3, 5) for n in keys]
    g = np.concatenate(([0+oct_cond], g))
    keys = np.concatenate(([keys[0]], keys))
    voice = 12*g+keys
    voice = voice_correction(voice)
    
    return voice

# Creates a 'stacked thirds' voicing
def basic_voicing(chord):
    offset = np.array([
        0, 
        chord[0]+1, 
        chord[0]+5, 
        chord[0]+9, 
        chord[0]+0, 
        chord[0]+4, 
        chord[0]+7
        ])
    notes = chord+offset+np.array([0, 36, 36, 36, 48, 48, 48])
    del_inds = [i for i in range(3,7) if chord[i] == 0]
    voicing=np.delete(notes, del_inds)
    voicing = voice_correction(voicing)
    return voicing



def play_sound(filename):
    with open(filename) as f:
        pygame.mixer.init()
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy(): # check if the file is playing
            pass
        pygame.mixer.music.unload()
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
    
    play_sound(str(clip_folder / 'temp.wav'))

# Function that returns index of input chord name
def chord_finder(chord_name):
    df = chords.loc[:, ~chords.columns.isin(['name', 'dissonance'])]
    return df.loc[(chords['name']==chord_name)].values.flatten()

def voice_correction(voicing):
    """
    Ensures voicing is within range of audio clips
    """
    copy = sorted(voicing)
    argsorted = np.argsort(voicing)

    # ensures highest notes are close
    if (copy[-1] - copy[-2]) > 6:
        voicing[argsorted[-1]]-=12

    # ensures lowest notes are distant
    if (copy[1] - copy[0]) < 7:
        voicing[argsorted[1]] += 12

    if voicing[0] < 4:
        voicing[0]+=12
        #voicing[1]+=12
    for i in range(len(voicing)):
        if voicing[i] < 27 and i > 1:
            voicing[i]+=12 #this adjustment might be useless/limiting
        elif voicing[i] > 58:
            voicing[i]-=12
        if voicing[i] // 12 > 4:
            voicing[i] -= 12
        if voicing[i] // 12 < 2 and i > 2:
            voicing[i] += 12

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
    return voicing_highs

def voicing_stat(voicing, metric: str):
    """
    Outputs a statistic about the voicing,
    metric can be 'max', 'min', 'range', 'avg'
    """
    max = np.max(voicing)
    min = np.min(voicing[1:])
    avg = 0
    count = 0
    for note in voicing:
        if note >= 28:
            avg += note
            count += 1
    if avg == 0:
        avg = max(voicing)
    else:
        avg = avg/count
    stat_dict = {
        "max": max,
        "min": min,
        "range": max-min,
        "avg": avg
        }
    return stat_dict[metric]

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
    avg1 = voicing_stat(voicing1, "avg")
    # the best current choice for voicing2
    voicing2 = basic_voicing(chord2)
    avg2 = voicing_stat(voicing2, "avg")
    ran2 = voicing_stat(voicing2, "range")

    # optimization objective is to minimize |avg1-avg|+ran
    for voice in all_voicings(chord2):
        root=voice[0]%12
        if root < 4:
            root += 12
        
        avg = voicing_stat(voice, "avg")
        ran = voicing_stat(voice, "range")

        # checks if the current voice is 'closer' to the first voicing
        # if true, we have found a new best voicing
        if abs(avg1-avg)+ran < abs(avg1-avg2)+ran2:
            # resets best current choice for voicing2 and its stats
            voicing2 = np.concatenate(([root], voice[1:]))
            avg2 = avg
            ran2 = ran
    return voice_correction(voicing2)

def voice_prog_completion(chord_prog, voice_prog=None):
    """
    Input(s): 
    chord_prog, a chord progression
    voice_prog, an incomplete sequence of voicings for chord_prog
    Outputs:
    voice_prog, the completed sequence of voicings for chord_prog
    """
    if type(chord_prog[0]) == str:
        chord_prog = [chord_finder(chord) for chord in chord_prog]
    
    if not voice_prog:
        voice_prog = []

    n = len(chord_prog)
    k = len(voice_prog)
    if k == 0:
        voicing_prev = basic_voicing(chord_prog[0])
    else:
        voicing_prev = voice_prog[-1]
    for j in range(n-k):
        # we have voice_prog[0], ..., voice_prog[k-1]
        # we need voice_prog[k], ..., voice_prog[n-1], which is n-k many items
        # voice_prog[k] = cond_voicing(voice_prog[k-1], chord_prog[k])
        # voice_prog[k +1] = cond_voicing(voice_prog[k-1 +1], chord_prog[k +1])
        # voice_prog[k +j] = cond_voicing(voice_prog[k-1 +j], chord_prog[k +j])
        chord_cur = chord_prog[k+j]
        voicing_cur = conditional_voicing2(voicing_prev, chord_cur)
        voice_prog=voice_prog+[voicing_cur]
        voicing_prev = voicing_cur
    return voice_prog

def complete_durations(chord_prog, durations=None):
    """
    Completes a sequence of durations
    """
    if not durations:
        durations = []

    durations = durations + [1 for _ in chord_prog[len(durations):]]

    return durations

def complete_ntuples(chord_prog, ntuples=None):
    """
    Completes a sequence of ntuples
    """
    if not ntuples:
        ntuples = []

    ntuples = ntuples + [1 for _ in chord_prog[len(ntuples):]]
    
    return ntuples

def file_from_audio_segment(segment):
    folder = os.getcwd()
    folder = os.path.join(folder, "Rhodes Notes")
    clip_folder = Path(folder)
    out = segment.export(clip_folder / 'progression.wav', format = 'wav')
    out.close()
    return str(clip_folder / 'progression.wav')

# Returns a pydub AudioSegment of a voicing
def audio_from_voicing(chord,
                       voice=None,
                       ntuple = 4,
                       duration = 4, 
                       tempo = 120, 
                      ):
    """
    Returns an audio segment from a chord and an optional voicing
    Input(s): 
    chord, a chord
    voice, a voicing for the chord, defaults to a stacked thirds voicing
    ntuple, the tuplet
    duration, how long the chord is played
    tempo, the tempo
    A chord will be played for some duration with some tuplet at some tempo
    The duration is how long the chord is audible
    The tuplet is how long the chord+silence lasts
    Thus the total duration of the audio segment should be ntuplet*tempo
    """
    if voice is None:
        voice = random_initial_root_voicing(chord)

    folder = os.getcwd()
    folder = os.path.join(folder, "Rhodes Notes")
    clip_folder = Path(folder)
    clip_folder = Path(folder)
    clips = [clip_folder / f'2RhodesNote-0{i}.wav' 
             if i < 10 else clip_folder / f'2RhodesNote-{i}.wav' 
             for i in voice]
    sounds = [AudioSegment.from_file(clip, format="wav")-20 
              for i, clip in enumerate(clips)]
    
    #merges sound clips into one sound
    s = sounds[0]
    for i in range(len(sounds)-1):
        s = s.overlay(sounds[i+1]-2, position = i*50)
    s = s+10
    
    #converts tempo to ms/beat
    #needs exception handling or something
    total_dur = (60000/tempo)*(4/ntuple) #can't be less than the duration
    duration = dur_of_tuple(duration, tempo)
    duration = min(duration, 2000) #ensures duration does not excede piano wav length
    
    if total_dur-duration <= 0.0:
        s = s[:duration]
    else:
        # the duration is less than the total duration, thus create silence
        # the silence has length total_duration - duration
        silence = AudioSegment.silent(duration=(total_dur-duration))
        s = s[:duration]+silence
    return s

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
            play_voicing(chord_cur, voice = voicing_cur)
            voicing_prev = voicing_cur
        voicing1 = conditional_voicing2(voicing_cur, chord1)

def dur_of_tuple(ntuple, tempo):
    """
    Returns the duration (in milliseconds) of a ntuple at a given tempo
    """
    return (60000/tempo)*(4/ntuple)

def play_chord_progression2(chord_prog, voice_prog=None, ntuples=None, durations=None, tempo=120):
    """
    Inputs: 
    chord_prog, an array of names of chords
    voice_prog, an array of voicings for each chord
    ntuples, an array of tuplet sizes for each chord
    durations, an array of durations (in tuple form) for each chord
    tempo, the desired tempo for playback
    
    Outputs:
    array of audio segments, summing the array gives audio for the chord progression"""
    
    folder = os.getcwd()
    folder = os.path.join(folder, "Rhodes Notes")
    clip_folder = Path(folder)

    if type(chord_prog[0]) == str:
        chord_prog = [chord_finder(chord) for chord in chord_prog]

    if voice_prog is None:
        voice_prog = []
    if durations is None:
        durations = []
    if ntuples is None:
        ntuples = []
        
    if len(voice_prog) < len(chord_prog):
        voice_prog = voice_prog_completion(chord_prog, voice_prog)
    if len(ntuples) < len(chord_prog):
        ntuples = complete_ntuples(chord_prog, ntuples)
    if len(durations) < len(chord_prog):
        durations = complete_durations(chord_prog, durations)

    chord_audio = [audio_from_voicing(chord_prog[i], voice_prog[i], ntuples[i], durations[i], tempo) 
                   for i in range(len(chord_prog))]
    s = chord_audio[0]
    for i in range(len(chord_audio)-1):
        s = s + chord_audio[i+1]
    
    out = s.export(clip_folder / 'progression.wav', format='wav')
    out.close()

    #play_sound(str(clip_folder / 'progression.wav'))

def midi_file_from_chord_prog(chord_prog, voice_prog=None, durations=None, ntuples=None):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    if voice_prog is None:
        voice_prog = []
    if durations is None:
        durations = []
    if ntuples is None:
        ntuples = []

    if len(voice_prog) < len(chord_prog):
        voice_prog = voice_prog_completion(chord_prog, voice_prog)
    if len(ntuples) < len(chord_prog):
        ntuples = complete_ntuples(chord_prog, ntuples)
    if len(durations) < len(chord_prog):
        durations = complete_durations(chord_prog, durations)

    for j, voice in enumerate(voice_prog):
        #voice is a tuple of notes
        if j == 0:
            on_messages = [Message('note_on', note=note+24, velocity=100, time=0) for note in voice]
        else:
            on_messages = [Message('note_on', note=voice[0]+24, velocity=100, time=0) 
            if i == 0 else Message('note_on', note=note+24, velocity=100, time=0) 
            for i, note in enumerate(voice)]
        
        off_messages = [Message('note_off', note=voice[0]+24, velocity=100, time=1920) 
            if i == 0 else Message('note_off', note=note+24, velocity=100, time=0) 
            for i, note in enumerate(voice)]
        
        # turn on notes in voice
        for message in on_messages:
            track.append(message)

        # turn off notes in voice
        for message in off_messages:
            track.append(message)

    mid.save('chord_prog.mid')

def voice_visualization(chord: list, voicing: list=None) -> None:
    """
    Input(s):
    chord
    voicing, a voice of the chord
    Creates an image of a piano with the notes of the voicing played
    """
    if voicing is None:
        voicing = basic_voicing(chord)

    #voicing = voicing.tolist()

    w1 = 35.5 # width of white key at top
    w2 = 20.6 # width of black key = 23.5/13.7 = w1/w2 => w2 = 13.7/23.5*w1 = 1/1.72*w1
    w3 = 21# width of white key at top = w1 - 7/10*w2 = 22.75

    # first create 720p black background
    img = Image.new(mode="RGB", size=(1280, 720), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)

    # pairs [m, n], corresponds to the horizontal value m*w2+n*w3
    # used to get the coordinates of the black keys, the values were adjusted due to rounding errors
    black_coords = [[0, 1.1], [1, 2.2], [2.1, 4], [3.1, 5], [4.1, 6]]
    def pair_coords_to_coords(coord: list, w2: int, w3: int) -> int:
        return w2*coord[0]+w3*coord[1]

    def create_piano_segment(coords: list, num: int, w_k, b_k, w2, w3):
        """
        Inputs: 
        coords, a list of coordinates for the black keys
        num, a starting horizontal value for the octave
        w_k, the number of white keys
        b_k, the number of black keys
        Create one octave from C to B, which has total width 7*w1
        """
        
        # creates white notes
        for i in range(w_k):
            draw.rectangle((1+w1*i+(7*w1)*num, 540, 1+w1+w1*i+(7*w1)*num, 180), fill = "white", outline="black")

        # creates black notes
        for i in range(b_k):
            coord = pair_coords_to_coords(black_coords[i], w2, w3)
            coord = num*(7*w1) + coord
            draw.rectangle((coord, 180, coord+w2, 396), fill ="black", outline="gray")

    # draws 5 octaves, C to B
    for i in range(5):
        create_piano_segment(black_coords, i, 7, 5, w2, w3)

    # draws one final C key
    draw.rectangle((35*w1, 540, 36*w1, 180), fill = "white", outline="gray")

    black_notes = [1, 3, 6, 8, 10]
    white_note_dict = {'C': 0, 'D': 1, 'E': 2, 'F': 3, 'G': 4, 'A': 5, 'B': 6}
    black_note_dict = {'C#': 0, 'Eb': 1, 'F#': 2, 'G#': 3, 'Bb': 4}
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

    # highlights notes being played
    for voice in voicing:
        # voice is a numerical note
        note = root_dict[voice % 12] # char, like an E for example
        octave = voice // 12 # technically the octave offset by 2 from a standard piano
        
        
        if voice % 12 in black_notes:
            # draw note on black key
            font = ImageFont.truetype("./arial.ttf", 13)
            text = f"{note}\n {octave+2}"
            value = black_note_dict[note] # numerical black key in the octave
            black_coord = black_coords[value]
            coord = pair_coords_to_coords(black_coord, w2, w3)
            coord = coord + (octave)*7*w1
            draw.rectangle((coord, 396, coord+w2, 250), fill="pink", outline="blue")
            draw.text((coord+2, 323), text, (0, 0, 0), font=font)

        else:
            # fill in note on white key
            font = ImageFont.truetype("./arial.ttf", 18)
            text = f"{note}{octave+2}"
            value = white_note_dict[note] # numerical white key
            coord = (octave)*7*w1 + value*w1
            draw.rectangle((coord+1, 540, coord+w1+1, 396), fill = "pink", outline='blue')
            draw.text((coord+7, 468), text, (0, 0, 0), font=font)

    return img

def chord_prog_vis(chord_prog: list, voice_prog: list=None):
    """
    Inputs:
    chord_prog, a chord progression
    voice_prog, a sequence of voicing for each chord
    Outputs photos of each chord
    """
    copies = 60
    if type(chord_prog[0]) == str:
        chord_prog = [chord_finder(chord) for chord in chord_prog]
    
    if voice_prog is None:
        voice_prog = voice_prog_completion(chord_prog)
    if len(voice_prog) < len(chord_prog):
        voice_prog = voice_prog_completion(chord_prog, voice_prog=voice_prog)

    play_chord_progression2(chord_prog=chord_prog, voice_prog=voice_prog)

    #img_folder = os.getcwd()
    #img_folder = os.path.join(img_folder, "imgs")
    img_folder = Path("imgs")
    #audio_folder = os.getcwd()
    #audio_folder = os.path.join(audio_folder, "Rhodes Notes")
    audio_folder = Path("Rhodes Notes")

    with open("jpgs_text.txt", "w") as file:
        for i, voicing in enumerate(voice_prog):
            img = voice_visualization(chord_prog[i], voicing=voicing)
            for j in range(copies):
                img_path = f"imgs/chord{i}copy{j+1}.jpg"
                img.save(img_path)
                if i == len(voice_prog)-1 and j == copies-1:
                    file.write(f"file \'{img_path}\'")
                else:
                    file.write(f"file \'{img_path}\'\n")




    input_audio = ffmpeg.input(audio_folder / "progression.wav")
    (
        ffmpeg
        .input('jpgs_text.txt', f='concat', safe='0', r='30')
        .output('movie.mp4', vcodec='libx264')
        .run(overwrite_output=True)
    )
    input_video = ffmpeg.input('movie.mp4')
    ffmpeg.concat(input_video, input_audio, v=1, a=1).output('movie.mp4').run(overwrite_output=True)

chord_prog_vis(["Bsus2 7", "Em #5 11", "C#m b13", "G#sus2 maj7"])