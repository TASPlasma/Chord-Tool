import pandas as pd
import numpy as np
import os
import cv2
import pygame
from pedalboard import Pedalboard, Chorus, Reverb, Delay, Convolution, Compressor, Gain, HighpassFilter, LowpassFilter, Limiter
from pedalboard.io import AudioFile
from moviepy.editor import ImageSequenceClip
from moviepy.editor import AudioFileClip
from PIL import Image, ImageDraw, ImageFont, ImageChops
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
            name += chord_array_dict[i][(chord[i]+shift) % 12]
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
def note_dict():
    note_dictionary = {
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
    return note_dictionary

def alph_note_dict():
    alph_note_dictionary = {
        'C': 0,
        'C#': 1,
        'D': 2,
        'Eb': 3,
        'E': 4,
        'F': 5,
        'F#': 6,
        'G': 7,
        'G#': 8,
        'A': 9,
        'Bb': 10,
        'B': 11
    }
    return alph_note_dictionary

def num_notes_to_alph(num_notes):
    return [note_dict()[num_note] for num_note in num_notes]

def alph_notes_to_num(alph_notes):
    return [alph_note_dict()[alph_note] for alph_note in alph_notes]

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

def scale_dissonance(notes):
    n = len(notes)

    scale_dissonance_value = 0

    # exception, augmented chords have dissonance
    for note in notes:
        if ((note+4 % 12) in notes) and ((note+8 % 12) in notes):
            scale_dissonance = 1
    
    for i in range(n):
        for j in range(i, n):
            if j > i:
                diss = dissonance(notes[i], notes[j])
                scale_dissonance_value += diss

    return scale_dissonance_value

def chord_dissonance(chord):
    """Inputs: chord, a chord with which the subjective dissonance will
    be measured.

    Outputs: a subjective value for the dissonance of an entire chord
    """
    notes = keys_of_chord(chord)

    return scale_dissonance(notes)

def alph_scale_dissonance(alph_notes):
    notes = [alph_note_dict()[alph] for alph in alph_notes]
    return scale_dissonance(notes)

def degree_of_chord(chord):
    dis = chord_dissonance(chord)
    ranges = [[0, 0], [1, 2], [3, 7], [8, 11], [12, 15], [16, 1000000000]]
    degree = -1
    for i, interval in enumerate(ranges):
        if dis >= interval[0] and dis <= interval[1]:
            degree = i+1
            return degree
    if degree == -1:
        print("Chord has dissonance higher than max and degree set to ", 7)
    return 7
        

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
        oct_cond = 1 # raises E, F, F# bass notes by an octave
    
    l = len(keys)
    if chord[2] != 2: # if 5th is not natural 5th
        # chooses octave of 4 for b5th and #5th
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

# pygame.mixer.init()
def play_sound(filename):
    with open(filename) as f:
        pygame.mixer.init()
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        # pygame.mixer.music.get_endevent()
        # pygame.event.wait()
    while pygame.mixer.music.get_busy(): # and cond: # check if the file is playing
        pass
    pygame.mixer.music.unload()
    pygame.mixer.quit()
    return

def stop_sound(filename):
    pygame.mixer.music.stop()
    pygame.mixer.music.unload()
    

def play_voicing(
    chord, 
    voice=None,
    degree=None
    ):
    if voice is None:
        voice = random_initial_root_voicing(chord)
    if degree is None:
        degree = degree_of_chord(chord)

    folder = os.getcwd()
    folder = os.path.join(folder, "Rhodes Notes")
    clip_folder = Path(folder)
    sorted_voice = sorted(voice)
    clips = [clip_folder / f'RhodesNote-0{i}.wav' 
             if i < 10 else clip_folder / f'RhodesNote-{i}.wav' 
             for i in sorted_voice]
    sounds = [AudioSegment.from_file(clip, format="wav")-20 
              for i, clip in enumerate(clips)]
    s = sounds[0]
    for i in range(len(sounds)-1):
        s=s.overlay(sounds[i+1]-2, position = i*30+i*degree*5) # spreads out notes
    
    combined = s+8
    
    out = combined.export(clip_folder / 'temp.wav', format = 'wav')
    out.close()
    
    play_sound(str(clip_folder / 'temp.wav'))

# Function that returns index of input chord name
def chord_finder(chord_name):
    df = chords.loc[:, ~chords.columns.isin(['name', 'dissonance'])]
    return df.loc[(chords['name']==chord_name)].values.flatten()

def transpose_chord(chord, shift):
    """
    Takes in a chord and a transposition shift value and transposes that chord by that shift
    e.g. [0, 1, 1, 0, 0, 0, 0], 4 |-> [4, 1, 1, 0, 0, 0, 0]
    """
    if type(chord[0]) == str:
        chord = chord_finder(chord)
    
    shift = shift % 12
    output = [chord[i] + shift if i == 0 else chord[i] for i in range(len(chord))]
    return output

def transpose_chord_name(chord, shift):
    if type(chord) == str:
        chord = chord_finder(chord)
    
    return chord_name(transpose_chord(chord, shift))

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

def inversion(voicing):
    inverted = [voice for voice in voicing]
    copy = sorted(inverted)
    
    argsorted = np.argsort(inverted)
    inverted[argsorted[1]] += 12

    return inverted

def un_inversion(voicing):

    un_inverted = [voice for voice in voicing]
    copy = sorted(un_inverted)
    
    argsorted = np.argsort(un_inverted)
    un_inverted[argsorted[-1]] -= 12

    return un_inverted

def kth_inversion(voicing, k):
    inverted = voicing
    for i in range(k):
        inverted = inversion(inverted)

    return voice_correction(voicing)

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
        avg = max
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

def voice_prog_completion(chord_prog, voice_prog: list=None):
    """
    Input(s): 
    chord_prog, a chord progression
    voice_prog, an incomplete sequence of voicings for chord_prog
    Outputs:
    voice_prog, the completed sequence of voicings for chord_prog
    """
    if type(chord_prog[0]) == str:
        chord_prog = [chord_finder(chord) for chord in chord_prog]
    
    if voice_prog is None:
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

def file_from_audio_segment(segment, filename, inst="Rhodes", effects=True):
    """
    Takes a pydub audio segment, exports it to the specified file in the path
    where audio files are saved
    Adds audio effects
    """
    inst_str = inst + " Notes" # for bass or rhodes
    folder = os.getcwd()
    folder = os.path.join(folder, inst_str)
    clip_folder = Path(folder)
    out = segment.export(clip_folder / filename, format = 'wav')
    out.close()

    if effects:
        add_effects(f'{clip_folder}\\{filename}', inst)
    return str(clip_folder / filename)

# Returns a pydub AudioSegment of a voicing
def audio_from_voicing(chord,
                       voice=None,
                       ntuple = 4,
                       duration = 4, 
                       tempo = 120, 
                       degree = None
                      ):
    """
    Returns an audio segment from a chord and an optional voicing
    Input(s): 
    chord, a chord
    voice, a voicing for the chord, defaults to a random voicing
    ntuple, the tuplet (default a quarter note)
    duration, how long the chord is played (presented as a tuple, default a quarter note)
    tempo, the tempo
    A chord will be played for some duration with some tuplet at some tempo
    The duration is how long the chord is audible
    The tuplet is how long the chord+silence lasts
    Thus the total duration of the audio segment should be ntuplet*tempo
    """
    if voice is None:
        voice = random_initial_root_voicing(chord)
    if degree is None:
        degree = degree_of_chord(chord)

    folder = os.getcwd()
    folder = os.path.join(folder, "Rhodes Notes")
    clip_folder = Path(folder)
    clip_folder = Path(folder)
    sorted_voice = sorted(voice)
    clips = [clip_folder / f'RhodesNote-0{i}.wav' 
             if i < 10 else clip_folder / f'RhodesNote-{i}.wav' 
             for i in sorted_voice]
    sounds = [AudioSegment.from_file(clip, format="wav")-20 
              for i, clip in enumerate(clips)]
    
    #merges sound clips into one sound
    s = sounds[0]
    for i in range(len(sounds)-1):
        s=s.overlay(sounds[i+1]-6, position = i*35+i*degree*5) # spreads out notes
    s = s+13
    
    #converts tempo to ms/beat
    #needs exception handling or something
    total_dur = dur_of_tuple(ntuple, tempo) #can't be less than the duration
    duration = dur_of_tuple(duration, tempo) # converts the tuplet format of duration into milliseconds
    duration = min(duration, 2000) #ensures duration does not excede piano wav length
    
    if total_dur-duration <= 0.0:
        s = s[:duration]
    else:
        silence = AudioSegment.silent(duration=(total_dur-duration))
        s = s[:duration]+silence
    return s

def add_effects(file, inst="Rhodes"):

    with AudioFile(file, 'r') as f:
        audio = f.read(f.frames)
        samplerate = f.samplerate

    if inst == "Rhodes":
        board = Pedalboard([
            Gain(gain_db=5),
            Compressor(threshold_db=-5, ratio=2, attack_ms=25, release_ms=400),
            # Limiter(threshold_db=2),
            # Gain(gain_db=8),
            HighpassFilter(cutoff_frequency_hz=60),
            LowpassFilter(cutoff_frequency_hz=2000),
            Delay(delay_seconds=0.5, feedback=0.3, mix=0.2),
            Reverb(room_size=0.6, wet_level=0.4)
            ])
        
    elif inst == "Bass":
        board = Pedalboard([
            Gain(gain_db=0.8),
            Compressor(threshold_db=-5, ratio=4, attack_ms=25, release_ms=400),
            Limiter(threshold_db=2),
            # Gain(gain_db=8),
            # HighpassFilter(cutoff_frequency_hz=60),
            LowpassFilter(cutoff_frequency_hz=2000),
            Delay(delay_seconds=0.5, feedback=0.3, mix=0.03),
            Reverb(room_size=0.5)
            ])
    elif inst == "Drum":
        board = Pedalboard([
            # Gain(gain_db=5),
            # Compressor(threshold_db=-5, ratio=1.1, attack_ms=25, release_ms=400),
            # Limiter(threshold_db=10),
            # Gain(gain_db=8),
            HighpassFilter(cutoff_frequency_hz=60),
            # LowpassFilter(cutoff_frequency_hz=6000),
            Delay(delay_seconds=0.5, feedback=0.3, mix=0.1),
            Reverb(
            room_size=0.3,
              damping=0.8,
                wet_level=0.11,
                  dry_level=0.4,
                    width=1,
                      freeze_mode=0
                      )
            ])

    effected = board(audio, samplerate)

    with AudioFile(file, 'w', samplerate, effected.shape[0]) as f:
        f.write(effected)

def bassline(chord_prog, tempo=120):
    """
    simple root -> octave bassline
    returns audiosegment of this bassline
    """

    if type(chord_prog[0]) == str:
        chord_prog = [chord_finder(chord) for chord in chord_prog]
    folder = os.getcwd()
    folder = os.path.join(folder, "Bass Notes")
    clip_folder = Path(folder)
    clip_folder = Path(folder)
    
    # voicings for each chord in prog.
    voices = [basic_voicing(chord)
              for chord in chord_prog] 
    
    # root for each voicing
    roots = [voicing[0] for voicing in voices]
    # octave for each voicing
    octaves = [voicing[0]+12 for voicing in voices]

    # Bass Note file paths for all roots
    root_clips = [clip_folder / f'BassNote-0{i}.wav' 
                  if i < 10 else clip_folder / f'BassNote-{i}.wav'
                  for i in roots]
    

    octave_clips = [clip_folder / f'BassNote-{i}.wav'
                  for i in octaves]

    root_sounds = [AudioSegment.from_file(root_clip, format="wav")-5
                   for root_clip in root_clips]
    octave_sounds = [AudioSegment.from_file(octave_clip, format="wav")-5
                   for octave_clip in octave_clips]
    
    # used to add staccato
    f = lambda x: note_with_pause(x, 5, 2, tempo, 16)

    for i in range(len(voices)):
        if i == 0:
            s = f(root_sounds[i])
            s = s + f(octave_sounds[i])
        else:
            root_sound = f(root_sounds[i])
            octave_sound = f(octave_sounds[i])
            s = s + root_sound
            s = s + octave_sound
        
    return s

def add_drums(chord_prog, tempo=120):
    """
    Creates a drum beat audio segment
    """
    folder = os.getcwd()
    folder = os.path.join(folder, "Drum Notes")
    clip_folder = Path(folder)
    clip_folder = Path(folder)

    kick_sounds = [AudioSegment.from_file(clip_folder / f'Kick.wav')-2]

    snare_sounds = [AudioSegment.from_file(clip_folder / f'Snare{i}.wav')-2 for i in range(1, 7)]

    hihat_sounds = [AudioSegment.from_file(clip_folder / f'HiHat{i}.wav') for i in range(1, 11)]

    ride_sounds = [AudioSegment.from_file(clip_folder / f'Ride{i}.wav') for i in range(1, 2)]

    f = lambda x: note_with_pause(x, 4, 4, 2*tempo, 32)

    drum_segment = f(kick_sounds[0])

    drum_segment = drum_segment.overlay(f(hihat_sounds[3]))

    drum_segment = drum_segment + f(hihat_sounds[3])

    drum_segment = drum_segment + f(hihat_sounds[3]).overlay(f(snare_sounds[0]))

    drum_segment = drum_segment + f(hihat_sounds[3])

    for i in range(2*(len(chord_prog))):
        drum_segment = drum_segment + f(hihat_sounds[3]).overlay(f(kick_sounds[0]))
        drum_segment = drum_segment + f(hihat_sounds[3])
        drum_segment = drum_segment + f(hihat_sounds[3]).overlay(f(snare_sounds[0]))
        drum_segment = drum_segment + f(hihat_sounds[3])

    return drum_segment

def apply_fadeout(segment, frac=16):
    """
    segment: an audio segment
    applies a fadeout to a pydub audio segment
    """
    audio_array = np.array(segment.get_array_of_samples(), dtype=np.float32) / 32768.0

    fadeout_duration = int(segment.duration_seconds * 1000 / frac)

    fadeout_samples = int(fadeout_duration * segment.frame_rate / 1000)

    fadeout_curve = np.concatenate(
        (np.ones(len(audio_array) - fadeout_samples), np.linspace(1.0, 0.0, fadeout_samples))
    )

    # Apply the fadeout curve to the audio array
    if segment.channels == 2 and False:
        # Stereo audio
        audio_array[::2] = np.multiply(audio_array[::2], fadeout_curve)
        audio_array[1::2] = np.multiply(audio_array[1::2], fadeout_curve)
    else:
        # Mono audio
        audio_array = np.multiply(audio_array, fadeout_curve)

    audio_array = np.int16(audio_array * 32767.0)

    faded_segment = AudioSegment(
        audio_array.tobytes(),
        frame_rate=segment.frame_rate,
        sample_width=segment.sample_width,
        channels=segment.channels
    )

    return faded_segment

def note_with_pause(segment, duration, ntuple, tempo, frac):
    """
    Takes in an audio segment with:
    -an intended duration that the segment should be audible
    -an intended total length that the segment should last (the ntuplet)
    duration is of the form of a tuplet and is converted to ms
    -frac is an argument used how deep into the clip to apply the fadeout
    """
    duration = dur_of_tuple(duration, tempo)
    total_dur = dur_of_tuple(ntuple, tempo)

    if total_dur-duration <= 0.0:
        segment = apply_fadeout(segment[:duration], frac)
    else:
        silence = AudioSegment.silent(duration=(total_dur-duration))
        segment = apply_fadeout(segment[:duration], frac) + silence

    return segment

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

def play_chord_progression2(chord_prog, voice_prog=None, ntuples=None, durations=None, tempo=120, backing=False):
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

    rhodes_file = file_from_audio_segment(s, "rhodes.wav", inst="Rhodes")
    s = AudioSegment.from_file(rhodes_file, format='wav')


    if backing:
        bass_segment = bassline(chord_prog, tempo=tempo)
        bass_file = file_from_audio_segment(bass_segment, "bass.wav", inst="Bass")
        bass_segment = AudioSegment.from_file(bass_file, format='wav')
        drum_segment = add_drums(chord_prog, tempo=tempo)
        drum_file = file_from_audio_segment(drum_segment, "drums.wav", inst="Drum")
        drum_segment = AudioSegment.from_file(drum_file, format='wav')

        s = s.overlay(bass_segment.overlay(drum_segment))
    
    return s


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
            draw.rectangle((1+w1*i+(7*w1)*num, 540, w1+w1*i+(7*w1)*num-1, 180), fill = "white", outline="black")

        # creates black notes
        for i in range(b_k):
            coord = pair_coords_to_coords(black_coords[i], w2, w3)
            coord = num*(7*w1) + coord
            draw.rectangle((coord, 180, coord+w2, 396), fill ="black", outline="black")

    # draws 5 octaves, C to B
    for i in range(5):
        create_piano_segment(black_coords, i, 7, 5, w2, w3)

    # draws one final C key
    draw.rectangle((35*w1+1, 540, 36*w1-2, 180), fill = "white", outline="black")

    black_notes = [1, 3, 6, 8, 10]
    white_note_dict = {'C': 0, 'D': 1, 'E': 2, 'F': 3, 'G': 4, 'A': 5, 'B': 6}
    black_note_dict = {u'C\u266F': 0, u'E\u266D': 1, u'F\u266F': 2, u'G\u266F': 3, u'B\u266D': 4}
    root_dict = {
    0: 'C',
    1: u'C\u266F',
    2: 'D',
    3: u'E\u266D',
    4: 'E',
    5: 'F',
    6: u'F\u266F',
    7: 'G',
    8: u'G\u266F',
    9: 'A',
    10: u'B\u266D',
    11: 'B'
    }

    top_text = f'Chord: "{chord_name(chord)}"'
    bottom_text = "Voicing: ("
    # highlights notes being played
    for voice in voicing:
        # voice is a numerical note
        note = root_dict[voice % 12] # char, like an E for example
        octave = voice // 12 # technically the octave offset by 2 from a standard piano
        bottom_text += f"{note}{octave+2}, "
        
        
        if voice % 12 in black_notes:
            # draw note on black key
            font = ImageFont.truetype("DejaVuSans.ttf", 13)
            text = f"{note}\n {octave+2}"
            
            value = black_note_dict[note] # numerical black key in the octave
            black_coord = black_coords[value]
            coord = pair_coords_to_coords(black_coord, w2, w3)
            coord = coord + (octave)*7*w1
            draw.rectangle((coord+2, 394, coord+w2-2, 250), fill="pink", outline="blue")
            draw.text((coord+2, 323), text, (0, 0, 0), font=font)

        else:
            # fill in note on white key
            font = ImageFont.truetype("DejaVuSans.ttf", 18)
            text = f"{note}{octave+2}"
            value = white_note_dict[note] # numerical white key
            coord = (octave)*7*w1 + value*w1
            draw.rectangle((coord+2, 539, coord+w1-2, 396), fill = "pink", outline='blue')
            draw.text((coord+7, 468), text, (0, 0, 0), font=font)


    bottom_text = bottom_text[:-2]+")"
    bottom_font = ImageFont.truetype("DejaVuSans.ttf", 34)
    width_b, height_b = draw.textsize(bottom_text, font=bottom_font)
    draw.text(((1280-width_b)/2, 5*(720-height_b)/6), bottom_text, (255, 255, 255), font=bottom_font)
    
    width_t, height_t = draw.textsize(top_text, font=bottom_font)
    draw.text(((1280-width_t)/2, (720-height_t)/6), top_text, (255, 255, 255), font=bottom_font)
    #img.show()

    return img

def verify_video_integrity(file):
    vidcap = cv2.VideoCapture(file)
    count = 1
    count2 = 0
    fps = int(vidcap.get(cv2.CAP_PROP_FPS))
    success, prev_img = vidcap.read()
    failure = False

    folder = os.getcwd()
    folder = os.path.join(folder, "ver_frames")
    ver_frame = Path(folder)

    for f in os.listdir(ver_frame):
        os.remove(os.path.join(ver_frame, f))

    remainder = 59
    while success:
        success, img = vidcap.read()
        print('read a new frame:', success)
        # if count == remainder:
        #     cv2.imwrite('ver_frames\\frame%d.jpg'%1, img)
        if count%(2*fps) == remainder: # and count > remainder: # executes every 60 frames
            cv2.imwrite('ver_frames\\frame%d.jpg'%count, img)
        
        count+=1
        
        

    n = len(os.listdir('ver_frames')) # number of verification frames
    img_files = [f'frame{60*i+remainder}.jpg' for i in range(n)] # ordered list of images

    prev_img = img_files[0] # should be frame1
    
    for i, img in enumerate(img_files[1:]):
        
        print("Previous frame: ", prev_img, "current frame: ", img)
        prev_img = Image.open(ver_frame / prev_img)
        img = Image.open(ver_frame / img)
        value = np.sum(np.array(ImageChops.difference(prev_img, img).getdata()))
        # ImageChops.difference(prev_img, img).show()
        print("Value: ", value)
        
        if np.sum(np.array(ImageChops.difference(prev_img, img).getdata())) <= 80000:
            print("Execute order 66")
            failure = True
        prev_img = img_files[i+1]
    
    if failure:
        print("Video failed integrity check or progression contains repeat chord")
    else:
        print("Video file successfully exported")
    return

def chord_prog_vis(chord_prog: list, voice_prog: list=None):
    """
    Inputs:
    chord_prog, a chord progression
    voice_prog, a sequence of voicing for each chord
    Outputs photos of each chord
    """
    
    if type(chord_prog[0]) == str:
        chord_prog = [chord_finder(chord) for chord in chord_prog]
    
    if voice_prog is None:
        voice_prog = voice_prog_completion(chord_prog)
    if len(voice_prog) < len(chord_prog):
        voice_prog = voice_prog_completion(chord_prog, voice_prog=voice_prog)

    s = play_chord_progression2(chord_prog=chord_prog, voice_prog=voice_prog)
    _1 = file_from_audio_segment(s, 'progression.wav')

    #folder = os.getcwd()
    #img_folder = os.path.join(img_folder, "imgs")
    #img_folder = Path("imgs")
    #audio_folder = os.path.join(folder, "Rhodes Notes")
    audio_folder = Path("Rhodes Notes")
    img_list = []

    with open("jpgs_text.txt", "w") as file:
        for i, voicing in enumerate(voice_prog):
            img = voice_visualization(chord_prog[i], voicing=voicing)
            img_path = f"imgs/chord{i}.jpg"
            img.save(img_path)
            chord_list = [img_path for _ in range(60)]
            img_list += chord_list

    audio_path = f"{audio_folder}/progression.wav"
    clip = ImageSequenceClip(img_list, fps=30)
    audio = (AudioFileClip(audio_path)
                .set_duration(clip.duration))
    clip = clip.set_audio(audio)
    clip.write_videofile("chord_progression.mp4", fps=30, codec='libx264', audio_codec='aac')

    verify_video_integrity("chord_progression.mp4")

def scales_of_chord(chord):
    notes = keys_of_chord(chord)

# segment = play_chord_progression2(["Em maj7 b9 11 13", "F#m 7 9 11 13", "Am 7 9 11 13", "F# maj7 9 #11 13"])
# file = file_from_audio_segment(segment, 'progression.wav')
# play_sound(file)
# chord_prog_vis(["Ebm maj7 9", "Ebm maj7 9", "D maj7 #11 13", "D maj7 #11 13"])

# chord = chord_finder("C 7 #11")
# print(f"Basic voicing of {chord_name(chord)}: {basic_voicing(chord)}, first inversion: {inversion(basic_voicing(chord))}")

# chord_prog = ["Am 7 9", "Cm 7 9"]
# rhodes_segment = play_chord_progression2(chord_prog, tempo=100)
# rhodes_file = file_from_audio_segment(rhodes_segment, "rhodes.wav", inst="Rhodes")
# rhodes_segment = AudioSegment.from_file(rhodes_file, format='wav')
# bass_segment = bassline(chord_prog, 100)
# bass_file = file_from_audio_segment(bass_segment, "bass.wav", inst="Bass")
# bass_segment = AudioSegment.from_file(bass_file, format='wav')
# drum_segment = add_drums(chord_prog, 100)
# drum_file = file_from_audio_segment(drum_segment, "drums.wav", inst="Drum")
# drum_segment = AudioSegment.from_file(drum_file, format='wav')
# segment = rhodes_segment.overlay(bass_segment.overlay(drum_segment))
# file = file_from_audio_segment(segment, "band.wav", inst="Bass", effects=False)
# file2 = file_from_audio_segment(drum_segment, "drumbeat.wav", inst="Bass")
# play_sound(file)
