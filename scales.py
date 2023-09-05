import numpy as np
import pandas as pd
import audio
import random
from PIL import Image, ImageDraw

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

def filename_from_num(num):
    naming_dict = {
        6: "hexa",
        7: "hepta",
        8: "octa",
        9: "nona",
        10: "deca"
    }

    return naming_dict[num] + "tonic_scales.txt"

def create_data(num):
    """
    Takes in a number num, outputs all scales (in step sequence form)
    that has num many notes
    """

    steps = [1, 2, 3]

    num_steps = [steps for i in range(num)]

    step_seq_data = cartesian_coord(*num_steps)

    step_seq_data = pd.DataFrame(
        step_seq_data,
        columns = [f'Interval {i+1}' for i in range(num)]
    )

    step_seq_data = step_seq_data[step_seq_data.sum(axis=1) == 12]

    filename = filename_from_num(num)

    step_seq_data.to_csv(
        filename,
        header='infer',
        index=None,
        sep=' ',
        mode='w'
    )

def step_seq_data_to_notes(num):
    """
    Takes in a number num
    and outputs a dataframe of all scales (in list of notes form) that have num many notes
    """
    filename = filename_from_num(num)
    column_names = [f'Interval {i+1}' for i in range(num)]
    step_seqs_df = pd.read_csv(filename, sep=' ', header=None, names=column_names, skiprows=1)

    # Initialize an empty list to store transformed rows
    transformed_rows = []

    # Apply the step_seq_to_scale function to each row with values 0 to 11
    for index, row in step_seqs_df.iterrows():
        for value in range(12):
            step_seq = row.tolist()
            step_seq_obj = StepSequence(step_seq)
            scale = step_seq_obj.step_seq_to_scale(value)
            notes = scale.alph_notes()
            transformed_rows += [notes]

    # Create a DataFrame from the transformed rows
    transformed_df = pd.DataFrame(transformed_rows, columns=[f'note {i+1}' for i in range(num)])

    # Remove duplicate rows
    transformed_df.drop_duplicates(inplace=True)

    # Reset the index of the DataFrame
    transformed_df.reset_index(drop=True, inplace=True)

    transformed_df['dissonance'] = transformed_df.apply(audio.alph_scale_dissonance, axis=1)
    return transformed_df

class StepSequence:
    """
    step_seq: a list of step sequences
    """
    def __init__(self, step_seq):
        self.step_seq = step_seq

    def step_seq_to_scale(self, root):
        """
        takes in a root note and a step sequence and
        produces a scale object
        """
        note = root
        notes = [root]
        for step in self.step_seq[:-1]:
            note += step
            note = note % 12
            notes += [note]

        # notes = list(set(notes))

        return Scale(notes)

class Scale:
    """
    scale: list of notes
    scale steps: list of elements in {1, 2} that sum to 12 indicating
    the step size starting at root, paired with a root
    e.g. C major: (0, [2, 2, 1, 2, 2, 2, 1])
    """
    def __init__(self, scale=[]):
        self.scale = scale

    def alph_notes(self):
        """
        Uses self.scale (which is a list of notes) and outputs the list of
        numerical notes
        """
        note_dict = audio.note_dict()
        return [note_dict[note] for note in sorted(self.scale)]

    def chords_of_scale(self):
        """
        Returns list of all chord names whose notes are
        in the list of notes in the scale
        """
        pass

    def alph_notes_to_num(self, alph_notes):
        """
        Input: alph_notes, a list of alphabetical musical notes, e.g. C# instead of 1
        
        Output: num_notes, the numerical musical notes of alph_notes
        """
        alph_note_dict = audio.alph_note_dict()
        return [alph_note_dict[alph_note] for alph_note in alph_notes]

    def is_mode_of(self, scale2):
        """
        A scale is a mode of another if they contain the same notes
        Returns true if scale2 is a mode of self.scale
        """
        note_set1 = set(self.notes_of_scale(self.scale))
        note_set2 = set(self.notes_of_scale(scale2))
        return note_set1 == note_set2
    
    def notes_to_coords(self, notes):
        """
        notes is a list of numerical notes
        returns a list of coordinates on a guitar fretboard that correspond to that note
        for example if note == 'E', coords = [(1, 12), (2, 7), (3, 2), (4, 9), (5, 5), (6, 12)]
        """
        notes = [(note % 12) for note in notes]

        coords = []
        for note in notes:
            for i in range(6):
                string = i+1
                if i <= 3:
                    fret1 = (note-5+7*i) % 12 + 1
                    fret2 = ((note-5+7*i) % 12 + 13) % 24
                    coords_on_string = list(set([(string, fret1), (string, fret2)]))
                else:
                    fret1 = (note-5+7*i) % 12 + 2
                    fret2 = ((note-5+7*i) % 12 + 14) % 24
                    coords_on_string = list(set([(string, fret1), (string, fret2)]))

                coords += coords_on_string

                coords = [coord for coord in coords if coord[1] in range(1,16)]

        return coords
    
    def piano_scale_visual(self):
        pass

    def guitar_scale_visual(self, x=15, alph_notes=[]):
        """
        Inputs: alph_notes, a list of alphabetical musical notes
        returns an image of a scale on a guitar
        """
        # Make sure the input value is at least 15
        x = max(x, 15)

        fret_indicators = [(3.5, 3), (3.5, 5), (3.5, 7), (3.5, 9), (2.5, 12), (4.5, 12), (3.5, 15)]

        dots = self.notes_to_coords(self.alph_notes_to_num(alph_notes))

        # Grid parameters
        rows = 5
        columns = x
        cell_size = 20
        grid_width = columns * cell_size
        grid_height = rows * cell_size

        # Padding values
        padding_x = 20
        padding_y = 20

        # Calculates image dimensions with padding
        image_width = grid_width + 2 * padding_x
        image_height = grid_height + 2 * padding_y

        # Creates a blank image with padding
        image = Image.new('RGB', (image_width, image_height), color='white')
        draw = ImageDraw.Draw(image)

        # Draw the outer rectangle
        draw.rectangle([(padding_x, padding_y), (image_width - padding_x, image_height - padding_y)], outline='black', width=2)

        # Draws grid lines with adjusted coordinates for padding
        for i in range(1, rows):
            y = padding_y + i * cell_size
            draw.line((padding_x, y, grid_width + padding_x, y), fill='black', width=2)

        for j in range(1, columns):
            x = padding_x + j * cell_size
            draw.line((x, padding_y, x, grid_height + padding_y), fill='black', width=2)

        # Draws centered circular dots on selected lines (strings) and boxes (frets)
        def circle_at_coords(coords, radius, color, indicator=False):
            for coord in coords:
                string, fret = coord
                x = padding_x + (fret - 0.5) * cell_size
                y = padding_y + (6-string) * cell_size
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)

        circle_at_coords(fret_indicators, 5 * cell_size/50, "black", indicator=True)

        circle_at_coords(dots, 9 * cell_size/50, "green")

        return image

    def play_scale(self):
        pass

# print(my_obj.chord)
# print(my_obj.scales_of_chord())

class Chord:
    """
    
    """
    def __init__(self, chord_name):
        self.chord_name = chord_name
        self.chord = audio.chord_finder(chord_name)
        self.dissonance = audio.chord_dissonance(self.chord)
        self.voicing = audio.basic_voicing(self.chord)

    def chord_completions(self):

        input_chord = self.chord

        chord = {
            'root': input_chord[0],
            '3rd': input_chord[1],
            '5th': input_chord[2],
            '7th': input_chord[3],
            '9th': input_chord[4],
            '11th': input_chord[5],
            '13th': input_chord[6]
        }

        chords = audio.chords
        completed_chords = chords[(chords['root'] == chord['root']) & (chords['5th'] == chord['5th'])]

        if chord['3rd'] in [2, 3]:
            completed_chords = completed_chords[completed_chords['3rd'] == chord['3rd']]
        elif (chord['3rd'] == 1): # triggers when chord has sus2
            condition = (
                ((completed_chords['3rd'] != 4)) &
                ((completed_chords['9th'] == chord['9th']) & (completed_chords['3rd'] == 1)) |
                ((completed_chords['9th'] == chord['9th']) & (completed_chords['3rd'].between(2, 3)) &
                 (not (chord['9th'] in [0, 1]))) |
                (((completed_chords['9th'] == 2) & (completed_chords['3rd'].between(2, 3)) &
                 (chord['9th'] == 0)))
            )
            completed_chords = completed_chords[condition]
        elif (chord['3rd'] == 4): # triggers when chord has sus4
            condition = (
                ((completed_chords['3rd'] != 1)) &
                ((completed_chords['11th'] == chord['11th']) & (completed_chords['3rd'] == 4)) |
                ((completed_chords['11th'] == chord['11th']) & (completed_chords['3rd'].between(2, 3)) &
                 (not (chord['11th'] in [0, 2]))) |
                (((completed_chords['11th'] == 1) & (completed_chords['3rd'].between(2, 3)) &
                 (chord['11th'] == 0))) |
                (((completed_chords['9th'] == 2) & (completed_chords['3rd'].between(1, 3)) &
                   completed_chords['11th'] == 1)) |
                (((completed_chords['9th'] == 0) & (completed_chords['3rd'].between(1, 1)) &
                   completed_chords['11th'] == 1))

            )
            completed_chords = completed_chords[condition]

        if chord['7th'] == 0:
            completed_chords = completed_chords[completed_chords['7th'] > 0]

        if chord['9th'] == 0 and chord['3rd'] != 1:
            completed_chords = completed_chords[completed_chords['9th'] > 0]

        if (chord['11th'] == 0) and (chord['3rd'] != 4):
            completed_chords = completed_chords[completed_chords['11th'] > 0]

        if chord['13th'] == 0:
            completed_chords = completed_chords[completed_chords['13th'] > 0]

        if chord['7th'] > 0:
            completed_chords = completed_chords[completed_chords['7th'] == chord['7th']]

        if (chord['9th'] > 0) and (chord['3rd'] != 1):
            completed_chords = completed_chords[completed_chords['9th'] == chord['9th']]

        if (chord['11th'] > 0) and (chord['3rd'] != 4):
            completed_chords = completed_chords[completed_chords['11th'] == chord['11th']]

        if chord['13th'] > 0:
            completed_chords = completed_chords[completed_chords['13th'] == chord['13th']]

        return completed_chords
    
    def find_matching_scales(self):
        chord_keys = self.alph_notes_of_chord()

        heptatonic_scales = step_seq_data_to_notes(7)
        
        # Filter rows where chord keys are a subset
        matching_scales = heptatonic_scales[
            heptatonic_scales.apply(lambda row: set(chord_keys).issubset(row[:-1]), axis=1)
        ]
        
        return matching_scales
    
    def notes_of_chord(self):
        return audio.keys_of_chord(self.chord)
    
    def alph_notes_of_chord(self):
        note_dict = audio.note_dict()
        keys = sorted(self.notes_of_chord())
        output = [note_dict[key] for key in keys]
        return output
    
    def minimal_diss_chords(self, chord_df):
        """
        Returns all completed chords that have minimal dissonance
        """
        # Find the minimum dissonance value
        min_diss = chord_df['dissonance'].min()

        # Filter the DataFrame to get all rows with minimal dissonance
        min_diss_chords = chord_df.loc[chord_df['dissonance'] == min_diss]

        return min_diss_chords

    def random_choice2(self, scale_df=None):
        if not scale_df:
            scale_df = self.find_matching_scales()

        random_index = random.randint(0, len(scale_df) - 1)

        selected_row = scale_df.iloc[random_index, :-1]
        alph_notes = selected_row.values.tolist()
        print(alph_notes)

        notes = audio.alph_notes_to_num(alph_notes)
        print(notes)

        scale = Scale(scale=notes)

        return scale


    def random_choice(self, chord_df):
        """
        Takes in an outputted list of chords, makes a random choice
        and converts it into a scale in the form of a list of notes, 
        outputs that scale
        """
        selected_row = chord_df[["root", "3rd", "5th", "7th", "9th", "11th", "13th"]].sample()

        # Convert the selected row to a list
        random_chord = selected_row.values.tolist()[0]

        # Convert the chord to a list of notes
        notes = audio.keys_of_chord(random_chord)

        # Convert the list of notes to a scale object
        scale = Scale(scale=notes)

        return scale
    
    def random_min_choice(self):
        """
        Makes a random choice of scale that has minimal dissonance among
        a collection of scales that contain self.chord
        example: self.chord = Cm 7,
        output would be C natural minor, phrygian, or dorian
        """
        chord_df = self.chord_completions()

        chord_df = self.minimal_diss_chords(chord_df)

        scale = self.random_choice(chord_df)

        return scale

    def global_scale(self, chord_prog):
        """
        Takes in a chord progression, completes self.chord
        using notes from the chord prog, makes random choices
        for the remaining missing notes
        returns the associated scale

        Details: while chord[i] doesn't resolve scale, check next
        if the last chord is reached, fill in the rest of the gaps based on dissonance
        """
        if type(chord_prog[0]) == str:
            chord_prog = [audio.chord_finder(chord) for chord in chord_prog]
        input_chord = self.chord

        completions = self.chord_completions()
        # chord dictionary
        chord_dict = {
            'root': input_chord[0],
            '3rd': input_chord[1],
            '5th': input_chord[2],
            '7th': input_chord[3],
            '9th': input_chord[4],
            '11th': input_chord[5],
            '13th': input_chord[6]
        }
        # need missing notes
        cols = ['7th', '9th', '11th', '13th']

        def choices_to_notes(root, degree, choices):
            """
            Takes in a root, an degree (7th, 9th, 11th, 13th) and a list of numerical choices
            and converts those into alphabetical note choices
            """

            # In C, lowest 7th is 10, lowest 9th is 1, etc.
            conversion_dict = {
                '7th': 9,
                '9th': 0,
                '11th': 4,
                '13th': 7
            }

            note_dict = audio.note_dict()

            # numerical notes following the following formula
            num_notes = [(root + conversion_dict[degree] + choice) % 12 for choice in choices]
            print(f'numerical notes (choices_to_notes): {num_notes}')
            return [note_dict[note] for note in num_notes]

        for chord in chord_prog:
            notes_of_chord = audio.keys_of_chord(chord)
            notes_of_chord = [audio.note_dict()[note] for note in notes_of_chord]
            print(f"Chord: {audio.chord_name(chord)}, notes: {notes_of_chord}")
            for col in cols:
                if chord_dict[col] == 0:
                    # a list of choices for the current extension (e.g. 7th)
                    deg_choices = completions[col].unique().tolist()
                    print(f'Degree choices for missing {col}: {deg_choices}')

                    # those choices but converted to a note
                    alph_choices = choices_to_notes(chord_dict['root'], col, deg_choices)
                    print(f'alph_choices for missing {col}: {alph_choices}')

                    # choice to alph note dict
                    # ex for 9th of Cm 7: {C#: 1, D: 2}
                    choice_dict = {alph_choices[i]: deg_choices[i] for i in range(len(deg_choices))}
                    print(f'choice_dict for missing {col}: {choice_dict}')

                    # notes in common of missing and the current neighbor chord
                    avail_choices = [note for note in notes_of_chord if note in alph_choices]
                    print(f'available choices for missing {col}: {avail_choices}')

                    if len(avail_choices):

                        # this is a alphabetical note chosen as an extension for our scale
                        random_choice = random.choice(avail_choices)
                        print(f'random choice: {random_choice}')

                        chord_dict[col] = choice_dict[random_choice]
                        print(f'choice as a num: {chord_dict[col]}')

        chord = list(chord_dict.values())

        updated_chord = Chord(audio.chord_name(chord))

        scale = updated_chord.random_choice(updated_chord.chord_completions())

        return scale
        

# my_chord = Chord(chord_name="Gm 7")
# scale = my_chord.random_choice2()
# # # scale = my_chord.global_scale(["Bb 7 #11", "G#m 13", "G 7 #11"])

# # print(scale.alph_notes())
# print(my_chord.find_matching_scales())

# print(f'Cum uwu: {my_chord.random_choice2().alph_notes()}')

# grid_image_with_dots = scale.guitar_scale_visual(x=15, alph_notes=scale.alph_notes())
# grid_image_with_dots.show()

# print(my_chord.chord_completions())
#  print(my_chord.chord_completions())

# step_seq = [2, 2, 2, 2, 2, 2]
# step_seq_obj = StepSequence(step_seq)
# scale = step_seq_obj.step_seq_to_scale(11)
# print(f'step sequence: {step_seq}, notes: {scale.alph_notes()}')

    







