from random import randint
import tkinter as tk
import chord_dicts
import audio
from ChordSearch import ChordSearch
from ChordProgEdit import ChordProgEditor
from RandomChord import RandomChord

window = tk.Tk()
window.title('Chord Tool')

e = tk.Entry(window, width=70, borderwidth=5)
e.grid(row=0, column=0, columnspan=3, padx=10, pady=10)

window.rowconfigure(0,weight=1)
window.columnconfigure(0, weight=1, uniform='equal')

def random_chord():
    ind = randint(0, 6948)
    chord = audio.chords.iloc[ind].values
    name = chord[-1]
    chord = chord[:-1]
    text_var = tk.StringVar()
    text_var.set(name)
    tk.Label(window, text=text_var.get()).grid(row=5, column=0, sticky='ew')
    audio.play_voicing(chord)

def show_frame(frame):
    frame.tkraise()

#====Menu
"""
Buttons to diplay pages for:
-Chord search
-Chord progression creator

Buttons to:
-Play a random chord
-Play a random chord progression

"""
menu_frame = tk.Frame(window)
menu_title=tk.Label(menu_frame, text='Menu')
menu_title.grid(row=0, column=0)

chord_srch_btn = tk.Button(
    menu_frame,
    text='Chord Search',
    command=lambda:show_frame(chord_search_display.frame),
    anchor='w'
    )
chord_srch_btn.grid(row=1, column=0, sticky='ew')

chord_prog_btn = tk.Button(
    menu_frame, 
    text='Chord Progression Editor', 
    anchor='w',
    command=lambda:show_frame(chord_prog_display.frame)
    )
chord_prog_btn.grid(row=2, column=0, sticky='ew')

# voice_edit_btn = tk.Button(
#     menu_frame, 
#     text='Voice Editor', 
#     command=lambda:show_frame(voice_edit_frame),
#     anchor='w'
#     )
# voice_edit_btn.grid(row=3, column=0, sticky='ew')

rndm_chord_btn = tk.Button(
    menu_frame, 
    text='Play Random Chord', 
    anchor='w',
    command=lambda:show_frame(random_chord_display.frame)
    )
rndm_chord_btn.grid(row=3, column=0, sticky='ew')

menu_frame.grid(row=0, column=0, sticky='nsew')
menu_frame.grid_columnconfigure(0, weight=1, uniform='equal')

#====Chord search
"""
Checkboxes for root, third, ..., 13th
Button to play the chord
"""

chord_search_display = ChordSearch(window)

#====Chord progression creator
"""
Dropdown for number of chords in progression

Then for each slot, textboxes for chord, tuple, duration

"""

chord_prog_display = ChordProgEditor(window)

#====Frame for voice editing
#voice_edit_frame = tk.Frame(window)

#====Frame for random chord playing
random_chord_display = RandomChord(window)

window.mainloop()
