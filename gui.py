from random import randint
import tkinter as tk
import audio
from chord_search import ChordSearch
from chord_prog_edit import ChordProgEditor
from random_chord import RandomChord

window = tk.Tk()
window.title('Chord Tool')

e = tk.Entry(window, width=70, borderwidth=5)
e.grid(row=0, column=0, columnspan=3, padx=10, pady=10)

window.rowconfigure(0,weight=1)
window.columnconfigure(0, weight=1, uniform='equal')

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
menu_color = '#2f2f2f'
btn_color = '#1a1a1a'
txt_color = '#ffffff'
menu_frame = tk.Frame(window, bg=menu_color)
menu_title=tk.Label(menu_frame, text='Menu', bg=menu_color, fg='#ffffff')
menu_title.grid(row=0, column=0)

chord_srch_btn = tk.Button(
    menu_frame,
    text='Chord Search',
    bg=btn_color,
    fg=txt_color,
    command=lambda:show_frame(chord_search_display.frame),
    anchor='w'
    )
chord_srch_btn.grid(row=1, column=0, sticky='ew')

chord_prog_btn = tk.Button(
    menu_frame, 
    text='Chord Progression Editor', 
    bg=btn_color,
    fg=txt_color,
    anchor='w',
    command=lambda:show_frame(chord_prog_display.frame)
    )
chord_prog_btn.grid(row=2, column=0, sticky='ew')

rndm_chord_btn = tk.Button(
    menu_frame, 
    text='Play Random Chord', 
    bg=btn_color,
    fg=txt_color,
    anchor='w',
    command=lambda:show_frame(random_chord_display.frame)
    )
rndm_chord_btn.grid(row=3, column=0, sticky='ew')

menu_frame.grid(row=0, column=0, sticky='nsew')
menu_frame.grid_columnconfigure(0, weight=1, uniform='equal')

#====Chord search
# window_color = '#013247'
window_color = '#264166'
chord_search_display = ChordSearch(window, color=window_color)

#====Chord progression editor
chord_prog_display = ChordProgEditor(window)

#====Frame for random chord playing
random_chord_display = RandomChord(window, color=window_color)


show_frame(chord_search_display.frame)

window.mainloop()
