import tkinter as tk

root = tk.Tk()
root.title('Chord Tool')

e = tk.Entry(root, width=70, borderwidth=5)
e.grid(row=0, column=0, columnspan=3, padx=10, pady=10)

root.rowconfigure(0,weight=1)
root.columnconfigure(0, weight=1)

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
menu_frame = tk.Frame(root)
menu_title=tk.Label(menu_frame, text='Menu')
menu_title.grid(row=0, column=0)

chord_srch_btn = tk.Button(
    menu_frame,
    text='Chord Search',
    command=lambda:show_frame(chord_search_frame)
    )
chord_srch_btn.grid(row=1, column=0)

chord_prog_btn = tk.Button(
    menu_frame, 
    text='Chord Progression Editor', 
    command=lambda:show_frame(chord_prog_frame)
    )
chord_prog_btn.grid(row=2, column=0)

voice_edit_btn = tk.Button(
    menu_frame, 
    text='Voice Editor', 
    command=lambda:show_frame(voice_edit_frame)
    )
voice_edit_btn.grid(row=3, column=0)

rndm_chord_btn = tk.Button(
    menu_frame, 
    text='Play Random Chord', 
    command=lambda:print("hi"))
rndm_chord_btn.grid(row=4, column=0)

menu_frame.grid(row=0, column=0, sticky='nsew')

#====Chord search
"""
Checkboxes for root, third, ..., 13th
Button to play the chord
"""
chord_search_frame = tk.Frame(root, bg='red')
chord_search_title = tk.Label(chord_search_frame, text='Chord Search', bg='red')
chord_search_title.grid(row=0, column=1)

chord_search_frame.grid(row=0, column=1, sticky='nsew')

tk.Label(chord_search_frame, text='Root').grid(row=1, column=1)
tk.Radiobutton(
    chord_search_frame,
    text='C', 
    value=1
    ).grid(row=2, column=1)
tk.Radiobutton(
    chord_search_frame,
    text='C#', 
    value=2
    ).grid(row=3, column=1)
tk.Radiobutton(
    chord_search_frame,
    text='D',
    value=3
    ).grid(row=4, column=1)
tk.Radiobutton(
    chord_search_frame,
    text='Eb',
    value=4
    ).grid(row=5, column=1)
tk.Radiobutton(
    chord_search_frame,
    text='E',
    value=5
    ).grid(row=6, column=1)
tk.Radiobutton(
    chord_search_frame,
    text='F', 
    value=6
    ).grid(row=7, column=1)
tk.Radiobutton(
    chord_search_frame,
    text='F#', 
    value =7
    ).grid(row=8, column=1)
tk.Radiobutton(
    chord_search_frame,
    text='G', 
    value=8
    ).grid(row=9, column=1)
tk.Radiobutton(
    chord_search_frame,
    text='G#', 
    value=9
    ).grid(row=10, column=1)
tk.Radiobutton(
    chord_search_frame,
    text='A',
    value=10
    ).grid(row=11, column=1)
tk.Radiobutton(
    chord_search_frame,
    text='Bb', 
    value=11
    ).grid(row=12, column=1)
tk.Radiobutton(
    chord_search_frame,
    text='B', 
    value=12
    ).grid(row=13, column=1)

play_chord_btn = tk.Button(
    chord_search_frame, 
    text='Play Chord', 
    command=lambda:print('play chord')
    )
play_chord_btn.grid(row=2, column=1)

#====Chord progression creator
"""
Dropdown for number of chords in progression

Then for each slot, textboxes for chord, tuple, duration

"""
chord_prog_frame = tk.Frame(root, bg='green')
chord_prog_title = tk.Label(chord_prog_frame, text='Chord Progression Editor', bg='green')
chord_prog_title.pack(fill='x')

chord_prog_frame.grid(row=0, column=1, sticky='nsew')

play_prog_btn = tk.Button(
    chord_prog_frame, 
    text='Play Chord Progression', 
    command=lambda:print('Playing Chord Progression')
    )
play_prog_btn.pack()

#====Frame for voice editing
voice_edit_frame = tk.Frame(root)

root.mainloop()