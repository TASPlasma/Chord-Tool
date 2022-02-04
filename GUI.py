import tkinter as tk
import chord_dicts

root = tk.Tk()
root.title('Chord Tool')

e = tk.Entry(root, width=70, borderwidth=5)
e.grid(row=0, column=0, columnspan=3, padx=10, pady=10)

root.rowconfigure(0,weight=1)
root.columnconfigure(0, weight=1, uniform='equal')

def show_frame(frame):
    frame.tkraise()

def create_radio_btn(frame, my_text, var, value, row, col, width=4):
    tk.Radiobutton(
        frame, 
        text=my_text, 
        variable=var,
        value=value,
        width=width, 
        anchor='w'
        ).grid(row=row, column=col, sticky='ew')

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
    command=lambda:show_frame(chord_search_frame),
    anchor='w'
    )
chord_srch_btn.grid(row=1, column=0, sticky='ew')

chord_prog_btn = tk.Button(
    menu_frame, 
    text='Chord Progression Editor', 
    anchor='w',
    command=lambda:show_frame(chord_prog_frame)
    )
chord_prog_btn.grid(row=2, column=0, sticky='ew')

voice_edit_btn = tk.Button(
    menu_frame, 
    text='Voice Editor', 
    command=lambda:show_frame(voice_edit_frame),
    anchor='w'
    )
voice_edit_btn.grid(row=3, column=0, sticky='ew')

rndm_chord_btn = tk.Button(
    menu_frame, 
    text='Play Random Chord', 
    anchor='w',
    command=lambda:print("hi")
    )
rndm_chord_btn.grid(row=4, column=0, sticky='ew')

menu_frame.grid(row=0, column=0, sticky='nsew')
menu_frame.grid_columnconfigure(0, weight=1, uniform='equal')

#====Chord search
"""
Checkboxes for root, third, ..., 13th
Button to play the chord
"""
chord_search_frame = tk.Frame(root, bg='red')
chord_search_title = tk.Label(chord_search_frame, text='Chord Search', bg='red')
chord_search_title.grid(row=0, column=4, sticky='ew')

chord_search_frame.grid(row=0, column=1, sticky='nsew')
chord_search_frame.grid_columnconfigure((1,2,3,4,5,6,7), weight=1, uniform='equal')

tk.Label(chord_search_frame, text='Root').grid(row=1, column=1, sticky='ew')
tk.Label(chord_search_frame, text='3rd').grid(row=1, column=2, sticky='ew')
tk.Label(chord_search_frame, text='5th').grid(row=1, column=3, sticky='ew')
tk.Label(chord_search_frame, text='7th').grid(row=1, column=4, sticky='ew')
tk.Label(chord_search_frame, text='9th').grid(row=1, column=5, sticky='ew')
tk.Label(chord_search_frame, text='11th').grid(row=1, column=6, sticky='ew')
tk.Label(chord_search_frame, text='13th').grid(row=1, column=7, sticky='ew')

# create_radio_btn(chord_search_frame, my_text='C', value=0, row=3, col=1)
# create_radio_btn(chord_search_frame, my_text='C#', value=1, row=4, col=1)
# create_radio_btn(chord_search_frame, my_text='D', value=2, row=5, col=1)
# create_radio_btn(chord_search_frame, my_text='Eb', value=3, row=6, col=1)
# create_radio_btn(chord_search_frame, my_text='E', value=4, row=7, col=1)
# create_radio_btn(chord_search_frame, my_text='F', value=5, row=8, col=1)
# create_radio_btn(chord_search_frame, my_text='F#', value=6, row=9, col=1)
# create_radio_btn(chord_search_frame, my_text='G', value=7, row=10, col=1)
# create_radio_btn(chord_search_frame, my_text='G#', value=8, row=11, col=1)
# create_radio_btn(chord_search_frame, my_text='A', value=9, row=12, col=1)
# create_radio_btn(chord_search_frame, my_text='Bb', value=10, row=13, col=1)
# create_radio_btn(chord_search_frame, my_text='B', value=11, row=14, col=1)

# for i in range(12):
#     create_radio_btn(
#         chord_search_frame, 
#         my_text=chord_dicts.root_dict[i], 
#         value=i, 
#         row=i+2, 
#         col=1
#         )

root_var = tk.IntVar()
third_var = tk.IntVar()
fifth_var = tk.IntVar()
seventh_var = tk.IntVar()
ninth_var = tk.IntVar()
eleventh_var = tk.IntVar()
thirteenth_var = tk.IntVar()

root_var.set(0)
third_var.set(0)
fifth_var.set(0)
seventh_var.set(0)
ninth_var.set(0)
eleventh_var.set(0)
thirteenth_var.set(0)

def restrict_selection():
    return "hi"

vars=[root_var, third_var, fifth_var, seventh_var, ninth_var, eleventh_var, thirteenth_var]

for i, (key, val) in enumerate(chord_dicts.chord_array_dict.items()):

    print("i=", i,"key=", key,"val=", val)
    dic = val
    for j, l in dic.items():
        print("j=", j, ", item=", l)
        if i in [1, 2]:
            create_radio_btn(
                chord_search_frame, 
                my_text=l, 
                var=vars[i],
                value=j+1, 
                row=j+2,
                col=i+1
                )
        else:
            create_radio_btn(
                chord_search_frame, 
                my_text=l, 
                var=vars[i],
                value=j, 
                row=j+2,
                col=i+1
                )


play_chord_btn = tk.Button(
    chord_search_frame, 
    text='Play Chord', 
    command=lambda:print('play chord'),
    anchor='w'
    )
play_chord_btn.grid(row=15, column=7, sticky='ew')

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