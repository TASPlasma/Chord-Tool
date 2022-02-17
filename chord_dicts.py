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

third_dict = {
    0: 'sus2',
    1: 'm',
    2: 'maj',
    3: 'sus4'
}

fifth_dict = {
    0: u' \u266D5',
    1: u' \u266E5',
    2: u' \u266F5'
}

seventh_dict = {
    0: 'no 7',
    1: ' 7',
    2: ' maj7'
}

ninth_dict = {
    0: 'no 9',
    1: u' \u266D9',
    2: u' \u266E9',
    3: u' \u266F9'
}

eleventh_dict = {
    0: 'no 11',
    1: u'\u266E11',
    2: u' \u266F11'
}

thirteenth_dict = {
    0: 'no 13',
    1: u' \u266D13',
    2: u' \u266E13'
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