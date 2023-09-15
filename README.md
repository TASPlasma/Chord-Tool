# Chord-Tool
 
# Table of Contents
1. [Intro](#section1)
2. [How To Install](#section2)
3. [How To Use](#section3)
4. [Chord Basics](#section4)
5. [Additional Features To Be Added](#section5)

# Intro

This program aims to assist users in creating chord progressions. Chord progression driven musical composition has advantages for the following reasons:
1. The vast choice of melodies gets drastically reduced by starting with a chord progression, thus making it easier to create a melody,
2. Creating a basic rhythm/beat is arguably one of the simpler processes in song writing, so starting with a chord progression can often get one of the hardest parts of song writing out of the way
3. Some tense chords are meant to be changed quickly (as a brief passing tone) and other chords are meant to last a bar or two, so writing beats/rhythm given a predefined sequence of chords can help guide certain rhythmic choices.

That said, claiming chord progression driven composition is the best way (in some sense) to write music is not a claim that will be made here. It could be argued however that the best way for a user to write music is the way that maximizes their enjoyment of listening to their song.

# How To Install
Requirements: 
1. Windows 10 (unclear if it works on other operating systems yet)
2. Python 3.10 (unclear if older versions work)
3. pandas
4. numpy
5. pydub
6. pygame
7. ttk
8. mido
9. opencv (`pip install opencv-python`)
10. pedalboard
11. moviepy

Download all files, save them to a folder called Chord-Tool. Run gui.py from the Chord-Tool folder.


# How To Use
This will serve as a brief tutorial on how to use the Chord Tool (name potentially subject to change). Users unaware of basic music theory feel free to read the section about chord basics.

The tool features a dataset of all possible "chords" (explained the Chord Basics section), and a graphical user interface (GUI for short). The GUI allows users to explore the chord dataset, and allows users to create chord progressions.

The specifics of the GUI are:
1. Chord Search, which allows users to input a choice of chord and then hear it by pressing the "Play Chord" button. The "Play Chord" buttons chooses a random voicing of the chord (with the root in the bass) and plays that voicing. It is encouraged to hear multiple different voicings of a chord. If the user defined chord is inputted in a way that is redundant (in some sense) with another, it may be corrected in the input fields. After playing the chord, the name (see the Chord Basics section to see the way chords are named) of the chord is displayed.
2. Chord Progression Editor, which allows users to create a 4-chord chord progression. The chords here are inputted by name. The chord progression can be played with the "Play Chord Progression" button. A random choice of voicing is assigned to the first chord in the progression, and then each following chord is given a voicing that is a close distance to the previous.
3. Play Random Chord, which groups chords into 6 different categories, based on complexity. Each of the categories has a button, Degree 1, Degree 2, etc. up to Degree 6. Degree 1 chords are the simplest sounding chords (pleasant triads), and degree 6 are extremely complicated (often very dissonant) chords meant to be used sparingly (or not, it's up to you!). Clicking one of these degree buttons will select a random chord with that degree of complexity, and choose a predetermined voicing (one that is not random this time). It is encouraged to press each button multiple times, it is completely possible to stumble upon surprisingly pleasing degree 6 chords!

If you want suggestions for chord progressions to try, here's a jazzy one: "Bm 13", "C#m 13", "Dm 13", "C#m 13".
If you want general advice on creating chord progressions and don't know where to start, try using the Chord Search and Play Random Chord sections.

If you aren't hearing quite what you want in the chord progression player, note that an additional feature of custom user voicings will be added in the near future.

# Chord Basics
A chord in the context of this software formally is a tuple `(c1, c3, c5, c7, c9, c11, c13)`, consisting of choices for the following:
1. Root note, the note in the bass (any note from C to B)
2. Third degree (minor, major, but also suspended 2nd or suspended 4th)
3. Fifth degree (natural fifth, flat fifth, sharp fifth)
4. Optional choice of seventh degree
5. Optional choice of ninth degree
6. Optional choice of 11th degree
7. Optional choice of 13th degree

What does any of this mean? Essentially a chord is a collection of notes to be played all at once. The most important components of the chord are called the root, third degree, and fifth degree. The root of a chord is often what is played by a bass player, or in the bass register of a piano. The third of a chord essentially dictates the mood of a chord. Chords with a major third are "happy" sounding, chords with a minor third sound "sad" (this is what we are often told at least). Try listening to a C major triad (abbreviated simply "C") by selecting C for the root, major third, and natural fifth in Chord Search, and then try listening to a C minor triad by selecting C for the root, minor third, and natural fifth. How do they make you feel?
Next there's the fifth degree, which almost always is set to natural fifth. Setting this to flat fifth (b5) gives what's called a "tritone" with the root note, a very dissonant sound.

After these three basic components of the chord come the higher extensions. These higher extensions are the basis of how jazz chords are formed. The more extensions that are added, the more flavors that are introduced. Using altered extensions (flat or sharp 9, sharp 11, flat 13) adds more and more complexity to the chord. Try experimenting first by using unaltered extensions, adding one in at a time, and then try altering the extensions. One nice example is a chord Jimi Hendrix often used, "Eb 7 #9". You can hear this by entering Chord Search and inputting "Eb" root, "maj" third, "natural" 5th, 7th, and sharp 9 (both 11 and 13 should be omitted).

A note about how the chords are named:

If a major third is chosen, the phrase "major" is omitted. For a example, a C with major third and natural fifth is simply written as "C". All instances of "natural" are omitted from the chord name.
Chords here are named a little differently than they would be in traditional musical notation. This is done to be as unambiguous as possible. What would be named a `C 7 9 11 13` here would be named a `C 13` in traditional musical notation. However, one musician might see a `C 13` and choose to omit the 9, the 11, or both, and another player might do none of that and play all extensions. The subtlety of artistic interpretation of a chord is not something easily captured by a computer program and as such it is best to treat these as different chords (in this app at least). It should be said that when one wants a `C 13` chord played and all extensions played and not up to interpretation, it is often written `C 13 (add 9, add 11)` which is essentially the naming system of the program.

For those with some knowledge of music theory, another unusual choice for the names of chords here is that diminished chords and augmented chords are named still by the choice of third and fifth. For example, a C dim chord is just named "Cm b5 13" because that's 'enharmonically' what it is. Please note that this particular naming convention may be subject to change.

Now that a little bit of chord theory has been explained, it is time to explain what a voicing of a chord is. A chord voicing is a choice of octaves for the notes in the chord. For example, for a "C" chord (which contains the notes C, E, G), one needs to choose which C to play on the instrument, and the same for the E and G.
The keyboard for the notes in Chord-Tool has 73 keys, it starts with an E. The tool adopts the convention that the C note just below this (that is not a note obtainable on the keyboard) is note 0. The full first octave would be:
0. C is note 0
1. C# is note 1
2. D is note 2
3. Eb is note 3
4. E is note 4
5. F is note 5
6. F# is note 6
7. G is note 7
8. G# is note 8
9. A is note 9
10. Bb is note 10
11. B is note 11

To know which note corresponds to a numerical key, simply take the number of the key and find its value modulo 12. For example, key 40 would be an E since 40=3(12)+4, so 40 mod 12 is 4, and 4 corresponds to E.

So an example of a voicing of a "C" chord would be `(12, 40, 43)`. Another example would be `(24, 43, 52)`.

# Additional Features To Be Added
Future improvements to the tool include:
1. Custom chord voicing editing.
2. Voicing visualization and MIDI file exporting
3. Custom durations of the chords in chord progression editing
4. AI assisted chord progression editing

