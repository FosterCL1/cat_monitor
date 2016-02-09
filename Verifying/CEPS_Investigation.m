pkg load signal
FileName = '/home/colin/Documents/Octave/CatMeow/NoCat/Validation/Day1_file_58.wav'
FileName2 = '/home/colin/Documents/Octave/CatMeow/NoCat/Validation/Night4_file_68.wav'

Data = wavread(FileName)

[mm,aspc] = melfcc(Data)

specgram(MFCC)

