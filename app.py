import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import numpy as np
# import imutils
import cv2
from music21 import converter, instrument, note, chord, stream
import pickle
import os
import time
import os
# from tensorflow.keras.preprocessing.image import img_to_array
# from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
# from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Bidirectional
from tensorflow.compat.v1.keras.layers import CuDNNLSTM, LSTM
from tensorflow.keras.layers import Activation
# from tensorflow.keras import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


alt=time.time()
cur=os.getcwd()

ferpath=cur+r"/MasterModel/FER.h5"
cascade=cur+r"/MasterModel/haarcascade_frontalface_default.xml"

emotion_dict = {0: "Agitated", 1: "Agitated", 2: "Sad", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Agitated"}

picklepath=""
modelpath=""
midioutpath=""
runpath=cur+r"/Database/run.npy"
predspath=cur+r"/Database/preds.npy"
lpath=cur+r"/Database/l.npy"
mpath=cur+r"/Database/m.npy"
bpath=cur+r"/Database/b.npy"
finallabelpath=cur+r"/Database/finallabel.npy"

cv2.ocl.setUseOpenCL(False)
detector = cv2.CascadeClassifier(cascade)

def FER():

    model1 = Sequential()
    model1.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    model1.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model1.add(MaxPooling2D(pool_size=(2, 2)))
    model1.add(Dropout(0.25))
    model1.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model1.add(MaxPooling2D(pool_size=(2, 2)))
    model1.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model1.add(MaxPooling2D(pool_size=(2, 2)))
    model1.add(Dropout(0.25))
    model1.add(Flatten())
    model1.add(Dense(1024, activation='relu'))
    model1.add(Dropout(0.5))
    model1.add(Dense(7, activation='softmax'))
    model1.load_weights(ferpath)
    return model1

def generate():
    
    with open(picklepath, 'rb') as filepath:
        notes = pickle.load(filepath)
    # st.write("And it begins!")
    pitchnames = sorted(set(item for item in notes))
    n_vocab = len(set(notes))

    # network_input= prepare_sequences(notes, pitchnames, n_vocab)
    # model2 = create_network(network_input, n_vocab)
    # prediction_output = generate_notes(model2, network_input, n_vocab,pitchnames)

    # midi_stream=create_midi(prediction_output)
    # midi_stream.write('midi', fp=midioutpath)

    # with st.spinner(f"Transcribing to FluidSynth"):
    #     midi_data = pretty_midi.PrettyMIDI(midioutpath)
    #     audio_data = midi_data.fluidsynth()
    #     audio_data = np.int16(
    #         audio_data / np.max(np.abs(audio_data)) * 32767 * 0.9
    #     )  

    #     virtualfile = io.BytesIO()
    #     wavfile.write(virtualfile, 44100, audio_data)

    # st.audio(virtualfile)
    with open(midioutpath, "rb") as file:
        col1, col2, col3 = st.columns(3)
        btn = col2.download_button(
                label="Download",
                data=file,
                file_name="output.mid",
                #  mime="image/png"
            )
    
    st.markdown("<h1 style='text-align: center; color: black;'>Want to experience more generated music?</h1>", unsafe_allow_html=True)
    col11, col22 = st.columns(2)

    with col11:
        st.markdown("<h1 style='text-align: center; color:  #2f2f1e;'>Cheerful</h1>", unsafe_allow_html=True)

        modelname=emotion_dict[2]
        ch2 = st.selectbox('',
        ('1', '2', '3','4', '5', '6','7', '8', '9','10'), index=0, key=2)
        if ch2:
            ss='(_'+str(modelname)+'_, '+str(ch2)+').wav'
            path1=cur+r"/Music/"+ss
            # path1=os.path.join(path1,ss)
            audio_file = open(path1, 'rb')
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/wav')

        st.markdown("<h1 style='text-align: center; color:  #2f2f1e;'>Any works</h1>", unsafe_allow_html=True)

        modelname=emotion_dict[4]
        ch4 = st.selectbox('',
        ('1', '2', '3','4', '5', '6','7', '8', '9','10'), index=0, key=4)
        if ch4:
            ss='(_'+str(modelname)+'_, '+str(ch4)+').wav'
            path1=cur+r"/Music/"+ss
            # path1=os.path.join(path1,ss)
            audio_file = open(path1, 'rb')
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/wav') 

    with col22:
        st.markdown("<h1 style='text-align: center; color:  #2f2f1e;'>Melancholic</h1>", unsafe_allow_html=True)

        modelname=emotion_dict[3]

        ch3 = st.selectbox('',
        ('1', '2', '3','4', '5', '6','7', '8', '9','10'),index=0, key=3)
        if ch3:
            ss='(_'+str(modelname)+'_, '+str(ch3)+').wav'
            path1=cur+r"/Music/"+ss
            # path1=os.path.join(path1,ss)
            audio_file = open(path1, 'rb')
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/wav') 

        st.markdown("<h1 style='text-align: center; color:  #2f2f1e;'>Intense</h1>", unsafe_allow_html=True)

        modelname=emotion_dict[1]
        ch1 = st.selectbox('',
        ('1', '2', '3','4', '5', '6','7', '8', '9','10'), index=0, key=1)
        if ch1:
            ss='(_'+str(modelname)+'_, '+str(ch1)+').wav'
            path1=cur+r"/Music/"+ss
            # path1=os.path.join(path1,ss)
            audio_file = open(path1, 'rb')
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/wav') 

def prepare_sequences(notes,pitchnames, n_vocab):
    sequence_length = 100

    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    network_input = []

    
    for i in range(0, len(notes)- sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
    
    n_patterns = len(network_input)
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    print(n_patterns)
    network_input = network_input / float(n_vocab)
    return network_input

def create_network(network_input, n_vocab):
    model2 = Sequential()
    # model2.add(CuDNNLSTM(512,input_shape=(network_input.shape[1], network_input.shape[2]),return_sequences=True))
    model2.add(LSTM(512,input_shape=(network_input.shape[1], network_input.shape[2]),return_sequences=True,activation='tanh',recurrent_activation='sigmoid'))
    model2.add(Dropout(0.3))
    # model2.add(Bidirectional(CuDNNLSTM(512, return_sequences=True)))
    model2.add(Bidirectional(LSTM(512, return_sequences=True,activation='tanh',recurrent_activation='sigmoid')))
    model2.add(Dropout(0.3))
    model2.add(Bidirectional(LSTM(512,activation='tanh',recurrent_activation='sigmoid')))
    # model2.add(Bidirectional(CuDNNLSTM(512)))
    model2.add(Dense(256))
    model2.add(Dropout(0.3))
    model2.add(Dense(n_vocab))
    model2.add(Activation('softmax'))
    model2.compile(loss='categorical_crossentropy', optimizer='adam')
    model2.load_weights(modelpath)

    return model2

def generate_notes(model2, network_input, n_vocab,pitchnames):
        
    start = np.random.randint(0, np.shape(network_input)[0]-1)
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
    pattern = network_input[start]
    prediction_output = []
    
    for note_index in range(125):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)
        prediction = model2.predict(prediction_input, verbose=1)
        
        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)
        
        pattern = np.append(pattern,index)
        pattern = pattern[1:len(pattern)]
    return prediction_output

def create_midi(prediction_output):

    offset = 0
    output_notes = []

    for pattern in prediction_output:
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        offset += 0.5
    midi_stream = stream.Stream(output_notes)
    return midi_stream

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized

class VideoProcessor:
    
    def recv(self, frame):
      run=np.load(runpath)
      preds=np.load(predspath)
      l=np.load(lpath)

      if(run==1):
        img = frame.to_ndarray(format="bgr24")

        frame = resize(img, width=300)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frameClone = frame.copy()

        rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
            minNeighbors=5, minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE)

        for (fX, fY, fW, fH) in rects:
            roi = gray[fY:fY + fH, fX:fX + fW]
            cropped_img=np.expand_dims(np.expand_dims(cv2.resize(roi, (48, 48)), -1),0)
            model1=FER()
            prediction = model1.predict(cropped_img)
            # print(prediction)
            maxindex = int(np.argmax(prediction))
            label=emotion_dict[maxindex]
            cv2.putText(frameClone, label, (fX, fY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),(0, 0, 255), 2)
            preds=np.append(preds,maxindex)
            l+=1
            np.save(predspath,preds)
            np.save(lpath,l)

            preds=np.load(predspath)
            if(l>duration and option=='Both'):
                finallabel=np.bincount(np.array(preds)).argmax()
                np.save(finallabelpath,finallabel)
                final=emotion_dict[int(finallabel)]
                final="Final Label: "+final
                cv2.putText(frameClone, final, (fX-20, fY - 30),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                np.save(runpath,0)
                run=0

        return av.VideoFrame.from_ndarray(frameClone, format="bgr24")




run=np.load(runpath)
preds=np.load(predspath)
np.save(lpath,0)
l=np.load(lpath)
m=np.load(mpath)
finallabel=np.load(finallabelpath)
b=np.save(bpath,0)
duration=30

st.set_page_config(page_title='Sentario')

option = st.sidebar.selectbox('Want to experience individual predictions?',
     ('Both', 'Facial Expression Recognition', 'Music Generation','Listen to Pregenerated Music'),index=0)


if option=='Both':
    a= st.sidebar.radio("Sequentially proceed to each state",('Halt','Recognize Expression', 'Generate Music'),index=0)
    if a == 'Recognize Expression':
        run=1
        np.save(runpath,run)
    if a=='Halt':
        np.save(runpath,0)
        np.save(predspath,0)
        np.save(runpath,0)
        np.save(lpath,0)
        np.save(mpath,0)
        m=0
        run=0
        preds=0
        but=0
        l=0
    if a=='Generate Music':
        np.save(runpath,0)
        run=0
        m=1
        s="Generating Some Fine Music"
        st.title(s)
        finallabel=np.load(finallabelpath)
        modelname=emotion_dict[int(finallabel)]
        picklepath=cur+fr"/MasterModel/{modelname}.pickle"
        modelpath=cur+fr"/MasterModel/{modelname}.hdf5"
        midioutpath=cur+fr"/{modelname}.mid"
        generate()


    if(1==run) and m==0:
        ctx=webrtc_streamer(key="FER", video_processor_factory=VideoProcessor)

    if m==1 and (0==run) and l>duration:
        finallabel=np.load(finallabelpath)
        modelname=emotion_dict[int(finallabel)]
        picklepath=cur+fr"/MasterModel/{modelname}.pickle"
        modelpath=cur+fr"/MasterModel/{modelname}.hdf5"
        midioutpath=cur+fr"/{modelname}.mid"
        generate()

if option=='Facial Expression Recognition':
    np.save(runpath,1)
    ctx=webrtc_streamer(key="FER", video_processor_factory=VideoProcessor)

if option=='Music Generation':
        np.save(runpath,0)
        a= st.sidebar.radio("Emotion of Music",('Melancholic','Cheerful','Intense','Any works'),index=0)
        but=st.sidebar.button('Generate Music')

        # audio_file = open(r"C:\Users\Axiomatize\Downloads\Cymatics-LIFE-Rain-Light-1.mp3", 'rb')
        # audio_bytes = audio_file.read()
        # st.audio(audio_bytes, format='audio/mp3')
        # audio_file = open(r"C:\Users\Axiomatize\Downloads\Cymatics-LIFE-Rain-Light-1.mp3", 'rb')
        # audio_bytes = audio_file.read()
        # st.audio(audio_bytes, format='audio/mp3')
        # audio_file = open(r"C:\Users\Axiomatize\Downloads\Cymatics-LIFE-Rain-Light-1.mp3", 'rb')
        # audio_bytes = audio_file.read()
        # st.audio(audio_bytes, format='audio/mp3')
        # audio_file = open(r"C:\Users\Axiomatize\Downloads\Cymatics-LIFE-Rain-Light-1.mp3", 'rb')
        # audio_bytes = audio_file.read()
        # st.audio(audio_bytes, format='audio/mp3')

        if but:
            if a=='Intense':
                modelname=emotion_dict[1]
            if a=='Cheerful':
                modelname=emotion_dict[3]
            if a=='Melancholic':
                modelname=emotion_dict[2]
            if a=='Any works':
                modelname=emotion_dict[4]
            s="Generating Some Fine Music"
            st.title(s)
            picklepath=cur+fr"/MasterModel/{modelname}.pickle"
            modelpath=cur+fr"/MasterModel/{modelname}.hdf5"
            midioutpath=cur+fr"/{modelname}.mid"
            generate()

if option=='Listen to Pregenerated Music':
    # np.save(runpath,0)
    st.markdown("<h1 style='text-align: center; color: black;'>Listen to some great Music</h1>", unsafe_allow_html=True)
    col11, col22 = st.columns(2)

    with col11:
        st.markdown("<h1 style='text-align: center; color:  #2f2f1e;'>Cheerful</h1>", unsafe_allow_html=True)

        modelname=emotion_dict[2]
        ch2 = st.selectbox('',
        ('1', '2', '3','4', '5', '6','7', '8', '9','10'), index=0, key=2)
        if ch2:
            ss='(_'+str(modelname)+'_, '+str(ch2)+').wav'
            path1=cur+r"/Music/"+ss
            # path1=os.path.join(path1,ss)
            audio_file = open(path1, 'rb')
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/wav')

        st.markdown("<h1 style='text-align: center; color:  #2f2f1e;'>Any works</h1>", unsafe_allow_html=True)

        modelname=emotion_dict[4]
        ch4 = st.selectbox('',
        ('1', '2', '3','4', '5', '6','7', '8', '9','10'), index=0, key=4)
        if ch4:
            ss='(_'+str(modelname)+'_, '+str(ch4)+').wav'
            path1=cur+r"/Music/"+ss
            # path1=os.path.join(path1,ss)
            audio_file = open(path1, 'rb')
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/wav') 

    with col22:
        st.markdown("<h1 style='text-align: center; color:  #2f2f1e;'>Melancholic</h1>", unsafe_allow_html=True)

        modelname=emotion_dict[3]

        ch3 = st.selectbox('',
        ('1', '2', '3','4', '5', '6','7', '8', '9','10'),index=0, key=3)
        if ch3:
            ss='(_'+str(modelname)+'_, '+str(ch3)+').wav'
            path1=cur+r"/Music/"+ss
            # path1=os.path.join(path1,ss)
            audio_file = open(path1, 'rb')
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/wav') 

        st.markdown("<h1 style='text-align: center; color:  #2f2f1e;'>Intense</h1>", unsafe_allow_html=True)

        modelname=emotion_dict[1]
        ch1 = st.selectbox('',
        ('1', '2', '3','4', '5', '6','7', '8', '9','10'), index=0, key=1)
        if ch1:
            ss='(_'+str(modelname)+'_, '+str(ch1)+').wav'
            path1=cur+r"/Music/"+ss
            # path1=os.path.join(path1,ss)
            audio_file = open(path1, 'rb')
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/wav') 