from flask import Flask,request,send_from_directory,render_template
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2
import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw   
import simpleaudio as sa

import re
os.chdir('E:\\web-omr')

FREQ = { "C1": 32, "C#1": 34, "Db1": 34, "D1": 36, "D#1": 38, "Eb1": 38, "E1": 41, "F1": 43, "F#1": 46, "Gb1": 46, "G1": 49, "G#1": 52, "Ab1": 52, "A1": 55, "A#1": 58, "Bb1": 58, "B1": 61, "C2": 65, "C#2": 69, "Db2": 69, "D2": 73, "D#2": 77, "Eb2": 77, "E2": 82, "F2": 87, "F#2": 92, "Gb2": 92, "G2": 98, "G#2": 104, "Ab2": 104, "A2": 110, "A#2": 116,"Bb2": 116, "B2": 123, "C3": 130, "C#3": 138, "Db3": 138, "D3": 146, "D#3": 155, "Eb3": 155, "E3": 164, "F3": 174, "F#3": 185, "Gb3": 185, "G3": 196, "G#3":208, "Ab3":208, "A3": 220, "A#3": 233, "Bb3": 233, "B3": 246, "C4": 261,"C#4": 277, "Db4": 277, "D4": 293, "D#4": 311, "Eb4": 311, "E4": 329, "F4": 349, "F#4": 369, "Gb4": 369, "G4": 392, "G#4": 415, "Ab4": 415, "A4": 440, "A#4": 466, "Bb4": 466, "B4": 493, "C5": 523,"C#5": 554, "Db5": 554, "D5": 587, "D#5": 622, "E5": 659, "Eb5": 659, "F5": 698, "F#5": 739, "Gb5": 739, "G5": 784, "G#5": 830,"Ab5": 830, "A5": 880, "A#5": 932, "Bb5": 932, "B5": 987, "rest": 0.0067,}

DUR = {
    "double": 4.0, 
    "whole": 2.0,
    "half": 1.0,
    "quarter": .50,
    "eighth": .25,
    "sixteenth": .06,
    "thirty_second": .03,
    "sixty_fourth": .02,
    "hundred_twenty_eighth": .01,
}

def music_str_parser(semantic):
    # finds string associated with symb
    found_str = re.compile(r'((note|gracenote|rest|multirest)(\-)(\S)*)'
                           ).findall(semantic)
    music_str = [i[0] for i in found_str]
    # finds the note's alphabets 
    fnd_notes = [re.compile(r'(([A-G](b|#)?[1-6])|rest)'
                    ).findall(note) for note in music_str]
    # stores the note's alphabets
    notes = [m[0][0] for m in fnd_notes]
    found_durs = [re.compile(r'((\_|\-)([a-z]|[0-9])+(\S)*)+'
                    ).findall(note) for note in music_str]
    #split by '_' every other string in list found in tuple of lists 
    durs = [i[0][0][1:].split('_') for i in found_durs]
    return notes, durs


def dur_evaluator(durations):
    note_dur_computed = []
    for dur in durations:
        # if dur_len in DUR dict, get. Else None 
        dur_len = [DUR.get(i.replace('.','').replace('.',''), 
                              None) for i in dur]
        # filter/remove None values, and sum list
        dur_len_actual = sum(list(filter(lambda a: a !=None, 
                                      dur_len)))
        # actual duration * 4 = quadruple
        if 'quadruple' in dur:
            dur_len_actual = dur_len_actual * 4
        # actual duration * 2 = fermata
        elif 'fermata' in dur:
            dur_len_actual = dur_len_actual * 2
        # actual duration + 1/2 of duration = .
        elif '.' in ''.join(dur):
            dur_len_actual = dur_len_actual + (dur_len_actual * 1/2)
        elif '..' in ''.join(dur):
            dur_len_actual = dur_len_actual +(2 *(dur_len_actual * 1/2))
        # if no special duration string
        elif dur[0].isnumeric():
            dur_len_actual = float(dur[0]) * .5
        note_dur_computed.append(dur_len_actual)
    return note_dur_computed

def get_music_note(semantic):
    notes, durations = music_str_parser(semantic)
    sample_rate = 44100
    timestep = []
    T = dur_evaluator(durations)
    for i in T:
        # gets timestep for each sample 
        timestep.append(np.linspace(0, i, int(i * sample_rate), 
                                    False))
    def get_freq(notes):
        # get pitchs frequency from dict
        pitch_freq = [FREQ[i] for i in notes]
        return pitch_freq
    return timestep, get_freq(notes)


def get_sinewave_audio(semantic):
    audio = []
    timestep, freq = get_music_note(semantic)
    for i in range(len(freq)):
        # calculates the sinewave
        audio.append(np.sin(
            freq[i] * timestep[i] * 2 * np.pi))
    return audio

app = Flask(__name__, static_url_path='')

def sparse_tensor_to_strs(sparse_tensor):
  indices= sparse_tensor[0][0]
  values = sparse_tensor[0][1]
  dense_shape = sparse_tensor[0][2]

  strs = [ [] for i in range(dense_shape[0]) ]

  string = []
  ptr = 0
  b = 0

  for idx in range(len(indices)):
      if indices[idx][0] != b:
          strs[b] = string
          string = []
          b = indices[idx][0]

      string.append(values[ptr])

      ptr = ptr + 1

  strs[b] = string

  return strs


def normalize(image):
  return (255. - image)/255.


def resize(image, height):
  width = int(float(height * image.shape[1]) / image.shape[0])
  sample_img = cv2.resize(image, (width, height))
  return sample_img

model = "E:\\web-omr\\Semantic-Model\\semantic_model.meta"

tf.reset_default_graph()
sess = tf.compat.v1.InteractiveSession()
# Read the dictionary
dict_file = open("E:/web-omr/vocabulary_semantic.txt",'r')
dict_list = dict_file.read().splitlines()
int2word = dict()
for word in dict_list:
  word_idx = len(int2word)
  int2word[word_idx] = word
dict_file.close()

# Restore weights
saver = tf.train.import_meta_graph(model)
saver.restore(sess,model[:-5])

graph = tf.get_default_graph()

input = graph.get_tensor_by_name("model_input:0")
seq_len = graph.get_tensor_by_name("seq_lengths:0")
rnn_keep_prob = graph.get_tensor_by_name("keep_prob:0")
height_tensor = graph.get_tensor_by_name("input_height:0")
width_reduction_tensor = graph.get_tensor_by_name("width_reduction:0")
logits = tf.get_collection("logits")[0]

# Constants that are saved inside the model itself
WIDTH_REDUCTION, HEIGHT = sess.run([width_reduction_tensor, height_tensor])

decoded, _ = tf.nn.ctc_greedy_decoder(logits, seq_len)

@app.route('/img/<filename>')
def send_img(filename):
   return send_from_directory('', filename)

@app.route("/")
def root():
    return render_template('index.html')

@app.route('/predict', methods = ['GET', 'POST'])
def predict():
   if request.method == 'POST':
      f = request.files['file']
      img = f
      print(img)
      image = Image.open(img).convert('L')
      image = np.array(image)
      image = resize(image, HEIGHT)
      image = normalize(image)
      image = np.asarray(image).reshape(1,image.shape[0],image.shape[1],1)

      seq_lengths = [ image.shape[2] / WIDTH_REDUCTION ]
      prediction = sess.run(decoded,
                            feed_dict={
                                input: image,
                                seq_len: seq_lengths,
                                rnn_keep_prob: 1.0,
                            })
      str_predictions = sparse_tensor_to_strs(prediction)

      array_of_notes = []
      SEMANTIC = ''

      for w in str_predictions[0]:
          array_of_notes.append(int2word[w])
          SEMANTIC += int2word[w] + '\n'

      print(array_of_notes)

      notes=[]
      for i in array_of_notes:
          if i[0:5]=="note-":
              if not i[6].isdigit():
                  notes.append(i[5:7])
              else:
                  notes.append(i[5])
      print(notes)

      audio = get_sinewave_audio(SEMANTIC)
      audio =  np.hstack(audio)
        
      audio *= 32767 / np.max(np.abs(audio))
         
      audio = audio.astype(np.int16)
           
      play_obj = sa.play_buffer(audio, 1, 2, 44100)
          
      if play_obj.is_playing():
        print("\nplaying...")
        print(f'\n{SEMANTIC}')
        play_obj.wait_done()

    #   img = Image.open(img).convert('L')
    #   size = (img.size[0], int(img.size[1]*1.5))
    #   layer = Image.new('RGB', size, (255,255,255))
    #   layer.paste(img, box=None)
    #   img_arr = np.array(layer)
    #   height = int(img_arr.shape[0])
    #   width = int(img_arr.shape[1])
    #   # print(img_arr.shape[0])
    #   draw = ImageDraw.Draw(layer)
    #   # font = ImageFont.truetype(<font-file>, <font-size>)
    #   font = ImageFont.truetype("E:\\OMR\\web-omr-master\\Aaargh.ttf", 20)
    #   # draw.text((x, y),"Sample Text",(r,g,b))
    #   j = width / 9
    #   for i in notes:
    #       draw.text((j, height-40), i, (0,0,0), font=font)
    #       j+= (width / (len(notes) + 4))
    #   layer.save("annotated.png")
      return render_template('index.html')

if __name__=="__main__":
    app.run()
