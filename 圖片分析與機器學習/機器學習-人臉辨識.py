#pip install face_recognition

import face_recognition
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw

fileName = '/content/drive/MyDrive/face/rain.jpg'
image = face_recognition.load_image_file(fileName)
box = face_recognition.face_locations(image)
print(box)

img = Image.open(fileName)
draw = ImageDraw.Draw(img)
for i in range(len(box)):
  draw.rectangle((box[i][3],box[i][0],box[i][1],box[i][2]),outline='red',width=2)

plt.imshow(img)
plt.show



判斷是否為同一個人

face1 = '/content/drive/MyDrive/face/lai.jpg'
face2 = '/content/drive/MyDrive/face/lai2.jpg'
faceimg1 = face_recognition.load_image_file(face1)
faceimg2 = face_recognition.load_image_file(face2)

know_face = face_recognition.face_encodings(faceimg1)
test_face = face_recognition.face_encodings(faceimg2)
know_face

face_distance = face_recognition.face_distance([know_face[0]],test_face[0])
face_distance

if face_distance[0] < 0.4:
  print('同一個人')
else:
  print('不同人')

# 判斷是不是同一個人，亞洲人適用0.4，西方人用0.6

peo1 = '/content/drive/MyDrive/face/colud.jpg'
peo2 = '/content/drive/MyDrive/face/lai.jpg'
peo3 = '/content/drive/MyDrive/face/water.jpg'
peo4 = '/content/drive/MyDrive/face/uncle.jpg'
peo5 = '/content/drive/MyDrive/face/rain.jpg'
name = ["宋云樺","賴清德","陳水扁","柯文哲","夏雨喬"]

pimg1 = face_recognition.load_image_file(peo1)
pimg2 = face_recognition.load_image_file(peo2)
pimg3 = face_recognition.load_image_file(peo3)
pimg4 = face_recognition.load_image_file(peo4)
pimg5 = face_recognition.load_image_file(peo5)

encoding1 = face_recognition.face_encodings(pimg1)[0]
encoding2 = face_recognition.face_encodings(pimg2)[0]
encoding3 = face_recognition.face_encodings(pimg3)[0]
encoding4 = face_recognition.face_encodings(pimg4)[0]
encoding5 = face_recognition.face_encodings(pimg5)[0]

know_faces = [encoding1,encoding2,encoding3,encoding4,encoding5]

result = face_recognition.compare_faces(know_faces,encoding1,0.4)
result

facename = ''
for i in range(len(result)):
  if result[i]:
    facename += name[i]+','
if facename == '':
  print('查無此人')
else:
  print(facename)

# 臉部網點辨識

images = face_recognition.face_landmarks(pimg1)
images



from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode

def take_photo(filename='photo.jpg', quality=0.8):
  js = Javascript('''
    async function takePhoto(quality) {
      const div = document.createElement('div');
      const capture = document.createElement('button');
      capture.textContent = 'Capture';
      div.appendChild(capture);

      const video = document.createElement('video');
      video.style.display = 'block';
      const stream = await navigator.mediaDevices.getUserMedia({video: true});

      document.body.appendChild(div);
      div.appendChild(video);
      video.srcObject = stream;
      await video.play();

      // Resize the output to fit the video element.
      google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

      // Wait for Capture to be clicked.
      await new Promise((resolve) => capture.onclick = resolve);

      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      stream.getVideoTracks()[0].stop();
      div.remove();
      return canvas.toDataURL('image/jpeg', quality);
    }
    ''')
  display(js)
  data = eval_js('takePhoto({})'.format(quality))
  binary = b64decode(data.split(',')[1])
  with open(filename, 'wb') as f:
    f.write(binary)
  return filename

from IPython.display import Image
try:
  filename = take_photo()
  print('Saved to {}'.format(filename))

  # Show the image which was just taken.
  display(Image(filename))
except Exception as err:
  # Errors will be thrown if the user does not have a webcam or if they do not
  # grant the page permission to access it.
  print(str(err))

