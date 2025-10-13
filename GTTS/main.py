from gtts import gTTS
import os

text = '''
Ще не вмерла Україна, і слава, і воля,
Ще нам, браття молодії, усміхнеться доля.
Згинуть наші вороженьки, як роса на сонці,
Запануєм і ми, браття, у своїй сторонці
'''

tts = gTTS(text=text,lang='uk')

tts.save("my.mp3")

os.system("start my.mp3")