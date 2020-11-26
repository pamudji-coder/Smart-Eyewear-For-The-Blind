import os
import gtts

text = 'bye'
#description = ', '.join(texts)
tts = gtts.tts.gTTS(text, lang='en')
tts.save(text+'.mp3')
os.system('mpg123 '+text+'.mp3')