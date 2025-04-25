from tkinter import font
from PIL import Image, ImageTk
import tkinter as tk
import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
import operator
import threading as thr
from string import ascii_uppercase
from spellchecker import SpellChecker
from time import time

CORAL_BLUE = '#3ABEFF'
BUTTON_BG = '#3ABEFF'
BUTTON_FG = 'white'
BUTTON_FONT = ("Helvetica", 14, "bold")

class Application:
    def __init__(self):
        self.directory = 'C:/live-sign-translate-main/model'
        # With:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, 'signlang', 'static', 'model', 'translator-bw-vgg16.h5')
        self.modelThread = thr.Thread(target=self.modelLoader, args=(model_path,))

        if os.name == 'nt':  # Windows
            os.environ['DISPLAY'] = ':0.0'
        # Load model in separate thread
        self.modelThread = thr.Thread(target=self.modelLoader, args=('C:/sign_language/TalkToMyHand/signlang/static/model/translator-bw-vgg16.h5',))
        self.modelThread.start()
        while self.modelThread.is_alive():
            print("model_loading")

        self.spell = SpellChecker()
        self.vs = cv2.VideoCapture(0)
        self.current_image = None
        self.current_image2 = None
        self.ct = {'blank': 0}
        self.blank_flag = 0
        for i in ascii_uppercase:
            self.ct[i] = 0

        self.letter_timer = {}
        self.last_letter = ""
        self.last_letter_time = time()

        self.root = tk.Tk()
        self.root.title("Live Sign Language Translator")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.geometry("1500x800")

        # Background
        self.canvas = tk.Canvas(self.root, width=1500, height=800)
        self.bg_image = ImageTk.PhotoImage(Image.open("C:/sign_language/TalkToMyHand/signlang/static/images/background.jpg").resize((1500, 800)))
        self.canvas.create_image(0, 0, anchor='nw', image=self.bg_image)
        self.canvas.place(x=0, y=0)

        # Title
        self.T = tk.Label(self.root, text="Live Sign Translate", font=("Courier", 40, "bold"), bg=CORAL_BLUE, fg='white')
        self.T.place(x=500, y=10)

        # Video feed
        self.panel = tk.Label(self.root, bg="white")
        self.panel.place(x=100, y=80, width=640, height=480)

        # Processed image
        self.panel2 = tk.Label(self.root, bg="white")
        self.panel2.place(x=800, y=80, width=310, height=310)

        # Letter
        self.T1 = tk.Label(self.root, text="Letter:", font=("Courier", 30, "bold"), bg='white')
        self.T1.place(x=1150, y=100)

        self.panel3 = tk.Label(self.root, bg='white')
        self.panel3.place(x=1150, y=160, width=150, height=60)

        # Word
        self.T2 = tk.Label(self.root, text="Word:", font=("Courier", 30, "bold"), bg='white')
        self.T2.place(x=1150, y=250)

        self.panel4 = tk.Label(self.root, bg='white')
        self.panel4.place(x=1150, y=310, width=300, height=50)

        # Sentence
        self.T3 = tk.Label(self.root, text="Sentence:", font=("Courier", 30, "bold"), bg='white')
        self.T3.place(x=100, y=580)

        self.panel5 = tk.Label(self.root, text="", font=("Courier", 20), bg='white', anchor='w', justify='left')
        self.panel5.place(x=100, y=640, width=1300, height=50)

        # Loading message
        self.loadingMsg = tk.Label(self.root, text='Please Wait While Model is Loading...', font=('Courier', 16), bg='white')
        self.loadingMsg.place(x=100, y=720)

        # Suggestion buttons
        self.suggestion_buttons = [
            self.make_button(self.root, lambda idx=i: self.select_suggestion(idx), "", x=800 + i*140, y=420)
            for i in range(5)
        ]

        # Control buttons
        self.btcall_next = self.make_button(self.root, self.nextWord, text="Next", x=800, y=500)
        self.btcall_clear = self.make_button(self.root, self.clearWord, text="Clear", x=950, y=500)

        self.str = ""
        self.word = ""
        self.current_symbol = "Empty"
        self.photo = "Empty"

        self.video_loop()
        self.root.mainloop()

    def make_button(self, master, command, text="", x=0, y=0):
        btn = tk.Button(master, text=text, command=command,
                        font=BUTTON_FONT, bg=BUTTON_BG, fg=BUTTON_FG,
                        activebackground='#2EA3D8',
                        bd=0, padx=20, pady=10,
                        relief="raised", highlightthickness=0,
                        cursor="hand2")
        btn.place(x=x, y=y, width=120, height=40)
        return btn

    def video_loop(self):
        ok, frame = self.vs.read()
        if ok:
            cv2image = cv2.flip(frame, 1)
            x1 = int(0.5 * frame.shape[1])
            y1 = 10
            x2 = frame.shape[1] - 10
            y2 = int(0.5 * frame.shape[1])
            cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255, 0, 0), 1)
            cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGBA)
            self.current_image = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=self.current_image)
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)

            roi = cv2image[y1:y2, x1:x2]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 2)
            th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 2)
            ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            self.predict(res)

            self.current_image2 = Image.fromarray(res)
            imgtk2 = ImageTk.PhotoImage(image=self.current_image2)
            self.panel2.imgtk = imgtk2
            self.panel2.config(image=imgtk2)

            self.panel3.config(text=self.current_symbol, font=("Courier", 40, 'bold'))
            self.panel4.config(text=self.word, font=("Courier", 30))
            self.panel5.config(text=self.str)

            suggestions = list(self.spell.candidates(self.word))
            for i in range(5):
                if i < len(suggestions):
                    self.suggestion_buttons[i].config(text=suggestions[i], state='normal')
                else:
                    self.suggestion_buttons[i].config(text="", state='disabled')

        self.root.after(30, self.video_loop)

    def predict(self, test_image):
        test_image = cv2.resize(test_image, (224, 224))
        try:
            result = self.loaded_model.predict(test_image.reshape(1, 224, 224, 1))
            self.loadingMsg.config(text="")
        except:
            return

        prediction = {'blank': result[0][0]}
        for i, letter in enumerate(ascii_uppercase):
            prediction[letter] = result[0][i + 1]

        prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        self.current_symbol = prediction[0][0]

        if self.current_symbol == 'blank':
            for i in ascii_uppercase:
                self.ct[i] = 0

        self.ct[self.current_symbol] += 1

        if self.ct[self.current_symbol] > 10:
            current_time = time()
            if self.current_symbol == self.last_letter:
                if current_time - self.last_letter_time >= 5:
                    if self.current_symbol != 'blank':
                        self.word += self.current_symbol
                        self.last_letter = ""
                        self.last_letter_time = current_time
            else:
                self.last_letter = self.current_symbol
                self.last_letter_time = current_time

            self.ct['blank'] = 0
            for i in ascii_uppercase:
                self.ct[i] = 0

            if self.current_symbol == 'blank':
                if self.blank_flag == 0:
                    self.blank_flag = 1
            else:
                self.blank_flag = 0

    def modelLoader(self, path):
        self.loaded_model = load_model(path)

    def select_suggestion(self, index):
        suggestions = list(self.spell.candidates(self.word))
        if len(suggestions) > index:
            self.word = suggestions[index]

    def nextWord(self):
        if len(self.word.strip()) > 0:
            if len(self.str) > 0:
                self.str += " " + self.word.strip()
            else:
                self.str = self.word.strip()
            self.word = ""

    def clearWord(self):
        self.word = ""

    def destructor(self):
        self.root.destroy()
        self.vs.release()
        # cv2.destroyAllWindows()


def main():
    Application()
if __name__ == "__main__":
    if os.name == 'nt':
        import ctypes
        ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)
    main()
