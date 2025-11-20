import tensorflow as tf
import numpy as np
import os
from PIL import Image, ImageSequence, ImageTk
import customtkinter as ctk
from tkinter import filedialog
import threading
import time
import random

#konfiguracja modelu
MODEL_PATH = 'model/saved_model/model_efficientnetB1_13.keras'
IMG_SIZE = (240, 240)
CLASSES = sorted(os.listdir('dataset/test'))
model = tf.keras.models.load_model(MODEL_PATH)
THRESHOLD = 85.0

#wiadomości wyświetlające się podczas ładowania zdjęcia
LOADING_MESSAGES = [
    "Szukamy ryb w głębinach...",
    "Fale niosą dane...",
    "Model nurkuje..."
]

# ======= GUI =======
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("Rozpoznawanie zwierząt morskich")
app.geometry("1000x650")
app.minsize(860, 540)

#główna ramka - panel
main_frame = ctk.CTkFrame(app, corner_radius=20, fg_color="#0f172a")
main_frame.pack(expand=True, fill="both", padx=40, pady=30)

#elementy UI - przyciski, itd.
button_select = ctk.CTkButton(main_frame, text="Wybierz zdjęcie", width=200)
button_select.pack(pady=20)

label_result = ctk.CTkLabel(main_frame, text="Wybierz obraz do klasyfikacji", font=ctk.CTkFont(size=16, weight="bold"))
label_result.pack(pady=10)

label_image = ctk.CTkLabel(main_frame, text="", width=320, height=320, corner_radius=12, fg_color="#2a2a2a")
label_image.pack(pady=10)

button_details = ctk.CTkButton(main_frame, text="🔍 Szczegóły", command=lambda: show_top3(), state="disabled")
button_details.pack(pady=5)
button_details.pack_forget()

threshold_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
threshold_frame.pack(pady=10)
ctk.CTkLabel(threshold_frame, text="Próg ufności (%)").pack(side="left", padx=(0, 10))
entry_threshold = ctk.CTkEntry(threshold_frame, width=80, justify='center')
entry_threshold.insert(0, str(THRESHOLD))
entry_threshold.pack(side="left")
ctk.CTkButton(threshold_frame, text="Zastosuj", command=lambda: apply_threshold(), width=100).pack(side="left", padx=(10, 0))

#obsługa GIF - ikonki ładowania
gif_frames = []
current_gif_frame = 0
gif_label = None
gif_running = False

last_prediction = None


def load_gif():
    global gif_frames
    spinner = Image.open("spinner.gif")
    gif_frames = [ImageTk.PhotoImage(frame.copy().resize((128, 128))) for frame in ImageSequence.Iterator(spinner)]

def start_gif_animation():
    global gif_running, gif_label
    gif_running = True
    gif_label = ctk.CTkLabel(label_image, text="")
    gif_label.place(relx=0.5, rely=0.5, anchor="center")
    animate_gif()


def animate_gif():
    global current_gif_frame, gif_running
    if not gif_running:
        return
    frame = gif_frames[current_gif_frame]
    gif_label.configure(image=frame)
    gif_label.image = frame
    current_gif_frame = (current_gif_frame + 1) % len(gif_frames)
    app.after(100, animate_gif)


def stop_gif_animation():
    global gif_running
    gif_running = False
    if gif_label:
        gif_label.destroy()


#klasyfikacja obrazu
def prepare_image(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)


def show_top3():
    global last_prediction
    if last_prediction is None:
        return
    top3_indices = np.argsort(last_prediction)[-3:][::-1]
    top3_classes = [(CLASSES[i], last_prediction[i]*100) for i in top3_indices]
    best_class = top3_classes[0][0]

    top_window = ctk.CTkToplevel()
    top_window.title("Szczegóły klasyfikacji")
    top_window.geometry("320x200")
    top_window.attributes("-topmost", True)

    for name, conf in top3_classes:
        weight = "bold" if name == best_class else "normal"
        label = ctk.CTkLabel(top_window, text=f"{name}: {conf:.1f}%", font=ctk.CTkFont(weight=weight))
        label.pack(pady=5)


def classify_and_display(file_path):
    global last_prediction
    time.sleep(0.3)
    img = Image.open(file_path).convert("RGB")
    img_array = prepare_image(file_path)
    pred = model.predict(img_array)[0]
    last_prediction = pred
    predicted_index = np.argmax(pred)
    predicted_class = CLASSES[predicted_index]
    confidence = pred[predicted_index] * 100

    stop_gif_animation()

    if confidence < THRESHOLD:
        label_result.configure(text=f"Nie rozpoznano gatunku", text_color="#C33535")
    else:
        label_result.configure(text=f"Gatunek: {predicted_class}", text_color="#38bdf8")

    button_details.configure(state="normal")
    button_details.pack()

    img_resized = img.resize((320, 320))
    img_ctk = ctk.CTkImage(light_image=img_resized, size=(320, 320))
    label_image.configure(image=img_ctk, text="")
    label_image.image = img_ctk


def classify_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Obrazy", "*.jpg *.jpeg *.png *.bmp *.webp")]
    )
    if not file_path:
        return

    label_result.configure(text=random.choice(LOADING_MESSAGES), text_color="#FFFFFF")
    label_image.configure(image="", text="")
    button_details.pack_forget()
    start_gif_animation()
    threading.Thread(target=classify_and_display, args=(file_path,)).start()


button_select.configure(command=classify_image)

#ustawianie progu
def apply_threshold():
    global THRESHOLD
    try:
        value = float(entry_threshold.get())
        if 0 <= value <= 100:
            THRESHOLD = value
            warning_msg = f"Ustawiono próg: {THRESHOLD:.1f}%"
            col = "#FFFFFF"
            if value < 50:
                warning_msg += "\n(Uwaga: niski próg może prowadzić do niepewnych wyników)"
                col = "#FFB300"
            label_result.configure(text=warning_msg, text_color=col)
        else:
            label_result.configure(text="Wartość powinna być 0–100", text_color="#FFFFFF")
    except ValueError:
        label_result.configure(text="Niepoprawna liczba", text_color="#FFFFFF")


#uruchomienie
load_gif()
app.mainloop()
