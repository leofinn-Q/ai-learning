import tkinter as tk
from time import strftime


def update_time():
    current_time = strftime("%H:%M:%S")
    current_date = strftime("%Y-%m-%d %A")
    time_label.config(text=current_time)
    date_label.config(text=current_date)
    root.after(1000, update_time)


root = tk.Tk()
root.title("数字时钟")
root.resizable(False, False)
root.configure(bg="#1a1a2e")

time_label = tk.Label(
    root,
    font=("Helvetica", 72, "bold"),
    bg="#1a1a2e",
    fg="#e94560",
)
time_label.pack(padx=40, pady=(30, 10))

date_label = tk.Label(
    root,
    font=("Helvetica", 20),
    bg="#1a1a2e",
    fg="#a2a8d3",
)
date_label.pack(padx=40, pady=(0, 30))

update_time()
root.mainloop()