import tkinter as tk

def detect_screen_size():
    root = tk.Tk()
    screen_width = int(root.winfo_screenmmwidth()/10)
    screen_height = int(root.winfo_screenmmheight()/10)
    
    root.destroy()
    return (screen_width, screen_height)

