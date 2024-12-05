import tkinter as tk
from PIL import Image, ImageTk
import os
import json

json_file = "ground_truth.json"
data = {}
with open(json_file, "w") as f:
    json.dump(data, f, indent=4)

coordinates = []
image_name = None

def on_click(event):
    """Save coordinates and draw nodes/edges."""
    global coordinates
    x, y = event.x, event.y
    coordinates.append([x, y])
    canvas.create_oval(x-5, y-5, x+5, y+5, fill="red", outline="black")  # Node
    if len(coordinates) > 1:
        canvas.create_line(coordinates[-2][0], coordinates[-2][1], x, y, fill="blue", width=2)
    print(f"Coordinates: ({x}, {y})")

def on_key(event):
    """Write JSON and load next image on 'Enter' key."""
    global coordinates, image_name
    if event.keysym == "Return":
        if len(coordinates) > 2:
            shape_type = classify_shape(len(coordinates))
            formatted_coords = ", ".join([f"[{x}, {y}]" for x, y in coordinates])
            data[image_name] = {
                "type": shape_type,
                "coordinates": formatted_coords
            }
            with open(json_file, "w") as f:
                json.dump(data, f, indent=4)
            print(f"{image_name}: {shape_type}, {formatted_coords}")
        coordinates = []
        canvas.delete("all")
        load_next_image()

def classify_shape(corner_count):
    """Detect shape from corner numbers."""
    shapes = {3: "Triangular", 4: "Rectangular", 5: "Pentagonal", 8: "Octagonal"}
    if corner_count > 8:
        return "Circle"
    elif corner_count < 3:
        return "Unknown"
    return shapes.get(corner_count, f"{corner_count}-sided Polygon")

def load_next_image():
    """Load next image."""
    global image_name, coordinates
    if image_files:
        image_name = image_files.pop(0)
        img_path = os.path.join(folder_path, image_name)
        
        img = Image.open(img_path)
        img = img.resize((800, 600))
        tk_img = ImageTk.PhotoImage(img)
        
        canvas.image = tk_img
        canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
        print(f"Uploaded image: {image_name}")
    else:
        print("All images done.")
        root.destroy()

folder_path = "traffic_Data/Data/mix"

if os.path.exists(folder_path):
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print("No image found in folder!")
    else:
        root = tk.Tk()
        canvas = tk.Canvas(root, width=800, height=600)
        canvas.pack()

        canvas.bind("<Button-1>", on_click)
        root.bind("<Key>", on_key)

        load_next_image()

        root.mainloop()
else:
    print(f"No folder found: {folder_path}")
