from visualizer.GUI import GUI

if __name__ == "__main__":
    size = 0.6
    gui = GUI("./video/input.avi", size=(size, size))
    gui.run(save_video=True)
