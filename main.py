from visualizer.GUI import GUI

if __name__ == "__main__":
    size = 0.7
    gui = GUI("./video/input.avi", size=(size, size))
    gui.run()
