try:
    import pyglet
except ImportError as e:
    reraise(suffix="HINT: you can install pyglet directly via 'pip install pyglet'. But if you really just want to install all Gym dependencies and not have to think about it, 'pip install -e .[all]' or 'pip install gym[all]' will do it.")

try:
    from pyglet.gl import *
except ImportError as e:
    reraise(prefix="Error occured while running `from pyglet.gl import *`",suffix="HINT: make sure you have OpenGL install. On Ubuntu, you can run 'apt-get install python-opengl'. If you're running on a server, you may need a virtual frame buffer; something like this should work: 'xvfb-run -s \"-screen 0 1400x900x24\" python <your_script.py>'")

from gym.envs.classic_control.rendering import Geom
import os

class ImageData(Geom):
    def __init__(self, array, width, height):
        Geom.__init__(self)
        self.width = width
        self.height = height
        img = pyglet.image.ImageData(width, height, 'G', array.tobytes())
        self.img = img
        self.flip = False
    def render1(self):
        self.img.blit(-self.width/2, -self.height/2, width=self.width, height=self.height)

class MovieCapture(object):

    def __init__(self, dirname):
        assert os.path.isdir(dirname), (
            "%s is not a valid directory" % dirname)
        self.dirname = dirname
        self.frame_no = 0

    def capture(self):
        """Capture the current window a file, 
        return the file name"""
        filename = 'frame-%05d.png'
        while os.path.exists(os.path.join(
            self.dirname, filename % self.frame_no)):
            self.frame_no += 1
        mgr = pyglet.image.get_buffer_manager()
        mgr.get_color_buffer().save(
            os.path.join(self.dirname, 
            filename % self.frame_no))
        return filename % self.frame_no

class GrayImageViewer(object):

    def __init__(self, display=None):
        self.window = None
        self.isopen = False
        self.display = display

    def imshow(self, arr):
        if self.window is None:
            height, width= arr.shape
            self.window = pyglet.window.Window(width=width, height=height, display=self.display)
            self.width = width
            self.height = height
            self.isopen = True
        assert arr.shape == (self.height, self.width), "You passed in an image with the wrong number shape"
        image = pyglet.image.ImageData(self.width, self.height, 'L', arr.tobytes())
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        image.blit(0,0)
        self.window.flip()
    def close(self):
        if self.isopen:
            self.window.close()
            self.isopen = False
    def __del__(self):
        self.close()