import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import os
import io
import random
import glob
import math
import base64
import json
import numpy as np
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML
from linedraw import *
from IPython import display
import pickle
import svgwrite

def create_from_text(text):
  width = 256
  height = 256
  black = (0,0,0)
  white = (255,255,255)
  font = ImageFont.truetype("Humor-Sans.ttf",50)
  img = Image.new("RGBA", (width,height),white)
  draw = ImageDraw.Draw(img)
  w, h = draw.textsize(text, font)
  draw.text(((width-w)/2,(height-h)/2), text, black, font=font, stroke_width = 1)
  draw = ImageDraw.Draw(img)
  img.save("result.png")
 

def create_animation(drawing, fps = 30, idx = 0, lw = 5): 
  
  seq_length = 0 
  
  xmax = 0 
  ymax = 0 
  
  xmin = math.inf
  ymin = math.inf
  
  #retreive min,max and the length of the drawing  
  for k in range(0, len(drawing)):
    x = drawing[k][0]
    y = drawing[k][1]

    seq_length += len(x)
    xmax = max([max(x), xmax]) 
    ymax = max([max(y), ymax]) 
    
    xmin = min([min(x), xmin]) 
    ymin = min([min(y), ymin]) 
    
  i = 0 
  j = 0
  
  # First set up the figure, the axis, and the plot element we want to animate
  fig = plt.figure()
  ax = plt.axes(xlim=(xmax+lw, xmin-lw), ylim=(ymax+lw, ymin-lw))
  ax.set_facecolor("white")
  line, = ax.plot([], [], lw=lw)

  #remove the axis 
  ax.grid = False
  ax.set_xticks([])
  ax.set_yticks([])
  
  # initialization function: plot the background of each frame
  def init():
      line.set_data([], [])
      return line, 

  # animation function.  This is called sequentially
  def animate(frame):    
    nonlocal i, j, line
    x = drawing[i][0]
    y = drawing[i][1]
    line.set_data(x[0:j], y[0:j])
    
    if j >= len(x):
      i +=1
      j = 0 
      line, = ax.plot([], [], lw=lw)
      
    else:
      j += 1
    return line,
  
  # call the animator.  blit=True means only re-draw the parts that have changed.
  anim = animation.FuncAnimation(fig, animate, init_func=init,
                                 frames= seq_length + len(drawing), blit=True)
  plt.close()
  
  # save the animation as an mp4.  
  anim.save(f'video.mp4', fps=fps, extra_args=['-vcodec', 'libx264'])

  import svgwrite
from IPython.display import SVG, display

def get_bounds(data, factor=10):
  """Return bounds of data."""
  min_x = 0
  max_x = 0
  min_y = 0
  max_y = 0

  abs_x = 0
  abs_y = 0
  for i in range(len(data)):
    x = float(data[i, 0]) / factor
    y = float(data[i, 1]) / factor
    abs_x += x
    abs_y += y
    min_x = min(min_x, abs_x)
    min_y = min(min_y, abs_y)
    max_x = max(max_x, abs_x)
    max_y = max(max_y, abs_y)

  return (min_x, max_x, min_y, max_y)

def draw_strokes(data, factor=0.2, svg_filename = '/tmp/sketch_rnn/svg/sample.svg'):
  os.makedirs(os.path.dirname(svg_filename), exist_ok=True)
  min_x, max_x, min_y, max_y = get_bounds(data, factor)
  dims = (50 + max_x - min_x, 50 + max_y - min_y)
  dwg = svgwrite.Drawing(svg_filename, size=dims)
  dwg.add(dwg.rect(insert=(0, 0), size=dims,fill='white'))
  lift_pen = 1
  abs_x = 25 - min_x 
  abs_y = 25 - min_y
  p = "M%s,%s " % (abs_x, abs_y)
  command = "m"
  for i in range(len(data)):
    if (lift_pen == 1):
      command = "m"
    elif (command != "l"):
      command = "l"
    else:
      command = ""
    x = float(data[i,0])/factor
    y = float(data[i,1])/factor
    lift_pen = data[i, 2]
    p += command+str(x)+","+str(y)+" "
  the_color = "black"
  stroke_width = 1
  dwg.add(dwg.path(p).stroke(the_color,stroke_width).fill("none"))
  dwg.save()
  display(SVG(dwg.tostring()))

def add_z(data):
  new_data = []
  x_prev, y_prev = data[0][0]
  for segment in data:
    x_data = []
    y_data = []
    segments = []
    for i, point in enumerate(segment):
      x, y = point
      if i == len(segment) - 1:
        z = 1
      else:
        z = 0

      if i >=0:
        segments.append([x-x_prev, y-y_prev, z])
      x_prev, y_prev = [x, y]
    new_data += segments
  return np.array(new_data)
