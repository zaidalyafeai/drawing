import json 
from matplotlib import animation
import math
import matplotlib.pyplot as plt
import numpy as np
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

def convert_format(drawing):
    new_data = []
    for item in drawing:
        char = list(item.keys())[0]
        stroke = item[char]
        xs = []
        ys = []
        segments = []
        for i, point in enumerate(stroke):
            x, y = point
            xs.append(500 - x)
            ys.append(y)
        new_data.append([xs, ys])
    return new_data

def convert_format_normalized(drawing):
    new_data = []
    xs = []
    ys = []
    x_prev, y_prev, _ = drawing[0]

    for x, y, z in drawing[1:]:
        xs.append(-(x + x_prev))
        ys.append(y + y_prev)
        if z:
          new_data.append([xs, ys])
          xs = []
          ys = []
        
        x_prev = x + x_prev 
        y_prev = y + y_prev
    return new_data

if __name__ == "__main__":
    drawing = np.load('data.npy')
    new_data = convert_format_normalized(drawing)
    create_animation(new_data)

# if __name__ == "__main__":
#     drawing = json.load(open('AbdulRasoul_عبد الرسول.json'))
#     new_data = convert_format(drawing)
#     create_animation(new_data)
