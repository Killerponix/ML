import math

class kreis():

  def __init__(self,r,x,y):
      if r<0 :
          r=0
      self.r = r
      self.mx = x
      self.my = y
  
  def abstand(self,x,y):
      d = math.sqrt((self.mx-x)**2 + (self.my-y)**2)
      d = d - self.r;
      return(d)