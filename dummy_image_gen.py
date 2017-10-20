import numpy as np

class dummy_image_gen:

    def __init__(self,params):
        # Extract the necessary parameter values
        self.w = params["w"]
        self.h = params["h"]
    
    def generate(self,args):
        # unfold the input arguments
        seed,q = (args[0],args[1]);

        img = np.ones([self.h,self.w])*seed;
        
        if q is not None:
            q.put( (img,) )
        else:
            return (img,)



