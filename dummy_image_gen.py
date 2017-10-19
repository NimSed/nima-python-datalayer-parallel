import numpy as np

class dummy_image_gen:

    def __init__(self):
        pass
    
    def generate(self,args):
        # unfold the input arguments
        seed,q = (args[0],args[1]);

        img = np.ones([100,100])*seed;
        
        if q is not None:
            q.put( (img,) )
        else:
            return (img,)



