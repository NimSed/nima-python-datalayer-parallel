import sys, os
#sys.path.insert(0, os.environ['CAFFE_ROOT']+'/python')

import caffe
import numpy as np
import random
import multiprocessing
from multiprocessing import Process, Queue
import time

def generator_daemon(q,initial_parent_id,image_gen):

    def check_pid(pid):        
        """ Check For existence of a unix pid. """
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        else:
            return True
        
    m = 100
    
    while True:
       
        #-- check if parent is still alive
        ppid = os.getppid();
        if ppid != initial_parent_id :
            sys.exit('generator: parent changed => should commit suicide!')
        elif not check_pid(ppid):
            sys.exit('generator: parent not running any more!')        
            
        #-- check the current status of the queue
        sz = q.qsize()

        if sz < m/2:
            print(multiprocessing.current_process().name,'queue size: %d'%sz)

        if sz < m:
            argset = (random.randint(0,100000),q)
            image_gen.generate(argset)
        else:
            time.sleep(0.01);
            

class NimaParallelDataLayer(caffe.Layer):

    def setup(self,bottom,top):

        import random
        from datetime import datetime
        random.seed(datetime.now())

        #--- read parameters from `self.param_str`
        self.params = eval(self.param_str)

        #--- Create the image_gen object(s)
        go_parallel = self.params.get("go_parallel",False);
        if not go_parallel:
            self.background_scene = background_scene(self.params);
        else:
            pool_size = multiprocessing.cpu_count()-1
            self.background_scenes = [];
            for i in range(pool_size):
                self.background_scenes.append(background_scene(self.params));
            
        #-- if parallel, start the image generator daemons
        if go_parallel:
            self.manager = multiprocessing.Manager()
            self.q = self.manager.Queue()
            self.pool = multiprocessing.Pool(pool_size);
            
            from functools import partial
            g = partial(generator_daemon,self.q,os.getpid());

            self.pool.map_async(g,[back_scene for back_scene in self.background_scenes])

        #-- we also have an internal iteration counter which is probably only useful in non-parallel mode
        self.internal_iter_count = 0;
                                              
    def reshape(self,bottom,top):

        # no "bottom"s for input layer
        if len(bottom)>0:
            raise Exception('cannot have bottoms for input layer')

        # make sure you have the right number of "top"s
        if not len(top) in range(1,7):
            raise Exception('Number of tops for the nima_data_layer should be in range [1,7) !')

        for i in range(len(top)):
            top[i].reshape(self.params["batch_size"],1,self.params["final_image_size"],self.params["final_image_size"])
    
    def forward(self,bottom,top):

        go_parallel = self.params.get("go_parallel",False);
        batch_size = self.params["batch_size"];

        if go_parallel:
            for i in range(batch_size):
                imgs = self.q.get()
                for j in range(len(top)):
                    top[j].data[i,...] = imgs[j]
        else:
            for i in range(batch_size):

                scene_counter = batch_size * self.internal_iter_count + i;
                
                imgs = self.background_scene.generate((scene_counter,None)) #let's pass the scene_counter instead of seed in non-parallel mode
                for j in range(len(top)):
                    top[j].data[i,...] = imgs[j]


        # update the internal iteration counter
        self.internal_iter_count += 1;
        
        
    def backward(self, top, propagate_down, bottom):
        
        pass
