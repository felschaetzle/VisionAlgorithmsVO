import matplotlib.pyplot as plt
import numpy as np

class Plotter:
    # use ggplot style for more sophisticated visuals
    plt.style.use('ggplot')
    
    def __init__(self):
        # plt.ion()
        self.plot1 = plt.subplot2grid((2, 3), (0, 0), colspan=2)
        self.plot2 = plt.subplot2grid((2, 3), (0, 2), rowspan=2, colspan=2)
        self.plot3 = plt.subplot2grid((2, 3), (1, 1), rowspan=1)
        self.plot4 = plt.subplot2grid((2, 3), (1, 0), rowspan=1)

    def live_plotter(self,num,image,x_vec,y1_data,line1,identifier='',pause_time=0.2):
        if line1==[]:
            # this is the call to matplotlib that allows dynamic plotting
            plt.ion()
            # fig = plt.figure()
            # ax = fig.add_subplot(111)
            self.plot1 = plt.subplot2grid((2, 3), (0, 0), colspan=2)
            self.plot2 = plt.subplot2grid((2, 3), (0, 2), rowspan=2, colspan=2)
            self.plot3 = plt.subplot2grid((2, 3), (1, 1), rowspan=1)
            self.plot4 = plt.subplot2grid((2, 3), (1, 0), rowspan=1)

            # create a variable for the line so we can later update it
            line1, = self.plot3.plot(x_vec,y1_data,'-o',alpha=0.8)      
            #update plot label/title
            self.plot1.set_title('Current frame ({})'.format(num))
            self.plot2.set_title('Trajectory of last 20 frames')
            self.plot3.set_title('Full trajectory')
            self.plot4.set_title('Keypoints and candidates in last frames')
            self.plot1.grid(None)
            # Packing all the plots and displaying them
            plt.tight_layout()
            plt.show()
        else:
            self.plot3.set_xlim(np.floor(np.min(x_vec)-np.std(x_vec)),np.ceil(np.max(x_vec)+np.std(x_vec)))
            self.plot3.set_ylim(np.floor(np.min(y1_data-np.std(y1_data))),np.ceil(np.max(y1_data)+np.std(y1_data)))
        # after the figure, axis, and line are created, we only need to update the x- and y-data
        line1.set_data(x_vec,y1_data)
        self.plot1.imshow(image)
        # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
        plt.pause(pause_time)
        
        # return line so we can update it again in the next iteration
        return line1