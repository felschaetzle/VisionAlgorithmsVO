import matplotlib.pyplot as plt
import numpy as np

class Plotter:
    # use ggplot style for more sophisticated visuals
    plt.style.use('ggplot')
    
    def __init__(self):
        self.ax = []
        self.plot1 = plt.subplot2grid((2, 3), (0, 0), colspan=2)
        self.plot2 = plt.subplot2grid((2, 3), (0, 2), rowspan=2, colspan=2)
        self.plot3 = plt.subplot2grid((2, 3), (1, 1), rowspan=1)
        self.plot4 = plt.subplot2grid((2, 3), (1, 0), rowspan=1)
        self.num_arr = np.zeros(20)
        self.number_of_candidates = np.zeros(20)
        self.number_of_keypoints = np.zeros(20)
        self.line2 = []
        self.line3 = []
        self.line4 = []
        self.xmin2 = 0
        self.xmax2 = 0
        self.xmin3 = 0
        self.xmax3 = 0
        self.ymin2 = 0
        self.ymax2 = 0
        self.ymin3 = 0
        self.ymax3 = 0
        self.ymin4 = 0
        self.ymax4 = 0
        

    def live_plotter(self, num, image, candidate_keypoints, keypoints, x_vec, y_vec, line1, pause_time=0.1):
        if line1==[]:
            # Call to matplotlib that allows dynamic plotting
            plt.ion()
            self.plot1 = plt.subplot2grid((2, 3), (0, 0), colspan=2)
            self.plot2 = plt.subplot2grid((2, 3), (0, 2), rowspan=2, colspan=2)
            self.plot3 = plt.subplot2grid((2, 3), (1, 1), rowspan=1)
            self.plot4 = plt.subplot2grid((2, 3), (1, 0), rowspan=1)

            # Create a variable for the line so we can later update it
            line1, = self.plot1.plot(keypoints[:], keypoints[:], 'gx', alpha=0.8)
            self.ax = self.plot1.imshow(image, cmap="gray", vmin=0, vmax=255)
            self.line2, = self.plot2.plot(x_vec, y_vec,'-bo', alpha=0.8)      
            self.line3, = self.plot3.plot(x_vec, y_vec,'-bo', alpha=0.8, markersize=1)
            self.line4, = self.plot4.plot(self.num_arr, self.number_of_candidates, '-rx', alpha=0.8, label='candidates')
            self.line5, = self.plot4.plot(self.num_arr, self.number_of_keypoints, '-gx', alpha=0.8, label='keypoints')

            # Update plot label/title
            self.plot1.grid(None)
            self.plot2.set_title('Zoomed in trajectory')
            self.plot3.set_title('Full trajectory')
            self.plot4.set_title('Keypoints and candidates in last frames')
            self.plot4.set_xlabel('frame number')
            self.plot4.legend()
            self.ymin4 = 0
            self.ymax4 = 0

            plt.tight_layout()
            plt.show()
            
        else:
            line1.set_data(keypoints[:,0], keypoints[:,1])
            if x_vec[-1]<self.xmin2 or x_vec[-1]>self.xmax2:
                self.xmin2 = np.floor(np.min(x_vec[-15:num+1])-np.std(x_vec[-15:num+1]))
                self.xmax2 = np.ceil(np.max(x_vec[-15:num+1])+np.std(x_vec[-15:num+1]))
                self.plot2.set_xlim(self.xmin2,self.xmax2)
                self.plot2.set_aspect('equal','datalim')
                self.plot2.relim()
            if y_vec[-1]<self.ymin2 or y_vec[-1]>self.ymax2:
                self.ymin2 = np.floor(np.min(y_vec[-15:num+1])-np.std(y_vec[-15:num+1]))
                self.ymax2 = np.ceil(np.max(y_vec[-15:num+1])+np.std(y_vec[-15:num+1]))
                self.plot2.set_ylim(self.ymin2,self.ymax2)
                self.plot2.set_aspect('equal','datalim')
                self.plot2.relim()
            if x_vec[-1]<self.xmin3 or x_vec[-1]>self.xmax3:
                self.xmin3 = np.floor(np.min(x_vec)-np.std(x_vec))
                self.xmax3 = np.ceil(np.max(x_vec)+np.std(x_vec))
                self.plot3.set_xlim(self.xmin3,self.xmax3)
                self.plot3.set_aspect('equal','datalim')
                self.plot3.relim()
            if y_vec[-1]<self.ymin3 or y_vec[-1]>self.ymax3:
                self.ymin3 = np.floor(np.min(y_vec)-np.std(y_vec))
                self.ymax3 = np.ceil(np.max(y_vec)+np.std(y_vec))
                self.plot3.set_ylim(self.ymin3,self.ymax3)
                self.plot3.set_aspect('equal','datalim')
                self.plot3.relim()

        self.ax.set_data(image)
        self.plot1.set_title('Current frame ({}) with keypoints'.format(num+1))
        self.line2.set_data(x_vec[-25:num+1], y_vec[-25:num+1])
        self.line3.set_data(x_vec, y_vec)
        self.num_arr = np.append(self.num_arr[1:], num+1)
        self.number_of_candidates = np.append(self.number_of_candidates[1:], len(candidate_keypoints))
        self.number_of_keypoints = np.append(self.number_of_keypoints[1:], len(keypoints))
        self.line4.set_data(self.num_arr, self.number_of_candidates)
        self.line5.set_data(self.num_arr, self.number_of_keypoints)
        self.plot4.set_xlim(num-20,num+1)
        if self.number_of_keypoints[-1]<self.ymin4 or self.number_of_candidates[-1]>self.ymax4:
                self.ymin4 = np.floor(np.min(self.number_of_keypoints))
                self.ymax4 = np.ceil(np.max(self.number_of_candidates))
                self.plot4.set_ylim(self.ymin4, self.ymax4)

        self.plot4.relim()

        # This pauses the data so the figure/axis can catch up - can not be 0
        plt.pause(pause_time)

        # Return line so we can update it again in the next iteration
        return line1