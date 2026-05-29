import numpy as np
import matplotlib.pyplot as plt


class utilities:

    def theta_generation(self,batch_size):
        """
        This function generates the vectors of angles in d-dimsion length 
        being d-1. 
        - First (d-2) angles are uniformly sampled in [0,pi]
        - (d-1)th angle is sampeld uniformly in [0,2*pi]
        - it generates batch_size number of such samples.
        """
        samples=np.random.normal(0, 1, (batch_size,self.d))
        r = np.linalg.norm(samples,axis=1)
        return samples/r[:,None]
    

    def orthant_theta_generator(self,orthant_id,batch_size):
        """
        generate random uniform directions, change signs to match the signs of the orthants exactly. 
        """
        batch_size=np.ceil(batch_size).astype(int)
        # 1. unpack original orthant id in 1s and 0s
        original_orthant_id=np.unpackbits(orthant_id)
        # 2. generate batch of random directions.
        theta_batch=self.theta_generation(batch_size)
        # 3. change signs in batch so that it matches that of given orthant.
        target_bits=original_orthant_id[:self.d]
        target_bits=target_bits.astype(int)
        target_signs=(target_bits*2)-1 #mapping 0,1 to -1,1
        orthant_thetas=np.abs(theta_batch)*target_signs
        return orthant_thetas 

    def R(self,thetas):
        """
        Returns the maximum length 'R' along that direcions of the 
        thetas given in the batch for CUBE of side length a.
        thetas: it is in Cartesian coordinate system, not spherical. 
        """
        inf_norm = np.max(np.abs(thetas),axis=1)
        R_vec=self.a/(2*inf_norm)
        return R_vec


    def get_orthant(self,thetas):
        mask= thetas>=0
        #axis=1 : for whole batch of theta, it makes sure we look at them horizontally along columns. 
        # packbits: for each 8 elements in array---0s,1s a 8bit number, it assigns a decimal number based on that 8 bit array.
        orthant_ids=np.packbits(mask,axis=1)

        #check orthants of two points using np.array_equal(id_a,id_b)
        return orthant_ids
        
    def projection_testing(self,accepted,g_r):
        """
        - Generte random direction.
        - Take projection of samples along that direction.
        - Projection of density (dont know how to do this).
        - measure distance (Primarily KS, but could use others too).
        """

        return
        
    def x_y_view(self, accepted):
        """
        This function visualizes the 2D projection of samples generated in
        d-dimensional Cartesian directional form.

        Parameters:
            accepted : list of tuples (a, r)
                a : unit direction vector (array-like of length d)
                r : radial distance
        """

        x_accepted, y_accepted = [], []

        for i in accepted:
            a=np.array([accepted[i][0][:2] for i in range(len(accepted))])
            r=np.array([accepted[i][-1] for i in range(len(accepted))]) 
            
            corrdinates=r[:,None] * a
            x_accepted.append(corrdinates[:,0])
            y_accepted.append(corrdinates[:,-1])
            
            # if len(a) > 1:
            #     y_accepted.append(r * a[1])
            # else:
            #     y_accepted.append(0)

        # is this correct projection corrdinates?
        plt.figure(figsize=(8, 8))
        plt.plot([-(self.a/2), (self.a/2), (self.a/2), -(self.a/2), -(self.a/2)], [-(self.a/2), -(self.a/2), (self.a/2), (self.a/2), -(self.a/2)], color='black', lw=2)
        r_boundary = self.a 


        plt.scatter(x_accepted, y_accepted, color='green', s=10 )
        plt.gca().set_aspect('equal')
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.title("Projection of d-dimsional Samples onto 2D Plane")
        plt.show()
        return
