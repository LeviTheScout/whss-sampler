import numpy as np
import matplotlib.pyplot as plt


class utilities:

    def theta_generation(self):
        """
        This function generates the vector of angles in d-dimsion length 
        being d-1. 
        - First (d-2) angles are uniformly sampled in [0,pi]
        - (d-1)th angle is sampeld uniformly in [0,2*pi]
        """
        sample=np.random.normal(0, 1, self.d)
        r = np.linalg.norm(sample)
        return sample/r
    

    def R(self,theta):
        """
        This function genertes unit vecotor r along the randomly generated
        angle theta in dimisional space. 
        Also, returns the maximum length 'R' along that direcion in the cube.
        """
        x=np.array(theta)
        inf_norm = np.max(np.abs(x))
        return (self.a / (2 * inf_norm))
    

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

        for a, r in accepted:

            a = np.asarray(a)

            x_accepted.append(r * a[0])

            if len(a) > 1:
                y_accepted.append(r * a[1])
            else:
                y_accepted.append(0)

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
