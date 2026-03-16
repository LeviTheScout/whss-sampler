import numpy as np

class sampling:
    def get_samples(self,f_r,alpha=0.1):
        """
        alpha: if we want k samples, we will check k+alpha%k samples. eg: alpha=0.01
        """
        def f_max_along_theta(r_vec,f_r):
            return
        
        possible_samples=[]
        maximums=[]
    
        for i in range(self.k+round(alpha*self.k)):
            theta=self.theta_generation(self.d)
            r_vec,R_=self.R(self.a,theta) 
            
            local_f_max=f_max_along_theta(r_vec,f_r)
            sampled_r=np.random.uniform(0,R_) #use importance sampling in R later
            f_value=(sampled_r**(self.d-1))*f_r(sampled_r)
            u=np.random.uniform(0,1)
            possible_samples.append((r_vec,u,sampled_r,f_value))
            maximums.append(local_f_max)
        emperical_f_max=np.max(np.array(maximums))
        accepted=[]
        rejected=[]
        for i in range(len(possible_samples)):
            if possible_samples[i][1]*emperical_f_max<=possible_samples[i][3]:
                accepted.append((possible_samples[i][0],possible_samples[2]))
            else:
                rejected.append((possible_samples[i][0],possible_samples[2]))

        # if i am get less than k samples, then i will do normal rejection sampling with emperical f_max for remaining samples, i will deploy the importance sampling in r, hence it should not come to this case more often.
        while len(accepted)<self.k:
            theta=self.theta_generation(self.d)
            r_vec,R_=self.R(self.a,theta)
            sampled_r=np.random.uniform(0,R_)
            sampled_f=np.random.uniform(0,emperical_f_max)
            if sampled_f<=(sampled_r**(self.d-1))*f_r(theta,sampled_r):
                accepted.append((theta,sampled_r))
            else:
                rejected.append((theta,sampled_r))
        return accepted,rejected
            
