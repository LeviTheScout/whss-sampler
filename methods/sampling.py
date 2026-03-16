import numpy as np

class sampling:

    def sampling_f_r(self,f_r):
        """
        The main sampling function utilising the concept of n-sphere Monte
        Carlo technique and rejection sampling.
        Returns:
            Two arrays, accepted samples and rejected samples.
        f_r: gives two things, a density function and f_max value in this one.
        
        Note: r^d-1 is alrady multplied in this f_r function.
        """
        
        
        density=f_r[0]
        f_max=f_r[1]+0.0
        #f_max give along with f_r , also doing +0,01 in f_max, might want to remove that later
        
        accepted=[]
        rejected=[]
        while len(accepted)<self.k:

            theta=self.theta_generation()
            _,R_=self.R(theta)
            
            a,b,total_mass=self.importance_r(density, R_,theta,0.01,0.98)
            #print(a,b,R_)
            sampled_r=np.random.uniform(a,b)
            #sampled_r=np.random.uniform(np.sqrt(self.d)-(5/np.sqrt(2)),np.sqrt(self.d)+(5/np.sqrt(2)))
            #sampled_r=np.random.uniform(0,R_)
            sampled_f=np.random.uniform(0,f_max)

            if sampled_f<=density(sampled_r,theta):
                accepted.append((theta,sampled_r))
            else:
                rejected.append((theta,sampled_r))
        
        return accepted,rejected

    


    def sampling_f_r_new(self,f_r,alpha=0.1):
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
            
