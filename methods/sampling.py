import numpy as np

class sampling:

    def sampling_f_r(self,f_r,batch_size=128):
        """
        The main sampling function utilising the concept of n-sphere Monte
        Carlo technique and rejection sampling.
        Returns:
            Two arrays, accepted samples and rejected samples.
        f_r: gives two things, a density function and f_max value in this one.
        
        Note: r^d-1 is alrady multplied in this f_r function.
        """
        
        
        density=f_r[0]
        f_max=f_r[1]+0.01
        #f_max give along with f_r , also doing +0,01 in f_max, 
        # might want to remove that later
        
        accepted_theta_list,accepted_r_list=[],[]
        rejected_theta_list,rejected_r_list=[],[]
        accepted_count=0
        while accepted_count<self.k:
            if np.abs(accepted_count-self.k)<batch_size:
                batch_size=2*np.abs(accepted_count-self.k)            

            theta_batch=self.theta_generation(batch_size)
            R_batch=self.R(theta_batch)
            a_batch,b_batch,total_mass_batch=self.importance_r(density,R_batch,theta_batch)
            sampled_r_batch=np.random.uniform(a_batch,b_batch)
            sampled_f=np.random.uniform(0,f_max,size=batch_size)
            density_vals=density(sampled_r_batch,theta_batch)
            mask= sampled_f <= density_vals
            accepted_theta_list.append(theta_batch[mask])
            accepted_r_list.append(sampled_r_batch[mask])
            rejected_theta_list.append(theta_batch[~mask])
            rejected_r_list.append(sampled_r_batch[~mask])
            print(np.std(a_batch),np.std(b_batch))
            accepted_count+=np.sum(mask)
        a_theta=np.concatenate(accepted_theta_list, axis=0)
        a_r=np.concatenate(accepted_r_list, axis=0)
        r_theta=np.concatenate(rejected_theta_list, axis=0)
        r_r=np.concatenate(rejected_r_list, axis=0)
        accepted=np.concatenate([a_theta, a_r[:, None]], axis=1)
        rejceted=np.concatenate([r_theta, r_r[:, None]], axis=1)

        return accepted,rejceted


        # while len(accepted)<self.k:
        #
        #     theta=self.theta_generation()
        #     R_=self.R(theta)
        #
        #     a,b,total_mass=self.importance_r(density, R_,theta,0.01,0.98)
        #     #print(a,b,R_)
        #     sampled_r=np.random.uniform(a,b)
        #     #sampled_r=np.random.uniform(np.sqrt(self.d)-(5/np.sqrt(2)),np.sqrt(self.d)+(5/np.sqrt(2)))
        #     #sampled_r=np.random.uniform(0,R_)
        #     sampled_f=np.random.uniform(0,f_max)
        #
        #     if sampled_f<=density(sampled_r,theta):
        #         accepted.append((theta,sampled_r))
        #     else:
        #         rejected.append((theta,sampled_r))
        #
        #return accepted,rejected

    


    def sampling_f_r_new(self,f_r,alpha=0.1):
        """
        alpha: if we want k samples, we will check k+alpha%k samples. eg: alpha=0.01
        """
        def f_max_along_theta(a,b,theta,f_r):

            return  

        
        possible_samples=[]
        maximums=[]
    
        for i in range(self.k+round(alpha*self.k)):
            theta=self.theta_generation(self.d)
            R_=self.R(self.a,theta)

            #a,b,total_mass=self.importance_r(f_r, R_,theta,0.01,0.98)
            
            local_f_max=f_max_along_theta(a,b,theta,f_r)
            sampled_r=np.random.uniform(0,R_) #use importance sampling in R , [a,b]
            f_value=(sampled_r**(self.d-1))*f_r(sampled_r,theta)
            u=np.random.uniform(0,1)
            possible_samples.append((theta,u,sampled_r,f_value))
            maximums.append(local_f_max)
        emperical_f_max=np.max(np.array(maximums))+0.01 #added a bit of buffer.
        accepted=[]
        rejected=[]
        for i in range(len(possible_samples)):
            if possible_samples[i][1]*emperical_f_max<=possible_samples[i][3]:
                accepted.append((possible_samples[i][0],possible_samples[2]))
            else:
                rejected.append((possible_samples[i][0],possible_samples[2]))

        '''
        if i get less than k samples, then i will do normal rejection sampling 
        with emperical f_max for remaining samples, i will deploy the importance sampling in r, 
        hence it should not come to this case more often.
        '''
        
        while len(accepted)<self.k:
            theta=self.theta_generation(self.d)
            R_=self.R(self.a,theta)
            sampled_r=np.random.uniform(0,R_)  #change to [a,b]
            sampled_f=np.random.uniform(0,emperical_f_max)
            if sampled_f<=(sampled_r**(self.d-1))*f_r(sampled_r,theta):
                accepted.append((theta,sampled_r))
            else:
                rejected.append((theta,sampled_r))
        return accepted,rejected
                
