# Rohitash Chandra, 2017 c.rohitash@gmail.com
# Cleaned by Ratneel Deo, 2018 deo.ratneel@gmail.com 

#!/usr/bin/python

# ref: http://iamtrask.github.io/2015/07/12/basic-python-network/  
 

#Sigmoid units used in hidden and output layer. gradient descent and stocastic gradient descent functions implemented with momentum. Note:
#  Classical momentum:

#vW(t+1) = momentum.*Vw(t) - scaling .* gradient_F( W(t) )
#W(t+1) = W(t) + vW(t+1)

#W Nesterov momentum is this: http://cs231n.github.io/neural-networks-3/

#vW(t+1) = momentum.*Vw(t) - scaling .* gradient_F( W(t) + momentum.*vW(t) )
#W(t+1) = W(t) + vW(t+1)

#http://jmlr.org/proceedings/papers/v28/sutskever13.pdf
 

# Numpy used: http://cs231n.github.io/python-numpy-tutorial/#numpy-arrays
 

 

import matplotlib.pyplot as plt
import numpy as np 
import random
import time
import Tkinter, tkFileDialog,  tkMessageBox

#An example of a class
class Network:

	def __init__(self, Topo, Train, Test, MaxTime, Samples, MinPer): 
		self.Top  = Topo  # NN topology [input, hidden, output]
		self.Max = MaxTime # max epocs
		self.TrainData = Train
		self.TestData = Test
		self.NumSamples = Samples

		self.lrate  = 0 # will be updated later with BP call

		self.momenRate = 0
		self.useNesterovMomen = 0 #use nestmomentum 1, not use is 0

		self.minPerf = MinPer
										#initialize weights ( W1 W2 ) and bias ( b1 b2 ) of the network
		np.random.seed() 
		
		self.W1 = np.random.randn(self.Top[0]  , self.Top[1])  / np.sqrt(self.Top[0] ) 
		#self.B1 = np.random.randn(1, self.Top[1])  / np.sqrt(self.Top[1] ) # bias first layer
		self.B1 = np.ones( self.Top[1]) # / np.sqrt(self.Top[1] ) # bias first layer

		self.BestB1 = self.B1
		self.BestW1 = self.W1 
		self.W2 = np.random.randn(self.Top[1] , self.Top[2]) / np.sqrt(self.Top[1] )
		#self.B2 = np.random.randn(1  , self.Top[2])  / np.sqrt(self.Top[1] ) # bias second layer
		self.B2 = np.ones( self.Top[2])  # bias second layer

		self.BestB2 = self.B2
		self.BestW2 = self.W2 
		self.hidout = np.zeros((1, self.Top[1] )) # output of first hidden layer
		self.out = np.zeros((1, self.Top[2])) #  output last layer

  
	def sigmoid(self,x):
		return 1 / (1 + np.exp(-x))

	def sampleEr(self,actualout):
		error = np.subtract(self.out, actualout)
		sqerror= np.sum(np.square(error))/self.Top[2] 
		#print sqerror
		return sqerror
  
	def ForwardPass(self, X ): 
		 z1 = X.dot(self.W1) - self.B1  
		 self.hidout = self.sigmoid(z1) # output of first hidden layer   
		 z2 = self.hidout.dot(self.W2)  - self.B2 
		 self.out = self.sigmoid(z2)  # output second hidden layer
	 
 
  
	def BackwardPassMomentum(self, Input, desired, vanilla):   
			out_delta =   (desired - self.out)*(self.out*(1-self.out))  
			hid_delta = out_delta.dot(self.W2.T) * (self.hidout * (1-self.hidout))
			
			if vanilla == 1: #no momentum 
				self.W2+= (self.hidout.T.dot(out_delta) * self.lrate)  
				#self.B2+=  (-1 * self.lrate * out_delta)
				self.W1 += (Input.T.dot(hid_delta) * self.lrate) 
				#self.B1+=  (-1 * self.lrate * hid_delta)
			  
			else:
				v2 = self.W2 #save previous weights http://cs231n.github.io/neural-networks-3/#sgd
				v1 = self.W1 
				b2 = self.B2
				b1 = self.B1 

				v2 = ( v2 *self.momenRate) + (self.hidout.T.dot(out_delta) * self.lrate)       # velocity update
				v1 = ( v1 *self.momenRate) + (Input.T.dot(hid_delta) * self.lrate)   
				v2 = ( v2 *self.momenRate) + (-1 * self.lrate * out_delta)       # velocity update
				v1 = ( v1 *self.momenRate) + (-1 * self.lrate * hid_delta)   

				if self.useNesterovMomen == 0: # use classical momentum 
					self.W2+= v2
					self.W1 += v1 
					#self.B2+= b2
					#self.B1 += b1 

				else: # useNesterovMomen http://cs231n.github.io/neural-networks-3/#sgd
					v2_prev = v2
					v1_prev = v1 
					self.W2+= (self.momenRate * v2_prev + (1 + self.momenRate) )  * v2
					self.W1 += ( self.momenRate * v1_prev + (1 + self.momenRate) )  * v1 

		   

	def TestNetwork(self, Data, testSize, erTolerance):
		Input = np.zeros((1, self.Top[0])) # temp hold input
		Desired = np.zeros((1, self.Top[2])) 
		nOutput = np.zeros((1, self.Top[2]))
		clasPerf = 0
		sse = 0  
		self.W1 = self.BestW1
		self.W2 = self.BestW2 #load best knowledge
		self.B1 = self.BestB1
		self.B2 = self.BestB2 #load best knowledge
	 
		for s in xrange(0, testSize):
				  
				Input[:]  =   Data[s,0:self.Top[0]] 
				Desired[:] =  Data[s,self.Top[0]:] 
				self.ForwardPass(Input ) 
				sse = sse+ self.sampleEr(Desired)  


				if(np.isclose(self.out, Desired, atol=erTolerance).any()):
					clasPerf =  clasPerf +1  

		return ( sse/testSize, float(clasPerf)/testSize * 100 )

 
	def saveKnowledge(self):
		self.BestW1 = self.W1
		self.BestW2 = self.W2
		self.BestB1 = self.B1
		self.BestB2 = self.B2  
 
	def BP_GD(self, learnRate, mRate,  useNestmomen , stocastic, vanilla, trainTolerance): # BP with SGD (Stocastic BP)
		self.lrate = learnRate
		self.momenRate = mRate
		self.useNesterovMomen =  useNestmomen  
	 
		Input = np.zeros((1, self.Top[0])) # temp hold input
		Desired = np.zeros((1, self.Top[2])) 
		Er = []#np.zeros((1, self.Max)) 
		epoch = 0
		bestmse = 100
		bestTrain = 0

		minibatchsize = int(0.30* self.NumSamples)
	
	
		#print '\n-------------------------------------------------\n'

		while  epoch < self.Max and bestTrain < self.minPerf :           
			sse = 0
			sample = []
			if(stocastic): # create a minibatch of samples 
				sample = np.array(self.TrainData).tolist()
				np.random.shuffle(sample)
				array = []
				for iteratable in xrange (0, minibatchsize):
					pat = random.randint(0, len(self.TrainData)-1)
					array.append(sample[pat])		   	
				sample = np.asarray(array)

				for s in xrange(0, len(sample)):
					Input[:]  =  sample[s,0:self.Top[0]]  
					Desired[:] = sample[s,self.Top[0]:]  
					self.ForwardPass(Input )  
					self.BackwardPassMomentum(Input , Desired, vanilla)
					sse = sse+ self.sampleEr(Desired)

			else:
				sample = self.TrainData
				Input[:] 
				Desired[:]
				for s in xrange(0, len(sample)):
					Input[:]  =  sample[s,0:self.Top[0]]  
					Desired[:] = sample[s,self.Top[0]:]  
					self.ForwardPass(Input )  
				self.BackwardPassMomentum(Input , Desired, vanilla)
				sse = sse+ self.sampleEr(Desired)
			

			
				 
			mse = np.sqrt(sse/len(sample)*self.Top[2])
			
			if mse < bestmse:
				bestmse = mse
				self.saveKnowledge() 
				(x,bestTrain) = self.TestNetwork(self.TrainData, self.NumSamples, trainTolerance)
				#print (epoch, bestTrain)
				  

			Er.append(mse)
 

			epoch=epoch+1  

		return (Er,bestmse, bestTrain, epoch) 



def normalisedata(data, inputsize, outsize): # normalise the data between [0,1]
	traindt = data[:,np.array(range(0,inputsize))]	
	dt = np.amax(traindt, axis=0)
	tds = abs(traindt/dt) 
	return np.concatenate(( tds[:,range(0,inputsize)], data[:,range(inputsize,inputsize+outsize)]), axis=1)



def main(): 
		
	f = open('out_run.txt', 'w')
	f1 = open('out_mean.txt', 'w')


	problem = 1 # [1,2,3] choose your problem 

	

	root = Tkinter.Tk()
	root.withdraw()

	# tkMessageBox.showinfo("Input", "Choose Training Data File")
	# tr_file_path = tkFileDialog.askopenfilename()

	# tkMessageBox.showinfo("Input", "Choose Testing Data File")
	# ts_file_path = tkFileDialog.askopenfilename()

	tr_file_path = "train.txt"
	ts_file_path = "test.txt"
	if problem == 1:

		TrDat  = np.loadtxt(tr_file_path) #  Iris classification problem (UCI dataset)
		TesDat  = np.loadtxt(ts_file_path) #  
		Hidden = 6
		Input = 4
		Output = 2
		learnRate = 0.1 
		mRate = 0.01   
		TrainData  = normalisedata(TrDat, Input, Output) 
		TestData  = normalisedata(TesDat, Input, Output)
		MaxTime = 1000




	TrSamples =  len(TrainData)
	TestSize = len(TestData)

	Topo = [Input, Hidden, Output] 
	MaxRun = 5#input("Enter Number of Experimental Runs : ") # number of experimental runs 
	 
	MinCriteria = 95 #stop when learn 95 percent

	trainTolerance = 0.2 # [eg 0.15 would be seen as 0] [ 0.81 would be seen as 1]
	testTolerance = 0.3

	useStocasticGD =  1#input("Choose 0 for Vanilla BP, or 1 for Stochastic BP : ")# 0 for vanilla BP. 1 for Stocastic BP
	useVanilla = 1 # 1 for Vanilla Gradient Descent, 0 for Gradient Descent with momentum (either regular momentum or nesterov momen) 
	useNestmomen = 0 # 0for regular momentum, 1 for Nesterov momentum

	trainPerf = np.zeros(MaxRun)
	testPerf =  np.zeros(MaxRun)

	trainMSE =  np.zeros(MaxRun)
	testMSE =  np.zeros(MaxRun)
	Epochs =  np.zeros(MaxRun)
	Time =  np.zeros(MaxRun)


	for run in xrange(0, MaxRun  ): 
		 print run
		 fnnSGD = Network(Topo, TrainData, TestData, MaxTime, TrSamples, MinCriteria) # Stocastic GD
		 start_time=time.time()
		 (erEp,  trainMSE[run] , trainPerf[run] , Epochs[run]) = fnnSGD.BP_GD(learnRate, mRate, useNestmomen,  useStocasticGD, useVanilla, trainTolerance)   

		 Time[run]  =time.time()-start_time


		 (testMSE[run], testPerf[run]) = fnnSGD.TestNetwork(TestData, TestSize, testTolerance)
		 
		 print (  trainPerf[run] , testPerf[run] , Epochs[run])

		 f.write(str(problem) + '\t')
		 f.write(str(run) + '\t' + str(trainPerf[run])+ '\t' + str(testPerf[run]) + '\t' + str( Time[run])+ '\n')

	print trainPerf 
	print testPerf
	#print trainMSE
	#print testMSE

	print Epochs
	print Time
	print(np.mean(trainPerf), np.std(trainPerf))
	print(np.mean(testPerf), np.std(testPerf))
	print(np.mean(Time), np.std(Time))
	
	f1.write(str(problem) + ' \t')
	f1.write('%5.2f'% np.mean(trainPerf) + ' +- ' +'%5.4f'% np.std(trainPerf) )
	f1.write(' \t'+ '%5.2f'% np.mean(testPerf) + ' +- ' +'%5.4f'% np.std(testPerf)  )
	f1.write(' \t'+ '%5.2f'% np.mean(Time) + ' +- ' +'%5.4f'% np.std(Time) +'\n' )
	


		
	plt.figure()
	plt.plot(erEp )
	plt.ylabel('error')  
	plt.savefig('out.png')
	#plt.show()
 
if __name__ == "__main__": main()

