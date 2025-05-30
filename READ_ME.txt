
NOTE: This repo only has a PORTION of the code base.

--->>>> Previous Scripts before first commit are not included for cleanliness.


 Functionality:
 
 
	BASIC FOR ALL:
	-) Filter Order???
	-) Circuit Repersentation
 
	
	BASIC Electrical (CONVENTIONAL)
	-) Generate G-Coefs
	-) Scale Coef to filter values
	-) Generate s-parameters from circuit elements
	

	BASIC ML:
	-) Simulate circuit quickly on GPU
	-) Compare output of circuit response, SO you can back prop through it...
	
	
	
	Limitations:
		-) 5th order ladder (series first) { Exact topology defined }
		-) Frequency range limited ( 0 - 5 GHz )
		-) Chebyshev data real data ONLY
		-) All computation done in Python
		-) s-parameters repersented as data points ( 1000 )
		-) limited to 50 ohm enviroment
		
		
		
	Generate Training Sets:
		-) Compute G-coefs
				-> filter type (chebyshev, butterworth)
				-> ripple (chebyshev)
		-) Element Scaling
				-> filter type (LP, HP, BP)  {BP only here}
				-> center freq
				-> Fractional BW
				
				
	Python Libraries:
		*) numpy
		*) pandas
		*) matplotlib
		*) pyTorch
		*) torchViz
		*) notebook
		
	Virtual_Env
	
	Functions:
		*) Filters.py
			-) circuit repersentation
			-) G-coefs
			-) Element Scaling
			
		*) Simulator.py
			-) convention sim
			-) GPU sim
	
	Generate Training Data
		*) Gen_data.py
		
	Training_Data
		*) ONLY DATA SAVED HERE
		
	ML_work
		*) AI Stuff goes here...
		
		
		
		



