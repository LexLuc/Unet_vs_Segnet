# Unet_vs_Segnet 
> README FOR ASSIGNMENT2, COMP9417, 2018 S1

## Structure of Submitted Files
	| report.pdf
	| files.zip
	|    | unet.py    	"Implementation of U-Net"
	|    | segnet.py	"Implementation of SegNet"


## External URL
Because of the size limitation of submission system, dataset and large-sized model file (*.hdf5) has to be uploaded to 
3rd-party cloud drive service. Rather than instruct you to combine and reconstruct the submitted file both in CSE Give system 
and cloud drive by yourself, we've already prepared a full version of our project file structure on Dropbox. The code are 
absolutely the same as that submitted on CSE server. Here is the link:

https://www.dropbox.com/sh/2untz9nygcoly9f/AAA7rCpNbigqm2bLkDcL3q_ya?dl=0


## Instruction of Running Program
### Environment
After the preparation of required data, the next step is to provide an environment to run the code.

	python=3.6
	matplotlib=2.0.2
	numpy=1.12.1
	keras=2.1.6
	scikit-learn=0.19.1
	scikit-image=0.13.1
	tensorflow=1.5.0
		
### Running Code
If both data and environment are ready, then it is time to run and test the code.
Note: Because the training process requires quite long time, you would better run it on a powerful GPU machine.

Command line:
`$python segnet.py`

`$python unet.py`

### Inspecting Result
When we have finished the pipleline (training data - predicting data - validate performance), the result should be found
in directory:

* Predicted masks: ./data/test/label/car/*.gif
* Ground Truth: ./data/test/gt/*.gif

* Accuracy/Loss Curve of U-Net: ./loss_dicecoef_segnet.png
* Accuracy/Loss Curve of SegNet: ./loss_dicecoef_segnet.png
		
