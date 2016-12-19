## Game Plan

- design 3 distinct collimators:
	- current design (review motivations/style, possibly diagram it)
	- half target with half beams, with focus on edges...?
	- ?


PART I
1) for each beamlet produce each angle of dose contribution IN PROGRESS
2) produce summed 3ddose for basic beam
	similar to sample_combine, create if necessary, what about error rate? forget it for now
3) produce summed 3ddose for arc therapy
	similar to sample_combine, create if necessary, what about error rate? forget it for now
4) produce matplotlib contour plot like magdalena's from compressed 3ddose
	it would help if we had the damn files, but guess_weights has them
	so let's see what we can do :)
5) automatically embed contour plots like magdalena's into report
6) calculate and report target-to-skin ratio for non-arc and arc therapy respectively
7) calculate and report Paddick's conformation number
8) create 3 fundamentally different collimator designs with 3 variations each and the reports
9) apply beam weighting techniques to the 9 above:
	- one 2nd order poly and just play with it
	- non negative least squares to increase paddick's conformation number
	- non negative least squares to reduce target-to-skin ratio

arc weighting? not sure. arc weighting we leave for now, but we produce the components necessary

PART II
1) edit this list based on feedback
2) generate beam profiles like kVRT_mbc_manuscript pg16 and put into report
3) generate DVH like magdalena's and put into report
4) write custom distance averaging per Wu's conformation distance index
5) embed diagram(s) of collimator design
6) add more beamdp/grace plots
7) add arguments used to produce grace plots
8) add separate diagrams for beam weighting

PART III
1) 

NICE TO HAVES
- beamlet rotation should use pathos.multiprocessing

- calculate and report target-to-skin ratio
- generate beam profiles a la magdalena's
- generate DVH a la magdalena's
- generate contour plots in report per magdalena's using matplotlib
- conformation number per paddick's
- write custom surface distance averaging mechanism
- diagram(s) of collimator (matplotlib?)
- select more beamdp/grace plots and
	- small chunk of collimator row
	- anything that looks interesting :)
	- with arguments!
- additionally, diagrams after beam weighting

- more collimators
- more weighting
