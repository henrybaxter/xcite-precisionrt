## Game Plan

- questions for magdalena
- collimator diagram

1) RUN SIMULATION WITH MULTIPLE ROWS, 3 AND 5, on EGS1
2) MULTIPLE TARGET SIZES
3) AUTO-WEIGHT
4) COLLIMATOR DIAGRAM
5) QUESTIONS FOR MAG

- keyed beamlet stats
- keyed grace plots
- don't read 3ddose files over and over again
- multiple rows!
- links to all reports
- two-sided more ideas
- multiple target sizes
- beam profiles and dvh
- grays in 30 minutes
- check email of recap
- automatically fit a poly to 'smooth' the skin curve
- think hard about new collimator designs




collimator figure from design
most important right now?
- beam weighting
- isodose difference / beam profile
- 

- put results / efficiency / conformity at end of report
- show x ray tube layer materials and sizes
- embed collimator design diagrams
- check target to skin ratio (does it change predictably?)
- check paddicks conformity index (possibly describe what it means)
- constructed weighted dose and weighted arc dose

- design 3 distinct collimators:
	- current design
	- half target with half beams


PART I
6) calculate and report target-to-skin ratio for non-arc and arc therapy respectively
7) calculate and report Paddick's conformation number
8) create 3 fundamentally different collimator designs with 3 variations each and the reports
9) apply beam weighting techniques to the 9 above:
	- one 2nd order poly and just play with it
	- non negative least squares to increase paddick's conformation number
	- non negative least squares to reduce target-to-skin ratio


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
