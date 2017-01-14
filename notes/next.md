NEXT

- fix the dosing. i think it might be to do with phi/theta or distance
- need to login to a server and run the dose calculations over and over
- combination takes a long time, so fix that first


- describe the collimator better
- show quantitative results
- show in a realistic CT
- show dose histograms
- ideal stationary weighting
- ideal arc weighting


- collimator scad
	- 3 general shots
		- perspective
		- anode
		- phantom side

beam profiles
	https://www.dropbox.com/s/yzrlttlhvaffoye/Screenshot%202017-01-13%2014.29.17.png?dl=0
dvh
	https://www.dropbox.com/s/a8lw919o6fv6krd/Screenshot%202017-01-13%2014.30.21.png?dl=0
another dvh (oar)
	https://www.dropbox.com/s/r7rv9ijte2vlntx/Screenshot%202017-01-13%2014.30.56.png?dl=0
- dose volume histogram
	- find a good example
	- need a 3ddose file (preferrably .npz file) that we can work on
	- from there we generate the dvh
	- generate one ffs
- dose values table
	- figure out what the sheet says!
	- implement it, hopefully standalone
- beautiful weighting
