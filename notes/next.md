NEXT

need to take those collimated beamlets and construct the right stuff out of them:


if we get non-equal groups, what does that mean? the 3ddose stuff is ok
so why not just use non-equal groups? this normalization will continue to work.
ok, so we just divide by 24 and hope for the best?
and what about reflection? we can use reflection to some degree. but for now, let's stick with not
reflecting the phantom
1) combine them in equal groups. it is essential we have

1) stop running so many damn dosxyzs
	- use 0.2cm / 2mm beam width
	- use 0.1cm / 1mm beam width
	- use combinations that reduce it to 2cm. now since we center it, we end up with 1-3, 3-5, ... etc
	- so then we'd have 37 different spots, built out of many different beamlets
	- now when we combine them all together, we end up with the correct number...
	- ok so now

(3) beam profile plot
	(3a) with depth
	(4a) with width (both ways)
(4) weighting
	(4a) stationary
	(4b) arc
(5) skin to target
	2cmx2cmx4mm (of skin!)
(6) screenshots
	3 shots in report OR STL file with tikz
(7) realistic CT


- model electron beam with no overlap
- model electron beam with more beamlets
- wording of collimator geometry
- geometry of entire monte carlo setup
- are there more sophisticated models we can use?



DROP the conformity index

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
