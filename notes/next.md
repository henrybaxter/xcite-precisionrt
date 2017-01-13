NEXT

- grace
	- hashed
	- toml
	- standalone
- collimator scad
	- 3 general shots
		- perspective
		- anode
		- phantom side
- dose volume histogram
	- find a good example
	- generate one ffs
- dose values table
	- figure out what the sheet says!
	- implement it, hopefully standalone
- beautiful weighting
- create directory in s3 at start of run
- check for directory in s3 and abort if there
- upload report to directory when done
- also upload simulation.toml and attendant files


how do we run a specific simulation? do we bother?
on a system we run a 'job finder'
basically it picks a job, checks if the directory is there, if so it moves on
keeps going until it finds one, and it reserves it
the only problem is
yes there is a race condition here, however, in practice it should rarely be a problem
in the file, the thing records its name
