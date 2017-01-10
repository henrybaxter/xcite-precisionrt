reflection: 1.9 per file => 10gb
filter: .36 gb per file => 1.7gb
collimator: 1.8 mb per file => 10mb





lesion sizes
1cm target
2cm target
4cm target

beam height and #rows
2mm single row
10mm five rows
10mm single row

collimator length = 10
collimator to lesion = 40

ideal beam weight
ideal beam angle

~500 gigabytes

AMI to boot up a new instance very quickly
script to boot up a new once from AMI and associate it properly for ssh, have it start a job, save results, and stop the instance.

can we put an AMI on an ebs volume? which volume do we associate it with? what if we have multiple spot instances?

what about the weighting? how do we make it a flat energy fluence?

we could weight it 'already', and/or make it a sinusoidal guy.

flat energy fluence would give us very similar beamlet sizes.

now do we want smaller beamlets? less data? how would that look?

gun distance.

so we can get much larger individual files

50mb/s over say 10gb is 200 seconds, so that's still pretty quick actually.

uploading 1tb at 12mb/s is 23 hours. jesus.

$100/month, that's not too terrible.

we

so what if we had less of them, what about a 75cm target with 75 beams each one 1cm. then 5 degree angle increments, each with equal probability, and over 120 degrees we would get 24

so 75 * 24 = 1800 simulations.

now to get a similar number of particles, we would use 135 million electrons per beam. that seems more reasonable.

now, what about parallelization
if we have 74 beams, we have overlap here.

so we need to let it keep going don't we.

16 32 64 80. that seems pretty good

we could generate the simulations, and put them through a multiprocessing 'pipe' so that one core is doing each one.

now, the steps are to:

1) reflection target result
2) translate it
3) rotate it
4) filter it
5) collimate it
6) run dose calculations through 24 arcs of 5 degrees each

so we need to generate the various simulations as we go

so they should operate on a single beamlet. that makes a lot of sense :)

then we can run the combination of each set at the end, when *everything* is done.

as for stats etc, we include a meta file .json for each one. this includes information about the file that might take a while to generate

now we also make it fault tolerant by running a simulation using a t prefix on all the filenames, and only when the simulation is finished and the meta information has been gathered (verifying the integrity of the file) do we move it
to the 'correct' filename and generate the json. 

no, instead we always generate the file, but we write the json file when we're done. and a version of the metadata?

we could write the entire json file, what if it gets out of sync?

so we write it, by convention, once the file has been completely finalized.

ok

this should also increase the likelihood that we're spreading our file munging stuff around, and it may even be in-os-cache

when it comes time to sample and combine though, it could get difficult...

if we randomize the file when we rotate it or whatever, w could end up . hmmm, how do you randomize a file? scary.

but i think we did that.

anyway, if we randomize like that, does that help? no. so we don't bother.

but by doing this, we ensure we use the server for the whole time.

now by running less simulations, do we gain anything? we get better stats=-09990009

sampling and combining so that we have a nice plot of it?


analyze magdaelena's code?

build the 'obvious' version that converges to a single focal spot using a single row width equal to 2mm
build the 'obvious' version that converges to a single focal spot using 5 rows, row width 10mm / 5

then converge to a range of points along the lesion (unweighted, just linear focus points)
then converge to a range of points along the lesion (weighted, run a few)

next to the 'whole lesion'


1) cover the target as normal
2) what about divergence in all directinos? does that happen? yes, and it should.
3) 
