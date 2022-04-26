## Image Segmentation

Hi there!

The reuqired packages can be installed with the ``requirement.txt``

My Segmentation program has the following arguments:

- `-f FILEPATH` (required) file to the image that should be segmented
- `-r INT` (required) the radius r
- `-pos` (optional) if you want to use positional information 
-  `-p` (optional) if you would like to use my pre-processing
- `-b` (optional) if you want to use Basin  of Attraction
- `-c INT` (optional) the wished c parameter. If parsed, the second optimization is selected
- default is classic mean shift --> expect it to take up to 4 min for a 200x200 image

An example could be: ``-f exp2.jpg -c 4 -r 25 -p``

The program creates the segmented image in your current folder. Happy segmenting!

